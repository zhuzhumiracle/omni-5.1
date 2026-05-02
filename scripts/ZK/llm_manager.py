import os
import re
import time
import threading
import torch
import importlib.util
from openai import OpenAI
from tqdm import tqdm
from llm_config import get_llm_settings

class LLMManager:
    def __init__(self, 
                 api_key=None, 
                 base_url=None, 
                 model_name=None,
                 relabel_sample_count=30):
        """
        连接商业/中转 API 管理类
        """
        settings = get_llm_settings()
        api_key = api_key or settings["api_key"]
        base_url = base_url or settings["base_url"]
        model_name = model_name or settings["model_name"]

        # 注意：SDK 会自动处理末尾的 /chat/completions，所以 base_url 只写到 /v1
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.reward_file = "generated_reward_fn.py"
        self.relabel_sample_count = int(relabel_sample_count)
        self.call_index = 0
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("llm_txt_logs", ts)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 预加载 Prompt 模板（假设你的文件夹叫 prompts）
        self.prompt_dir = "prompts"
        self.initial_system = self._file_to_string("initial_system.txt")
        self.initial_user_template = self._file_to_string("new_initial_user.txt")
        self.output_tip = self._file_to_string("new_code_output_tip.txt")
        self.feedback_template = self._file_to_string("code_feedback.txt")

    def _to_float_tensor(self, value):
        if value is None:
            return None
        tensor = torch.as_tensor(value)
        if tensor.numel() == 0:
            return None
        return tensor.float().reshape(-1)

    def _sample_priority(self, info, obs):
        score = 0.0

        if info.get("collision"):
            score += 4.0
        if info.get("success"):
            score += 2.0

        z = self._to_float_tensor(obs.get("z_pos"))
        if z is not None:
            z_mean = float(z.mean().item())
            z_max = float(z.max().item())
            score += max(0.0, z_mean - 3.0) * 2.0
            score += max(0.0, z_max - 3.5) * 4.0
            score += max(0.0, 0.8 - z_mean) * 0.5

        v_norm = self._to_float_tensor(obs.get("v_norm"))
        if v_norm is not None:
            score += max(0.0, float(v_norm.mean().item()) - 4.0) * 1.5

        v = self._to_float_tensor(obs.get("v"))
        if v is not None and v.numel() % 3 == 0:
            v_vec = v.reshape(-1, 3)
            vz = float(v_vec[:, 2].abs().mean().item())
            score += vz * 0.8

        tilt_cos = self._to_float_tensor(obs.get("tilt_cos"))
        if tilt_cos is not None:
            score += max(0.0, 0.8 - float(tilt_cos.mean().item())) * 2.5

        action = self._to_float_tensor(obs.get("action"))
        if action is not None:
            score += float(action.abs().mean().item()) * 0.2

        return score

    def _format_sample_summary(self, idx, info, obs, score_text):
        tags = []
        if info.get("collision"):
            tags.append("collision")
        if info.get("success"):
            tags.append("success")
        if not tags:
            tags.append("neutral")

        z = self._to_float_tensor(obs.get("z_pos"))
        v_norm = self._to_float_tensor(obs.get("v_norm"))
        v = self._to_float_tensor(obs.get("v"))
        tilt_cos = self._to_float_tensor(obs.get("tilt_cos"))
        action = self._to_float_tensor(obs.get("action"))

        parts = [f"- sample {idx:02d} [{'|'.join(tags)}] score={score_text}"]
        if z is not None:
            parts.append(f"z_mean={float(z.mean().item()):.3f}")
            parts.append(f"z_max={float(z.max().item()):.3f}")
        if v_norm is not None:
            parts.append(f"v_norm={float(v_norm.mean().item()):.3f}")
        if v is not None and v.numel() % 3 == 0:
            vz = v.reshape(-1, 3)[:, 2]
            parts.append(f"v_z={float(vz.mean().item()):.3f}")
        if tilt_cos is not None:
            parts.append(f"tilt_cos={float(tilt_cos.mean().item()):.3f}")
        if action is not None:
            parts.append(f"action_abs_mean={float(action.abs().mean().item()):.3f}")
        return " ".join(parts)

    def _write_txt(self, filename, content):
        path = os.path.join(self.log_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _next_tag(self, stage):
        self.call_index += 1
        return f"{self.call_index:03d}_{stage}"

    def _chat_with_heartbeat(self, *, messages, temperature, max_tokens=None, stage_label="LLM"):
        """Blocking chat call with periodic progress logs to avoid 'stuck' feeling."""
        stop_event = threading.Event()
        start_t = time.time()

        def _heartbeat():
            while not stop_event.wait(5.0):
                elapsed = time.time() - start_t
                print(f"⏱️ [{stage_label}] 等待模型响应中... {elapsed:.1f}s", flush=True)

        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return self.client.chat.completions.create(**kwargs)
        finally:
            stop_event.set()
            t.join(timeout=0.1)

    def _chat_with_progress(self, *, messages, temperature, max_tokens=None, stage_label="LLM"):
        """Prefer streaming with tqdm progress; fallback to heartbeat blocking call."""
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            stream = self.client.chat.completions.create(**kwargs)
            pbar = tqdm(
                total=100,
                desc=f"LLM {stage_label}",
                dynamic_ncols=True,
                leave=False,
                mininterval=0.2,
            )
            content_parts = []
            chunk_count = 0
            char_count = 0

            for chunk in stream:
                delta = None
                if chunk.choices and chunk.choices[0].delta is not None:
                    delta = chunk.choices[0].delta.content
                if delta:
                    content_parts.append(delta)
                    char_count += len(delta)

                chunk_count += 1
                if pbar.n < pbar.total:
                    pbar.update(1)
                else:
                    pbar.total += 1
                    pbar.update(1)

                if chunk_count % 10 == 0:
                    pbar.set_postfix({"chunks": chunk_count, "chars": char_count})

            pbar.close()
            return "".join(content_parts)
        except Exception:
            response = self._chat_with_heartbeat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stage_label=stage_label,
            )
            return response.choices[0].message.content

    def _file_to_string(self, filename):
        path = os.path.join(self.prompt_dir, filename)
        if not os.path.exists(path):
            # 兼容性处理：如果主程序运行目录下没有文件夹，尝试当前目录
            path = filename 
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract_code(self, response_text):
        """精准提取 Markdown 中的 Python 代码块"""
        patterns = [r'```python\s*(.*?)\s*```', r'```\s*(.*?)\s*```']
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return response_text.strip()

    def save_reward_code(self, code_str):
        """保存代码并自动注入 PyTorch 环境"""
        with open(self.reward_file, "w", encoding="utf-8") as f:
            f.write("import torch\nimport numpy as np\nimport math\n\n")
            f.write(code_str)
        print(f"💾 奖励函数已保存至 {self.reward_file}")

    def generate_initial_reward(
        self,
        task_desc,
        state_info,
        baseline_reward_reference: str = "",
        reward_param_context: str = "",
    ):
        """第 0 代：生成初始奖励函数"""
        user_content = self.initial_user_template.format(
            task=task_desc, 
            input_dict_string=state_info
        ) + "\n" + self.output_tip

        if reward_param_context:
            user_content += (
                "\n\n[Environment Reward Parameters]\n"
                + reward_param_context
                + "\nPlease keep reward magnitude compatible with these parameter scales."
            )

        if baseline_reward_reference:
            user_content += (
                "\n\n[Baseline Reward Reference - Inherit Good Parts, Fix Weak Parts]\n"
                "Use this as a starting point and improve it. Keep vectorized shape-safe implementation.\n"
                "```python\n"
                + baseline_reward_reference
                + "\n```"
            )
        tag = self._next_tag("initial")
        self._write_txt(
            f"{tag}_prompt.txt",
            "[SYSTEM]\n" + self.initial_system + "\n\n[USER]\n" + user_content,
        )

        print(f"🧠 正在请求 {self.model_name} 生成初始代码...")
        raw_text = self._chat_with_progress(
            messages=[
                {"role": "system", "content": self.initial_system},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,  # 商业模型通常 0.3 生成代码最稳
            max_tokens=2048,
            stage_label="initial",
        )
        code = self.extract_code(raw_text)
        self._write_txt(f"{tag}_response_raw.txt", raw_text)
        self._write_txt(f"{tag}_code.txt", code)
        self.save_reward_code(code)
        return code

    def generate_feedback_with_relabel(
        self,
        reflection_buffer,
        last_code,
        current_success_rate,
        reward_param_context: str = "",
        baseline_reward_reference: str = "",
    ):
        """
        🌟 创新点：经验池重标（Relabeling）反馈
        """
        print("🔍 正在通过离线经验池进行重标分析 (Relabeling)...")
        
        # 1. 尝试动态加载当前函数以便本地测试（如果需要计算分数回传给 LLM）
        try:
            spec = importlib.util.spec_from_file_location("temp_reward", self.reward_file)
            temp_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_mod)
        except Exception as e:
            print(f"⚠️ 无法加载旧代码进行重标，将仅进行成功率反馈。错误: {e}")
            temp_mod = None

        # 2. 构造错题分析文本
        relabel_reports = []
        sample_count = max(1, self.relabel_sample_count)
        recent_window = min(len(reflection_buffer.storge), max(sample_count * 3, sample_count))
        candidate_samples = reflection_buffer.storge[-recent_window:] if recent_window > 0 else []

        scored_samples = []
        for local_idx, sample in enumerate(candidate_samples):
            info, obs, act, reward_list, obs_, done = sample
            if isinstance(info, dict):
                info_dict = info
            else:
                info_dict = {}
            obs_with_action = dict(obs)
            obs_with_action["action"] = act
            priority = self._sample_priority(info_dict, obs_with_action)
            priority += (local_idx + 1) / max(1, len(candidate_samples))
            scored_samples.append((priority, local_idx, sample))

        scored_samples.sort(key=lambda item: item[0], reverse=True)
        samples = [sample for _, _, sample in scored_samples[:sample_count]]

        collision_total = sum(1 for info, _, _, _, _, _ in samples if isinstance(info, dict) and info.get("collision"))
        success_total = sum(1 for info, _, _, _, _, _ in samples if isinstance(info, dict) and info.get("success"))
        z_max_values = []
        vz_values = []
        for _, obs, act, _, _, _ in samples:
            z = self._to_float_tensor(obs.get("z_pos"))
            if z is not None:
                z_max_values.append(float(z.max().item()))
            v = self._to_float_tensor(obs.get("v"))
            if v is not None and v.numel() % 3 == 0:
                vz_values.append(float(v.reshape(-1, 3)[:, 2].abs().mean().item()))

        if samples:
            relabel_reports.append(
                f"- selected {len(samples)} / {len(candidate_samples)} samples | collisions={collision_total} | successes={success_total}"
            )
            if z_max_values:
                relabel_reports.append(f"- selected altitude stats: z_max_mean={sum(z_max_values)/len(z_max_values):.3f} z_max_peak={max(z_max_values):.3f}")
            if vz_values:
                relabel_reports.append(f"- selected vertical speed stats: v_z_mean={sum(vz_values)/len(vz_values):.3f} v_z_peak={max(vz_values):.3f}")
        
        if temp_mod:
            for i, (info, obs, act, _, _, _) in enumerate(samples):
                # 将保存的 numpy 数据转回 Tensor 给函数打分
                # 假设 obs 是一个 dict，且包含 get_clean_states 返回的所有 key
                obs_tensor = {k: torch.as_tensor(v).unsqueeze(0).cuda() for k, v in obs.items()}
                obs_tensor["action"] = torch.as_tensor(act).unsqueeze(0).cuda()
                
                with torch.no_grad():
                    try:
                        score = temp_mod.compute_reward(**obs_tensor).item()
                        score_text = f"{score:.4f}"
                        relabel_reports.append(self._format_sample_summary(i, info, obs_tensor, score_text))
                    except Exception as e:
                        relabel_reports.append(self._format_sample_summary(i, info, obs_tensor, f"ERR({e})"))
        elif samples:
            for i, (info, obs, act, _, _, _) in enumerate(samples):
                obs_tensor = dict(obs)
                obs_tensor["action"] = act
                relabel_reports.append(self._format_sample_summary(i, info, obs_tensor, "N/A"))

        relabel_str = "\n".join(relabel_reports) if relabel_reports else "无历史数据反馈"
        
        # 3. 填入反馈模板
        feedback_str = self.feedback_template.format(
            win_rate=f"{current_success_rate * 100:.1f}%",
            current_score="N/A", 
            current_our_score="See internal logs",
            current_output=relabel_str
        )

        if reward_param_context:
            feedback_str += (
                "\n\n[Environment Reward Parameters]\n"
                + reward_param_context
                + "\nKeep reward scales compatible with these values."
            )

        if baseline_reward_reference:
            feedback_str += (
                "\n\n[Baseline Reward Reference]\n"
                "You may borrow stable parts from this baseline while fixing failure cases.\n"
                "```python\n"
                + baseline_reward_reference
                + "\n```"
            )
        tag = self._next_tag("feedback")
        self._write_txt(
            f"{tag}_prompt.txt",
            "[SYSTEM]\n"
            + self.initial_system
            + "\n\n[ASSISTANT_LAST_CODE]\n"
            + f"```python\n{last_code}\n```"
            + "\n\n[USER]\n"
            + feedback_str
            + "\n"
            + self.output_tip,
        )
        self._write_txt(f"{tag}_relabel.txt", relabel_str)

        print(f"🧠 正在发送反馈给 {self.model_name} 进行逻辑进化...")
        raw_text = self._chat_with_progress(
            messages=[
                {"role": "system", "content": self.initial_system},
                {"role": "assistant", "content": f"```python\n{last_code}\n```"},
                {"role": "user", "content": feedback_str + "\n" + self.output_tip}
            ],
            temperature=0.2,  # 迭代修正时进一步降低随机性
            stage_label="feedback",
        )
        new_code = self.extract_code(raw_text)
        self._write_txt(f"{tag}_response_raw.txt", raw_text)
        self._write_txt(f"{tag}_code.txt", new_code)
        self.save_reward_code(new_code)
        return new_code