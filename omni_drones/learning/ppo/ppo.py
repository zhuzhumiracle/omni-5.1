# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
import einops

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE

@dataclass
class PPOConfig:
    name: str = "ppo"
    train_every: int = 256 #32
    ppo_epochs: int = 4
    num_minibatches: int = 8 #16
    clip_param: float = 0.2
    value_clip_param: float = 0.2
    policy_mode: str = "ppo"  # "ppo" or "spo"
    spo_epsilon: float = 0.2

    # whether to use privileged information
    priv_actor: bool = False
    priv_critic: bool = False

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")
cs.store("ppo_spo", node=PPOConfig(policy_mode="spo"), group="algo")
cs.store("ppo_priv", node=PPOConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ppo_priv_critic", node=PPOConfig(priv_critic=True), group="algo")


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


def _scalar_stat(x: torch.Tensor, op: str):
    x = x.reshape(-1)
    if op == "mean":
        return x.mean()
    if op == "std":
        return x.std(unbiased=False)
    if op == "max":
        return x.max()
    if op == "min":
        return x.min()
    raise ValueError(f"Unsupported stat op: {op}")


def _safe_quantile(x: torch.Tensor, q: float):
    x = x.reshape(-1)
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    return torch.quantile(x, q)


class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        # scale = torch.exp(self.actor_std).expand_as(loc)
        # 关键：限制 log_std，防止后期策略方差漂移
        log_std = self.actor_std.clamp(-5.0, 1.0)
        scale = torch.exp(log_std).expand_as(loc)
        return loc, scale


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.002 #0.01
        self.policy_mode = str(getattr(cfg, "policy_mode", "ppo")).lower()
        if self.policy_mode not in {"ppo", "spo"}:
            raise ValueError(f"Unsupported policy_mode: {self.policy_mode}. Expected 'ppo' or 'spo'.")
        self.clip_param = float(getattr(cfg, "clip_param", 0.2))
        self.spo_epsilon = float(getattr(cfg, "spo_epsilon", self.clip_param))
        self.value_clip_param = float(getattr(cfg, "value_clip_param", self.clip_param))
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec[("agents", "action")].shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        if self.cfg.priv_actor:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            actor_module = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([128, 128]), Actor(self.action_dim)),
                    ["feature"], ["loc", "scale"]
                )
            )
        else:
            actor_module=TensorDictModule(
                nn.Sequential(make_mlp([128, 128, 128]), Actor(self.action_dim)),
                [("agents", "observation")], ["loc", "scale"]
            )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="sample_log_prob"
        ).to(self.device)

        if self.cfg.priv_critic:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            self.critic = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([128, 128]), nn.LazyLinear(1)),
                    ["feature"], ["state_value"]
                )
            ).to(self.device)
        else:
            self.critic = TensorDictModule(
                nn.Sequential(make_mlp([128, 128, 128]), nn.LazyLinear(1)),
                [("agents", "observation")], ["state_value"]
            ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        # else:
        #     def init_(module):
        #         if isinstance(module, nn.Linear):
        #             nn.init.orthogonal_(module.weight, 0.01)
        #             nn.init.constant_(module.bias, 0.)

        #     self.actor.apply(init_)
        #     self.critic.apply(init_)
        else:
            def init_weights(module):
                # 1. 隐藏层必须使用标准的正交初始化增益 (对于 ReLU 及其变体，通常用 sqrt(2))
                if isinstance(module, (nn.Linear, nn.LazyLinear)):
                    nn.init.orthogonal_(module.weight, 1.414) # np.sqrt(2) 约等于 1.414
                    nn.init.constant_(module.bias, 0.)

            # 给所有网络打上健康的基础初始化
            self.actor.apply(init_weights)
            self.critic.apply(init_weights)

            # 2. PPO 专属 Trick：只对 Actor 动作输出层进行 0.01 的微小初始化
            for name, param in self.actor.named_parameters():
                if "actor_mean.weight" in name:
                    nn.init.orthogonal_(param, 0.01)
                    
            # Critic 最后一层通常用 1.0，这步可选但推荐
            for name, param in self.critic.named_parameters():
                if "module.0.1.weight" in name: # 根据你的网络结构定位最后一步
                    nn.init.orthogonal_(param, 1.0)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4) #1e-4
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)# 3e-4
        self.value_norm = ValueNorm1(reward_spec[("agents", "reward")].shape[-2:]).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        # rewards = tensordict[("next", "agents", "reward")]
        # 算法层的最后一道防火墙，把 reward 强行压在合理区间
        rewards = tensordict[("next", "agents", "reward")].clamp(-10000.0, 10000.0)
        dones = einops.repeat(
            tensordict[("next", "terminated")],
            "t e 1 -> t e a 1",
            a=self.n_agents
        )
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        rollout_info = TensorDict({
            "rollout_adv_mean": _scalar_stat(adv, "mean"),
            "rollout_adv_std": _scalar_stat(adv, "std"),
            "rollout_adv_abs_max": adv.abs().max(),
            "rollout_adv_p95": _safe_quantile(adv, 0.95),
            "rollout_adv_p99": _safe_quantile(adv, 0.99),
            "rollout_ret_mean": _scalar_stat(ret, "mean"),
            "rollout_ret_std": _scalar_stat(ret, "std"),
            "rollout_ret_abs_max": ret.abs().max(),
            "rollout_ret_p95": _safe_quantile(ret, 0.95),
            "rollout_ret_p99": _safe_quantile(ret, 0.99),
            "rollout_reward_mean": _scalar_stat(rewards, "mean"),
            "rollout_reward_std": _scalar_stat(rewards, "std"),
            "rollout_reward_abs_max": rewards.abs().max(),
            "rollout_value_mean": _scalar_stat(values, "mean"),
            "rollout_value_std": _scalar_stat(values, "std"),
            "rollout_next_value_mean": _scalar_stat(next_values, "mean"),
            "rollout_next_value_std": _scalar_stat(next_values, "std"),
        }, [])

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        kl_early_stop = False
        for epoch in range(self.cfg.ppo_epochs):
            if kl_early_stop:
                break
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                info = self._update(minibatch)
                infos.append(info)
                # Epoch-level KL early stopping: 一旦某个 minibatch KL 过大，
                # 立即停止当前 epoch 及后续所有 epoch
                if info["approx_kl"].item() > 0.05:
                    kl_early_stop = True
                    break

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        infos.update(rollout_info)
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]

        log_ratio_raw = log_probs - tensordict["sample_log_prob"]
        # Keep policy-ratio clamp for optimization stability only.
        log_ratio = log_ratio_raw.clamp(-10.0, 10.0)
        ratio = torch.exp(log_ratio).unsqueeze(-1)
        ratio_squeezed = ratio.squeeze(-1)

        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        if self.policy_mode == "spo":
            eps = self.spo_epsilon
            # J_spo = E[ r*A - |A|/(2*eps) * (r-1)^2 ]
            spo_obj = ratio * adv - (adv.abs() / (2.0 * eps)) * (ratio - 1.0).pow(2)
            policy_loss = - torch.mean(spo_obj)
        else:
            eps = self.clip_param
            surr1 = ratio * adv
            surr2 = ratio.clamp(1. - eps, 1. + eps) * adv
            policy_loss = - torch.mean(torch.min(surr1, surr2))

        clip_fraction = ((ratio_squeezed - 1.0).abs() > eps).float().mean()
        ratio_deviation = (ratio_squeezed - 1.0).abs().mean()
        # Use unclamped log-ratio for KL diagnostics and early stopping.
        # KL(old||new) Monte-Carlo estimator: E[exp(log_ratio) - 1 - log_ratio], where
        # log_ratio = log pi_new(a|s) - log pi_old(a|s).
        ratio_kl = torch.exp(log_ratio_raw.clamp(-60.0, 60.0))
        approx_kl = (ratio_kl - 1.0 - log_ratio_raw).mean()
        approx_kl = torch.nan_to_num(approx_kl, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

        ratio_max = ratio_squeezed.max()
        ratio_p95 = _safe_quantile(ratio_squeezed, 0.95)
        ratio_p99 = _safe_quantile(ratio_squeezed, 0.99)
        log_ratio_abs_max = log_ratio_raw.abs().max()
        log_ratio_p95 = _safe_quantile(log_ratio_raw.abs(), 0.95)
        log_ratio_p99 = _safe_quantile(log_ratio_raw.abs(), 0.99)
        adv_abs_max = adv.abs().max()
        adv_p95 = _safe_quantile(adv, 0.95)
        adv_p99 = _safe_quantile(adv, 0.99)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.value_clip_param, self.value_clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)
        value_err_abs = (values - b_returns).abs()
        value_err_abs_mean = value_err_abs.mean()
        value_err_abs_max = value_err_abs.max()
        value_err_abs_p95 = _safe_quantile(value_err_abs, 0.95)
        value_err_abs_p99 = _safe_quantile(value_err_abs, 0.99)
        explained_var_denom = b_returns.var().clamp_min(1e-6)
        explained_var = 1 - F.mse_loss(values, b_returns) / explained_var_denom

        # KL early stopping: skip this minibatch update before backward/step.
        if approx_kl.item() > 0.05:
            zero = torch.zeros((), device=policy_loss.device)
            return TensorDict({
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "clip_fraction": clip_fraction,
                "ratio_deviation": ratio_deviation,
                "approx_kl": approx_kl,
                "ratio_max": ratio_max,
                "ratio_p95": ratio_p95,
                "ratio_p99": ratio_p99,
                "log_ratio_abs_max": log_ratio_abs_max,
                "log_ratio_abs_p95": log_ratio_p95,
                "log_ratio_abs_p99": log_ratio_p99,
                "mb_adv_abs_max": adv_abs_max,
                "mb_adv_p95": adv_p95,
                "mb_adv_p99": adv_p99,
                "value_err_abs_mean": value_err_abs_mean,
                "value_err_abs_max": value_err_abs_max,
                "value_err_abs_p95": value_err_abs_p95,
                "value_err_abs_p99": value_err_abs_p99,
                "actor_grad_norm": zero,
                "critic_grad_norm": zero,
                "explained_var": explained_var
            }, [])

        loss = policy_loss + entropy_loss + value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 1.0)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.actor_opt.step()
        self.critic_opt.step()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "clip_fraction": clip_fraction,
            "ratio_deviation": ratio_deviation,
            "approx_kl": approx_kl,
            "ratio_max": ratio_max,
            "ratio_p95": ratio_p95,
            "ratio_p99": ratio_p99,
            "log_ratio_abs_max": log_ratio_abs_max,
            "log_ratio_abs_p95": log_ratio_p95,
            "log_ratio_abs_p99": log_ratio_p99,
            "mb_adv_abs_max": adv_abs_max,
            "mb_adv_p95": adv_p95,
            "mb_adv_p99": adv_p99,
            "value_err_abs_mean": value_err_abs_mean,
            "value_err_abs_max": value_err_abs_max,
            "value_err_abs_p95": value_err_abs_p95,
            "value_err_abs_p99": value_err_abs_p99,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
