"""Supervised warm-up and RLVR training for a population of transformers.

Each population member is an independent run: its own prompts, its own rollouts, its own
optimiser state. The only thing they share is the kernel they are computed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from caliper.control.drift import DriftController
from caliper.envs.modsum import ModSum
from caliper.objectives.advantages import weight_table
from caliper.population.model import PopulationTransformer
from caliper.population.optim import PopulationAdam, PopulationSGD


@dataclass
class SFTConfig:
    steps: int = 400
    batch: int = 64
    learning_rate: float = 3e-3
    target_pass_rate: float | None = 0.25
    eval_batch: int = 256
    eval_every: int = 10


@dataclass
class RLConfig:
    estimator: str = "rloo"
    prompts: int = 32
    group_size: int = 8
    steps: int = 100
    drift_target: float = 2e-3
    initial_step_size: float = 0.1
    optimiser: str = "sgd"
    temperature: float = 1.0
    instrument_every: int = 10
    eval_batch: int = 256
    eval_every: int = 5
    controller_gain: float = 0.5
    max_step_ratio: float = 1.5
    history: dict = field(default_factory=dict)


def _sequence_log_probs(model, tokens, prompt_len) -> torch.Tensor:
    return model.log_probs(tokens, prompt_len).sum(dim=-1)


@torch.no_grad()
def pass_rate(model, task: ModSum, batch: int, generator: torch.Generator) -> torch.Tensor:
    n = model.config.population
    prompts = task.sample_prompts(n * batch, generator).reshape(n, batch, task.prompt_len)
    out = model.generate(prompts, task.response_len, generator, temperature=1.0)
    return task.reward(prompts, out[:, :, task.prompt_len :]).mean(dim=-1)


def supervised_warmup(
    model: PopulationTransformer,
    task: ModSum,
    config: SFTConfig,
    generator: torch.Generator,
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    n = model.config.population
    device = model.token_embedding.device
    active = torch.ones(n, device=device)
    history = {"step": [], "pass_rate": []}
    for step in range(config.steps):
        prompts = task.sample_prompts(n * config.batch, generator).reshape(
            n, config.batch, task.prompt_len
        )
        targets = task.targets(prompts)
        tokens = torch.cat([prompts, targets], dim=-1)
        logits = model(tokens[:, :, :-1])[:, :, task.prompt_len - 1 :, :]
        per_member = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(n, -1).mean(dim=1)
        # members that already reached the target pass rate are frozen, so a population can be
        # brought to a common starting point rather than to a common number of updates
        loss = (per_member * active).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if config.target_pass_rate is not None and step % config.eval_every == 0:
            rate = pass_rate(model, task, config.eval_batch, generator)
            history["step"].append(step)
            history["pass_rate"].append(rate.tolist())
            active = (rate < config.target_pass_rate).float()
            if float(active.sum()) == 0:
                break
    return history


class RLVRTrainer:
    """One RLVR run per population member, with drift-controlled step size."""

    def __init__(
        self,
        model: PopulationTransformer,
        task: ModSum,
        config: RLConfig,
        generator: torch.Generator,
    ):
        self.model = model
        self.task = task
        self.config = config
        self.generator = generator
        device = model.token_embedding.device
        self.weights = torch.tensor(
            weight_table(config.estimator, config.group_size), device=device, dtype=torch.float32
        )
        sub = config.group_size // 2
        self.sub_weights = torch.tensor(
            weight_table(config.estimator, sub), device=device, dtype=torch.float32
        )
        self.controllers = [
            DriftController(
                target=config.drift_target,
                step_size=config.initial_step_size,
                gain=config.controller_gain,
                max_ratio=config.max_step_ratio,
            )
            for _ in range(model.config.population)
        ]
        self.optimiser = self._make_optimiser()

    def _make_optimiser(self):
        population = self.model.config.population
        if self.config.optimiser == "sgd":
            return PopulationSGD(self.model.parameters(), population)
        if self.config.optimiser == "adam":
            return PopulationAdam(self.model.parameters(), population)
        raise KeyError(self.config.optimiser)

    def rollout(self):
        cfg, task = self.config, self.task
        n = self.model.config.population
        prompts = self.task.sample_prompts(n * cfg.prompts, self.generator).reshape(
            n, cfg.prompts, task.prompt_len
        )
        wide = prompts.repeat_interleave(cfg.group_size, dim=1)
        tokens = self.model.generate(
            wide, task.response_len, self.generator, temperature=cfg.temperature
        )
        rewards = task.reward(wide, tokens[:, :, task.prompt_len :])
        return tokens, rewards.reshape(n, cfg.prompts, cfg.group_size)

    def advantages(self, rewards: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        counts = rewards.sum(dim=-1).long()
        idx = rewards.long()
        return weights[counts.unsqueeze(-1).expand_as(idx), idx]

    def _loss(self, tokens, advantage, mask=None):
        n, cfg = self.model.config.population, self.config
        logp = _sequence_log_probs(self.model, tokens, self.task.prompt_len)
        logp = logp.reshape(n, cfg.prompts, cfg.group_size)
        term = advantage * logp
        if mask is not None:
            term = term * mask
            denom = mask.sum(dim=(1, 2)).clamp(min=1)
        else:
            denom = torch.full((n,), float(cfg.prompts * cfg.group_size), device=term.device)
        return -(term.sum(dim=(1, 2)) / denom).sum()

    def _flat_grad(self) -> torch.Tensor:
        return torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(
                    self.model.config.population, -1
                )
                for p in self.model.parameters()
            ],
            dim=1,
        )

    def _gradient_for(self, tokens, advantage, mask=None) -> torch.Tensor:
        self.optimiser.zero_grad()
        self._loss(tokens, advantage, mask).backward()
        return self._flat_grad().detach().clone()

    @torch.no_grad()
    def _response_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.model(tokens[:, :, :-1])
        return logits[:, :, self.task.prompt_len - 1 :, :]

    @staticmethod
    def _token_kl(old_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
        """Mean KL(new || old) per response token, per population member."""
        new_log = torch.log_softmax(new_logits, dim=-1)
        old_log = torch.log_softmax(old_logits, dim=-1)
        kl = (new_log.exp() * (new_log - old_log)).sum(dim=-1)
        return kl.mean(dim=(1, 2))

    def _instrument(self, tokens, rewards):
        """Fill the four buffers and return the hierarchical noise estimate per member."""
        cfg = self.config
        half_p = cfg.prompts // 2
        sub = cfg.group_size // 2

        full_adv = self.advantages(rewards, self.weights)
        mask = torch.zeros_like(full_adv)
        mask[:, :half_p, :] = 1.0
        g_pa = self.optimiser.precondition(self._gradient_for(tokens, full_adv, mask))
        g_pb = self.optimiser.precondition(self._gradient_for(tokens, full_adv, 1.0 - mask))

        sub_a = self.advantages(rewards[:, :, :sub], self.sub_weights)
        sub_b = self.advantages(rewards[:, :, sub:], self.sub_weights)
        pad_a = torch.cat([sub_a, torch.zeros_like(sub_b)], dim=-1)
        pad_b = torch.cat([torch.zeros_like(sub_a), sub_b], dim=-1)
        roll_mask_a = torch.zeros_like(full_adv)
        roll_mask_a[:, :, :sub] = 1.0
        g_ra = self.optimiser.precondition(
            self._gradient_for(tokens, pad_a, roll_mask_a)
        )
        g_rb = self.optimiser.precondition(
            self._gradient_for(tokens, pad_b, 1.0 - roll_mask_a)
        )
        return g_pa, g_pb, g_ra, g_rb

    def step(self, instrument: bool) -> dict:
        tokens, rewards = self.rollout()
        old_logits = self._response_logits(tokens)

        buffers = self._instrument(tokens, rewards) if instrument else None

        advantage = self.advantages(rewards, self.weights)
        self.optimiser.zero_grad()
        self._loss(tokens, advantage).backward()
        step_sizes = torch.tensor(
            [c.step_size for c in self.controllers],
            device=tokens.device,
            dtype=torch.float32,
        )
        self.optimiser.step(step_sizes)

        drift = self._token_kl(old_logits, self._response_logits(tokens))
        for member, value in enumerate(drift.tolist()):
            self.controllers[member].update(value)

        info = {
            "drift": drift.tolist(),
            "step_size": step_sizes.tolist(),
            "mean_reward": rewards.mean(dim=(1, 2)).tolist(),
            "pass_rate_spread": rewards.mean(dim=2).std(dim=1).tolist(),
        }
        if buffers is not None:
            info["noise"] = self.noise_estimates(*buffers)
        return info

    def noise_estimates(self, g_pa, g_pb, g_ra, g_rb) -> list[dict]:
        cfg = self.config
        sub = cfg.group_size // 2
        signal = (g_pa * g_pb).sum(dim=1)
        tau_total = 0.25 * cfg.prompts * ((g_pa - g_pb) ** 2).sum(dim=1)
        tau_w_sub = 0.5 * cfg.prompts * ((g_ra - g_rb) ** 2).sum(dim=1)
        tau_w_scaled = tau_w_sub * (sub - 1) if sub > 1 else tau_w_sub
        between = (g_ra * g_rb).sum(dim=1) * cfg.prompts
        tau_b = between - signal
        return [
            {
                "signal": float(signal[i]),
                "tau_total": float(tau_total[i]),
                "tau_w_scaled": float(tau_w_scaled[i]),
                "tau_b": float(tau_b[i]),
            }
            for i in range(signal.shape[0])
        ]

    def run(self) -> dict:
        cfg = self.config
        history = {"step": [], "pass_rate": [], "drift": [], "step_size": [], "noise": []}
        for t in range(cfg.steps):
            info = self.step(instrument=(t % cfg.instrument_every == 0))
            if t % cfg.eval_every == 0 or t == cfg.steps - 1:
                rate = pass_rate(self.model, self.task, cfg.eval_batch, self.generator)
                history["step"].append(t)
                history["pass_rate"].append(rate.tolist())
                history["drift"].append(info["drift"])
                history["step_size"].append(info["step_size"])
            if "noise" in info:
                history["noise"].append({"step": t, "terms": info["noise"]})
        history["final_pass_rate"] = pass_rate(
            self.model, self.task, cfg.eval_batch * 2, self.generator
        ).tolist()
        return history
