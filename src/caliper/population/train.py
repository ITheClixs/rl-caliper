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
from caliper.estimators.torch_splits import decompose_batched
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
    pool_size: int | None = None
    prompts: int = 32
    group_size: int = 8
    steps: int = 100
    drift_target: float = 2e-3
    initial_step_size: float = 0.1
    optimiser: str = "sgd"
    temperature: float = 1.0
    instrument_every: int = 10
    blocks: int = 8
    eval_batch: int = 256
    eval_every: int = 5
    controller_gain: float = 0.5
    max_step_ratio: float = 1.5
    control: bool = True
    history: dict = field(default_factory=dict)


def _sequence_log_probs(model, tokens, prompt_len, temperature=1.0) -> torch.Tensor:
    return model.log_probs(tokens, prompt_len, temperature).sum(dim=-1)


def make_pool(task, population: int, pool_size: int, generator) -> torch.Tensor:
    """A fixed set of prompts per population member."""
    return task.sample_prompts(population * pool_size, generator).reshape(
        population, pool_size, task.prompt_len
    )


def draw_prompts(task, pool, n, count, generator):
    """Draw `count` distinct prompts per population member from a fixed pool.

    Distinctness matters: a prompt appearing in two accumulation blocks makes the between-block
    inner product pick up a same-prompt term, which biases the between-prompt noise estimate down.
    """
    if pool is None:
        return task.sample_prompts(n * count, generator).reshape(n, count, task.prompt_len)
    if count > pool.shape[1]:
        raise ValueError("cannot draw more distinct prompts than the pool holds")
    order = torch.rand(n, pool.shape[1], generator=generator, device=pool.device).argsort(dim=1)
    idx = order[:, :count]
    return torch.gather(pool, 1, idx.unsqueeze(-1).expand(-1, -1, task.prompt_len))


@torch.no_grad()
def pass_rate(
    model, task: ModSum, batch: int, generator: torch.Generator, pool: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean pass rate over the prompt distribution the run is training on.

    A pool smaller than the evaluation batch is covered by repeated passes rather than by
    sampling prompts twice within one pass.
    """
    n = model.config.population
    per_pass = batch if pool is None else min(batch, pool.shape[1])
    passes = max(1, -(-batch // per_pass))
    total = 0.0
    for _ in range(passes):
        prompts = draw_prompts(task, pool, n, per_pass, generator)
        out = model.generate(prompts, task.response_len, generator, temperature=1.0)
        total = total + task.reward(prompts, out[:, :, task.prompt_len :]).mean(dim=-1)
    return total / passes


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
        pool: torch.Tensor | None = None,
    ):
        self.model = model
        self.task = task
        self.config = config
        self.generator = generator
        device = model.token_embedding.device
        self.weights = torch.tensor(
            weight_table(config.estimator, config.group_size), device=device, dtype=torch.float32
        )
        # the crossed split needs two sub-groups of at least two rollouts each, so a run with
        # G < 4 can train but cannot be instrumented
        sub = config.group_size // 2
        self.sub_weights = (
            torch.tensor(
                weight_table(config.estimator, sub), device=device, dtype=torch.float32
            )
            if sub >= 2
            else None
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
        self.blocks = self._usable_blocks()
        self.optimiser = self._make_optimiser()
        self.pool = pool
        if pool is None and config.pool_size is not None:
            self.pool = make_pool(task, model.config.population, config.pool_size, generator)

    def _usable_blocks(self) -> int | None:
        """Largest block count dividing the prompt count, or None if no split is possible."""
        cfg = self.config
        for blocks in range(min(cfg.blocks, cfg.prompts // 2), 1, -1):
            if cfg.prompts % blocks == 0:
                return blocks
        return None

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
        prompts = draw_prompts(task, self.pool, n, cfg.prompts, self.generator)
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
        logp = _sequence_log_probs(
            self.model, tokens, self.task.prompt_len, self.config.temperature
        )
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
        logits = self.model(tokens[:, :, :-1]) / self.config.temperature
        return logits[:, :, self.task.prompt_len - 1 :, :]

    @staticmethod
    def _token_kl(old_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
        """Mean KL(new || old) per response token, per population member."""
        new_log = torch.log_softmax(new_logits, dim=-1)
        old_log = torch.log_softmax(old_logits, dim=-1)
        kl = (new_log.exp() * (new_log - old_log)).sum(dim=-1)
        return kl.mean(dim=(1, 2))

    def _instrument(self, tokens, rewards) -> torch.Tensor:
        """Gradients on a K x 2 crossing of prompt block and rollout sub-group."""
        if self.sub_weights is None or self.blocks is None:
            raise ValueError("instrumentation needs G >= 4 and at least two prompt blocks")
        cfg = self.config
        blocks = self.blocks
        per_block = cfg.prompts // blocks
        sub = cfg.group_size // 2

        sub_rewards = (rewards[:, :, :sub], rewards[:, :, sub:])
        cells = []
        for block in range(blocks):
            row = []
            for piece in (0, 1):
                advantage = torch.zeros_like(rewards)
                columns = slice(0, sub) if piece == 0 else slice(sub, cfg.group_size)
                advantage[:, :, columns] = self.advantages(sub_rewards[piece], self.sub_weights)
                mask = torch.zeros_like(rewards)
                mask[:, block * per_block : (block + 1) * per_block, columns] = 1.0
                row.append(
                    self.optimiser.precondition(self._gradient_for(tokens, advantage, mask))
                )
            cells.append(torch.stack(row))
        return torch.stack(cells)

    def step(self, instrument: bool) -> dict:
        tokens, rewards = self.rollout()
        old_logits = self._response_logits(tokens)

        cells = self._instrument(tokens, rewards) if instrument else None

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
        if self.config.control:
            for member, value in enumerate(drift.tolist()):
                self.controllers[member].update(value)

        info = {
            "drift": drift.tolist(),
            "step_size": step_sizes.tolist(),
            "mean_reward": rewards.mean(dim=(1, 2)).tolist(),
            "pass_rate_spread": rewards.mean(dim=2).std(dim=1).tolist(),
        }
        if cells is not None:
            info["noise"] = self.noise_estimates(cells)
        return info

    def noise_estimates(self, cells: torch.Tensor) -> list[dict]:
        corpus = None if self.pool is None else self.pool.shape[1]
        terms = decompose_batched(
            cells,
            self.config.prompts // self.blocks * self.blocks,
            self.config.group_size,
            corpus_size=corpus,
        )
        population = cells.shape[2]
        return [
            {name: float(value[i]) for name, value in terms.items()} for i in range(population)
        ]

    def run(self) -> dict:
        cfg = self.config
        history = {"step": [], "pass_rate": [], "drift": [], "step_size": [], "noise": []}
        realised_drift = []
        for t in range(cfg.steps):
            measurable = (
                self.sub_weights is not None
                and self.blocks is not None
                and cfg.instrument_every > 0
            )
            info = self.step(instrument=measurable and t % cfg.instrument_every == 0)
            if t % cfg.eval_every == 0 or t == cfg.steps - 1:
                rate = pass_rate(
                    self.model, self.task, cfg.eval_batch, self.generator, self.pool
                )
                history["step"].append(t)
                history["pass_rate"].append(rate.tolist())
                history["drift"].append(info["drift"])
                history["step_size"].append(info["step_size"])
            realised_drift.append(info["drift"])
            if "noise" in info:
                history["noise"].append({"step": t, "terms": info["noise"]})
        history["mean_drift"] = (
            torch.tensor(realised_drift).mean(dim=0).tolist() if realised_drift else []
        )
        history["final_pass_rate"] = pass_rate(
            self.model, self.task, cfg.eval_batch * 2, self.generator, self.pool
        ).tolist()
        return history


@torch.no_grad()
def snapshot(model: PopulationTransformer) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def restore(model: PopulationTransformer, state: dict) -> None:
    for key, value in model.state_dict().items():
        value.copy_(state[key])


def measure_noise(
    model: PopulationTransformer,
    task: ModSum,
    config: RLConfig,
    generator: torch.Generator,
    batches: int,
    pool: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Estimate the noise terms at a frozen policy, averaging raw moments over batches.

    No parameters are updated, so this measures the state the sweep starts from rather than a
    moving target. Ratios are formed only after averaging.
    """
    trainer = RLVRTrainer(model, task, config, generator, pool=pool)
    totals = None
    for _ in range(batches):
        tokens, rewards = trainer.rollout()
        terms = decompose_batched(
            trainer._instrument(tokens, rewards), config.prompts, config.group_size
        )
        totals = terms if totals is None else {k: totals[k] + v for k, v in terms.items()}
    trainer.optimiser.zero_grad()
    return {k: v / batches for k, v in totals.items()}
