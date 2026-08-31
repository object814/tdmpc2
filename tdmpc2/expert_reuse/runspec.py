"""Discovery + config reconstruction for finished sequential-CL runs.

A `RunSpec` points at one `sweep/logdir/prismatic_seq_progressive/{taskset}/
{seed_dir}` directory and knows:

  - the ordered task list and each task's checkpoint directory
  - whether the run actually finished every task
  - how to rebuild the exact per-task TD-MPC2 cfg that produced the
    checkpoints (needed only for the rollout tier; the router tier reads
    the raw state_dict directly)

Config reconstruction notes
---------------------------
`run_continual_sequential` builds one `shared` override dict and applies it
to every task via `build_task_cfg`. We mirror that dict here. Three training
knobs are deliberately NOT mirrored because they are irrelevant once we load
a checkpoint and never take a gradient step:

  pretrained_encoder  the encoder weights are overwritten by the checkpoint
                      state_dict immediately after construction
  freeze_encoder      only affects which params enter the optimizer
  sdp_freeze          only affects which params enter the optimizer

`encoder_type` IS mirrored, because 'dino' swaps the rgb branch for a
different architecture. Any mismatch we get wrong surfaces immediately as a
strict `load_state_dict` failure, which is why the loader keeps strict=True.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Repo layout: .../third_party/tdmpc2/tdmpc2/expert_reuse/runspec.py
_HERE = Path(__file__).resolve().parent
TDMPC2_PKG = _HERE.parent                      # .../tdmpc2/tdmpc2
REPO_ROOT = TDMPC2_PKG.parent                  # .../tdmpc2
SWEEP_LOGDIR = REPO_ROOT / 'sweep' / 'logdir' / 'prismatic_seq_progressive'
SWEEP_YAML = REPO_ROOT / 'sweep' / 'sweep_prismatic_seq_progressive.yaml'
SLURM_DIR = REPO_ROOT / 'sweep' / 'slurm_run_dir_sequential'


@dataclass
class RunSpec:
    """One sequential-CL run (one taskset x one seed x one encoder variant)."""

    taskset: str
    seed_dir: str                      # e.g. 'seed_2_preEncFr'
    root: Path
    tasks: list[str]                   # e.g. ['metaworld_reach-v3', ...]
    task_dirs: list[Path]
    completed: list[bool]
    task_steps: list[int] = field(default_factory=list)
    # Number of tasks the taskset DEFINES (from the sweep yaml). The MoE was
    # built for this many, so it fixes the gate geometry; `tasks` only lists
    # the ones that have actually started.
    yaml_num_tasks: int = 0
    _geom_cache: dict | None = field(default=None, repr=False)

    # ---- identity -----------------------------------------------------
    @property
    def name(self) -> str:
        return f'{self.taskset}__{self.seed_dir}'

    @property
    def seed(self) -> int:
        m = re.search(r'seed_(\d+)', self.seed_dir)
        return int(m.group(1)) if m else 0

    @property
    def variant(self) -> str:
        """Encoder variant tag: 'preEncFr', 'dino', or 'scratch'."""
        if 'preEncFr' in self.seed_dir:
            return 'preEncFr'
        if 'dino' in self.seed_dir:
            return 'dino'
        return 'scratch'

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def final_task_idx(self) -> int:
        return self.num_tasks - 1

    def short_tasks(self) -> list[str]:
        return [t.replace('metaworld_', '') for t in self.tasks]

    # ---- completeness -------------------------------------------------
    def backbone_path(self, task_idx: int) -> Path:
        return self.task_dirs[task_idx] / 'backbone.pt'

    def task_modules_path(self, task_idx: int) -> Path:
        return self.task_dirs[task_idx] / 'task_modules.pt'

    def is_finished(self) -> bool:
        """Every task marked complete in sequential_progress.json AND every
        end-of-task backbone/task_modules checkpoint present on disk.

        `save_task` runs just before `save_progress(completed=True)`, so the
        two can disagree if a job died in between; requiring both is the
        conservative read.
        """
        return (
            len(self.tasks) > 1
            and all(self.completed)
            and all(self.backbone_path(i).exists() for i in range(self.num_tasks))
            and all(self.task_modules_path(i).exists() for i in range(self.num_tasks))
        )

    def missing(self) -> list[str]:
        out = []
        for i in range(self.num_tasks):
            if not self.completed[i]:
                out.append(f'task{i + 1} not marked complete')
            if not self.backbone_path(i).exists():
                out.append(f'task{i + 1} backbone.pt missing')
            if not self.task_modules_path(i).exists():
                out.append(f'task{i + 1} task_modules.pt missing')
        return out

    # ---- runs still training their final task -------------------------
    def latest_backbone(self, task_idx: int) -> tuple[Path | None, int, bool]:
        """Newest usable backbone checkpoint for one task.

        Returns `(path, step, is_end_of_task)`. Prefers the end-of-task
        `backbone.pt`; falls back to the newest `models/<step>/backbone.pt`
        snapshot so a task still in flight can still be analysed. `step` is
        the task's full budget when the end-of-task file is used, the
        snapshot's step otherwise, and -1 when nothing exists.
        """
        end = self.backbone_path(task_idx)
        if end.exists():
            budget = (self.task_steps[task_idx]
                      if task_idx < len(self.task_steps) else -1)
            return end, int(budget), True
        periodic = self.periodic_checkpoints(task_idx)
        if periodic:
            step, path = periodic[-1]
            return path, int(step), False
        return None, -1, False

    def final_task_progress(self) -> tuple[int, int]:
        """`(steps_trained, budget)` for the last started task."""
        _, step, _ = self.latest_backbone(self.final_task_idx)
        budget = (self.task_steps[self.final_task_idx]
                  if self.final_task_idx < len(self.task_steps) else -1)
        return step, budget

    def reached_final_task(self, min_final_steps: int = 0) -> bool:
        """The run has started the LAST task of its taskset, has at least
        `min_final_steps` of training into it, and every earlier task closed
        out with a proper end-of-task checkpoint.

        The earlier-task requirement is not negotiable: their gate columns
        are what the final task's routing is compared against, and a task
        that never wrote `backbone.pt` has no settled column to compare to.
        """
        if self.yaml_num_tasks and self.num_tasks != self.yaml_num_tasks:
            return False          # still on an earlier task of the sequence
        if self.num_tasks < 2:
            return False
        for i in range(self.final_task_idx):
            if not (self.completed[i] and self.backbone_path(i).exists()):
                return False
        path, step, _ = self.latest_backbone(self.final_task_idx)
        return path is not None and step >= min_final_steps

    # ---- intermediate (periodic) checkpoints --------------------------
    def periodic_checkpoints(self, task_idx: int) -> list[tuple[int, Path]]:
        """`models/<step>/backbone.pt` snapshots written every `save_freq`
        steps during the task, sorted by step. Lets us watch the router move
        during training (and works on runs still in flight)."""
        models = self.task_dirs[task_idx] / 'models'
        if not models.is_dir():
            return []
        out = []
        for d in models.iterdir():
            if d.is_dir() and d.name.isdigit() and (d / 'backbone.pt').exists():
                out.append((int(d.name), d / 'backbone.pt'))
        return sorted(out)

    # ---- geometry read straight off the checkpoint --------------------
    def moe_geometry(self) -> dict:
        """total_K / K_per_task / active_K / tau, read from the final
        backbone.pt rather than reconstructed from flags."""
        if self._geom_cache is not None:
            return self._geom_cache
        path, _, _ = self.latest_backbone(self.final_task_idx)
        if path is None:
            raise FileNotFoundError(
                f'{self.name}: no backbone checkpoint for task '
                f'{self.final_task_idx + 1}')
        try:
            ck = torch.load(path, map_location='cpu', weights_only=False,
                            mmap=True)
        except (TypeError, RuntimeError):
            ck = torch.load(path, map_location='cpu', weights_only=False)
        sd = ck['model']
        W = sd['_dynamics.gate.weight']
        total_K = int(W.shape[0])
        gate_num_tasks = int(W.shape[1]) - 1      # last column is the tau context
        expected = self.yaml_num_tasks or self.num_tasks
        assert gate_num_tasks == expected, (
            f'{self.name}: gate width implies {gate_num_tasks} tasks but the '
            f'taskset defines {expected}')
        assert total_K % gate_num_tasks == 0, (
            f'{self.name}: total_K={total_K} not divisible by '
            f'num_tasks={gate_num_tasks}')
        self._geom_cache = dict(
            total_K=total_K,
            K_per_task=total_K // gate_num_tasks,
            active_K=int(sd['_dynamics._active_K_buf']),
            frozen_K=int(sd['_dynamics._frozen_K_buf']),
            tau=float(sd['_dynamics._tau_buf']),
        )
        return self._geom_cache

    # ---- cfg reconstruction (rollout tier only) -----------------------
    def build_cfg(self, task_idx: int, *, defaults: dict, geom: dict):
        """Rebuild the parsed cfg for one task, mirroring
        `run_continual_sequential`'s `shared` dict.

        The returned cfg still has placeholder `obs_shape` / `action_dim`;
        `make_env(cfg)` fills those in (it mutates cfg in place), exactly as
        the trainer relies on.
        """
        from omegaconf import OmegaConf
        from common.parser import parse_cfg

        cfg = OmegaConf.load(TDMPC2_PKG / 'config.yaml')
        cfg.task = self.tasks[task_idx]
        cfg.steps = int(self.task_steps[task_idx]) if self.task_steps else 100_000
        cfg.seed = self.seed
        cfg.work_dir = str(self.task_dirs[task_idx].resolve())
        cfg.data_dir = str(self.task_dirs[task_idx].resolve())

        overrides = dict(
            model_size=defaults['model_size'],
            batch_size=defaults['batch_size'],
            horizon=defaults['horizon'],
            mpc=defaults['mpc'],
            image_size=defaults['image_size'],
            max_episode_steps=defaults['max_episode_steps'],
            action_repeat=defaults['action_repeat'],
            cameras=list(defaults['cameras']),
            episodic=defaults['episodic'],
            enable_wandb=False,
            save_video=False,
            save_agent=False,
            compile=False,
            save_episodes=False,
            episode_dir=None,
            use_moe=True,
            num_tasks=self.num_tasks,
            progressive_moe=True,
            use_orthogonal=True,
            K_per_task=geom['K_per_task'],
            num_experts=geom['total_K'],
            encoder_type='default' if self.variant != 'dino' else 'dino',
            # Deliberately neutral — see module docstring.
            pretrained_encoder=None,
            freeze_encoder=False,
            sdp_freeze=False,
        )
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
        return parse_cfg(cfg)


# =======================================================================
# Discovery
# =======================================================================

def load_sweep_defaults(yaml_path: Path = SWEEP_YAML) -> dict:
    from omegaconf import OmegaConf
    y = OmegaConf.load(yaml_path)
    return OmegaConf.to_container(y['defaults'], resolve=True)


def load_taskset_tasks(taskset: str, yaml_path: Path = SWEEP_YAML) -> list[str]:
    """The task list the taskset DEFINES, which is what the MoE geometry was
    built from -- a run part-way through the sequence has fewer started
    tasks than this."""
    from omegaconf import OmegaConf
    y = OmegaConf.load(yaml_path)
    try:
        return list(y['tasksets'][taskset]['tasks'])
    except Exception:
        return []


def load_task_steps(taskset: str, yaml_path: Path = SWEEP_YAML) -> list[int]:
    from omegaconf import OmegaConf
    y = OmegaConf.load(yaml_path)
    try:
        return list(y['methods']['prismatic_seq_progressive']
                    ['tasksets'][taskset]['steps'])
    except Exception:
        return []


def discover_runs(logdir: Path = SWEEP_LOGDIR,
                  variant: str | None = 'preEncFr') -> list[RunSpec]:
    """Scan the sweep logdir for run directories.

    `variant=None` returns every seed dir; the default keeps only the
    pretrained-frozen-encoder runs, which are the paper's main setting.
    """
    runs: list[RunSpec] = []
    if not logdir.is_dir():
        return runs
    for taskset_dir in sorted(p for p in logdir.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in taskset_dir.iterdir() if p.is_dir()):
            if variant is not None and variant not in seed_dir.name:
                continue
            prog_file = seed_dir / 'sequential_progress.json'
            if not prog_file.exists():
                continue
            try:
                prog = json.loads(prog_file.read_text())['tasks']
            except Exception:
                continue
            keys = sorted(prog, key=int)
            tasks = [prog[k]['task_name'] for k in keys]
            completed = [bool(prog[k].get('completed')) for k in keys]
            task_dirs = [seed_dir / f'task{i + 1}_{t}'
                         for i, t in enumerate(tasks)]
            runs.append(RunSpec(
                yaml_num_tasks=len(load_taskset_tasks(taskset_dir.name)),
                taskset=taskset_dir.name,
                seed_dir=seed_dir.name,
                root=seed_dir,
                tasks=tasks,
                task_dirs=task_dirs,
                completed=completed,
                task_steps=load_task_steps(taskset_dir.name),
            ))
    return runs


def finished_runs(**kw) -> list[RunSpec]:
    return [r for r in discover_runs(**kw) if r.is_finished()]


def final_task_runs(min_final_steps: int = 0, **kw) -> list[RunSpec]:
    """Runs that have reached the last task of their taskset with at least
    `min_final_steps` of training into it -- finished or not."""
    return [r for r in discover_runs(**kw)
            if r.reached_final_task(min_final_steps)]
