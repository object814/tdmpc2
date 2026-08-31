"""Analytic router (gate) analysis for the progressive-MoE dynamics block.

Why this is analytic and not sampled
------------------------------------
In `PrismBackbone` the dynamics MoE gate is wired as

    gate_dim  = num_tasks
    gate_base = task_one_hot            (NOT [z, a])
    gate_in   = [task_one_hot, tau]     -> shape (num_tasks + 1,)
    logits    = gate.weight[:active_K] @ gate_in / tau
    weights   = softmax(logits)

so the gate never sees the latent or the action. For a fixed task index the
routing distribution is a single constant vector over experts, identical at
every state, timestep and episode. Sampling it from rollouts would return
thousands of copies of the same row, so we compute it in closed form from
`gate.weight` and `_tau_buf`.

The expert *inputs* do depend on (z, a) — only the mixing coefficients are
constant. Because `use_orthogonal=True` runs Gram-Schmidt over the expert
feature stack, those coefficients sit in an orthonormal basis, so
w_k is exactly the coordinate of expert k's (orthogonalized) contribution to
the combined feature: ||combined||^2 = sum_k w_k^2. Router weight and
contribution magnitude coincide here; they would not in a non-orthogonal MoE.

Baselines
---------
Two nulls are reported for "how much mass sits on old experts":

  uniform null   t / (t + 1)
      what you get if the router never learned anything, since a fresh
      column of `gate.weight` is initialised at ~1e-3 * randn and softmax
      over (t+1)*K near-equal logits is near-uniform.

  start-of-task null   (preferred)
      the router distribution obtained from the *previous* task's
      checkpoint, evaluated at task t's column. Because a one-hot gate input
      zeroes the gradient on every column but the active one, column t is
      untouched until task t begins -- so `W_after_task_{t-1}[:, t]` is
      literally the initial condition of task t's router. This is an exact
      per-run null, not an assumption.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .runspec import RunSpec

TAU_COL = -1  # last gate input column is the scalar tau context


def _torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False,
                          mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location='cpu', weights_only=False)


# =======================================================================
# Core: reconstruct routing distributions from a checkpoint
# =======================================================================

def _load_gate(path) -> tuple[torch.Tensor, float, int, int]:
    # mmap: we need four small tensors out of a ~150MB checkpoint, and the
    # router-trajectory figure touches ~20 of them per task.
    sd = _torch_load(path)['model']
    W = sd['_dynamics.gate.weight'].float()
    tau = float(sd['_dynamics._tau_buf'])
    active_K = int(sd['_dynamics._active_K_buf'])
    frozen_K = int(sd['_dynamics._frozen_K_buf'])
    return W, tau, active_K, frozen_K


def routing_weights(W: torch.Tensor, tau: float, task_idx: int,
                    active_K: int) -> np.ndarray:
    """Softmax routing distribution for one task over its active experts.

    Returns a length-`active_K` array. Mirrors
    `MaskedMoEBlock._route_masked` exactly: slice the gate rows to
    `active_K` *before* the linear, divide logits by tau, softmax.
    """
    num_tasks = W.shape[1] - 1
    gate_in = torch.zeros(num_tasks + 1)
    gate_in[task_idx] = 1.0
    gate_in[TAU_COL] = tau
    logits = F.linear(gate_in, W[:active_K]) / max(tau, 1e-6)
    return F.softmax(logits, dim=-1).numpy()


def gate_table(run: RunSpec) -> dict:
    """Full routing picture for one run.

    Every row is taken from that task's *own* end-of-task checkpoint, using
    that task's own tau and its own active_K window -- i.e. each row is the
    distribution the model actually used while that task was live.

    Returns a dict with:
      W            [total_K, num_tasks+1]  final gate weight matrix
      weights      [num_tasks, total_K]    routing rows, NaN outside the
                                           task's active window
      weights_start[num_tasks, total_K]    same, but from the *previous*
                                           task's checkpoint (the exact
                                           start-of-task null); row 0 is NaN
      taus, active_Ks, frozen_Ks           per task
      columns_frozen  bool  whether every prior task's gate column is
                            bit-identical in the final checkpoint (routing
                            isolation check)
      column_drift    per-task max |column change| after that task ended
    """
    geom = run.moe_geometry()
    total_K, K = geom['total_K'], geom['K_per_task']
    nT = run.num_tasks

    Ws, taus, aKs, fKs = [], [], [], []
    steps, at_end = [], []
    for i in range(nT):
        # Earlier tasks always have their end-of-task checkpoint (enforced by
        # `reached_final_task`); the final one may still be in flight, in
        # which case its newest periodic snapshot stands in.
        path, step, is_end = run.latest_backbone(i)
        if path is None:
            raise FileNotFoundError(
                f'{run.name}: no backbone checkpoint for task {i + 1}')
        W, tau, aK, fK = _load_gate(path)
        Ws.append(W)
        taus.append(tau)
        aKs.append(aK)
        fKs.append(fK)
        steps.append(step)
        at_end.append(is_end)

    weights = np.full((nT, total_K), np.nan, dtype=np.float64)
    weights_start = np.full((nT, total_K), np.nan, dtype=np.float64)
    for t in range(nT):
        aK = aKs[t]
        weights[t, :aK] = routing_weights(Ws[t], taus[t], t, aK)
        if t > 0:
            # Column t inside the PREVIOUS checkpoint == task t's initial
            # router, evaluated over task t's (already opened) window.
            weights_start[t, :aK] = routing_weights(Ws[t - 1], taus[t], t, aK)

    # Routing isolation: once task j ends, column j must never move again,
    # because every later task feeds a one-hot that zeroes its gradient.
    drift = []
    for j in range(nT - 1):
        d = (Ws[-1][:, j] - Ws[j][:, j]).abs().max().item()
        drift.append(d)
    tau_col_drift = (Ws[-1][:, TAU_COL] - Ws[0][:, TAU_COL]).abs().max().item()

    final_step, final_budget = run.final_task_progress()
    return dict(
        run=run.name,
        taskset=run.taskset,
        seed_dir=run.seed_dir,
        tasks=run.short_tasks(),
        num_tasks=nT,
        K_per_task=K,
        total_K=total_K,
        W=Ws[-1].numpy(),
        weights=weights,
        weights_start=weights_start,
        taus=taus,
        active_Ks=aKs,
        frozen_Ks=fKs,
        # Checkpoint provenance per row: which step it came from, and whether
        # it is the settled end-of-task file or a mid-training snapshot.
        row_steps=steps,
        row_at_end=at_end,
        final_complete=bool(at_end[-1]),
        final_step=int(final_step),
        final_budget=int(final_budget),
        final_frac=(float(final_step) / final_budget
                    if final_budget and final_budget > 0 else float('nan')),
        column_drift=drift,
        tau_col_drift=tau_col_drift,
        columns_frozen=bool(all(d == 0.0 for d in drift)),
        tau_col_frozen=bool(tau_col_drift == 0.0),
    )


# =======================================================================
# Metrics
# =======================================================================

def _block(k_idx: int, K: int) -> int:
    """Which task's expert block index `k_idx` belongs to."""
    return k_idx // K


def block_masses(w: np.ndarray, t: int, K: int) -> np.ndarray:
    """Total routing mass task t puts on each expert block 0..t."""
    return np.array([w[j * K:(j + 1) * K].sum() for j in range(t + 1)])


def entropy(w: np.ndarray) -> float:
    w = w[np.isfinite(w)]
    w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def task_metrics(tab: dict, t: int) -> dict:
    """Reuse metrics for task t (0-indexed). Task 0 has no prior experts, so
    the reuse fields are NaN there."""
    K, nT = tab['K_per_task'], tab['num_tasks']
    aK = tab['active_Ks'][t]
    w = tab['weights'][t, :aK]
    w_start = tab['weights_start'][t, :aK]

    n_old = t * K
    old_mass = float(w[:n_old].sum()) if n_old else float('nan')
    new_mass = float(w[n_old:].sum())
    uniform_null = t / (t + 1) if t > 0 else float('nan')
    start_null = float(w_start[:n_old].sum()) if (t > 0 and np.isfinite(w_start).all()) else float('nan')

    H = entropy(w)
    eff = math.exp(H)
    top = int(np.argmax(w))

    out = dict(
        task_idx=t,
        task=tab['tasks'][t],
        active_K=aK,
        tau=tab['taus'][t],
        # --- headline reuse numbers ---
        old_mass=old_mass,
        new_mass=new_mass,
        uniform_null=uniform_null,
        start_null=start_null,
        # >1 means the router moved mass ONTO old experts relative to the null
        reuse_index_uniform=old_mass / uniform_null if t > 0 else float('nan'),
        reuse_index_start=old_mass / start_null if (t > 0 and start_null == start_null and start_null > 0) else float('nan'),
        # --- concentration ---
        entropy=H,
        effective_experts=eff,
        effective_experts_norm=eff / aK,
        top_expert=top,
        top_expert_weight=float(w[top]),
        top_expert_is_old=bool(top < n_old),
        # uniform weight is 1/aK, so this is "how many times uniform"
        top_expert_ratio=float(w[top] * aK),
        # --- how much the router MOVED during this task ---
        router_l1_movement=(float(np.abs(w - w_start).sum())
                            if t > 0 and np.isfinite(w_start).all()
                            else float('nan')),
        # --- per-block breakdown ---
        block_mass=block_masses(w, t, K).tolist(),
        # block mass relative to its uniform share 1/(t+1)
        block_index=(block_masses(w, t, K) * (t + 1)).tolist(),
        weights=w.tolist(),
    )

    # Best OLD expert specifically: a single prior expert carrying far more
    # than its uniform share is much stronger evidence of targeted reuse
    # than diffuse leftover mass.
    if n_old:
        old_top = int(np.argmax(w[:n_old]))
        out.update(
            top_old_expert=old_top,
            top_old_expert_block=_block(old_top, K),
            top_old_expert_weight=float(w[old_top]),
            top_old_expert_ratio=float(w[old_top] * aK),
        )
    return out


def _norm(v):
    s = v.sum()
    return v / s if s > 0 else v


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def router_copy_similarity(tab: dict, t: int) -> dict:
    """Does task t reproduce task j's *internal ordering* over task j's own
    experts?

    For each prior block j we take task t's weights restricted to block j,
    renormalise, and compare against task j's own weights over the same
    block. High similarity means the later task re-derived the preference
    structure the original task learned -- a stronger reuse signal than raw
    mass, because it is about WHICH expert inside the block, not how much of
    the block.

    The shared-tau confound
    -----------------------
    The gate logit for expert k under task t is

        W[k, t] + tau * W[k, -1]

    and the tau column is SHARED by every task (and frozen after task 0 under
    sdp_freeze). If the task columns were small, every task's distribution
    would be dominated by that common term and would agree with every other
    task's for free -- the similarity would measure nothing.

    So three numbers are reported per block:

      cosine            the full comparison, as the model actually routes
      cosine_no_tau     the same comparison with the shared tau term dropped,
                        i.e. driven only by the two independently-trained
                        task columns. This is the honest number.
      cosine_tau_only   how close each task's block distribution is to the
                        one the tau column ALONE would induce. If this is
                        ~1 for both tasks, the agreement is an artifact and
                        `cosine` should be discarded.
    """
    K = tab['K_per_task']
    aK = tab['active_Ks'][t]
    W = tab['W']
    tau = tab['taus'][t]
    w_t = tab['weights'][t, :aK]
    out = {}
    for j in range(t):
        sl = slice(j * K, (j + 1) * K)
        a = _norm(w_t[sl])
        b = _norm(tab['weights'][j, sl])
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue

        # Task-column-only distributions: same softmax, tau term removed.
        def soft(x):
            x = x - x.max()
            e = np.exp(x / max(tau, 1e-6))
            return e / e.sum()

        a_nt = soft(W[sl, t])
        b_nt = soft(W[sl, j])
        tau_only = soft(tau * W[sl, TAU_COL])

        corr = (float(np.corrcoef(a, b)[0, 1])
                if K > 1 and a.std() > 0 and b.std() > 0 else float('nan'))
        out[f'block{j}'] = dict(
            cosine=_cos(a, b),
            pearson=corr,
            argmax_match=bool(np.argmax(a) == np.argmax(b)),
            cosine_no_tau=_cos(a_nt, b_nt),
            argmax_match_no_tau=bool(np.argmax(a_nt) == np.argmax(b_nt)),
            cosine_tau_only_vs_late=_cos(a, tau_only),
            cosine_tau_only_vs_orig=_cos(b, tau_only),
        )
    return out


def settling(traj: dict) -> dict:
    """How much the router is still moving near the end of training.

    Runs analysed mid-final-task need this: a routing distribution read at
    40% of the step budget is only worth quoting if it has stopped moving.
    Snapshots are evenly spaced (`save_freq`), so the L1 distance between
    consecutive snapshots is directly comparable within a run.
    """
    W, steps = traj['weights'], traj['steps']
    nan = float('nan')
    if W.shape[0] < 3:
        return dict(recent_drift=nan, peak_drift=nan, drift_ratio=nan,
                    settled=None, n_snapshots=int(W.shape[0]))
    aK = int(np.isfinite(W[-1]).sum())
    rows = W[:, :aK]
    d = np.abs(np.diff(rows, axis=0)).sum(-1)     # L1 per snapshot interval
    recent = float(d[-3:].mean())
    peak = float(d.max())
    return dict(
        recent_drift=recent,
        peak_drift=peak,
        drift_ratio=float(recent / peak) if peak > 0 else nan,
        # Below ~2% total L1 movement per snapshot the row has stopped
        # meaningfully changing.
        settled=bool(recent < 0.02),
        n_snapshots=int(W.shape[0]),
        snapshot_steps=[int(s) for s in steps],
    )


def analyze_run(run: RunSpec) -> dict:
    """Complete router analysis for one run."""
    tab = gate_table(run)
    tab['per_task'] = [task_metrics(tab, t) for t in range(tab['num_tasks'])]
    tab['copy_similarity'] = {
        f'task{t}': router_copy_similarity(tab, t)
        for t in range(1, tab['num_tasks'])
    }
    tab['final'] = tab['per_task'][-1]
    tab['trajectory'] = router_trajectory(run, run.final_task_idx)
    tab['settling'] = settling(tab['trajectory'])
    return tab


# =======================================================================
# Router trajectory during a single task (uses periodic checkpoints)
# =======================================================================

def router_trajectory(run: RunSpec, task_idx: int) -> dict:
    """Routing distribution over training steps within one task, read from
    the `models/<step>/backbone.pt` snapshots `save_freq` leaves behind.

    Also prepends the exact start-of-task router (previous task's
    checkpoint) at step 0. Works on runs still in flight.
    """
    geom = run.moe_geometry()
    total_K = geom['total_K']
    steps: list[int] = []
    rows: list[np.ndarray] = []

    if task_idx > 0 and run.backbone_path(task_idx - 1).exists():
        # The previous task's end-of-task checkpoint holds this task's gate
        # column at its initial value, and the tau it starts from.
        W_prev, tau_prev, _, _ = _load_gate(run.backbone_path(task_idx - 1))
        aK = (task_idx + 1) * geom['K_per_task']
        row = np.full(total_K, np.nan)
        row[:aK] = routing_weights(W_prev, tau_prev, task_idx, aK)
        steps.append(0)
        rows.append(row)

    for step, path in run.periodic_checkpoints(task_idx):
        W, tau, aK, _ = _load_gate(path)
        row = np.full(total_K, np.nan)
        row[:aK] = routing_weights(W, tau, task_idx, aK)
        steps.append(step)
        rows.append(row)

    # End-of-task row, when the task actually finished. A task still in
    # flight is already represented by its newest periodic snapshot above.
    if run.backbone_path(task_idx).exists():
        W, tau, aK, _ = _load_gate(run.backbone_path(task_idx))
        row = np.full(total_K, np.nan)
        row[:aK] = routing_weights(W, tau, task_idx, aK)
        final_step = (run.task_steps[task_idx]
                      if task_idx < len(run.task_steps)
                      else (steps[-1] if steps else 0))
        if not steps or final_step > steps[-1]:
            steps.append(int(final_step))
            rows.append(row)

    return dict(steps=steps,
                weights=np.array(rows) if rows else np.zeros((0, total_K)),
                K_per_task=geom['K_per_task'],
                total_K=total_K,
                task_idx=task_idx)
