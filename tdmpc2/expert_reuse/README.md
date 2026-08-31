# Expert-reuse diagnostics

Does the router reuse earlier tasks' experts when learning the final task of a
sequence?

This package is **read-only analysis**. It loads finished checkpoints from
`sweep/logdir/prismatic_seq_progressive/` and writes only into
`expert_reuse/results/` and wandb. No training file is imported by the trainer,
no checkpoint is modified, and no intervention outlives its context manager.

```
runspec.py        find finished runs; rebuild the exact per-task cfg
gate.py           analytic router weights + reuse metrics
interventions.py  temporary expert ablations on the MoE routing
rollout.py        env rollouts, open-loop prediction error, closed-loop MPC eval
plots.py          figures
analyze.py        CLI driver (+ wandb)
submit_slurm.sh   sbatch wrapper for the GPU tier
```

## Running it

```bash
# router tier — pure checkpoint arithmetic, CPU, ~1 min for all runs
cd third_party/tdmpc2/tdmpc2
python expert_reuse/analyze.py --tier router

# + causal tier — needs a GPU (MuJoCo + MPC)
bash expert_reuse/submit_slurm.sh
```

Useful flags: `--tasksets grasp reach`, `--variant all`, `--no-wandb`,
`--episodes N`, `--closedloop-episodes N` (`0` skips the expensive part).

**Which runs get analysed.** By default: every run that has reached the LAST
task of its taskset with at least `--min-final-steps` (200k) of training into
it, finished or not. A run still training its final task is read from its
newest `models/<step>/backbone.pt` snapshot; earlier tasks must still have
closed out properly, since their gate columns are the reference the final task
is compared against. Every figure and table marks in-flight runs with the
fraction of the step budget reached, and reports a `settled` flag (router
moving < 0.02 L1 per snapshot over the last three snapshots) so a number read
at 20% of budget is never mistaken for a converged one. `--finished-only`
restores the strict behaviour; the rollout tier implies it, because it needs
the final task's `task_modules.pt`, which only exists once the task closes
out.

Results: `expert_reuse/results/<taskset>__<seed_dir>/` (figures + `metrics.json`)
and one wandb run per taskset in project `prismatic_moe_reuse`, grouped by
taskset.

## What the model actually does, and what follows for the analysis

`PrismBackbone` holds **one** MoE — the dynamics block. Reward/Q/π/termination
live in the per-task `TaskModules`, so there is exactly one router to study.

The gate is wired (`backbone.py`) as

```
gate_dim  = num_tasks
gate_base = task_one_hot          # NOT [z, a]
gate_in   = [task_one_hot, tau]
logits    = gate.weight[:active_K] @ gate_in / tau
weights   = softmax(logits)
```

**The router never sees the latent or the action.** For a fixed task index the
routing distribution is one constant vector over experts — identical at every
state, timestep and episode. Sampling it from rollouts would return thousands of
copies of the same row, so `gate.py` computes it in closed form. The expert
*inputs* are still `[z, a, one_hot]`; only the mixing coefficients are constant.

Two consequences worth stating in any write-up:

- **Reuse here is a per-task static mixture, not a per-timestep dynamic one.**
  The claim the figures support is "the router allocates a fixed reuse
  coefficient per task", not "the router queries old experts as the state
  demands".
- **Router weight equals contribution magnitude**, because `use_orthogonal=True`
  makes the expert feature stack orthonormal before mixing, so
  `||combined||² = Σ w_k²`. That identity would not hold in a non-orthogonal
  MoE. The caveat is that Gram-Schmidt runs in index order, so old experts
  supply the raw directions and new experts contribute only the residual
  orthogonal to them — a structural advantage for the old experts that the
  causal tier is what actually tests.

## The nulls

A naive "how much mass sits on old experts" number is not evidence on its own.
At the start of task *t*, `gate.weight[:, t]` is still at its `1e-3 * randn`
initialisation, so the softmax over `(t+1)·K` experts starts near-uniform and
old-expert mass starts at ≈ `t/(t+1)` by construction. Both nulls are reported:

- **uniform null** `t/(t+1)`.
- **start-of-task null** (preferred, exact). A one-hot gate input zeroes the
  gradient on every column but the active one, so column *t* is untouched until
  task *t* begins: `W_after_task_{t-1}[:, t]` *is* task *t*'s initial router. The
  analysis reconstructs it from the previous checkpoint and verifies the
  isolation property directly (`check/prior_gate_columns_frozen`).

Because the aggregate is anchored near the null by construction, the informative
statistics are the **per-expert** ones: which single expert dominates, at how
many times its uniform share, and whether it belongs to an earlier task.

## Within-block ordering agreement, and its confound

The strongest available signal is not how much mass the final task puts on old
experts but whether it puts that mass on the *same* experts the original task
preferred. For each prior block, `copy_similarity` compares the final task's
routing restricted to that block against the block's own task's routing over
the same experts.

That comparison has a confound that must be tested, not assumed away. The gate
logit for expert *k* under task *t* is `W[k, t] + tau * W[k, -1]`, and the tau
column is **shared** across tasks (and frozen after task 0 under `sdp_freeze`).
If the task columns were small, every task's distribution would be dominated by
that common term and would agree with every other task's for free. So three
numbers are reported per block:

- `cosine` — the full comparison, as the model actually routes.
- `cosine_no_tau` — the same comparison with the shared tau term dropped, so
  only the two independently-trained gate columns contribute. **This is the
  number to quote.**
- `cosine_tau_only_*` — how close each task's block distribution is to the one
  the tau column alone would induce. If these are ~1 for both tasks, the
  agreement is an artifact and `cosine` must be discarded.

## The causal tier

Router weights show intent, not load-bearing-ness: a large coefficient on an old
expert is still compatible with that expert contributing nothing useful.
`interventions.py` rebinds `MaskedMoEBlock._route_masked` on one instance for the
duration of a `with` block and masks the gate logits.

By default the ablation **keeps the expert stack** and masks only the mixing
coefficients, so every surviving expert's Gram-Schmidt basis vector stays
bit-identical to the unablated run and the resulting error is attributable to the
removed experts alone. `keep_stack=False` gives the alternative "true removal"
reading, where survivors are re-orthogonalised among themselves; both are
computed and reported so the ordering caveat stays visible.

Cost is split deliberately. Episodes are collected **once** with the unablated
agent and their latents cached — the encoder is untouched by any routing
intervention, so only the dynamics rollout is recomputed per condition. That
makes leave-one-expert-out over all K experts nearly free, which is what produces
the per-expert causal row plotted directly under the router row. Only the
headline conditions pay for closed-loop MPC evaluation.

Conditions: `full`, `new_only`, `old_only`, `drop_block{j}`, `keep_block{j}`,
`drop_expert{k}`, plus two controls — `ctl_uniform` (learned routing replaced by
uniform: does the learned mixture matter at all?) and `ctl_shuffle` (coefficients
permuted across experts: does the *assignment* matter, or only the shape?).

## Figures

| file | what it shows |
|---|---|
| `router_heatmap` | tasks × experts routing weights; the headline figure |
| `final_start_vs_end` | final task's router before vs after training (the exact null) |
| `router_trajectory` | router across final-task training steps |
| `router_vs_causal` | router weight above leave-one-out damage, shared expert axis |
| `openloop_ablation`, `closedloop_success`, `closedloop_reward` | ablation conditions |
| `reuse_summary` | old-expert mass vs both nulls, across runs; in-flight runs hatched |
