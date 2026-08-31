"""Driver for the expert-reuse analysis.

    # router tier only — pure checkpoint maths, runs on a CPU node in seconds
    python expert_reuse/analyze.py --tier router

    # add the causal tier — needs a GPU (MuJoCo rollouts + MPC)
    python expert_reuse/analyze.py --tier all

Run from `third_party/tdmpc2/tdmpc2/`. Results land in
`expert_reuse/results/<taskset>__<seed_dir>/` and, unless `--no-wandb`, in one
wandb run per taskset.

Nothing here writes to the training logdirs.
"""

from __future__ import annotations

import os

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('LAZY_LEGACY_OP', '0')
os.environ.setdefault('XDG_RUNTIME_DIR', '/tmp')
os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
for p in (str(_PKG), str(_PKG.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from expert_reuse import gate as gate_mod          # noqa: E402
from expert_reuse import plots                     # noqa: E402
from expert_reuse.runspec import (                 # noqa: E402
    discover_runs, load_sweep_defaults, RunSpec)

DEFAULT_OUTDIR = _HERE / 'results'
DEFAULT_PROJECT = 'prismatic_moe_reuse'
DEFAULT_ENTITY = 'haoyu-a2i'


# =======================================================================
# JSON helper — numpy types are not JSON-serialisable
# =======================================================================

def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


# =======================================================================
# Tier 1 — router
# =======================================================================

def run_router_tier(run: RunSpec, outdir: Path, wb=None) -> dict:
    tab = gate_mod.analyze_run(run)
    figs = {}
    figs['router_heatmap'] = plots.router_heatmap(tab, outdir)
    figs['final_start_vs_end'] = plots.start_vs_end(tab, outdir)

    f = plots.router_trajectory(tab['trajectory'], tab, outdir)
    if f is not None:
        figs['router_trajectory'] = f

    final = tab['final']
    st = tab['settling']
    if tab['final_complete']:
        print(f'  final task trained to completion '
              f'({tab["final_step"]:,} steps)')
    else:
        print(f'  final task STILL TRAINING: analysed at '
              f'{tab["final_step"]:,}/{tab["final_budget"]:,} steps '
              f'({tab["final_frac"]:.0%} of budget)')
    print(f'  router settling: recent drift {st["recent_drift"]:.4f} L1 per '
          f'snapshot (peak {st["peak_drift"]:.3f}) -> '
          f'{"settled" if st["settled"] else "STILL MOVING"}')
    print(f'  routing isolation: prior gate columns unchanged='
          f'{tab["columns_frozen"]} (max drift '
          f'{max(tab["column_drift"]) if tab["column_drift"] else 0:.2e}), '
          f'tau column frozen={tab["tau_col_frozen"]}')
    print(f'  final task "{final["task"]}": old-expert mass='
          f'{final["old_mass"]:.3f}  (uniform null {final["uniform_null"]:.3f},'
          f' start-of-task null {final["start_null"]:.3f})')
    print(f'  top expert #{final["top_expert"]} @ {final["top_expert_weight"]:.3f}'
          f' ({final["top_expert_ratio"]:.1f}x uniform, '
          f'{"OLD" if final["top_expert_is_old"] else "new"})')
    print(f'  router L1 movement during final task: '
          f'{final["router_l1_movement"]:.3f} (max possible 2.0)')
    cp = copy_stats(tab)
    if cp['n']:
        print(f'  within-block ordering agreement with the original task: '
              f'cos={cp["cos"]:.3f} (tau removed {cp["cos_no_tau"]:.3f}, '
              f'tau-only baseline {cp["tau_only"]:.3f}), '
              f'top expert agrees in {cp["match_no_tau"]}/{cp["n"]} blocks')

    if wb is not None:
        wb.log({f'router/{k}': wb_image(wb, v) for k, v in figs.items()})
        wb.summary.update({
            'final/old_mass': final['old_mass'],
            'final/new_mass': final['new_mass'],
            'final/uniform_null': final['uniform_null'],
            'final/start_null': final['start_null'],
            'final/reuse_index_uniform': final['reuse_index_uniform'],
            'final/reuse_index_start': final['reuse_index_start'],
            'final/entropy': final['entropy'],
            'final/effective_experts': final['effective_experts'],
            'final/top_expert': final['top_expert'],
            'final/top_expert_weight': final['top_expert_weight'],
            'final/top_expert_ratio': final['top_expert_ratio'],
            'final/top_expert_is_old': final['top_expert_is_old'],
            'final/router_l1_movement': final['router_l1_movement'],
            'check/prior_gate_columns_frozen': tab['columns_frozen'],
            'check/tau_column_frozen': tab['tau_col_frozen'],
            'maturity/final_complete': tab['final_complete'],
            'maturity/final_step': tab['final_step'],
            'maturity/final_budget': tab['final_budget'],
            'maturity/final_frac': tab['final_frac'],
            'maturity/recent_drift': tab['settling']['recent_drift'],
            'maturity/settled': tab['settling']['settled'],
            'copy/cosine': copy_stats(tab)['cos'],
            'copy/cosine_no_tau': copy_stats(tab)['cos_no_tau'],
            'copy/cosine_tau_only_baseline': copy_stats(tab)['tau_only'],
            'copy/argmax_match': copy_stats(tab)['match_no_tau'],
            'copy/n_blocks': copy_stats(tab)['n'],
        })
        cols = ['task'] + [f'e{k}' for k in range(tab['total_K'])]
        rows = [[tab['tasks'][t]] +
                [None if not np.isfinite(v) else float(v)
                 for v in tab['weights'][t]]
                for t in range(tab['num_tasks'])]
        wb.log({'router/weights_table': wb_table(cols, rows)})

    return dict(table=tab, figures={k: str(v) for k, v in figs.items()})


def copy_stats(tab) -> dict:
    """Aggregate the within-block ordering agreement between the final task
    and each prior task, over that run's prior blocks."""
    cs = tab['copy_similarity'].get(f'task{tab["num_tasks"] - 1}', {})
    if not cs:
        return dict(n=0, cos=float('nan'), cos_no_tau=float('nan'),
                    match=0, match_no_tau=0, tau_only=float('nan'))
    v = list(cs.values())
    return dict(
        n=len(v),
        cos=float(np.mean([b['cosine'] for b in v])),
        cos_no_tau=float(np.mean([b['cosine_no_tau'] for b in v])),
        match=int(sum(b['argmax_match'] for b in v)),
        match_no_tau=int(sum(b['argmax_match_no_tau'] for b in v)),
        tau_only=float(np.mean([b['cosine_tau_only_vs_late'] for b in v])),
    )


def wb_image(wb, path):
    # `Image`/`Table` are module-level factories; the Run object only logs.
    import wandb
    return wandb.Image(str(path))


def wb_table(columns, data):
    import wandb
    return wandb.Table(columns=columns, data=data)


def stable_run_id(name: str) -> str:
    """Deterministic wandb id so re-running the analysis updates the same
    run instead of littering the project with duplicates."""
    import hashlib
    return hashlib.md5(name.encode()).hexdigest()[:8]


# =======================================================================
# Tier 2 — causal
# =======================================================================

def run_rollout_tier(run: RunSpec, tab: dict, outdir: Path, args,
                     wb=None) -> dict:
    from expert_reuse import rollout as ro
    from expert_reuse.interventions import build_conditions, make_weight_fns

    defaults = load_sweep_defaults()
    geom = run.moe_geometry()
    t = run.final_task_idx
    aK = geom['active_K']
    K = geom['K_per_task']

    agent, cfg, env = ro.load_final_agent(run, defaults, geom)
    learned = tab['weights'][t, :aK]

    try:
        print(f'  collecting {args.episodes} episodes on the final task...')
        eps = ro.collect_episodes(agent, env, args.episodes,
                                  seed=args.seed)
        collected = dict(
            success=float(np.mean([e['success'] for e in eps])),
            ep_reward=float(np.mean([e['ep_reward'] for e in eps])),
            n_transitions=int(sum(e['length'] for e in eps)),
        )
        print(f'  collected {collected["n_transitions"]} transitions '
              f'(success={collected["success"]:.2f})')

        conds = build_conditions(aK, K, t, per_expert=True)
        horizon = int(cfg.horizon)

        # ---- open-loop, every condition, both ablation semantics ----
        openloop = {}
        for keep_stack in (True, False):
            tag = 'keepstack' if keep_stack else 'trueremoval'
            fns = make_weight_fns(conds, renormalize=True,
                                  keep_stack=keep_stack,
                                  learned_weights=learned, active_K=aK,
                                  seed=args.seed)
            res = {}
            for name, fn in fns.items():
                res[name] = ro.openloop_error(
                    agent, eps, horizon, weight_fn=fn,
                    max_starts=args.max_starts)
            openloop[tag] = res
            base = res['full']['mse']
            print(f'  open-loop ({tag}): baseline MSE={base:.5f}')

        # Relative damage vs unablated, on the primary (keep-stack) semantics
        primary = openloop['keepstack']
        base_mse = primary['full']['mse']
        rel = {name: (r['mse'] - base_mse) / max(base_mse, 1e-12)
               for name, r in primary.items()}

        figs = {}
        figs['router_vs_causal'] = plots.causal_heatmap(tab, rel, outdir)
        figs['openloop_ablation'] = plots.ablation_bars(
            primary, tab, outdir, metric='mse',
            ylabel='open-loop latent MSE', name='openloop_ablation',
            higher_is_better=False)

        # ---- closed-loop, headline conditions only (expensive) ----
        closedloop = {}
        if args.closedloop_episodes > 0:
            names = ['full']
            if t > 0:
                names += ['new_only', 'old_only']
            names += ['ctl_uniform', 'ctl_shuffle']
            names += [f'drop_block{j}' for j in range(t + 1)]
            fns = make_weight_fns(conds, renormalize=True, keep_stack=True,
                                  learned_weights=learned, active_K=aK,
                                  seed=args.seed)
            for name in names:
                if name not in fns:
                    continue
                print(f'  closed-loop [{name}] '
                      f'({args.closedloop_episodes} episodes)...')
                closedloop[name] = ro.closedloop_eval(
                    agent, env, args.closedloop_episodes,
                    weight_fn=fns[name], seed=args.seed, tag=name)
            figs['closedloop_success'] = plots.ablation_bars(
                closedloop, tab, outdir, metric='episode_success',
                ylabel='success rate', name='closedloop_success',
                higher_is_better=True)
            figs['closedloop_reward'] = plots.ablation_bars(
                closedloop, tab, outdir, metric='episode_reward',
                ylabel='episode return', name='closedloop_reward',
                higher_is_better=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    out = dict(collected=collected, openloop=openloop,
               relative_mse_damage=rel, closedloop=closedloop,
               figures={k: str(v) for k, v in figs.items() if v})

    if wb is not None:
        wb.log({f'causal/{k}': wb_image(wb, v)
                for k, v in figs.items() if v})
        summ = {
            'causal/openloop_mse_full': base_mse,
            'causal/collect_success': collected['success'],
            'causal/collect_reward': collected['ep_reward'],
        }
        for name, r in primary.items():
            summ[f'causal/openloop_mse/{name}'] = r['mse']
            summ[f'causal/openloop_reldamage/{name}'] = rel[name]
        for name, r in closedloop.items():
            summ[f'causal/success/{name}'] = r['episode_success']
            summ[f'causal/reward/{name}'] = r['episode_reward']
        wb.summary.update(summ)

        # per-expert causal table, aligned with the router weights
        rows = [[k, float(learned[k]),
                 float(rel.get(f'drop_expert{k}', np.nan)),
                 'old' if k < t * K else 'new', k // K]
                for k in range(aK)]
        wb.log({'causal/per_expert': wb_table(
            ['expert', 'router_weight', 'rel_mse_damage', 'era',
             'owner_task'],
            [[a, b, (None if not np.isfinite(c) else c), d, e]
             for a, b, c, d, e in rows])})

    return out


# =======================================================================
# Main
# =======================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tier', choices=['router', 'rollout', 'all'],
                   default='router')
    p.add_argument('--variant', default='preEncFr',
                   help="Encoder variant substring to keep ('preEncFr', "
                        "'dino', or 'all').")
    p.add_argument('--tasksets', nargs='*', default=None,
                   help='Restrict to these tasksets (default: all found).')
    p.add_argument('--seed-dirs', nargs='*', default=None,
                   help='Restrict to these seed dir names.')
    p.add_argument('--min-final-steps', type=int, default=200_000,
                   help='Minimum training steps into the FINAL task for a run '
                        'to be analysed. Runs still training use their newest '
                        'models/<step>/backbone.pt snapshot. Set 0 for no '
                        'floor.')
    p.add_argument('--finished-only', action='store_true',
                   help='Only analyse runs that completed every task '
                        '(the old behaviour).')
    p.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    p.add_argument('--episodes', type=int, default=20,
                   help='Episodes collected once for the open-loop tier.')
    p.add_argument('--closedloop-episodes', type=int, default=10,
                   help='Episodes per ablation condition in the env. 0 skips '
                        'the closed-loop tier.')
    p.add_argument('--max-starts', type=int, default=None,
                   help='Subsample rollout start indices per episode.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wandb-project', default=DEFAULT_PROJECT)
    p.add_argument('--wandb-entity', default=DEFAULT_ENTITY)
    p.add_argument('--no-wandb', action='store_true')
    args = p.parse_args()

    variant = None if args.variant in ('all', 'any') else args.variant
    runs = discover_runs(variant=variant)
    if args.tasksets:
        runs = [r for r in runs if r.taskset in args.tasksets]
    if args.seed_dirs:
        runs = [r for r in runs if r.seed_dir in args.seed_dirs]

    # Selection is per TIER, not global. The router tier runs on anything that
    # reached the final task; the rollout tier additionally needs that task's
    # `task_modules.pt`, which only exists once the task closes out. Gating
    # BOTH on `is_finished` would silently shrink the router results whenever
    # the rollout tier is requested, and overwrite the cross-run figure with
    # the smaller set.
    if args.finished_only:
        selected = [r for r in runs if r.is_finished()]
    else:
        selected = [r for r in runs
                    if r.reached_final_task(args.min_final_steps)]
    rejected = [r for r in runs if r not in selected]
    wants_rollout = args.tier in ('rollout', 'all')
    causal_ready = [r for r in selected if r.is_finished()]

    print('=' * 70)
    print(f'expert-reuse analysis   tier={args.tier}   variant={args.variant}')
    if args.finished_only:
        print('selection: fully finished runs only')
    else:
        print(f'selection: reached final task with >= '
              f'{args.min_final_steps:,} steps into it')
    if wants_rollout:
        print(f'causal tier: {len(causal_ready)}/{len(selected)} runs are '
              f'finished and therefore eligible (needs task_modules.pt)')
    print('=' * 70)
    print(f'analysing ({len(selected)}):')
    for r in selected:
        step, budget = r.final_task_progress()
        frac = f'{step / budget:.0%}' if budget > 0 else '?'
        mark = 'complete' if r.is_finished() else f'in flight {frac}'
        if wants_rollout:
            mark += ', router+causal' if r.is_finished() else ', router only'
        print(f'  {r.name:<32} {" -> ".join(r.short_tasks())}  [{mark}]')
    if rejected:
        print(f'skipped ({len(rejected)}):')
        for r in rejected:
            step, budget = r.final_task_progress()
            if r.yaml_num_tasks and r.num_tasks != r.yaml_num_tasks:
                why = (f'only on task {r.num_tasks}/{r.yaml_num_tasks} '
                       f'of the sequence')
            elif step < args.min_final_steps:
                why = (f'final task only {max(step, 0):,} steps '
                       f'(< {args.min_final_steps:,})')
            else:
                why = (r.missing()[0] if r.missing() else 'incomplete')
            print(f'  {r.name:<32} {why}')
    print()
    if not selected:
        print('nothing to analyse.')
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = []

    for run in selected:
        print(f'--- {run.name} ---')
        outdir = args.outdir / run.name
        wb = None
        if not args.no_wandb:
            import wandb
            wb = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=run.name, group=run.taskset, reinit=True,
                id=stable_run_id(run.name), resume='allow',
                dir=str(args.outdir),
                tags=['expert_reuse', run.taskset, run.variant,
                      f'seed{run.seed}', args.tier],
                config=dict(taskset=run.taskset, seed_dir=run.seed_dir,
                            seed=run.seed, variant=run.variant,
                            tasks=run.short_tasks(),
                            num_tasks=run.num_tasks,
                            tier=args.tier,
                            episodes=args.episodes,
                            closedloop_episodes=args.closedloop_episodes,
                            **run.moe_geometry()))
        t0 = time.time()
        record = dict(run=run.name, taskset=run.taskset,
                      seed_dir=run.seed_dir, tasks=run.short_tasks())
        try:
            r1 = run_router_tier(run, outdir, wb=wb)
            record['router'] = r1['table']
            record['router_figures'] = r1['figures']

            if wants_rollout:
                if run.is_finished():
                    record['causal'] = run_rollout_tier(
                        run, r1['table'], outdir, args, wb=wb)
                else:
                    step, budget = run.final_task_progress()
                    why = (f'final task still training '
                           f'({step:,}/{budget:,} steps); the causal tier '
                           f'needs its task_modules.pt, written only at task '
                           f'end')
                    print(f'  causal tier SKIPPED: {why}')
                    record['causal_skipped'] = why
        except Exception:
            traceback.print_exc()
            record['error'] = traceback.format_exc()
        record['seconds'] = time.time() - t0

        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / 'metrics.json').write_text(
            json.dumps(_jsonable(record), indent=2))
        summary.append(record)
        if wb is not None:
            wb.finish()
        print()

    ok = [r for r in summary if 'router' in r and 'error' not in r]
    if ok:
        tables = [r['router'] for r in ok]
        plots.reuse_summary(tables, args.outdir)
        print(f'cross-run summary figure -> {args.outdir}/reuse_summary.png')

        print('\n' + '=' * 88)
        print(f'{"run":<30} {"final":>7} {"old-mass":>9} {"null":>7} '
              f'{"index":>6} {"topexp":>7} {"x-unif":>7} {"settled":>8} '
              f'{"copy-cos":>9} {"top-agree":>10}')
        print('-' * 106)
        for r in ok:
            tab = r['router']
            f = tab['final']
            null = (f['start_null'] if np.isfinite(f['start_null'])
                    else f['uniform_null'])
            idx = f['old_mass'] / null if null else float('nan')
            mat = ('done' if tab['final_complete']
                   else f'{tab["final_frac"]:.0%}')
            st = tab['settling']['settled']
            cp = copy_stats(tab)
            print(f'{r["run"]:<30} {mat:>7} {f["old_mass"]:>9.3f} '
                  f'{null:>7.3f} {idx:>6.2f} '
                  f'{("#" + str(f["top_expert"]) + ("*" if f["top_expert_is_old"] else "")):>7} '
                  f'{f["top_expert_ratio"]:>7.2f} '
                  f'{("yes" if st else "NO" if st is not None else "?"):>8} '
                  f'{cp["cos_no_tau"]:>9.3f} '
                  f'{(str(cp["match_no_tau"]) + "/" + str(cp["n"])):>10}')
        print('=' * 106)
        print('* = the most-used expert on the final task is a frozen expert '
              'from an earlier task')
        print('final = steps into the final task, as a fraction of its budget')
        print('settled = router moved < 0.02 L1 per snapshot over the last 3 '
              'snapshots.')
        print('         For a `done` run the number is final either way; the '
              'flag only qualifies')
        print('         the in-flight rows, whose value can still move.')
        print('copy-cos  = mean cosine between the final task\'s routing '
              'ordering WITHIN each prior')
        print('            block and that block\'s own task\'s ordering, '
              'with the shared tau term')
        print('            removed so only independently-trained gate '
              'columns contribute.')
        print('top-agree = prior blocks where the final task\'s favourite '
              'expert is the same one')
        print('            that block\'s own task favoured (chance = 1/K).')

    (args.outdir / 'summary.json').write_text(
        json.dumps(_jsonable(summary), indent=2))
    print(f'\nwrote {args.outdir}/summary.json')


if __name__ == '__main__':
    main()
