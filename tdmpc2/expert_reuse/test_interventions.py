"""CPU self-test for the routing interventions.

Runs against a real `MaskedMoEBlock` (no GPU, no env, no checkpoints), so the
riskiest part of the analysis -- the monkey-patched forward -- is checked
independently of the cluster.

    python expert_reuse/test_interventions.py
"""

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from progmoe_training.masked_moe import MaskedMoEBlock          # noqa: E402
from expert_reuse.interventions import (                        # noqa: E402
    routing_intervention, build_conditions, make_weight_fns)

FAILED = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'  {detail}' if detail else ''))
    if not cond:
        FAILED.append(name)


def main():
    torch.manual_seed(0)
    num_tasks, K_per_task = 3, 3
    total_K = num_tasks * K_per_task
    latent, action = 16, 4
    task_idx = num_tasks - 1
    active_K = (task_idx + 1) * K_per_task

    block = MaskedMoEBlock(
        latent + action + num_tasks, 32, latent,
        num_experts=total_K, gate_dim=num_tasks, use_orthogonal=True,
    )
    block.set_active_K(active_K)
    block.eval()
    with torch.no_grad():
        block.gate.weight.normal_(0, 1.0)

    N = 8
    z = torch.randn(N, latent)
    a = torch.randn(N, action)
    oh = torch.zeros(N, num_tasks)
    oh[:, task_idx] = 1.0

    with torch.no_grad():
        ref_out, ref_w = block.forward_with_gate(z, a, oh)

    print('constant-routing property')
    check('gate weights identical across all states',
          bool((ref_w - ref_w[0:1]).abs().max() < 1e-6),
          f'max spread {float((ref_w - ref_w[0:1]).abs().max()):.2e}')
    check('routes over active_K only', ref_w.shape[-1] == active_K,
          f'shape {tuple(ref_w.shape)}')

    print('\northonormality (why weight == contribution)')
    with torch.no_grad():
        _, _, feats = block.forward_with_diagnostics(z, a, oh)
    gram = feats[0] @ feats[0].T
    check('expert features orthonormal after Gram-Schmidt',
          bool((gram - torch.eye(active_K)).abs().max() < 1e-4),
          f'max |G - I| = {float((gram - torch.eye(active_K)).abs().max()):.2e}')

    conds = build_conditions(active_K, K_per_task, task_idx, per_expert=True)
    fns = make_weight_fns(conds, renormalize=True, keep_stack=True,
                          learned_weights=ref_w[0], active_K=active_K)

    print('\nintervention correctness')
    keep_all = torch.ones(active_K, dtype=torch.bool)
    from expert_reuse.interventions import _mask_fn
    with torch.no_grad(), routing_intervention(block, _mask_fn(
            keep_all, renormalize=True, keep_stack=True)):
        out_noop, w_noop = block.forward_with_gate(z, a, oh)
    check('ablating nothing reproduces the unablated output exactly',
          bool((out_noop - ref_out).abs().max() < 1e-6),
          f'max diff {float((out_noop - ref_out).abs().max()):.2e}')

    with torch.no_grad(), routing_intervention(block, fns['new_only']):
        _, w_new = block.forward_with_gate(z, a, oh)
    n_old = task_idx * K_per_task
    check('new_only zeroes every prior-task coefficient',
          bool(w_new[:, :n_old].abs().max() < 1e-8))
    check('new_only renormalises to 1',
          bool((w_new.sum(-1) - 1).abs().max() < 1e-5))
    check('new_only preserves relative weights among survivors',
          bool((w_new[0, n_old:] / w_new[0, n_old:].sum()
                - ref_w[0, n_old:] / ref_w[0, n_old:].sum()).abs().max() < 1e-5))

    with torch.no_grad(), routing_intervention(block, fns['old_only']):
        _, w_old = block.forward_with_gate(z, a, oh)
    check('old_only zeroes every new-task coefficient',
          bool(w_old[:, n_old:].abs().max() < 1e-8))

    with torch.no_grad(), routing_intervention(block, fns['ctl_uniform']):
        _, w_u = block.forward_with_gate(z, a, oh)
    check('ctl_uniform is uniform over active experts',
          bool((w_u - 1.0 / active_K).abs().max() < 1e-6))

    with torch.no_grad(), routing_intervention(block, fns['ctl_shuffle']):
        _, w_s = block.forward_with_gate(z, a, oh)
    check('ctl_shuffle permutes the learned coefficients',
          bool(torch.allclose(w_s[0].sort().values, ref_w[0].sort().values,
                              atol=1e-6))
          and not bool(torch.allclose(w_s[0], ref_w[0], atol=1e-6)))

    print('\ntrue-removal semantics')
    fns_tr = make_weight_fns(conds, renormalize=True, keep_stack=False,
                             learned_weights=ref_w[0], active_K=active_K)
    with torch.no_grad(), routing_intervention(block, fns_tr['drop_expert0']):
        out_tr, w_tr = block.forward_with_gate(z, a, oh)
    check('true removal drops the expert from the stack',
          w_tr.shape[-1] == active_K - 1, f'shape {tuple(w_tr.shape)}')
    with torch.no_grad(), routing_intervention(block, fns['drop_expert0']):
        out_ks, _ = block.forward_with_gate(z, a, oh)
    check('keep-stack and true-removal differ under Gram-Schmidt',
          bool((out_tr - out_ks).abs().max() > 1e-6),
          'expected: dropping expert 0 re-bases the survivors')

    print('\nrestoration')
    check('_route_masked restored to the class method after the with-block',
          '_route_masked' not in block.__dict__)
    with torch.no_grad():
        again_out, _ = block.forward_with_gate(z, a, oh)
    check('block returns identical output after all interventions',
          bool((again_out - ref_out).abs().max() < 1e-6))

    print('\nguard')
    block.train()
    try:
        with routing_intervention(block, fns['new_only']):
            pass
        check('refuses to run in train mode', False)
    except AssertionError:
        check('refuses to run in train mode', True)
    block.eval()

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
