"""Non-invasive interventions on the dynamics MoE routing.

Everything here works by temporarily rebinding `MaskedMoEBlock._route_masked`
on a single *instance*, then restoring it. No file in `progmoe_training/`,
`common/` or `sequential/` is touched, and no checkpoint is modified.

Ablation semantics
------------------
The default (`keep_stack=True`) masks the gate *logits* before the softmax
and leaves the expert feature stack untouched. That matters because
`use_orthogonal=True` runs Gram-Schmidt across the K experts in index order:
if we dropped experts from the stack, the surviving experts' basis vectors
would change too, and we could no longer attribute the resulting error to
the removed experts alone. Keeping the stack holds every surviving expert's
feature vector bit-identical to the unablated run and changes only the
mixing coefficients -- a clean attribution.

`keep_stack=False` gives the alternative "true removal" reading, where the
survivors are re-orthogonalised among themselves. Both are reported so the
Gram-Schmidt ordering caveat is visible rather than hidden.

Renormalisation
---------------
`renormalize=True` (default) redistributes the ablated mass across the
survivors -- the standard MoE ablation, asking "can the remaining experts do
the job if forced to carry the full mixture?". `renormalize=False` zeroes the
ablated coefficients without redistributing, shrinking the combined feature
vector; that asks "how much of the output signal did these experts supply?".
"""

from __future__ import annotations

import contextlib
import types

import torch
import torch.nn.functional as F


# =======================================================================
# Low-level: swap in a patched _route_masked for the duration of a block
# =======================================================================

@contextlib.contextmanager
def routing_intervention(block, weight_fn):
    """Temporarily replace `block._route_masked` with a version that routes
    through `weight_fn(weights, raw_feats, ortho) -> (weights, feats)`.

    The RAW (pre-Gram-Schmidt) expert stack is handed to `weight_fn` along
    with the block's own orthogonalisation module, because where the
    orthogonalisation happens relative to the ablation is exactly what
    separates the two ablation semantics -- see `_mask_fn`. A transform that
    receives already-orthonormal vectors cannot express "true removal" at
    all: re-orthonormalising an orthonormal subset is a no-op.

    `block` must be a `MaskedMoEBlock` in eval mode; the patched path
    deliberately drops the tau entropy-feedback update, so running it under
    `.train()` would silently diverge from training behaviour.
    """
    assert not block.training, (
        'routing_intervention must run with the MoE block in eval mode '
        '(the tau schedule update is intentionally not replicated).')

    def patched(self, expert_in, gate_base):
        K_active = self._active_K
        tau_col = torch.full_like(gate_base[..., :1], self.tau)
        gate_in = torch.cat([gate_base, tau_col], dim=-1)
        W_active = self.gate.weight[:K_active]
        logits = F.linear(gate_in, W_active) / max(self.tau, 1e-6)
        weights = F.softmax(logits, dim=-1)

        feats = torch.stack(
            [self.experts[k](expert_in) for k in range(K_active)], dim=-2)

        weights, feats = weight_fn(weights, feats, self._ortho)

        combined = (weights.unsqueeze(-1) * feats).sum(dim=-2)
        return combined, weights, feats

    had_own = '_route_masked' in block.__dict__
    saved = block.__dict__.get('_route_masked')
    block._route_masked = types.MethodType(patched, block)
    try:
        yield
    finally:
        if had_own:
            block._route_masked = saved
        else:
            del block._route_masked


# =======================================================================
# Weight transforms
# =======================================================================

def _mask_fn(keep: torch.Tensor, *, renormalize: bool, keep_stack: bool):
    """keep: bool tensor of shape [active_K].

    The two semantics differ only in where the ablation sits relative to
    Gram-Schmidt:

      keep_stack=True   orthogonalise the FULL stack, then zero the ablated
                        coefficients. Every surviving expert's basis vector
                        is bit-identical to the unablated run, so the error
                        is attributable to the removed experts alone.
      keep_stack=False  drop the ablated experts from the RAW stack, then
                        orthogonalise the survivors among themselves. This
                        is "true removal": the survivors get re-based, and
                        experts that were previously only contributing a
                        residual direction can now claim the raw one.
    """

    def fn(weights, feats, ortho):
        k = keep.to(weights.device)
        if keep_stack:
            if ortho is not None:
                feats = ortho(feats)
            w = weights * k
            if renormalize:
                w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return w, feats
        idx = torch.nonzero(k, as_tuple=False).squeeze(-1)
        f = feats.index_select(-2, idx)
        if ortho is not None:
            f = ortho(f)
        w = weights.index_select(-1, idx)
        if renormalize:
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return w, f

    return fn


def _override_fn(new_weights: torch.Tensor):
    """Replace the learned routing distribution wholesale (controls)."""

    def fn(weights, feats, ortho):
        if ortho is not None:
            feats = ortho(feats)
        w = new_weights.to(weights.device, weights.dtype)
        w = w.expand_as(weights) if w.ndim == 1 else w
        return w, feats

    return fn


# =======================================================================
# Condition catalogue
# =======================================================================

def build_conditions(active_K: int, K_per_task: int, task_idx: int, *,
                     learned_weights=None, per_expert=True,
                     seed: int = 0) -> dict:
    """The ablation conditions evaluated on the final task.

    Returns {name: weight_fn or None}. `None` means "no intervention"
    (the unablated reference).

    Groups:
      full            reference
      new_only        every prior task's experts ablated -> can the new
                      block alone still model the final task?
      old_only        the new block ablated -> how far do the frozen prior
                      experts get on their own?
      drop_block{j}   leave-one-block-out
      keep_block{j}   only block j survives
      drop_expert{k}  leave-one-expert-out (the causal per-expert heatmap)
      ctl_uniform     learned routing replaced by uniform -> does the
                      learned mixture matter at all?
      ctl_shuffle     learned coefficients permuted across experts -> does
                      the *assignment* matter, or only the shape?
    """
    n_old = task_idx * K_per_task
    conds: dict = {'full': None}

    def keep_vec(idxs):
        k = torch.zeros(active_K, dtype=torch.bool)
        k[list(idxs)] = True
        return k

    if n_old:
        conds['new_only'] = keep_vec(range(n_old, active_K))
        conds['old_only'] = keep_vec(range(0, n_old))
    for j in range(task_idx + 1):
        lo, hi = j * K_per_task, (j + 1) * K_per_task
        if task_idx > 0:
            conds[f'drop_block{j}'] = keep_vec(
                [i for i in range(active_K) if not (lo <= i < hi)])
            conds[f'keep_block{j}'] = keep_vec(range(lo, hi))
    if per_expert:
        for k in range(active_K):
            conds[f'drop_expert{k}'] = keep_vec(
                [i for i in range(active_K) if i != k])
    return conds


def make_weight_fns(conds: dict, *, renormalize=True, keep_stack=True,
                    learned_weights=None, active_K=None, seed=0) -> dict:
    """Turn the condition catalogue into callables usable with
    `routing_intervention`."""
    out = {}
    for name, keep in conds.items():
        if keep is None:
            out[name] = None
        else:
            out[name] = _mask_fn(keep, renormalize=renormalize,
                                 keep_stack=keep_stack)
    if learned_weights is not None and active_K is not None:
        lw = torch.as_tensor(learned_weights, dtype=torch.float32)
        out['ctl_uniform'] = _override_fn(
            torch.full((active_K,), 1.0 / active_K))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(active_K, generator=g)
        out['ctl_shuffle'] = _override_fn(lw[perm])
    return out
