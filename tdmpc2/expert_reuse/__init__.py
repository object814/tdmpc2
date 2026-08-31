"""Expert-reuse diagnostics for the progressive-MoE (PRISM-WM) sequential runs.

Read-only analysis package: it loads finished training checkpoints from
`sweep/logdir/prismatic_seq_progressive/` and never mutates them. Nothing in
here is imported by the training pipeline.

See README.md.
"""
