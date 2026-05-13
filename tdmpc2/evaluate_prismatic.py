import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _entropy(p, eps=1e-9):
	p = np.clip(p, eps, 1.0)
	return -(p * np.log(p)).sum(axis=-1)


def _top1(w):
	"""Per-step argmax of gate weights. w: [T, K] -> [T]."""
	return w.argmax(axis=-1)


def _specialization(w):
	"""Episode-mean weight of the most-used expert.
	1/K = perfectly uniform routing, 1.0 = collapsed to one expert."""
	return float(w.mean(axis=0).max())


def _switch_rate(w):
	"""Fraction of consecutive timesteps where the top-1 expert changed.
	0 = perfectly sticky (single expert for whole episode); high = flickering."""
	t = _top1(w)
	if len(t) < 2:
		return 0.0
	return float((t[1:] != t[:-1]).mean())


def _pairwise_cosine_sim(feats):
	"""feats: [T, K, H] -> [K, K] mean pairwise cosine similarity over T.
	Diagonal is 1. Off-diagonal close to 1 means experts are *functionally*
	collapsed (their feature vectors are nearly parallel) — the failure
	mode Gram-Schmidt is meant to prevent."""
	if isinstance(feats, torch.Tensor):
		feats = feats.detach().cpu().numpy()
	norm = np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-9
	unit = feats / norm                                   # [T, K, H]
	sim = np.einsum('tih,tjh->tij', unit, unit)           # [T, K, K]
	return sim.mean(axis=0)                               # [K, K]


def _mean_offdiag(mat):
	K = mat.shape[0]
	mask = ~np.eye(K, dtype=bool)
	return float(mat[mask].mean())


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def _try_import_matplotlib():
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		return plt
	except Exception as e:
		print(colored(f'  matplotlib unavailable ({e}); plots skipped.', 'red'))
		return None


def _expert_colors(plt, K):
	cmap = plt.get_cmap('tab10' if K <= 10 else 'tab20')
	return [cmap(i) for i in range(K)]


def _plot_episode(fp, w_dyn, w_rew, sim_dyn, sim_rew, title):
	plt = _try_import_matplotlib()
	if plt is None:
		return False
	from matplotlib import gridspec

	T, K = w_dyn.shape
	t = np.arange(T)
	colors = _expert_colors(plt, K)
	labels = [f'E{i}' for i in range(K)]

	fig = plt.figure(figsize=(14, 9))
	gs = gridspec.GridSpec(4, 2, figure=fig,
	                       height_ratios=[3, 0.35, 1, 2.4], hspace=0.45, wspace=0.25)

	for col, (name, w, sim) in enumerate([
		('dynamics gate (residual ∆z)', w_dyn, sim_dyn),
		('reward gate', w_rew, sim_rew),
	]):
		ax_area = fig.add_subplot(gs[0, col])
		ax_area.stackplot(t, w.T, labels=labels, colors=colors)
		ax_area.set_title(f'{title} — {name}')
		ax_area.set_ylabel('expert weight')
		ax_area.set_ylim(0, 1)
		ax_area.set_xlim(0, max(T - 1, 1))
		ax_area.legend(loc='upper right', fontsize=8, ncol=K)

		# Top-1 ribbon strip directly under the stacked area.
		ax_strip = fig.add_subplot(gs[1, col], sharex=ax_area)
		top1 = _top1(w)
		ax_strip.imshow(top1[None, :], aspect='auto', interpolation='nearest',
		                cmap=plt.get_cmap('tab10', K),
		                vmin=-0.5, vmax=K - 0.5,
		                extent=(0, max(T - 1, 1), 0, 1))
		ax_strip.set_yticks([])
		ax_strip.set_xlim(0, max(T - 1, 1))
		ax_strip.set_ylabel('top-1', rotation=0, ha='right', va='center', fontsize=8)
		spec = _specialization(w)
		sr = _switch_rate(w)
		ent = (_entropy(w) / np.log(K)).mean()
		ax_strip.set_xlabel(f'step  (spec={spec:.2f}, switch={sr:.2f}, H̄={ent:.2f})',
		                    fontsize=9)

	# Entropy lines (both blocks on one axis).
	ax_ent = fig.add_subplot(gs[2, 0])
	ax_ent.plot(t, _entropy(w_dyn) / np.log(K), label='dynamics', color='C0')
	ax_ent.plot(t, _entropy(w_rew) / np.log(K), label='reward', color='C3', alpha=0.7)
	ax_ent.axhline(1.0, color='k', linestyle=':', linewidth=0.5)
	ax_ent.set_xlim(0, max(T - 1, 1))
	ax_ent.set_ylim(0, 1.05)
	ax_ent.set_ylabel('normalized entropy')
	ax_ent.set_xlabel('step')
	ax_ent.legend(loc='upper right', fontsize=8)

	# Dyn-vs-rew top-1 agreement timeline.
	ax_agree = fig.add_subplot(gs[2, 1], sharex=ax_ent)
	agree = (_top1(w_dyn) == _top1(w_rew)).astype(np.float32)
	ax_agree.fill_between(t, 0, agree, step='pre', color='C2', alpha=0.5)
	ax_agree.set_ylim(0, 1.05)
	ax_agree.set_xlim(0, max(T - 1, 1))
	ax_agree.set_ylabel('dyn==rew top-1')
	ax_agree.set_xlabel('step')
	ax_agree.set_title(f'agreement rate = {agree.mean():.2f}', fontsize=9)

	# Expert feature cosine-similarity heatmaps.
	for col, (name, sim) in enumerate([('dynamics', sim_dyn), ('reward', sim_rew)]):
		ax = fig.add_subplot(gs[3, col])
		im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
		ax.set_xticks(range(K)); ax.set_xticklabels(labels)
		ax.set_yticks(range(K)); ax.set_yticklabels(labels)
		ax.set_title(f'{name} — pairwise expert-feature cos sim   '
		             f'(off-diag mean={_mean_offdiag(sim):.2f})', fontsize=9)
		for i in range(K):
			for j in range(K):
				ax.text(j, i, f'{sim[i, j]:.2f}',
				        ha='center', va='center', fontsize=8,
				        color='white' if abs(sim[i, j]) > 0.5 else 'black')
		fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

	fig.savefig(fp, dpi=120, bbox_inches='tight')
	plt.close(fig)
	return True


def _plot_aggregate(fp, agg, K, task):
	plt = _try_import_matplotlib()
	if plt is None:
		return False
	from matplotlib import gridspec

	colors = _expert_colors(plt, K)
	labels = [f'E{i}' for i in range(K)]
	fig = plt.figure(figsize=(14, 8))
	gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

	# Top: per-step mean weight (over episodes, only where defined),
	# one line per expert, with std band.
	for col, name in enumerate(('dynamics', 'reward')):
		ax = fig.add_subplot(gs[0, col])
		mean = agg[f'{name}_mean']     # [T_max, K]
		std  = agg[f'{name}_std']
		cnt  = agg[f'{name}_count']    # [T_max]
		T_max = mean.shape[0]
		t = np.arange(T_max)
		valid = cnt > 0
		for k in range(K):
			ax.plot(t[valid], mean[valid, k], color=colors[k], label=labels[k])
			ax.fill_between(t[valid], mean[valid, k] - std[valid, k],
			                mean[valid, k] + std[valid, k],
			                color=colors[k], alpha=0.15)
		ax.set_xlim(0, max(T_max - 1, 1))
		ax.set_ylim(0, 1)
		ax.set_title(f'{task} — {name}: mean expert weight vs step (±1 std across episodes)')
		ax.set_xlabel('step')
		ax.set_ylabel('weight')
		ax.legend(loc='upper right', fontsize=8, ncol=K)

	# Top-right: top-1 frequency histogram (across all steps, all episodes).
	ax = fig.add_subplot(gs[0, 2])
	x = np.arange(K)
	w = 0.4
	ax.bar(x - w/2, agg['dyn_top1_freq'], width=w, color='C0', label='dynamics')
	ax.bar(x + w/2, agg['rew_top1_freq'], width=w, color='C3', alpha=0.8, label='reward')
	ax.set_xticks(x); ax.set_xticklabels(labels)
	ax.set_ylabel('top-1 frequency (all eps × steps)')
	ax.set_ylim(0, 1)
	ax.axhline(1.0 / K, color='k', linestyle=':', linewidth=0.7,
	           label=f'uniform = 1/K = {1/K:.2f}')
	ax.legend(loc='upper right', fontsize=8)
	ax.set_title('expert top-1 frequency')

	# Bottom: similarity matrices (mean across episodes).
	for col, name in enumerate(('dynamics', 'reward')):
		ax = fig.add_subplot(gs[1, col])
		sim = agg[f'{name}_sim_mean']
		im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
		ax.set_xticks(range(K)); ax.set_xticklabels(labels)
		ax.set_yticks(range(K)); ax.set_yticklabels(labels)
		ax.set_title(f'{name}: mean pairwise cos sim   '
		             f'(off-diag={_mean_offdiag(sim):.2f})', fontsize=10)
		for i in range(K):
			for j in range(K):
				ax.text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center',
				        fontsize=8, color='white' if abs(sim[i, j]) > 0.5 else 'black')
		fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

	# Bottom-right: text summary.
	ax = fig.add_subplot(gs[1, 2])
	ax.axis('off')
	lines = [
		f'task: {task}',
		f'episodes: {agg["num_episodes"]}',
		f'success rate: {agg["success"]:.2f}',
		f'mean reward:  {agg["reward"]:.1f}',
		'',
		f'dynamics  specialization (mean over eps): {agg["dyn_spec_mean"]:.2f}',
		f'dynamics  switch rate    (mean over eps): {agg["dyn_switch_mean"]:.2f}',
		f'dynamics  entropy        (mean over eps): {agg["dyn_ent_mean"]:.2f}',
		f'dynamics  expert-sim off-diag (mean):     {_mean_offdiag(agg["dynamics_sim_mean"]):.2f}',
		'',
		f'reward    specialization (mean over eps): {agg["rew_spec_mean"]:.2f}',
		f'reward    switch rate    (mean over eps): {agg["rew_switch_mean"]:.2f}',
		f'reward    entropy        (mean over eps): {agg["rew_ent_mean"]:.2f}',
		f'reward    expert-sim off-diag (mean):     {_mean_offdiag(agg["reward_sim_mean"]):.2f}',
		'',
		f'dyn↔rew top-1 agreement (mean over eps):  {agg["dr_agree_mean"]:.2f}',
	]
	ax.text(0.0, 1.0, '\n'.join(lines), va='top', ha='left', family='monospace',
	        fontsize=9)

	fig.suptitle(f'PRISM-MoE aggregate diagnostics — {task}', fontsize=12)
	fig.savefig(fp, dpi=120, bbox_inches='tight')
	plt.close(fig)
	return True


# ----------------------------------------------------------------------
# Aggregation across episodes
# ----------------------------------------------------------------------
def _aggregate_episodes(per_episode, K, task):
	"""per_episode: list of dicts with keys w_dyn, w_rew, sim_dyn, sim_rew,
	reward, success. Episodes can have different T; we right-pad with NaN
	for the per-step mean/std plot and compute count to gray out empty bins."""
	T_max = max(e['w_dyn'].shape[0] for e in per_episode)
	def stack_pad(key):
		out = np.full((len(per_episode), T_max, K), np.nan, dtype=np.float32)
		for i, e in enumerate(per_episode):
			w = e[key]; out[i, :w.shape[0]] = w
		return out
	W_dyn = stack_pad('w_dyn')   # [N, T_max, K]
	W_rew = stack_pad('w_rew')
	dyn_count = np.isfinite(W_dyn[..., 0]).sum(axis=0).astype(np.int64)
	rew_count = np.isfinite(W_rew[..., 0]).sum(axis=0).astype(np.int64)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', category=RuntimeWarning)
		dyn_mean = np.nanmean(W_dyn, axis=0); dyn_std = np.nanstd(W_dyn, axis=0)
		rew_mean = np.nanmean(W_rew, axis=0); rew_std = np.nanstd(W_rew, axis=0)

	# Top-1 frequency across all (episode, step).
	all_dyn = np.concatenate([e['w_dyn'] for e in per_episode], axis=0)
	all_rew = np.concatenate([e['w_rew'] for e in per_episode], axis=0)
	dyn_top1_freq = np.bincount(_top1(all_dyn), minlength=K) / max(all_dyn.shape[0], 1)
	rew_top1_freq = np.bincount(_top1(all_rew), minlength=K) / max(all_rew.shape[0], 1)

	# Mean similarity matrix across episodes.
	dyn_sim = np.stack([e['sim_dyn'] for e in per_episode], axis=0).mean(axis=0)
	rew_sim = np.stack([e['sim_rew'] for e in per_episode], axis=0).mean(axis=0)

	def mean_of(fn, key):
		return float(np.mean([fn(e[key]) for e in per_episode]))
	dr_agree = float(np.mean([
		(_top1(e['w_dyn']) == _top1(e['w_rew'])).mean() for e in per_episode
	]))
	rewards   = np.array([e['reward'] for e in per_episode])
	successes = np.array([e['success'] for e in per_episode])

	return {
		'num_episodes': len(per_episode),
		'reward':   float(rewards.mean()),
		'success':  float(successes.mean()),
		'dynamics_mean': dyn_mean, 'dynamics_std': dyn_std,
		'reward_mean': rew_mean,   'reward_std': rew_std,
		'dynamics_count': dyn_count, 'reward_count': rew_count,
		'dyn_top1_freq': dyn_top1_freq, 'rew_top1_freq': rew_top1_freq,
		'dynamics_sim_mean': dyn_sim, 'reward_sim_mean': rew_sim,
		'dyn_spec_mean':   mean_of(_specialization, 'w_dyn'),
		'rew_spec_mean':   mean_of(_specialization, 'w_rew'),
		'dyn_switch_mean': mean_of(_switch_rate, 'w_dyn'),
		'rew_switch_mean': mean_of(_switch_rate, 'w_rew'),
		'dyn_ent_mean':    float(np.mean([(_entropy(e['w_dyn'])/np.log(K)).mean()
		                                  for e in per_episode])),
		'rew_ent_mean':    float(np.mean([(_entropy(e['w_rew'])/np.log(K)).mean()
		                                  for e in per_episode])),
		'dr_agree_mean':   dr_agree,
		'task': task,
	}


# ----------------------------------------------------------------------
# Diagnostics extraction
# ----------------------------------------------------------------------
@torch.no_grad()
def _step_diagnostics(agent, obs, action, task_idx):
	"""Encode obs, run gate_diagnostics for the executed action. Returns
	(w_dyn [K], w_rew [K], f_dyn [K, H], f_rew [K, H])."""
	device = agent.device
	if isinstance(obs, dict):
		obs_b = {k: v.to(device, non_blocking=True).unsqueeze(0) for k, v in obs.items()}
	elif hasattr(obs, 'keys') and not isinstance(obs, torch.Tensor):
		obs_b = obs.to(device, non_blocking=True).unsqueeze(0)
	else:
		obs_b = obs.to(device, non_blocking=True).unsqueeze(0)
	task_t = torch.tensor([task_idx], device=device) if task_idx is not None else None
	z = agent.model.encode(obs_b, task_t)
	a = action.to(device, non_blocking=True).unsqueeze(0)
	d = agent.model.gate_diagnostics(z, a, task_t)
	return (d['dynamics']['weights'][0].cpu().numpy(),
	        d['reward']['weights'][0].cpu().numpy(),
	        d['dynamics']['features'][0].cpu().numpy(),
	        d['reward']['features'][0].cpu().numpy())


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Evaluate a Prismatic World Model TD-MPC2 checkpoint and visualise the
	dynamics + reward MoE routers AND check functional expert collapse.

	Outputs under {work_dir}/eval_prism/:
	  Per episode:
	    {task}-{i}.mp4                — rollout video
	    {task}-{i}-gates.npz          — raw [T, K] weights + similarity matrices
	    {task}-{i}-gates.png          — stacked area + top-1 ribbon +
	                                    entropy + dyn↔rew agreement +
	                                    expert-feature similarity heatmaps
	  Aggregate (all episodes for this task):
	    aggregate-{task}.png          — mean weights per step, top-1 hist,
	                                    mean similarity matrices, summary table
	    aggregate-{task}.npz          — raw aggregate stats
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)

	if not getattr(cfg, 'use_moe', False):
		print(colored('use_moe is False — this script is for PRISM-WM checkpoints. '
		              'Pass `use_moe=true num_experts=K` matching the training config.',
		              'red', attrs=['bold']))
		return

	K = int(cfg.num_experts)
	print(colored(f'Task:        {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size:  {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint:  {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored(f'PRISM-WM:    K={K}, residual_dynamics={cfg.moe_residual_dynamics}',
	              'blue', attrs=['bold']))

	env = make_env(cfg)
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found.'
	agent.load(cfg.checkpoint)
	agent.model.eval()

	out_dir = os.path.join(cfg.work_dir, 'eval_prism')
	os.makedirs(out_dir, exist_ok=True)

	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		per_ep = []
		for i in range(cfg.eval_episodes):
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			frames = [env.render()] if cfg.save_video else None
			w_dyn_seq, w_rew_seq, f_dyn_seq, f_rew_seq = [], [], [], []
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = agent.act(obs, t0=(t == 0), task=task_idx, eval_mode=True)
				w_dyn, w_rew, f_dyn, f_rew = _step_diagnostics(agent, obs, action, task_idx)
				w_dyn_seq.append(w_dyn); w_rew_seq.append(w_rew)
				f_dyn_seq.append(f_dyn); f_rew_seq.append(f_rew)
				obs, reward, done, info = env.step(action)
				ep_reward += reward; t += 1
				if cfg.save_video:
					frames.append(env.render())

			w_dyn = np.stack(w_dyn_seq, axis=0)              # [T, K]
			w_rew = np.stack(w_rew_seq, axis=0)
			f_dyn = np.stack(f_dyn_seq, axis=0)              # [T, K, H]
			f_rew = np.stack(f_rew_seq, axis=0)
			sim_dyn = _pairwise_cosine_sim(f_dyn)            # [K, K]
			sim_rew = _pairwise_cosine_sim(f_rew)

			tag = f'{task}-{i}'
			np.savez(os.path.join(out_dir, f'{tag}-gates.npz'),
			         dynamics=w_dyn, reward=w_rew,
			         dynamics_sim=sim_dyn, reward_sim=sim_rew,
			         episode_reward=float(ep_reward),
			         success=float(info['success']))
			_plot_episode(
				os.path.join(out_dir, f'{tag}-gates.png'),
				w_dyn, w_rew, sim_dyn, sim_rew,
				title=f'{tag}  R={float(ep_reward):.1f}  S={float(info["success"]):.0f}',
			)
			if cfg.save_video:
				imageio.mimsave(os.path.join(out_dir, f'{tag}.mp4'), frames, fps=15)

			per_ep.append(dict(
				w_dyn=w_dyn, w_rew=w_rew, sim_dyn=sim_dyn, sim_rew=sim_rew,
				reward=float(ep_reward), success=float(info['success']),
			))

			# Console one-liner per episode.
			print(colored(
				f'  {tag:<24} R={ep_reward:.1f}  S={info["success"]:.0f}  '
				f'dyn(spec={_specialization(w_dyn):.2f}, switch={_switch_rate(w_dyn):.2f}, '
				f'sim̄={_mean_offdiag(sim_dyn):+.2f})  '
				f'rew(spec={_specialization(w_rew):.2f}, switch={_switch_rate(w_rew):.2f}, '
				f'sim̄={_mean_offdiag(sim_rew):+.2f})  '
				f'agree={float((_top1(w_dyn)==_top1(w_rew)).mean()):.2f}',
				'yellow'))

		# Aggregate across episodes for this task.
		agg = _aggregate_episodes(per_ep, K, task)
		np.savez(os.path.join(out_dir, f'aggregate-{task}.npz'), **agg)
		_plot_aggregate(os.path.join(out_dir, f'aggregate-{task}.png'), agg, K, task)
		print(colored(
			f'{task:<24}  N={agg["num_episodes"]}  mean R={agg["reward"]:.1f}  '
			f'S={agg["success"]:.2f}   '
			f'dyn(spec={agg["dyn_spec_mean"]:.2f}, sim̄={_mean_offdiag(agg["dynamics_sim_mean"]):+.2f})  '
			f'rew(spec={agg["rew_spec_mean"]:.2f}, sim̄={_mean_offdiag(agg["reward_sim_mean"]):+.2f})  '
			f'agree={agg["dr_agree_mean"]:.2f}',
			'green', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
