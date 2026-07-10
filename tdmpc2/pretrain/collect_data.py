"""Collect an offline Meta-World dataset for encoder pretraining.

Rolls scripted expert policies on Meta-World tasks that are NOT used anywhere
in the PRISM-WM evaluation suite (single-task `sweep_single_prism.yaml` + the
four sequential sweeps), and dumps per-step observations to a single HDF5 file.
The encoder pretrainer (`pretrain/pretrain_encoder.py`) then learns a plain
autoencoder over those observations, so the resulting encoder is general to the
Meta-World visual/proprio distribution without ever having seen an eval task.

Observation format MATCHES the TD-MPC2 metaworld pipeline
(`envs/metaworld_custom.py:_DictObsAdapter`):
    state : (7,)        float32   proprio (ee pos 3 + ee vel 3 + gripper 1)
    rgb   : (3*N,H,W)   uint8     N cameras stacked channel-wise, CHW layout
with image_size / cameras / action_repeat identical to the training default,
so the pretrained encoder drops into `layers.enc(cfg)` with no shape mismatch.

The dataset is capped at `--total-transitions` evenly distributed across the
pretraining tasks. Images are written incrementally to a resizable, gzip-
compressed HDF5 dataset, so peak RAM stays O(one episode) regardless of cap.

--task-set selects which tasks to collect:
    nonEval  (default) held-out tasks (never seen by the world models),
    excluded           the eval-sweep + visual-twin tasks (all single +
                       composition tasks the world models train on),
    all                everything.

--append-to extends an existing dataset: the base file is copied to --out and
the new tasks are appended, with their task_ids continuing the base numbering
(the base file is left untouched). This lets you grow a dataset without
regenerating it.

Usage (inside the apptainer container with metaworld deps):
    # fresh held-out dataset
    python pretrain/collect_data.py --out /Metaworld/.../pretrain_data.h5 \
        --total-transitions 300000
    # append the eval/composition tasks onto an existing dataset
    python pretrain/collect_data.py \
        --append-to /Metaworld/.../pretrain_no_eval/pretrain_data.h5 \
        --out       /Metaworld/.../pretrain_full/pretrain_data.h5 \
        --task-set excluded --total-transitions 200000
    python pretrain/collect_data.py --out /tmp/smoke.h5 --smoke   # tiny test

This script imports only metaworld + gymnasium + numpy + h5py (no torch).
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_LOG_LEVEL", "fatal")

import argparse
import importlib
import json
import math
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Constant.*may be too high")

# Repo root so `import metaworld` resolves regardless of cwd.
_BASE_DIR = Path(__file__).resolve().parents[4]
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import gymnasium as gym  # noqa: E402

gym.logger.min_level = gym.logger.ERROR

import metaworld  # noqa: E402,F401  # registers "Meta-World/MT1"
from metaworld.wrappers import ProprioMultiImageObsWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# env_name -> (policy_module, policy_class). Keys are the exact strings
# accepted by gym.make("Meta-World/MT1", env_name=...) (env_dict.ALL_V3).
# Only tasks with a scripted policy are listed.
# ---------------------------------------------------------------------------
POLICY_REGISTRY = {
    "assembly-v3": ("sawyer_assembly_v3_policy", "SawyerAssemblyV3Policy"),
    "basketball-v3": ("sawyer_basketball_v3_policy", "SawyerBasketballV3Policy"),
    "bin-picking-v3": ("sawyer_bin_picking_v3_policy", "SawyerBinPickingV3Policy"),
    "bin-picking-redblue-v3": ("sawyer_bin_picking_v3_policy", "SawyerBinPickingV3Policy"),
    "bin-picking-yellowblue-v3": ("sawyer_bin_picking_v3_policy", "SawyerBinPickingV3Policy"),
    "bin-picking-redpurple-v3": ("sawyer_bin_picking_v3_policy", "SawyerBinPickingV3Policy"),
    "bin-picking-yellowpurple-v3": ("sawyer_bin_picking_v3_policy", "SawyerBinPickingV3Policy"),
    "box-close-v3": ("sawyer_box_close_v3_policy", "SawyerBoxCloseV3Policy"),
    "button-press-topdown-v3": ("sawyer_button_press_topdown_v3_policy", "SawyerButtonPressTopdownV3Policy"),
    "button-press-topdown-wall-v3": ("sawyer_button_press_topdown_wall_v3_policy", "SawyerButtonPressTopdownWallV3Policy"),
    "button-press-v3": ("sawyer_button_press_v3_policy", "SawyerButtonPressV3Policy"),
    "button-press-wall-v3": ("sawyer_button_press_wall_v3_policy", "SawyerButtonPressWallV3Policy"),
    "coffee-button-v3": ("sawyer_coffee_button_v3_policy", "SawyerCoffeeButtonV3Policy"),
    "coffee-pull-v3": ("sawyer_coffee_pull_v3_policy", "SawyerCoffeePullV3Policy"),
    "coffee-push-v3": ("sawyer_coffee_push_v3_policy", "SawyerCoffeePushV3Policy"),
    "dial-turn-v3": ("sawyer_dial_turn_v3_policy", "SawyerDialTurnV3Policy"),
    "disassemble-v3": ("sawyer_disassemble_v3_policy", "SawyerDisassembleV3Policy"),
    "door-close-v3": ("sawyer_door_close_v3_policy", "SawyerDoorCloseV3Policy"),
    "door-lock-v3": ("sawyer_door_lock_v3_policy", "SawyerDoorLockV3Policy"),
    "door-open-v3": ("sawyer_door_open_v3_policy", "SawyerDoorOpenV3Policy"),
    "door-unlock-v3": ("sawyer_door_unlock_v3_policy", "SawyerDoorUnlockV3Policy"),
    "drawer-close-v3": ("sawyer_drawer_close_v3_policy", "SawyerDrawerCloseV3Policy"),
    "drawer-open-v3": ("sawyer_drawer_open_v3_policy", "SawyerDrawerOpenV3Policy"),
    "faucet-close-v3": ("sawyer_faucet_close_v3_policy", "SawyerFaucetCloseV3Policy"),
    "faucet-open-v3": ("sawyer_faucet_open_v3_policy", "SawyerFaucetOpenV3Policy"),
    "grasp-v3": ("sawyer_grasp_policy", "SawyerGraspV3Policy"),
    "hammer-v3": ("sawyer_hammer_v3_policy", "SawyerHammerV3Policy"),
    "hand-insert-v3": ("sawyer_hand_insert_v3_policy", "SawyerHandInsertV3Policy"),
    "handle-press-side-v3": ("sawyer_handle_press_side_v3_policy", "SawyerHandlePressSideV3Policy"),
    "handle-press-v3": ("sawyer_handle_press_v3_policy", "SawyerHandlePressV3Policy"),
    "handle-pull-side-v3": ("sawyer_handle_pull_side_v3_policy", "SawyerHandlePullSideV3Policy"),
    "handle-pull-v3": ("sawyer_handle_pull_v3_policy", "SawyerHandlePullV3Policy"),
    "lever-pull-v3": ("sawyer_lever_pull_v3_policy", "SawyerLeverPullV3Policy"),
    "peg-insert-side-v3": ("sawyer_peg_insertion_side_v3_policy", "SawyerPegInsertionSideV3Policy"),
    "peg-unplug-side-v3": ("sawyer_peg_unplug_side_v3_policy", "SawyerPegUnplugSideV3Policy"),
    "pick-out-of-hole-v3": ("sawyer_pick_out_of_hole_v3_policy", "SawyerPickOutOfHoleV3Policy"),
    "pick-place-block-v3": ("sawyer_pick_place_block_v3_policy", "SawyerPickPlaceBlockV3Policy"),
    "pick-place-redblock-v3": ("sawyer_pick_place_block_v3_policy", "SawyerPickPlaceBlockV3Policy"),
    "pick-place-greenblock-v3": ("sawyer_pick_place_block_v3_policy", "SawyerPickPlaceBlockV3Policy"),
    "pick-place-v3": ("sawyer_pick_place_v3_policy", "SawyerPickPlaceV3Policy"),
    "pick-place-wall-v3": ("sawyer_pick_place_wall_v3_policy", "SawyerPickPlaceWallV3Policy"),
    "plate-slide-v3": ("sawyer_plate_slide_v3_policy", "SawyerPlateSlideV3Policy"),
    "plate-slide-side-v3": ("sawyer_plate_slide_side_v3_policy", "SawyerPlateSlideSideV3Policy"),
    "plate-slide-back-v3": ("sawyer_plate_slide_back_v3_policy", "SawyerPlateSlideBackV3Policy"),
    "plate-slide-back-side-v3": ("sawyer_plate_slide_back_side_v3_policy", "SawyerPlateSlideBackSideV3Policy"),
    "push-v3": ("sawyer_push_v3_policy", "SawyerPushV3Policy"),
    "push-wall-v3": ("sawyer_push_wall_v3_policy", "SawyerPushWallV3Policy"),
    "push-back-v3": ("sawyer_push_back_v3_policy", "SawyerPushBackV3Policy"),
    "reach-v3": ("sawyer_reach_v3_policy", "SawyerReachV3Policy"),
    "reach-wall-v3": ("sawyer_reach_wall_v3_policy", "SawyerReachWallV3Policy"),
    "reach-xy-v3": ("sawyer_reach_xy_v3_policy", "SawyerReachXYV3Policy"),
    "reach-xz-v3": ("sawyer_reach_xz_v3_policy", "SawyerReachXZV3Policy"),
    "reach-yz-v3": ("sawyer_reach_yz_v3_policy", "SawyerReachYZV3Policy"),
    "reach-xyz-v3": ("sawyer_reach_xyz_v3_policy", "SawyerReachXYZV3Policy"),
    "shelf-place-v3": ("sawyer_shelf_place_v3_policy", "SawyerShelfPlaceV3Policy"),
    "soccer-v3": ("sawyer_soccer_v3_policy", "SawyerSoccerV3Policy"),
    "stick-pull-v3": ("sawyer_stick_pull_v3_policy", "SawyerStickPullV3Policy"),
    "stick-push-v3": ("sawyer_stick_push_v3_policy", "SawyerStickPushV3Policy"),
    "sweep-into-v3": ("sawyer_sweep_into_v3_policy", "SawyerSweepIntoV3Policy"),
    "sweep-v3": ("sawyer_sweep_v3_policy", "SawyerSweepV3Policy"),
    "window-close-v3": ("sawyer_window_close_v3_policy", "SawyerWindowCloseV3Policy"),
    "window-open-v3": ("sawyer_window_open_v3_policy", "SawyerWindowOpenV3Policy"),
    "compo-assembly-disassembly": ("compo_assembly_disassembly_policy", "CompoAssemblyDisassemblyPolicy"),
    "compo-coffeepushbuttonpull": ("compo_coffee_push_button_pull_policy", "CompoCoffeePushButtonPullPolicy"),
    "compo-dooropen-doorclose": ("compo_dooropen_doorclose_policy", "CompoDoorOpenDoorClosePolicy"),
    # Eval composition tasks (used by the world-model sweeps). Only collected
    # when explicitly requested via --task-set {excluded,all}.
    "compo-draweropen-pickplace": ("compo_draweropen_pickplace_policy", "CompoDrawerOpenPickPlacePolicy"),
    "compo-pickplace-block": ("compo_pickplace_block_policy", "CompoPickPlaceBlockPolicy"),
    "compo-pickplace-boxclose": ("compo_pickplace_boxclose_policy", "CompoPickPlaceBoxClosePolicy"),
}

# Tasks that appear in any PRISM-WM eval sweep — NEVER used for pretraining.
EVAL_TASKS = frozenset({
    # single-task PRISM sweep
    "bin-picking-yellowpurple-v3", "box-close-v3", "drawer-open-v3",
    "grasp-v3", "pick-place-redblock-v3", "pick-place-v3",
    "reach-xy-v3", "reach-xyz-v3",
    "compo-draweropen-pickplace", "compo-pickplace-block", "compo-pickplace-boxclose",
    # sequential sweeps (union over binpnp/drawerpnp/pnpblock/reach/grasp/pnpboxclose)
    "bin-picking-redblue-v3", "bin-picking-yellowblue-v3", "bin-picking-redpurple-v3",
    "pick-place-greenblock-v3", "reach-xz-v3", "reach-yz-v3", "reach-v3",
})

# Visual near-twins of eval tasks (same objects/scene, uncoloured parents).
# Excluded by default for a clean held-out split; re-include with --include-twins.
VISUAL_TWINS = frozenset({
    "bin-picking-v3", "pick-place-block-v3", "pick-place-wall-v3",
})


def _fmt_dur(seconds):
    """Compact h/m/s formatting for progress + ETA lines."""
    s = int(max(seconds, 0))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def tasks_for(task_set, include_twins=False):
    """Select the env_name list to collect.

      nonEval  (default) : every task NOT in any eval sweep (held-out split).
                           `include_twins` also pulls in the visual twins.
      excluded           : exactly the previously-held-out tasks — the eval
                           sweep tasks + visual twins (all single + composition
                           tasks the world models train on).
      all                : every task in POLICY_REGISTRY (nonEval + excluded).
    """
    if task_set == "all":
        return list(POLICY_REGISTRY)
    if task_set == "excluded":
        return [t for t in POLICY_REGISTRY
                if t in EVAL_TASKS or t in VISUAL_TWINS]
    # nonEval
    exclude = set(EVAL_TASKS)
    if not include_twins:
        exclude |= VISUAL_TWINS
    return [t for t in POLICY_REGISTRY if t not in exclude]


def default_pretrain_tasks(include_twins=False):
    """Back-compat alias used by eval_reconstruction.py (nonEval split)."""
    return tasks_for("nonEval", include_twins=include_twins)


def _load_policy(env_name):
    module_name, class_name = POLICY_REGISTRY[env_name]
    mod = importlib.import_module(f"metaworld.policies.{module_name}")
    return getattr(mod, class_name)()


def _make_env(env_name, cameras, image_size, inner_max_steps):
    env = gym.make(
        "Meta-World/MT1", env_name=env_name, render_mode="rgb_array",
        max_episode_steps=inner_max_steps,
    )
    return ProprioMultiImageObsWrapper(
        env, image_height=image_size, image_width=image_size,
        camera_names=cameras,
    )


def _to_model_obs(obs):
    """proprio/image dict -> (state float32 (7,), rgb uint8 (3N,H,W) CHW)."""
    state = np.asarray(obs["proprio"], dtype=np.float32)
    rgb = np.transpose(np.asarray(obs["image"]), (2, 0, 1))  # HWC -> CHW
    return state, rgb.astype(np.uint8, copy=False)


def collect(args):
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is required: `pip install h5py` (inside the container).")

    cameras = list(args.cameras)
    num_cameras = len(cameras)
    img_c = 3 * num_cameras

    tasks = args.tasks if args.tasks else tasks_for(args.task_set, args.include_twins)
    unknown = [t for t in tasks if t not in POLICY_REGISTRY]
    if unknown:
        sys.exit(f"No scripted policy for: {unknown}")

    if args.smoke:
        args.total_transitions = min(args.total_transitions, 600)
        tasks = tasks[:3]

    n_tasks = len(tasks)
    per_task = math.ceil(args.total_transitions / n_tasks)
    inner_max_steps = args.max_episode_steps * args.action_repeat

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Append mode: copy an existing dataset, then extend it -------------
    # New tasks get task_ids continuing the base file's numbering, and the
    # base file is left untouched (we copy it to --out first).
    append = bool(args.append_to)
    base_tasks, base_total = [], 0
    if append:
        src = Path(args.append_to)
        if not src.exists():
            sys.exit(f"--append-to file not found: {src}")
        if out_path.resolve() == src.resolve():
            sys.exit("--out must differ from --append-to (the base is preserved).")
        with h5py.File(src, "r") as f0:
            if (int(f0.attrs["img_channels"]) != img_c
                    or int(f0.attrs["image_size"]) != args.image_size):
                sys.exit("append-to shape mismatch: base img_channels/image_size "
                         f"= {int(f0.attrs['img_channels'])}/"
                         f"{int(f0.attrs['image_size'])} vs requested "
                         f"{img_c}/{args.image_size}.")
            base_tasks = json.loads(f0.attrs["tasks"])
            base_total = int(f0.attrs.get("total_transitions", f0["state"].shape[0]))
        print(f">>> copying base dataset ({base_total} transitions, "
              f"{len(base_tasks)} tasks)\n      {src}\n   -> {out_path}", flush=True)
        shutil.copyfile(src, out_path)

    print("=" * 70)
    print(f">>> Pretrain data collection  (mode: "
          f"{'APPEND' if append else 'fresh'}, task_set={args.task_set})")
    print(f">>> new tasks:         {n_tasks}")
    print(f">>> new transitions:   {args.total_transitions}  (~{per_task}/task)")
    if append:
        print(f">>> base transitions:  {base_total}  ({len(base_tasks)} tasks, kept)")
        print(f">>> grand total ~      {base_total + per_task * n_tasks}")
    print(f">>> cameras:           {cameras}  ({img_c} channels)")
    print(f">>> image_size:        {args.image_size}  action_repeat={args.action_repeat}")
    print(f">>> output:            {args.out}")
    print("=" * 70)

    rng = np.random.default_rng(args.seed)
    task_names = []
    task_id_offset = len(base_tasks)   # new task_ids continue the numbering
    written = 0
    target_total = per_task * n_tasks   # planned NEW transitions
    t_start = time.time()
    last_progress = t_start

    with h5py.File(out_path, "a" if append else "w") as f:
        if append:
            d_state, d_rgb = f["state"], f["rgb"]
            d_action, d_task = f["action"], f["task_id"]
        else:
            d_state = f.create_dataset(
                "state", shape=(0, 7), maxshape=(None, 7), dtype="float32",
                chunks=(1024, 7))
            d_rgb = f.create_dataset(
                "rgb", shape=(0, img_c, args.image_size, args.image_size),
                maxshape=(None, img_c, args.image_size, args.image_size),
                dtype="uint8", chunks=(32, img_c, args.image_size, args.image_size),
                compression="gzip", compression_opts=4)
            d_action = f.create_dataset(
                "action", shape=(0, 4), maxshape=(None, 4), dtype="float32",
                chunks=(1024, 4))
            d_task = f.create_dataset(
                "task_id", shape=(0,), maxshape=(None,), dtype="int16",
                chunks=(1024,))

        def _append(states, rgbs, actions, tids):
            nonlocal written
            n = len(states)
            if n == 0:
                return
            for d, arr in ((d_state, states), (d_rgb, rgbs),
                           (d_action, actions), (d_task, tids)):
                d.resize(d.shape[0] + n, axis=0)
                d[-n:] = np.asarray(arr)
            written += n

        for task_idx, env_name in enumerate(tasks):
            task_names.append(env_name)
            try:
                policy = _load_policy(env_name)
                env = _make_env(env_name, cameras, args.image_size, inner_max_steps)
            except Exception as e:  # noqa: BLE001 — skip unbuildable task, keep going
                print(f"  [skip] {env_name}: {type(e).__name__}: {e}")
                continue

            collected = 0
            ep = 0
            successes = 0
            while collected < per_task:
                ep += 1
                obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                ep_success = False
                s_buf, r_buf, a_buf = [], [], []
                for _ in range(args.max_episode_steps):
                    action = np.asarray(
                        policy.get_action(obs["original_obs"]), dtype=np.float32)
                    state, rgb = _to_model_obs(obs)
                    s_buf.append(state); r_buf.append(rgb); a_buf.append(action)
                    # Mirror training action-repeat: apply the same action
                    # `action_repeat` times on the inner env, keep the last obs.
                    done = False
                    for _ in range(args.action_repeat):
                        obs, _rew, terminated, truncated, info = env.step(action)
                        done = bool(terminated or truncated)
                        if info.get("success", 0.0):
                            ep_success = True
                        if done:
                            break
                    if collected + len(s_buf) >= per_task or done:
                        break
                take = min(len(s_buf), per_task - collected)
                tids = np.full(take, task_id_offset + task_idx, dtype="int16")
                _append(s_buf[:take], r_buf[:take], a_buf[:take], tids)
                collected += take
                successes += int(ep_success)

                # Throttled intra-task progress (long tasks) with global ETA.
                now = time.time()
                if now - last_progress >= 20.0:
                    elapsed = now - t_start
                    rate = written / max(elapsed, 1e-9)
                    eta = (target_total - written) / max(rate, 1e-9)
                    pct = 100.0 * written / max(target_total, 1)
                    print(f"      ... {env_name} {collected}/{per_task} "
                          f"| total {written}/{target_total} ({pct:4.1f}%) "
                          f"| {rate:.0f} tr/s | elapsed {_fmt_dur(elapsed)} "
                          f"| ETA {_fmt_dur(eta)}", flush=True)
                    last_progress = now
            env.close()
            elapsed = time.time() - t_start
            rate = written / max(elapsed, 1e-9)
            eta = (target_total - written) / max(rate, 1e-9)
            pct = 100.0 * written / max(target_total, 1)
            print(f"  [{task_idx + 1:>2}/{n_tasks}] {env_name:<32} "
                  f"+{collected:>6} ({ep} eps, {successes} success) "
                  f"| total {written}/{target_total} ({pct:4.1f}%) "
                  f"| {rate:.0f} tr/s | elapsed {_fmt_dur(elapsed)} "
                  f"| ETA {_fmt_dur(eta)}", flush=True)
            last_progress = time.time()

        f.attrs["image_size"] = args.image_size
        f.attrs["num_cameras"] = num_cameras
        f.attrs["img_channels"] = img_c
        f.attrs["action_repeat"] = args.action_repeat
        f.attrs["cameras"] = json.dumps(cameras)
        f.attrs["tasks"] = json.dumps(base_tasks + task_names)
        f.attrs["total_transitions"] = base_total + written
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

    grand = base_total + written
    print("=" * 70)
    print(f">>> Done. Added {written} new transitions"
          + (f" to {base_total} base = {grand} total" if append else "")
          + f" in {_fmt_dur(time.time() - t_start)}.")
    print(f">>> File: {out_path}  ({len(base_tasks) + len(task_names)} tasks)")
    print("=" * 70)


def build_argparser():
    p = argparse.ArgumentParser(description="Collect Meta-World pretraining data.")
    p.add_argument("--out", type=str, required=True,
                   help="Output HDF5 path.")
    p.add_argument("--total-transitions", type=int, default=300000,
                   help="Total (new) transitions across the selected tasks, "
                        "evenly split.")
    p.add_argument("--task-set", choices=["nonEval", "excluded", "all"],
                   default="nonEval",
                   help="Which tasks to collect: nonEval (held-out, default), "
                        "excluded (the eval-sweep + twin tasks), or all.")
    p.add_argument("--append-to", type=str, default=None,
                   help="Existing HDF5 to extend: it is copied to --out and the "
                        "new tasks are appended with task_ids continuing the "
                        "base numbering (base file left untouched).")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Explicit env_name list; overrides --task-set.")
    p.add_argument("--include-twins", action="store_true",
                   help="With --task-set nonEval, also include the visual-twin "
                        f"tasks ({sorted(VISUAL_TWINS)}).")
    p.add_argument("--cameras", nargs="+",
                   default=["topview", "front", "gripperPOV"])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-episode-steps", type=int, default=250,
                   help="Outer (action-repeated) steps per episode.")
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run: 3 tasks, <=600 transitions.")
    return p


if __name__ == "__main__":
    collect(build_argparser().parse_args())
