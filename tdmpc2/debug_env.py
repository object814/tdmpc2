"""Minimal diagnostic for the mw custom env path.

Run inside the tdmpc2 conda env:

    cd /data/engs-a2i/catz0908/Metaworld/third_party/tdmpc2/tdmpc2
    python debug_env.py
"""
import os, sys, traceback
os.environ.setdefault("MUJOCO_GL", "egl")

print("=== Python ===")
print("exec:", sys.executable)
print("cwd :", os.getcwd())

print("\n=== Step 1: import gymnasium ===")
try:
    import gymnasium
    print("gymnasium", gymnasium.__version__, gymnasium.__file__)
except Exception:
    traceback.print_exc(); sys.exit(1)

print("\n=== Step 2: resolve + import metaworld from repo root ===")
from pathlib import Path
BASE = Path(__file__).resolve().parents[3]
print("BASE:", BASE)
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    import metaworld
    print("metaworld from:", metaworld.__file__)
except Exception:
    traceback.print_exc(); sys.exit(1)

print("\n=== Step 3: check that Meta-World/MT1 is registered ===")
from gymnasium.envs.registration import registry
mw_ids = [k for k in registry if "Meta-World" in k or "MT1" in k]
print("registered Meta-World* IDs:", mw_ids)
if "Meta-World/MT1" not in registry:
    print(">>> Meta-World/MT1 NOT registered. import metaworld did not register.")
    sys.exit(2)

print("\n=== Step 4: gymnasium.make Meta-World/MT1 ===")
try:
    env = gymnasium.make(
        "Meta-World/MT1",
        env_name="drawer-open-v3",
        render_mode="rgb_array",
        max_episode_steps=500,
    )
    print("gymnasium.make OK:", type(env))
except Exception:
    traceback.print_exc(); sys.exit(3)

print("\n=== Step 5: import our custom make_env and build the full env ===")
sys.path.insert(0, str(Path(__file__).parent))
try:
    from envs.metaworld_custom import make_env as make_mw
    class _Cfg:
        task = "metaworld_drawer-open-v3"
        def get(self, k, d=None): return getattr(self, k, d)
    env = make_mw(_Cfg())
    obs = env.reset()
    print("obs keys:", list(obs.keys()))
    print("state shape:", obs["state"].shape)
    print("rgb shape  :", obs["rgb"].shape)
    print("action_space:", env.action_space)
    print("max_episode_steps:", env.max_episode_steps)
    print("SUCCESS")
except Exception:
    traceback.print_exc(); sys.exit(4)
