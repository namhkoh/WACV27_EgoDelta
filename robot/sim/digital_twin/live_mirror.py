"""Live digital-twin mirror: stream the REAL RB-Y1 into the Isaac twin (arm + gripper + grasp).

READ-ONLY on the real robot -- it NEVER powers on, servos, or commands the robot. It reads
get_state() (arm/body encoders, gRPC) and the gripper bridge /status (finger encoder, HTTP),
then writes sim drive targets over the :8226 vscode socket. When the real gripper closes near a
tomato the sim carries the ACTUAL fruit (kinematic follow of the gripper); when it opens the
fruit's stem is cut and it physically DROPS (into the basket). The fruit then recovers to the
vine after a delay so the demo loops.

Prereqs:
  - Isaac Sim up with the twin loaded (prim /RBY1_A_v1_1), e.g. robot_home_v11_fcoll.usd.
  - Real RB-Y1 on Wi-Fi RBY1 -> 192.0.2.10:50051; gripper bridge default 192.0.2.20:8765.

Run from the repo root:
  python robot\\sim\\digital_twin\\live_mirror.py --hz 15
  python robot\\sim\\digital_twin\\live_mirror.py --no-robot         # sim-write self-test
  python robot\\sim\\digital_twin\\live_mirror.py --gripper-url ""    # arm only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
for _p in (_DIR, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import isaac_client  # same dir; talks to Isaac over 127.0.0.1:8226

PRIM = "/RBY1_A_v1_1"
MAG_OPEN, MAG_CLOSE = 0.0495, 0.0385  # sim finger prismatic magnitude (m): open vs closed
BASKET_XY = (-0.82, -0.97)            # matches isaac_sim_backend.SimRobot.basket_xy
_TOMATOES_PY = os.path.join(_DIR, "scene_add_tomatoes.py").replace("\\", "/")
_BASKET_PY = os.path.join(_DIR, "scene_add_basket.py").replace("\\", "/")
_RESET_TOMATOES = (
    "exec(compile(open(r'{p}','r',encoding='utf-8').read(), 'scene_add_tomatoes.py', 'exec'), globals())\n"
    "add_tomatoes()".format(p=_TOMATOES_PY)
)
_RESET_BASKET = (
    "exec(compile(open(r'{p}','r',encoding='utf-8').read(), 'scene_add_basket.py', 'exec'), globals())\n"
    "add_basket({bx}, {by})".format(p=_BASKET_PY, bx=BASKET_XY[0], by=BASKET_XY[1])
)

# Stiffen the arm drives so they hold the mirrored pose tightly (run once at startup).
_STIFFEN = '''
import omni.usd
from pxr import Sdf
st = omni.usd.get_context().get_stage()
_ns = 0
for _prim in st.Traverse():
    if not str(_prim.GetPath()).startswith("{prim}/joints/"):
        continue
    if "_arm_" not in _prim.GetName():
        continue
    for _k, _v in (("drive:angular:physics:stiffness", 60000.0), ("drive:angular:physics:damping", 2000.0)):
        _a = _prim.GetAttribute(_k)
        if not _a or not _a.IsValid():
            _a = _prim.CreateAttribute(_k, Sdf.ValueTypeNames.Float)
        _a.Set(float(_v))
    _ns += 1
print("STIFFENED", _ns)
'''

# play() is async and is dropped if it races a stop; confirm the clock ADVANCES before mirroring.
_PLAY_CHECK = '''
import omni.timeline
tl = omni.timeline.get_timeline_interface()
if not tl.is_playing():
    tl.play()
print("PLAY", tl.is_playing(), tl.get_current_time())
'''

# Clear any carried-fruit state from a prior run.
_CLEAR_GRASP = '''
import builtins
builtins._MIRROR_HELD = None
builtins._MIRROR_OFF = None
print("grasp cleared")
'''

# One frame: drive the revolute body joints (DEGREES) from `targets` and the prismatic finger
# joints (METERS) from `fingers`; then, if a fruit is held, keep it at its grasp offset from the
# gripper (kinematic follow of the ACTUAL tomato -- no synthetic duplicate).
_MIRROR = '''
import omni.usd as ou, builtins
from pxr import Sdf, Gf, UsdGeom
st = ou.get_context().get_stage()
_t = {targets}
_f = {fingers}
_n = 0
for _prim in st.Traverse():
    if not str(_prim.GetPath()).startswith("{prim}/joints/"):
        continue
    _nm = _prim.GetName()
    if _nm in _t:
        _k = "drive:angular:physics:targetPosition"; _src = _t
    elif _nm in _f:
        _k = "drive:linear:physics:targetPosition"; _src = _f
    else:
        continue
    _a = _prim.GetAttribute(_k)
    if not _a or not _a.IsValid():
        _a = _prim.CreateAttribute(_k, Sdf.ValueTypeNames.Float)
    _a.Set(float(_src[_nm]))
    _n += 1
_held = getattr(builtins, "_MIRROR_HELD", None)
_off = getattr(builtins, "_MIRROR_OFF", None)
if _held and _off:
    _eef = st.GetPrimAtPath("{prim}/{eef}")
    _tp = st.GetPrimAtPath("/World/tomatoes/" + _held)
    if _eef.IsValid() and _tp.IsValid():
        _M = Gf.Matrix4d(*_off) * ou.get_world_transform_matrix(_eef)
        _x = UsdGeom.Xformable(_tp); _x.ClearXformOpOrder(); _x.AddTransformOp().Set(_M)
print("MIRRORED", _n)
'''

# Grasp the ACTUAL nearest tomato: make it kinematic (so it follows the gripper, ignoring the
# vine stem + gravity) and record its pose in the gripper frame.
_GRASP_REAL = '''
import omni.usd as ou, builtins
from pxr import Gf, UsdGeom, UsdPhysics, Sdf
st = ou.get_context().get_stage()
eef = st.GetPrimAtPath("{prim}/{eef}")
fs = []
for fn in ("ee_finger_{side}1", "ee_finger_{side}2"):
    fp = st.GetPrimAtPath("{prim}/" + fn)
    if fp.IsValid():
        fs.append(Gf.Vec3d(ou.get_world_transform_matrix(fp).ExtractTranslation()))
jaw = (fs[0] + fs[1]) * 0.5 if len(fs) == 2 else Gf.Vec3d(ou.get_world_transform_matrix(eef).ExtractTranslation())
best = None; bd = 1e9
for nm in ("tomato0", "tomato1"):
    tp = st.GetPrimAtPath("/World/tomatoes/" + nm)
    if not tp.IsValid():
        continue
    d = (Gf.Vec3d(ou.get_world_transform_matrix(tp).ExtractTranslation()) - jaw).GetLength()
    if d < bd:
        bd = d; best = nm
if best is not None and bd < {max_dist}:
    tp = st.GetPrimAtPath("/World/tomatoes/" + best)
    ka = tp.GetAttribute("physics:kinematicEnabled")
    if not ka or not ka.IsValid():
        ka = tp.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool)
    ka.Set(True)
    off = ou.get_world_transform_matrix(tp) * ou.get_world_transform_matrix(eef).GetInverse()
    builtins._MIRROR_HELD = best
    builtins._MIRROR_OFF = [off[i][j] for i in range(4) for j in range(4)]
    print("GRASPED " + best + " " + str(round(bd, 3)))
else:
    print("GRASP_SKIP " + str(round(bd, 3)))
'''

# Release: cut the stem (RemovePrim) and flip the fruit dynamic so it physically DROPS.
_RELEASE_REAL = '''
import omni.usd as ou, builtins
from pxr import Gf, Sdf
st = ou.get_context().get_stage()
held = getattr(builtins, "_MIRROR_HELD", None)
if held:
    stem = st.GetPrimAtPath("/World/tomatoes/" + held + "_stem")
    if stem.IsValid():
        st.RemovePrim(stem.GetPath())
    tp = st.GetPrimAtPath("/World/tomatoes/" + held)
    if tp.IsValid():
        ka = tp.GetAttribute("physics:kinematicEnabled")
        if ka and ka.IsValid():
            ka.Set(False)
        # WAKE it: a sleeping dynamic body holds its pose and ignores gravity, so without a nudge the
        # fruit floats between the open fingers. A downward velocity above the sleep threshold wakes
        # it so it actually falls (verified: z 0.908 -> 0.040 in ~1.2s).
        va = tp.GetAttribute("physics:velocity")
        if not va or not va.IsValid():
            va = tp.CreateAttribute("physics:velocity", Sdf.ValueTypeNames.Vector3f)
        va.Set(Gf.Vec3f(0.0, 0.0, -0.6))
    builtins._MIRROR_HELD = None
    builtins._MIRROR_OFF = None
    print("RELEASED " + str(held))
else:
    print("RELEASE_NOOP")
'''


def _out(raw: str) -> str:
    try:
        return json.loads(raw).get("output", "") or ""
    except Exception:
        return raw


def ensure_playing(tries: int = 12, dwell_s: float = 0.5) -> bool:
    last_t = None
    playing = False
    for _ in range(tries):
        cur_t = None
        for ln in _out(isaac_client.send_code(_PLAY_CHECK)).splitlines():
            if ln.startswith("PLAY"):
                p = ln.split()
                playing = len(p) > 1 and p[1] == "True"
                try:
                    cur_t = float(p[2])
                except (IndexError, ValueError):
                    cur_t = None
        if playing and last_t is not None and cur_t is not None and cur_t > last_t:
            return True
        last_t = cur_t
        time.sleep(dwell_s)
    return playing


def push_init(prim: str, reset_scene: bool = True) -> str:
    s = _out(isaac_client.send_code(_STIFFEN.format(prim=prim))).strip()
    _out(isaac_client.send_code(_CLEAR_GRASP))
    tm = ""
    if reset_scene:
        _out(isaac_client.send_code(_RESET_TOMATOES))
        _out(isaac_client.send_code(_RESET_BASKET))
        tm = " | scene reset"
    return f"{s}{tm} | playing={ensure_playing()}"


def _finger_targets(mag: float, side: str) -> dict:
    return {f"gripper_finger_{side}1": -float(mag), f"gripper_finger_{side}2": float(mag)}


def push(targets: dict, fingers: dict, prim: str, eef: str) -> str:
    code = _MIRROR.format(targets=json.dumps(targets), fingers=json.dumps(fingers), prim=prim, eef=eef)
    return isaac_client.send_code(code)


def grasp(prim: str, side: str, eef: str, max_dist: float) -> str:
    return _out(isaac_client.send_code(_GRASP_REAL.format(prim=prim, side=side, eef=eef, max_dist=float(max_dist))))


def release() -> str:
    return _out(isaac_client.send_code(_RELEASE_REAL))


def recover() -> str:
    return _out(isaac_client.send_code(_RESET_TOMATOES))


def _read_gripper_mag(url: str, *, arm: str, grip_open: float, grip_close: float, timeout_s: float):
    target_id = 1 if arm == "left" else 0
    payload = None
    for _ in range(3):  # retry: the bridge drops reads under contention with the harvest
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/status", timeout=timeout_s) as r:
                payload = json.loads(r.read().decode("utf-8"))
            break
        except Exception:
            time.sleep(0.04)
    if payload is None:
        return None
    pos = None
    for s in payload.get("motor_states", []):
        if int(s.get("id", -1)) == target_id and s.get("position") is not None:
            pos = float(s["position"])
            break
    if pos is None:
        return None
    span = (grip_close - grip_open) or 1e-9
    frac = max(0.0, min(1.0, (pos - grip_open) / span))  # 0 = open, 1 = closed
    return MAG_OPEN - frac * (MAG_OPEN - MAG_CLOSE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live RB-Y1 -> Isaac twin mirror: arm + gripper + grasp (READ-ONLY on the robot).")
    ap.add_argument("--address", default="192.0.2.10:50051")
    ap.add_argument("--model", default="a")
    ap.add_argument("--arm", default="left", choices=["left", "right"])
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    ap.add_argument("--prim", default=PRIM)
    ap.add_argument("--gripper-url", default="http://192.0.2.20:8765", help='gripper bridge; "" to skip fingers')
    ap.add_argument("--grip-open", type=float, default=1.8, help="left-id1 encoder when open (the harvest opens to ~1.8)")
    ap.add_argument("--grip-close", type=float, default=3.4, help="left-id1 encoder at grasp (it stalls on the fruit ~3.4)")
    ap.add_argument("--grasp-dist-m", type=float, default=0.09, help="grasp the nearest tomato when the closing jaw is within this")
    ap.add_argument("--recover-after-s", type=float, default=5.0, help="re-grow the fruit on the vine this long after a drop; 0 = never")
    ap.add_argument("--no-reset-scene", action="store_true", help="do NOT re-author tomatoes + basket at startup")
    ap.add_argument("--no-sim-grasp", action="store_true", help="do NOT carry/drop fruit on gripper close/open")
    ap.add_argument("--no-robot", action="store_true", help="sim-write self-test: sweep arm + gripper, no real robot")
    a = ap.parse_args()
    side = a.arm[0]
    eef = "ee_left" if a.arm == "left" else "ee_right"
    period = 1.0 / max(1e-3, a.hz)

    if a.no_robot:
        print("self-test: sweeping left_arm_1 +/-20 deg and the gripper open<->close (no real robot). Ctrl-C to stop.")
        print("  " + push_init(a.prim, reset_scene=not a.no_reset_scene).strip()[:120])
        held = False
        for i in range(240):
            ang = 20.0 * math.sin(i * 0.08)
            frac = (math.sin(i * 0.05) + 1.0) / 2.0
            mag = MAG_OPEN - frac * (MAG_OPEN - MAG_CLOSE)
            r = push({"left_arm_1": ang}, _finger_targets(mag, side), a.prim, eef).strip()
            if not a.no_sim_grasp:
                if (not held) and frac > 0.6:
                    print("  " + grasp(a.prim, side, eef, a.grasp_dist_m).strip()[:60]); held = True
                elif held and frac < 0.3:
                    print("  " + release().strip()[:60]); held = False
            if i % 20 == 0:
                print(f"  i={i:3d} arm={ang:+6.1f}deg grip_frac={frac:.2f} -> {r[:50]}")
            time.sleep(period)
        return 0

    import rby1_sdk as rby
    from robot.kinematics.rby1_model import sdk_model_from_name

    names = [str(n) for n in sdk_model_from_name(a.model).robot_joint_names]
    robot = rby.create_robot(a.address, a.model)
    if not robot.connect():
        raise SystemExit(f"failed to connect to RB-Y1 at {a.address} (is Wi-Fi RBY1 up?)")
    # NOTE: deliberately NO power_on / servo_on / enable_control_manager / send_command.
    use_grip = bool(a.gripper_url)
    do_grasp = use_grip and not a.no_sim_grasp
    print(f"connected READ-ONLY; mirroring {len(names)} joints"
          f"{' + ' + a.arm + ' gripper/grasp' if use_grip else ''} @ {a.hz} Hz. Ctrl-C to stop.")
    print("  " + push_init(a.prim, reset_scene=not a.no_reset_scene).strip()[:120])
    t_end = (time.time() + a.duration) if a.duration > 0 else None
    # poll the gripper ~2 Hz: the encoder moves slowly and frequent polls contend with the harvest
    # on the same :8765 serial bus.
    grip_every = max(1, int(round(a.hz / 2.0)))
    n = 0
    last_mag = MAG_OPEN
    held = False
    drop_t = None
    try:
        while True:
            q = robot.get_state().position
            deg = {names[i]: math.degrees(float(q[i])) for i in range(min(len(names), len(q)))}
            fingers = {}
            if use_grip:
                if n % grip_every == 0:
                    mag = _read_gripper_mag(a.gripper_url, arm=a.arm, grip_open=a.grip_open,
                                            grip_close=a.grip_close, timeout_s=min(0.15, period))
                    if mag is not None:
                        last_mag = mag
                fingers = _finger_targets(last_mag, side)
                if do_grasp:
                    frac = (MAG_OPEN - last_mag) / (MAG_OPEN - MAG_CLOSE)
                    if (not held) and frac > 0.6:
                        r = grasp(a.prim, side, eef, a.grasp_dist_m)
                        held = "GRASPED" in r
                        print(f"  [grasp] frac={frac:.2f} {r.strip()[:48]}", flush=True)
                    elif held and frac < 0.45:
                        print(f"  [release] frac={frac:.2f} {release().strip()[:40]}", flush=True)
                        held = False
                        drop_t = time.time()
            push(deg, fingers, a.prim, eef)
            if drop_t and a.recover_after_s > 0 and (time.time() - drop_t) > a.recover_after_s:
                recover()
                drop_t = None
            n += 1
            if t_end and time.time() >= t_end:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        print()
    print(f"done; {n} frames mirrored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
