"""In-process Isaac Sim backend for ``guarded_bc_rollout`` (``--backend sim``).

Runs the harvest/deleaf rollout against the live Isaac Sim digital twin instead of the
real RB-Y1, with NO hardware:

* kinematics built OFFLINE from the reference URDF (``RBY1Kinematics.from_*`` path:
  ``rby.dynamics.load_robot_from_urdf`` + ``rby.Model_A`` -- proven <0.1mm vs live FK);
* robot STATE served from a cache seeded from ``homestate.json`` and advanced to the
  commanded pose each step (open-loop, sufficient for the receding loop);
* joint COMMANDS captured into a trajectory buffer; after the rollout the whole buffer
  is animated into ``/RBY1_A_v1_1`` over the :8226 vscode-exec socket in one continuous
  timeline play (no per-step stop/play thrash). The active arm follows the policy; torso,
  head and the other arm are pinned to ``home`` exactly like ``command_arm`` does.

``SimRobot`` duck-types just the ``rby`` surface the rollout touches: ``get_dynamics``,
``model``, ``get_state``, ``connect``/power/servo no-ops, and -- via the ``_send_arm``
seam in guarded_bc_rollout -- ``apply_q_arm`` (returns a real-enum ``Ok`` feedback).
"""
from __future__ import annotations

import ast
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rby1_sdk as rby

from robot.kinematics.rby1_model import BASE_LINK, sdk_model_from_name

# host-side socket client lives in robot/sim/digital_twin/
_DIGITAL_TWIN = Path(__file__).resolve().parents[1] / "sim" / "digital_twin"
if str(_DIGITAL_TWIN) not in sys.path:
    sys.path.insert(0, str(_DIGITAL_TWIN))
import isaac_client  # noqa: E402

REFERENCE_URDF = (
    Path(__file__).resolve().parents[2]
    / "submodules" / "rby1-sdk" / "models" / "rby1a" / "urdf" / "model_v1.1.urdf"
)
ROBOT_PRIM = "/RBY1_A_v1_1"
WHEEL_JOINTS = ("right_wheel", "left_wheel")
SCENE_ADD_TOMATOES = str(_DIGITAL_TWIN / "scene_add_tomatoes.py").replace("\\", "/")
SCENE_ADD_BASKET = str(_DIGITAL_TWIN / "scene_add_basket.py").replace("\\", "/")


class _SimFeedback:
    """Stand-in for rby RobotCommandFeedback: only ``finish_code`` is read."""

    def __init__(self) -> None:
        self.finish_code = rby.RobotCommandFeedback.FinishCode.Ok


class _SimState:
    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = position
        self.velocity = velocity


def _isaac(snippet: str, *, timeout: float = 60.0) -> str:
    """Run a snippet in the live Isaac Sim; raise on a Kit-side exception."""
    for attempt in range(3):
        raw = isaac_client.send_code(snippet, timeout=timeout)
        try:
            rep = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if rep.get("status") == "ok":
            return rep.get("output", "")
        # The vscode-exec server has NO message framing: data_received() compiles
        # each TCP segment as a complete source, so a large snippet that splits
        # across segments produces a spurious SyntaxError on truncated code.
        # compile() failed before exec -- nothing ran -- so a resend is safe.
        # Only retry when OUR snippet is locally valid (a real SyntaxError isn't).
        if rep.get("ename") == "SyntaxError" and attempt < 2:
            try:
                ast.parse(snippet)
            except SyntaxError:
                pass
            else:
                time.sleep(0.3)
                continue
        raise RuntimeError(
            "Isaac Sim exec failed: {}: {}\n{}".format(
                rep.get("ename"), rep.get("evalue"), "\n".join(rep.get("traceback", []))
            )
        )
    raise RuntimeError("Isaac Sim exec failed after retries")


def _ensure_playing(tries: int = 12, dwell_s: float = 0.6) -> bool:
    """``tl.play()`` after a ``tl.stop()`` intermittently fails to engage -- the play is
    dropped when it races the stop in the same Kit update, leaving physics frozen so the
    arm never leaves home (the root cause of the flaky / non-existent grasp). Call play()
    and confirm it actually STEPS (is_playing True AND the clock advancing between checks),
    retrying until it does."""
    last_t = None
    playing = False
    for _ in range(tries):
        out = _isaac(_PLAY_TEMPLATE)
        cur_t = None
        for ln in str(out).splitlines():
            if ln.startswith("PLAYING"):
                parts = ln.split()
                playing = len(parts) > 1 and parts[1] == "True"
                try:
                    cur_t = float(parts[2])
                except (IndexError, ValueError):
                    cur_t = None
        if playing and last_t is not None and cur_t is not None and cur_t > last_t:
            return True
        last_t = cur_t
        time.sleep(dwell_s)
    return bool(playing)


class SimRobot:
    """rby-robot-shaped shim that drives the Isaac Sim twin instead of hardware."""

    is_sim_backend = True

    def __init__(self, *, home_path: Path, arm: str, model_name: str = "a",
                 urdf_path: Path = REFERENCE_URDF, toolfit_path: Path | None = None) -> None:
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        self.arm = arm
        self._home = json.loads(Path(home_path).read_text(encoding="utf-8"))
        # tool-centre-point offset (ee_<arm> -> jaw centre), for placing a grasped fruit
        self._tcp_offset = (0.0, 0.0, -0.073)
        try:
            if toolfit_path is not None:
                _tf = json.loads(Path(toolfit_path).read_text(encoding="utf-8"))
                _off = _tf.get("estimated_tool_offset_m")
                if _off and len(_off) == 3:
                    self._tcp_offset = (float(_off[0]), float(_off[1]), float(_off[2]))
        except Exception:
            pass
        # offline kinematics model + dynamics (no robot connection)
        self._model = sdk_model_from_name(model_name)
        configuration = rby.dynamics.load_robot_from_urdf(str(Path(urdf_path).resolve()), BASE_LINK)
        self._dyn = rby.dynamics.Robot(configuration)
        self._joint_names = [str(n) for n in self._model.robot_joint_names]
        idx = self._model.left_arm_idx if arm == "left" else self._model.right_arm_idx
        self._arm_idx = [int(i) for i in idx]
        # state cache (full-DoF rad, joint_names order) seeded at home
        self._q_full = np.asarray(self._home["all_joints"]["positions_rad"], dtype=float).copy()
        # commanded trajectory buffer: {"q_full_rad", "minimum_time_s", "label"}
        self.frames: list[dict[str, Any]] = []
        # basket drop-off point (world x,y) ~50cm to the robot's left (robot at ~-0.32,-0.97
        # facing +y, so its left is -x). The release drops the fruit toward it.
        self.basket_xy = (-0.82, -0.97)
        # the twin rests at the calibrated real-robot-aligned base; the recorded grasp tail only
        # reaches the vine fruit from ~23cm closer, so the harvest DRIVES the base up to this
        # grasp spot for the run, then grasps. tomato1 (lower): (-0.105,-0.8805).
        self.grasp_base_xy = (-0.105, -0.8805)
        self.grasp_base_yaw_deg = 86.3

    # --- rby surface used by guarded_bc_rollout -------------------------------
    def get_dynamics(self) -> Any:
        return self._dyn

    def model(self) -> Any:
        return self._model

    def get_state(self) -> _SimState:
        return _SimState(self._q_full.copy(), np.zeros_like(self._q_full))

    def connect(self) -> bool:  # pragma: no cover - trivial
        return True

    def is_power_on(self, *_a, **_k) -> bool:
        return True

    def power_on(self, *_a, **_k) -> bool:
        return True

    def is_servo_on(self, *_a, **_k) -> bool:
        return True

    def servo_on(self, *_a, **_k) -> bool:
        return True

    def enable_control_manager(self, *_a, **_k) -> bool:
        return True

    def reset_fault_control_manager(self, *_a, **_k) -> bool:
        return True

    def get_control_manager_state(self) -> Any:
        class _CM:
            state = rby.ControlManagerState.State.Enabled
        return _CM()

    # --- the execution seam (called via _send_arm) ----------------------------
    def apply_q_arm(self, home: dict[str, Any], arm: str, q_arm: np.ndarray, *,
                    minimum_time_s: float = 0.5, label: str | None = None) -> _SimFeedback:
        q_arm = np.asarray(q_arm, dtype=float).reshape(-1)
        # full body: torso/head/other-arm pinned to home, active arm = q_arm
        q_full = np.asarray(home["all_joints"]["positions_rad"], dtype=float).copy()
        active = self._arm_idx if arm == self.arm else (
            [int(i) for i in (self._model.left_arm_idx if arm == "left" else self._model.right_arm_idx)]
        )
        q_full[active] = q_arm
        self._q_full = q_full
        self.frames.append({
            "q_full_rad": q_full.tolist(),
            "minimum_time_s": float(minimum_time_s),
            "label": label or "policy",
        })
        return _SimFeedback()

    # --- gripper (routed here via args._sim_robot in guarded_bc_rollout) -------
    def gripper(self, kind: str, *, arm: str, action: str, reason: str = "",
                phase_s: float = 0.0, phase_fraction: float = 0.0,
                tcp_goal_distance_m: Any = None) -> dict[str, Any]:
        kind = "close" if str(kind).lower().startswith("c") else "open"
        self.frames.append({"gripper": kind, "label": action, "minimum_time_s": 1.0})
        return {
            "backend": "sim", "action": action, "reason": reason, "arm": arm,
            "kind": kind, "phase_s": float(phase_s), "phase_fraction": float(phase_fraction),
            "tcp_goal_distance_m": tcp_goal_distance_m, "ok": True,
        }

    # --- animate the buffered trajectory into the twin ------------------------
    def replay_into_sim(self, *, segment_s: float = 10.0, interp_steps: int = 40,
                        grip_pause_s: float = 1.0, joint_vel_deg_s: float = 18.0,
                        grasp_settle_s: float = 4.0) -> dict[str, Any]:
        """Smoothly animate the buffered tail into the twin.

        Each waypoint segment is linearly interpolated (``interp_steps`` sub-frames over
        ``segment_s`` seconds) so the arm glides instead of snapping. On grasp-close the
        nearest fruit is attached to the left EEF and carried; on the basket open it is
        released (drops). The active arm follows the policy/tail; torso/head/other-arm
        stay at home.
        """
        if not self.frames:
            return {"frames": 0, "note": "no commanded frames to replay"}
        names = [n for n in self._joint_names if n not in WHEEL_JOINTS]
        eef = "ee_left" if self.arm == "left" else "ee_right"

        def deg_map(q_full) -> dict[str, float]:
            return {n: math.degrees(float(q_full[self._joint_names.index(n)])) for n in names}

        side = self.arm[0]
        home_q = np.asarray(self._home["all_joints"]["positions_rad"], dtype=float)
        # SurfaceGripper assembly first, while STOPPED (idempotent): the carry attach must
        # pre-exist before play so runtime needs zero USD authoring.
        print("[sgdbg]", str(_isaac(_SG_SETUP_TEMPLATE.format(prim=ROBOT_PRIM, eef=eef, armname=self.arm))).replace("\n", " | "))
        _isaac(_SETUP_TEMPLATE.format(prim=ROBOT_PRIM, home=json.dumps(deg_map(home_q)),
                                      vel=float(joint_vel_deg_s), tomatoes_py=SCENE_ADD_TOMATOES,
                                      basket_py=SCENE_ADD_BASKET,
                                      bx=float(self.basket_xy[0]), by=float(self.basket_xy[1]),
                                      gbx=float(self.grasp_base_xy[0]), gby=float(self.grasp_base_xy[1]),
                                      gyaw=float(self.grasp_base_yaw_deg)))
        playing = _ensure_playing()  # confirm physics is actually stepping before driving
        prev_q = home_q
        arm_frames = grip_frames = 0
        held = ""  # name of the fruit currently carried in the jaws ("" = none)
        released = ""  # fruit released at the basket (a successful carry ends here, not in held)
        release_fruit_z = None  # fruit height at the release instant (in-jaw ~ jaw height)
        grasp_dist_m = None  # jaw-to-fruit distance at the grasp instant
        grasp_residual_m = None  # (fruit - jaw) world vector at the grasp instant
        for fr in self.frames:
            if "gripper" in fr:
                if fr["gripper"] == "close" and not held:
                    # ENABLE-SEVER HANDOFF: settle at the fruit with fingers OPEN, identify
                    # the target, pre-attach the carry joint at the in-tongs pose, then
                    # enable the fruit's collider mid-play -- the body re-init transient
                    # LOADS the 60N stem past breakForce (the engine-internal severing
                    # action; this path loads joints without killing them, unlike prim
                    # removal which re-creates the body and orphans its joints into an
                    # unprotected window). The live carry joint catches the fruit at the
                    # pinch with zero free time; the fingers then close a quasi-static
                    # contact pinch on the held fruit.
                    _ensure_playing()
                    self._wait_arm_settled(prev_q, deg_map, max_iters=max(16, int(round(grasp_settle_s)) + 10))
                    out = _isaac(_GRASP_TEMPLATE.format(prim=ROBOT_PRIM, eef=eef, side=side))
                    for _ln in str(out).splitlines():
                        if _ln.startswith("GRASPED:"):
                            held = _ln.split(":", 1)[1].strip()
                        if "jaw-to-fruit dist" in _ln:
                            try:
                                grasp_dist_m = float(_ln.rsplit(None, 1)[-1])
                            except ValueError:
                                pass
                        if _ln.startswith("GRASPVEC"):
                            try:
                                _v = [float(x) for x in _ln.split()[1:7]]
                                grasp_residual_m = [round(_v[3 + i] - _v[i], 4) for i in range(3)]
                            except (ValueError, IndexError):
                                pass
                    # SANITY GATE: never attach to a fruit that is not actually between the
                    # tongs (a failed grasp must read as a failed grasp, not drag a stray
                    # body around the scene on an invisible joint).
                    if held and grasp_dist_m is not None and grasp_dist_m > 0.06:
                        print("sim grasp GATE FAILED: nearest fruit %.3fm from jaw -- grasp counted as failed" % grasp_dist_m)
                        held = ""
                    if held:
                        self._wait_arm_settled(prev_q, deg_map, max_iters=10)
                        # 1) enable the fruit collider (SAFE against the 1e38 stem --
                        #    verified across every earlier run; the fruit stays on the vine)
                        _isaac(_ENABLE_TEMPLATE.format(name=held))
                        time.sleep(1.0)
                        # 2) quasi-static close: the real contact pinch on the vine-anchored
                        #    fruit (force closure = the finger drives stall on the fruit)
                        _isaac(_GRIPPER_TEMPLATE.format(prim=ROBOT_PRIM, side=side, kind="close"))
                        time.sleep(2.5)  # close completes in ~1s; settle to steady state (demo-trim 5.0->2.5)
                        # 3) SurfaceGripper attach at the validated pinch: C++ PxJoint
                        #    retarget inside the physics step, zero USD, revalidated every
                        #    step. Gate on the gripped-objects report.
                        _sg_ok = False
                        for _try in range(3):
                            _isaac(_SG_CLOSE_TEMPLATE)
                            time.sleep(1.0)
                            if "tomato" in str(_isaac(_SG_STATE_TEMPLATE)):
                                _sg_ok = True
                                break
                        if not _sg_ok:
                            print("sim grasp SG GATE FAILED: no gripped object after 3 tries -- "
                                  "stem left intact, grasp counted as failed")
                            held = ""
                        else:
                            print("sim grasp: %s pinched + SurfaceGripper attached" % held)
                    if held:
                        # 4) sever the stem (RemovePrim + real-value flush) -- the removal
                        #    window is covered by the pinch (verified 2.5s hold) and the
                        #    gripper's per-step revalidation (retryInterval 1.0s)
                        _isaac(_STEM_REMOVE_TEMPLATE.format(name=held))
                        time.sleep(0.5)  # let the step-callback RemovePrim fire first
                        _nt = deg_map(prev_q)
                        _nj = "%s_arm_6" % self.arm
                        _nt[_nj] = _nt.get(_nj, 0.0) + 0.002
                        _isaac(_FRAME_TEMPLATE.format(prim=ROBOT_PRIM, targets=json.dumps(_nt)))  # drain the topology queue NOW, parked
                        time.sleep(1.0)
                        _isaac(_FRAME_TEMPLATE.format(prim=ROBOT_PRIM, targets=json.dumps(deg_map(prev_q))))  # restore exact target
                        time.sleep(1.0)  # settle: fruit transfers from stem to pinch+gripper (demo-trim 2.0->1.0)
                else:
                    if fr["gripper"] == "open" and held:
                        # release the SurfaceGripper over the basket BEFORE opening the
                        # fingers (pure C++ detach, no USD)
                        _isaac(_SG_OPEN_TEMPLATE)
                        time.sleep(0.5)
                    _isaac(_GRIPPER_TEMPLATE.format(prim=ROBOT_PRIM, side=side, kind=fr["gripper"]))
                    if fr["gripper"] == "open" and held:
                        _isaac(_RELEASE_TEMPLATE.format(prim=ROBOT_PRIM, eef=eef, side=side, name=held,
                                                        bx=float(self.basket_xy[0]), by=float(self.basket_xy[1])))
                        # honest carry metric: where is the released fruit relative to the jaw?
                        # If it is still at the tongs at the open, the carry held all the way.
                        release_fruit_z = None
                        for _ln in str(_isaac(_FRUITZ_TEMPLATE)).splitlines():
                            if _ln.startswith("FRUITZ " + held):
                                try:
                                    release_fruit_z = float(_ln.split("pos=(")[1].split(")")[0].split(",")[2])
                                except (IndexError, ValueError):
                                    pass
                        released = held
                        held = ""
                time.sleep(grip_pause_s)
                grip_frames += 1
                continue
            target_q = np.asarray(fr["q_full_rad"], dtype=float)
            seg = max(float(segment_s), float(fr["minimum_time_s"]))
            steps = max(1, int(interp_steps))
            for k in range(1, steps + 1):
                q = prev_q + (target_q - prev_q) * (k / steps)
                _isaac(_FRAME_TEMPLATE.format(prim=ROBOT_PRIM, targets=json.dumps(deg_map(q))))
                time.sleep(seg / steps)  # host paces; Kit keeps playing + rendering
            prev_q = target_q
            arm_frames += 1
        _isaac(_TEARDOWN_TEMPLATE)
        # "carried" = the fruit made it to a release (or is still in hand at the end); the old
        # bool(held) read false on a SUCCESSFUL run because release clears held at the basket.
        return {"arm_frames": arm_frames, "gripper_frames": grip_frames,
                "carried": bool(held or released), "released_fruit": released or None,
                "release_fruit_z_m": release_fruit_z,
                "grasp_jaw_to_fruit_m": grasp_dist_m, "grasp_residual_m": grasp_residual_m,
                "physics_playing": bool(playing)}

    def _wait_arm_settled(self, prev_q, deg_map, *, max_iters: int = 16, tol_m: float = 0.0015,
                          settle_hits: int = 3, min_hold: int = 7) -> Any:
        """Re-assert the current arm target and wait until the jaw has FULLY arrived at the
        commanded pose's steady state, so a grasp fires at the converged pose rather than
        mid-slew. Holds for at least ``min_hold`` ticks before trusting a 'stable' reading --
        the drives approach the extended grasp pose asymptotically and a loose threshold trips
        early while the jaw is still creeping in (the cause of the inconsistent grasp distance)."""
        side = self.arm[0]
        prev_jaw = None
        stable = 0
        for i in range(max(1, int(max_iters))):
            _isaac(_FRAME_TEMPLATE.format(prim=ROBOT_PRIM, targets=json.dumps(deg_map(prev_q))))
            time.sleep(1.0)
            jaw = None
            for ln in str(_isaac(_JAW_TEMPLATE.format(prim=ROBOT_PRIM, side=side))).splitlines():
                if ln.startswith("JAW"):
                    p = ln.split()
                    try:
                        jaw = (float(p[1]), float(p[2]), float(p[3]))
                    except (IndexError, ValueError):
                        jaw = None
            if jaw is not None and prev_jaw is not None:
                d = sum((jaw[i] - prev_jaw[i]) ** 2 for i in range(3)) ** 0.5
                stable = stable + 1 if d < tol_m else 0
                if (i + 1) >= min_hold and stable >= settle_hits:
                    return jaw
            prev_jaw = jaw
        return prev_jaw


# ---- Isaac-side snippet templates (run in Kit's global scope) ----------------
# play (if not already) and report whether it is stepping + the sim clock, so the host can
# confirm physics actually engaged before driving the arm.
_PLAY_TEMPLATE = '''
import omni.timeline
tl = omni.timeline.get_timeline_interface()
if not tl.is_playing():
    tl.play()
print("PLAYING", tl.is_playing(), round(tl.get_current_time(), 4))
'''

# report the current jaw (fingertip-midpoint) world position, for closed-loop settle.
_JAW_TEMPLATE = '''
import omni.usd as ou
from pxr import Gf
st = ou.get_context().get_stage()
_fs = []
for _fn in ("ee_finger_{side}1", "ee_finger_{side}2"):
    _fp = st.GetPrimAtPath("{prim}/" + _fn)
    if _fp.IsValid():
        _fs.append(Gf.Vec3d(ou.get_world_transform_matrix(_fp).ExtractTranslation()))
if len(_fs) == 2:
    _j = (_fs[0] + _fs[1]) * 0.5
    print("JAW", round(_j[0], 4), round(_j[1], 4), round(_j[2], 4))
'''

# diagnostic: report each fruit's world position + whether its stem joint is still enabled,
# so the host can log the fruit through the grasp sub-steps and see exactly when it is lost.
_FRUITZ_TEMPLATE = '''
import omni.usd as ou
from pxr import UsdPhysics
st = ou.get_context().get_stage()
for _n in ("tomato0", "tomato1"):
    _p = st.GetPrimAtPath("/World/tomatoes/" + _n)
    if not _p.IsValid():
        continue
    _t = ou.get_world_transform_matrix(_p).ExtractTranslation()
    _sj = UsdPhysics.Joint.Get(st, "/World/tomatoes/" + _n + "_stem")
    _en = None
    if _sj:
        _ea = _sj.GetJointEnabledAttr()
        _en = _ea.Get() if _ea and _ea.IsValid() else True
    print("FRUITZ %s pos=(%.3f,%.3f,%.3f) stem_enabled=%s" % (_n, _t[0], _t[1], _t[2], _en))
'''

_SETUP_TEMPLATE = '''
import omni.usd, omni.timeline, carb, math
from pxr import UsdPhysics, PhysxSchema, UsdShade, UsdGeom, Sdf, Gf
st = omni.usd.get_context().get_stage()
tl = omni.timeline.get_timeline_interface()
tl.stop()
# reposition the base to the grasp spot (the twin rests at the calibrated aligned position;
# the recorded grasp only reaches the vine fruit from here). Timeline is stopped -> the
# articulation root xform is authored cleanly.
_rb = st.GetPrimAtPath("{prim}")
_xf = UsdGeom.Xformable(_rb)
_set_t = _set_r = False
for _op in _xf.GetOrderedXformOps():
    if _op.GetOpName() == "xformOp:translate":
        _op.Set(Gf.Vec3d({gbx}, {gby}, 0.0)); _set_t = True
    elif _op.GetOpName() == "xformOp:rotateZ":
        _op.Set(float({gyaw})); _set_r = True
if not _set_t:
    _xf.AddTranslateOp().Set(Gf.Vec3d({gbx}, {gby}, 0.0))
if not _set_r:
    _xf.AddRotateZOp().Set(float({gyaw}))
# re-author BOTH fruit fresh on the vine (so every run starts clean -- a grasp removes a
# stem, which a re-lock alone cannot recreate)
_ns = {{}}
exec(compile(open(r"{tomatoes_py}", "r", encoding="utf-8").read(), "scene_add_tomatoes.py", "exec"), _ns)
_ns["add_tomatoes"](frame=False)
UsdGeom.Imageable(st.GetPrimAtPath("/World/tomatoes")).MakeVisible()
# high-friction material so a released fruit settles instead of sliding off the floor
if not st.GetPrimAtPath("/World/PhysicsMaterials").IsValid():
    UsdGeom.Scope.Define(st, "/World/PhysicsMaterials")
_mat = UsdShade.Material.Define(st, "/World/PhysicsMaterials/grip")
_pm = UsdPhysics.MaterialAPI.Apply(_mat.GetPrim())
_pm.CreateStaticFrictionAttr(1.2); _pm.CreateDynamicFrictionAttr(1.1); _pm.CreateRestitutionAttr(0.0)
# bind grip material to the ground, both fruit, AND the gripper finger pad colliders --
# both contact surfaces need the high friction for the fingers to hold the fruit.
_grip_targets = ["/World/GroundPlane/collisionPlane", "/World/tomatoes/tomato0", "/World/tomatoes/tomato1"]
for _fn in ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2"):
    _grip_targets.append("{prim}/" + _fn + "/collisions/pad_box")
for _t in _grip_targets:
    _pr = st.GetPrimAtPath(_t)
    if _pr.IsValid():
        UsdShade.MaterialBindingAPI.Apply(_pr).Bind(_mat, materialPurpose="physics")
        # torsional patch friction on the gripping surfaces: the spherical fruit otherwise
        # spins/rolls freely about the point contacts and works loose during the carry even
        # though linear friction holds (sphere-in-parallel-gripper instability).
        _pc = PhysxSchema.PhysxCollisionAPI.Apply(_pr)
        _tp = _pc.GetTorsionalPatchRadiusAttr() or _pc.CreateTorsionalPatchRadiusAttr()
        _tp.Set(0.01)
        _mtp = _pc.GetMinTorsionalPatchRadiusAttr() or _pc.CreateMinTorsionalPatchRadiusAttr()
        _mtp.Set(0.003)
        # tight contact offset on the GRIPPING surfaces only (pads + fruit). NOT on the
        # ground plane: its restOffset exceeds 0.002 and PhysX rejects the write with
        # "contact offset must be positive and greater than restOffset" console errors.
        if "collisionPlane" not in _t:
            _co = _pc.GetContactOffsetAttr() or _pc.CreateContactOffsetAttr()
            _co.Set(0.001)
# CRADLE rims: flat pads cannot cage a rigid sphere -- measured: at stem-break the fruit
# escapes along the pad plane and shoves the force-limited finger to its joint limit (a
# wedge beats an 8N drive every time). Real tong tips curve inward for exactly this reason.
# Thin rim lips on ALL FOUR pad edges make a pocket so an in-plane escape presses on the
# RIGID LINK (held by the arm), not the prismatic drive. Sizing from the measured grasp
# (pad face local x=-0.003 fruit side, fruit r=0.043): rim depth 4mm < sphere bulge at every
# rim plane (5mm at y=+-0.020, 4mm at z=+-0.018 from pad center) -> the FLAT pad faces
# always contact first (wedge angle ~0: friction holds, no funnel force) and the rims stay
# ~1mm clear, engaging only when the fruit tries to escape. A DEEPER rim contacts the bulge
# before the pads do and turns the pocket into a funnel that squirts the sphere out the
# open end (verified: 6mm y-rims ejected the fruit out the fingertips at stem-break).
# Authored while the timeline is STOPPED (this setup) -> cooked once at play, no re-cook.
# Xforms are re-asserted every setup so stale geometry from a previous session is corrected.
# y-rims (gravity direction; fruit y-centered so ~1mm clear at face contact) + ONE tip rim
# at 1.1mm rest clearance (inner face -0.0495; sphere bulge there is 5.1mm > 4mm rim). The
# tip rim catches the small constant down-channel ratchet (~0.4N from the pads' 1.7-deg tilt
# + drive chatter -- the force the stem had been canceling) after 1-2mm of drift; a z-load
# on a rim is PERPENDICULAR to the prismatic axis, so it transfers through the rigid link to
# the arm instead of spreading the force-limited finger. NO knuckle rim: a knife-edge z-PAIR
# brackets the sphere and walks it down the channel during the close (measured 1.2cm), and
# nothing pushes the fruit toward the knuckle anyway.
# y-rims ONLY. NO tip rim: every tip-rim variant (4mm knife-edge, shaved 3mm deepened)
# put an asymmetric contact on the channel axis that squirted the materialized fruit out
# (verified across six configurations -- identical 8cm endpoint). The carry pull exits the
# knuckle side, so the channel's tip end needs no guard.
_RIMS = (("pad_rim_yp", (-0.005, 0.022, -0.0295), (0.002, 0.002, 0.031)),
         ("pad_rim_yn", (-0.005, -0.022, -0.0295), (0.002, 0.002, 0.031)))
for _fn2 in ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2"):
    _cs = "{prim}/" + _fn2 + "/collisions"
    if not st.GetPrimAtPath(_cs).IsValid():
        continue
    # remove obsolete z-rims left on the live stage by earlier sessions
    for _zr in ("pad_rim_zk", "pad_rim_zt"):
        if st.GetPrimAtPath(_cs + "/" + _zr).IsValid():
            st.RemovePrim(_cs + "/" + _zr)
    for _rn, _rt, _rs in _RIMS:
        _rp = _cs + "/" + _rn
        _rc = UsdGeom.Cube.Define(st, _rp)
        _rc.GetSizeAttr().Set(2.0)
        _rx = UsdGeom.Xformable(_rc)
        _rx.ClearXformOpOrder()
        _rx.AddTranslateOp().Set(Gf.Vec3d(*_rt))
        _rx.AddScaleOp().Set(Gf.Vec3f(*_rs))
        _rpr = _rc.GetPrim()
        UsdPhysics.CollisionAPI.Apply(_rpr)
        UsdGeom.Imageable(_rpr).CreateVisibilityAttr().Set("invisible")
        _rpc = PhysxSchema.PhysxCollisionAPI.Apply(_rpr)
        _rpc.CreateContactOffsetAttr(0.001); _rpc.CreateRestOffsetAttr(0.0)
        _rpc.CreateTorsionalPatchRadiusAttr(0.01); _rpc.CreateMinTorsionalPatchRadiusAttr(0.003)
        UsdShade.MaterialBindingAPI.Apply(_rpr).Bind(_mat, materialPurpose="physics")
# Fruit colliders stay ON for the whole run (the calyx cup holds the fruit by contact).
# The reach's joint-space interpolation swings the big arm links THROUGH the vine volume
# Fruit colliders OFF for the transit (authored while STOPPED): the reach's joint-space
# blend sweeps the arm through the vine volume and a live-collider fruit (even the rigid
# 1e9-stem one) gets brushed/wedged. The grasp enables the collider mid-play at the grasp
# pose -- proven benign against the unbreakable stem. FilteredPairsAPI is a complete no-op
# in this build, so no filtering anywhere.
for _t in ("/World/tomatoes/tomato0", "/World/tomatoes/tomato1"):
    _pr = st.GetPrimAtPath(_t)
    if _pr.IsValid():
        _ce = _pr.GetAttribute("physics:collisionEnabled")
        if not _ce or not _ce.IsValid():
            _ce = UsdPhysics.CollisionAPI.Apply(_pr).CreateCollisionEnabledAttr()
        _ce.Set(False)
print("fruit: rigid-stem ghosts for transit (colliders off)")
# SCENE CONTACT HYGIENE (research-validated): CCD interactions have friction DISABLED by
# default in PhysX (the only documented frictionless contact path -- and the importer
# enabled CCD scene-wide); the default frictionCorrelationDistance (0.025) collapses the
# whole 4.3cm fruit into ONE friction anchor, destroying the pinch's self-locking. These
# are NVIDIA's own small-object-manipulation values.
_scene = st.GetPrimAtPath("/World/PhysicsScene")
if _scene.IsValid():
    for _an, _ty, _av in (("physxScene:enableCCD", Sdf.ValueTypeNames.Bool, False),
                          ("physxScene:frictionOffsetThreshold", Sdf.ValueTypeNames.Float, 0.01),
                          ("physxScene:frictionCorrelationDistance", Sdf.ValueTypeNames.Float, 0.00625)):
        _a = _scene.GetAttribute(_an)
        if not _a or not _a.IsValid():
            _a = _scene.CreateAttribute(_an, _ty)
        _a.Set(_av)
    print("physics scene: CCD off, frictionOffset 0.01, frictionCorrelation 0.00625")
# round metal-pail basket (40cm dia x 40cm) at the drop-off point
_nsb = {{}}
exec(compile(open(r"{basket_py}", "r", encoding="utf-8").read(), "scene_add_basket.py", "exec"), _nsb)
_nsb["add_basket"]({bx}, {by})
# snap every commanded joint to home (deg) + cap its velocity so it EASES (no snap)
_home = {home}
for _prim in st.Traverse():
    _p = str(_prim.GetPath())
    if not _p.startswith("{prim}/joints/"):
        continue
    _n = _prim.GetName()
    if _n not in _home:
        continue
    for _k in ("drive:angular:physics:targetPosition", "state:angular:physics:position"):
        _a = _prim.GetAttribute(_k)
        if not _a or not _a.IsValid():
            _a = _prim.CreateAttribute(_k, Sdf.ValueTypeNames.Float)
        _a.Set(float(_home[_n]))
    # clear any leftover velocity cap from a prior run (it destabilises the drive)
    _mv = _prim.GetAttribute("physxJoint:maxJointVelocity")
    if _mv and _mv.IsValid():
        _mv.Set(1.0e7)
    # stiffen the arm drives so the extended grasp pose is tracked tightly (the default
    # gains leave the jaw a few cm short of the commanded pose at full reach)
    if "_arm_" in _n:
        for _kk, _vv in (("drive:angular:physics:stiffness", 60000.0), ("drive:angular:physics:damping", 2000.0)):
            _aa = _prim.GetAttribute(_kk)
            if not _aa or not _aa.IsValid():
                _aa = _prim.CreateAttribute(_kk, Sdf.ValueTypeNames.Float)
            _aa.Set(float(_vv))
_vel_unused = float({vel})
# gentle, FORCE-LIMITED finger drives: the prismatic gripper joints (not in homestate, so
# untouched above) get a bounded maxForce so closing onto the fruit produces a soft squeeze
# (a few N -- plenty to hold a 50g fruit by friction) instead of crushing/flinging it. Start
# spread so the fruit fits between the pads before the reach. Validated in the grasp bench.
for _prim in st.Traverse():
    _gn = _prim.GetName()
    if "gripper_finger_" not in _gn or _prim.GetTypeName() != "PhysicsPrismaticJoint":
        continue
    for _kk, _vv in (("drive:linear:physics:stiffness", 4000.0),
                     ("drive:linear:physics:damping", 200.0),
                     ("drive:linear:physics:maxForce", 8.0),
                     # QUASI-STATIC close: cap the finger speed so the pad CONTACTS the
                     # stem-anchored fruit instead of STRIKING it. An uncapped close sweeps
                     # ~10mm fast and the finger's impact impulse transiently exceeds any
                     # stem threshold (verified: 5N and 8.5N stems snapped identically, the
                     # fruit punted to the same mm), while the 8N drive cap only bounds the
                     # steady force. At 2cm/s the contact is impact-free and peak force
                     # stays at the spring level (<=8N) -- below the 8.5N stem.
                     ("physxJoint:maxJointVelocity", 0.02)):
        _aa = _prim.GetAttribute(_kk)
        if not _aa or not _aa.IsValid():
            _aa = _prim.CreateAttribute(_kk, Sdf.ValueTypeNames.Float)
        _aa.Set(float(_vv))
    # spread to the joint limit so the rim channel clears the fruit on the approach
    _ot = -0.0495 if _gn.endswith("1") else 0.0495
    _at = _prim.GetAttribute("drive:linear:physics:targetPosition")
    if not _at or not _at.IsValid():
        _at = _prim.CreateAttribute("drive:linear:physics:targetPosition", Sdf.ValueTypeNames.Float)
    _at.Set(float(_ot))
# NOTE: do NOT re-apply PhysxArticulationAPI / toggle EnabledSelfCollisions here. The
# articulation root is {prim}/root_joint and it ALREADY carries PhysxArticulationAPI with
# EnabledSelfCollisions=False (baked into the robot asset). Applying PhysxArticulationAPI to
# the parent {prim} at runtime creates a CONFLICTING articulation definition that forces PhysX
# to re-cook the articulation mid-session -> the instanced visual meshes desync from the (still
# coherent) skeleton and the robot renders "decomposed". Self-collision is already off in the
# asset, so the grasp behaviour is preserved with no runtime re-cook. (Isolated: this single op
# was the sole cause of the decomposition; reposition/play/material-bind/finger-drives are clean.)
_s = carb.settings.get_settings()
# Fixed exposure + splat dim (VERIFIED necessary): the NuRec greenhouse splat is EMISSIVE
# (self-lit) and over-bright. With a fixed exposure tuned for the lit robot, an undimmed splat
# blows out and back-lights the robot into a dark silhouette; with auto-exposure the background
# washes to white. Dimming the splat's emissive at the SOURCE (ccm 0.45) + this fixed exposure
# (fNumber 6) is what keeps the white robot well-lit and readable against the greenhouse.
# (The robot "decomposition" was never this -- that is a stale dark recolor / collider-display
# state on the live stage from prior edits, cleared by a fresh stage load, not by these knobs.)
_s.set("/rtx/post/histogram/enabled", False); _s.set("/rtx/post/tonemap/fNumber", 6.0)
_s.set("/rtx/post/tonemap/filmIso", 100.0); _s.set("/rtx/post/tonemap/cameraShutter", 70.0)
# dim the greenhouse splat at the source so the robot reads against it (idempotent)
_ef = st.GetPrimAtPath("/World/greenhouse/gauss/gauss/emissive_color_field")
if _ef.IsValid():
    _ccm = 0.45
    for _k, _vv in (("omni:nurec:ccmR", (_ccm, 0, 0, 0)), ("omni:nurec:ccmG", (0, _ccm, 0, 0)),
                    ("omni:nurec:ccmB", (0, 0, _ccm, 0))):
        _a = _ef.GetAttribute(_k)
        if _a and _a.IsValid():
            _cur = _a.Get()
            _a.Set(type(_cur)(*_vv) if _cur is not None else Gf.Vec4f(*_vv))
tl.set_current_time(0.0); tl.play()
print("sim replay: setup done, playing")
'''

_FRAME_TEMPLATE = '''
import omni.usd
from pxr import Sdf
st = omni.usd.get_context().get_stage()
_t = {targets}
for _prim in st.Traverse():
    _p = str(_prim.GetPath())
    if not _p.startswith("{prim}/joints/"):
        continue
    _n = _prim.GetName()
    if _n not in _t:
        continue
    _a = _prim.GetAttribute("drive:angular:physics:targetPosition")
    if not _a or not _a.IsValid():
        _a = _prim.CreateAttribute("drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float)
    _a.Set(float(_t[_n]))
'''

_TEARDOWN_TEMPLATE = '''
import omni.timeline
omni.timeline.get_timeline_interface().pause()
# stems are NOT touched: runtime breakForce changes snap a live joint outright (the next
# setup re-authors both tomatoes from scratch while stopped anyway).
print("sim replay: paused at final pose")
'''

# best-effort: drive the active-arm gripper finger prismatic joints. open spreads the
# fingers, close brings them together. Range unknown -> nominal targets; no-op if absent.
_GRIPPER_TEMPLATE = '''
import omni.usd
from pxr import Sdf
st = omni.usd.get_context().get_stage()
# fingers move in OPPOSITE directions (l1: -limit..0, l2: 0..+limit). open spreads wide; close
# commands CONTACT + ~1mm preload, NOT a deep crush. Measured: the pads stall on the 8.5cm
# fruit at finger states summing to ~0.080, so 0.030 commanded 13mm of interference -- both
# drives saturated at maxForce, and a rigid sphere crushed between two flat pads at saturated
# force is laterally unstable: at stem-break the fruit shoved one finger to its joint limit
# and squirted out (verified by the GRIPSTATE trace). The gentle preload close puts the drive
# in its linear zone (~4-8N normal) like a real tong pinch; the pad cradle rims (setup) cage
# the sphere laterally. OPEN must spread to the joint limit so the rim channel (pad gap minus
# 2x6mm rims) clears the 85.4mm fruit on the approach and at the basket release (93mm at
# +-0.0495 vs 84mm at the old +-0.045, which would jam the fruit against the rims).
# "precage": a loose cage at ~1mm clearance around the still-KINEMATIC fruit. Contact with
# a kinematic (infinite-mass) body shoves the whole articulation backward via depenetration
# (the arm retreated 8cm with intact targets -- verified), so the fingers must surround the
# fruit WITHOUT touching it, the fruit is flipped dynamic into the waiting cage, and only
# then does the real squeeze close on a body that can yield.
_mag = 0.0495 if ("{kind}" == "open") else (0.0435 if "{kind}" == "precage" else 0.0385)
_n = 0
for _prim in st.Traverse():
    _p = str(_prim.GetPath())
    if not _p.startswith("{prim}/joints/"):
        continue
    _name = _prim.GetName()
    if "gripper_finger_{side}" not in _name:
        continue
    if _prim.GetTypeName() != "PhysicsPrismaticJoint":
        continue
    _t = -_mag if _name.endswith("1") else _mag
    _a = _prim.GetAttribute("drive:linear:physics:targetPosition")
    if not _a or not _a.IsValid():
        _a = _prim.CreateAttribute("drive:linear:physics:targetPosition", Sdf.ValueTypeNames.Float)
    _a.Set(float(_t)); _n += 1
print("gripper {kind}: drove", _n, "finger joints")
'''

# grasp: attach the fruit nearest the EEF to the EEF link (break its stem, fixed joint
# at the current relative pose so it is carried without a snap). Tracks the held fruit.
# grasp: hide the physics fruit nearest the tongs, free its stem, and place a VISUAL fruit
# between the jaws as a child of the gripper link -- it follows the gripper automatically.
_GRASP_TEMPLATE = '''
import omni.usd
import omni.usd as ou
from pxr import UsdGeom, UsdPhysics, Gf
st = omni.usd.get_context().get_stage()
_eL = st.GetPrimAtPath("{prim}/{eef}")
_emi = ou.get_world_transform_matrix(_eL).GetInverse()
_fs = []
for _fn in ("ee_finger_{side}1", "ee_finger_{side}2"):
    _fp = st.GetPrimAtPath("{prim}/" + _fn)
    if _fp.IsValid():
        _fs.append(Gf.Vec3d(ou.get_world_transform_matrix(_fp).ExtractTranslation()))
_jaw = (_fs[0] + _fs[1]) * 0.5 if len(_fs) == 2 else Gf.Vec3d(ou.get_world_transform_matrix(_eL).ExtractTranslation())
_jl = _emi.Transform(_jaw)
_best = None; _bd = 1e9
for _nm in ("tomato0", "tomato1"):
    _tp = st.GetPrimAtPath("/World/tomatoes/" + _nm)
    if not _tp.IsValid():
        continue
    _pp = Gf.Vec3d(ou.get_world_transform_matrix(_tp).ExtractTranslation())
    _d = (_pp - _jaw).GetLength()
    if _d < _bd:
        _bd = _d; _best = _nm
if _best is not None:
    # IDENTIFY ONLY -- the collider enable moved to _ENABLE_TEMPLATE, fired AFTER the
    # carry joint is live: the mid-play toggle's body-re-init transient LOADS attached
    # joints without killing them (it snapped an 8.5N stem at threshold -- a live-joint
    # event), so with a breakable stem the enable IS the severing action, and the
    # pre-attached carry joint catches the fruit at the pinch with zero free window.
    _fp2 = "/World/tomatoes/" + _best
    _tw = Gf.Vec3d(ou.get_world_transform_matrix(st.GetPrimAtPath(_fp2)).ExtractTranslation())
    print("GRASPED:" + _best)
    print("grasp: target identified. jaw-to-fruit dist", round(_bd, 3))
    print("GRASPVEC %.4f %.4f %.4f %.4f %.4f %.4f" % (_jaw[0], _jaw[1], _jaw[2], _tw[0], _tw[1], _tw[2]))
else:
    print("grasp: no fruit found near tongs")
'''

# mid-play collider enable on the held fruit, AFTER the carry joint is live: the body
# re-init transient loads the 60N stem past breakForce (the engine-internal severing
# action -- joints are LOADED by this path, not killed; verified by the 8.5N-era snap)
# while the carry joint, instantiated at the pinch and unbreakable, catches the fruit.
_ENABLE_TEMPLATE = '''
import omni.usd
st = omni.usd.get_context().get_stage()
_ce = st.GetPrimAtPath("/World/tomatoes/{name}").GetAttribute("physics:collisionEnabled")
if _ce and _ce.IsValid():
    _ce.Set(True)
    print("collider enabled (mid-play) on {name} -- safe against the 1e38 stem")
else:
    print("collider enable FAILED: attribute missing on {name}")
'''

# STEM REMOVAL release (research-validated): every joint-ATTRIBUTE edit routes through
# omni.physx's USD re-parse path whose re-init impulse ejects the fruit, but deleting the
# joint PRIM goes through PxConstraint::release(), which PhysX documents as impulse-free.
# Mid-play, no pause (pause/resume swings the arm ~8cm through the fruit -- verified).
_STEM_REMOVE_TEMPLATE = '''
import omni.usd, omni.physx, builtins
st = omni.usd.get_context().get_stage()
_sp = "/World/tomatoes/{name}_stem"
# Remove the stem INSIDE a physics-step callback (the SurfaceGripper pattern): a RemovePrim
# from the exec thread defers the topology change to the next structural flush, whose
# re-init impulse then ejects the fruit mid-carry (verified: held 2.5s, ejected the moment
# the first carry frame flushed). Inside the stepping pipeline the change applies cleanly.
def _remove_stem_once(_dt):
    try:
        if st.GetPrimAtPath(_sp).IsValid():
            st.RemovePrim(_sp)
    finally:
        _s = getattr(builtins, "_stem_remove_sub", None)
        if _s is not None:
            try:
                _s.unsubscribe()
            except Exception:
                pass
        builtins._stem_remove_sub = None
builtins._stem_remove_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(_remove_stem_once)
print("stem removal queued inside the physics step (SurfaceGripper pattern): {name}")
'''

# ---- Isaac SurfaceGripper integration (the engine-sanctioned runtime attach) ---------
# Attach/detach execute as PxJoint::setActors + setLocalPose inside the physics step in
# C++ with write-to-usd OFF: ZERO USD authoring at runtime, so the change-processing
# layer (lazy flushes, body re-cooks, joint-frame chaos -- all verified fatal for a
# USD-authored carry joint) never touches it, and the component revalidates the grip
# every step. Authored ONCE while STOPPED: a D6 attachment joint under the palm link
# (gantry template values) + the IsaacSurfaceGripper prim.
_SG_SETUP_TEMPLATE = '''
import omni.timeline, omni.usd
import omni.kit.app
from pxr import Gf, Sdf, UsdPhysics
_mgr = omni.kit.app.get_app().get_extension_manager()
for _x in ("isaacsim.robot.schema", "isaacsim.robot.surface_gripper"):
    if not _mgr.is_extension_enabled(_x):
        _mgr.set_extension_enabled_immediate(_x, True)
import usd.schema.isaac.robot_schema as robot_schema
import omni.usd as ou
omni.timeline.get_timeline_interface().stop()
st = omni.usd.get_context().get_stage()
# ROOT-CAUSE FIX: the URDF importer's forearm collision capsule (r=0.075, half-length
# 0.186, eef-local span z=[-0.145,+0.226]) ENVELOPES the entire grasp volume -- the
# pinched fruit sits inside it, and its depenetration push ejected every freed fruit to
# exactly capsule_tip - fruit_r (= the bit-identical -0.185 endpoint across 13 runs,
# including the zero-g era). It also blocked the SurfaceGripper raycast at dist 0.
# De-instance (proxies are read-only) and disable it, while STOPPED (clean re-cook).
_cw = st.GetPrimAtPath("{prim}/link_{armname}_arm_5/collisions")
if _cw.IsValid() and _cw.IsInstanceable():
    _cw.SetInstanceable(False)
_cap = st.GetPrimAtPath("{prim}/link_{armname}_arm_5/collisions/mesh_0/capsule")
if _cap.IsValid():
    _ca = _cap.GetAttribute("physics:collisionEnabled")
    if not _ca or not _ca.IsValid():
        _ca = _cap.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
    _ca.Set(False)
    print("forearm capsule collider DISABLED (was swallowing the grasp volume)")
else:
    print("WARN: forearm capsule prim not found")
_jp = "{prim}/{eef}/AttachmentJoint"
if True:  # re-assert every setup (Define + Set are idempotent; fixes partial authoring)
    _j = UsdPhysics.Joint.Define(st, _jp)
    _j.CreateBody0Rel().SetTargets(["{prim}/{eef}"])
    # ray origin on the palm centerline, 8mm behind the mid-pad fruit surface; joint +Z
    # (forwardAxis) aimed out the palm via the gantry 180deg-about-X localRot0
    _lp0 = Gf.Vec3d(0.0, 0.002, -0.0535)
    _lr0 = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 180.0)
    _j.CreateLocalPos0Attr().Set(Gf.Vec3f(_lp0))
    _q0 = _lr0.GetQuat()
    _j.CreateLocalRot0Attr().Set(Gf.Quatf(_q0.GetReal(), Gf.Vec3f(_q0.GetImaginary())))
    # PARKED rest state (gantry pattern): body1 = the palm's mount link, frames computed
    # from current transforms so the parked constraint is exactly satisfied. body1 EMPTY
    # means joint-to-world in USD-Physics -- omni.physx never instantiated it and
    # close_gripper found no PxJoint at the attachment point (verified: Closing->Open,
    # gripped stayed empty).
    _ep = st.GetPrimAtPath("{prim}/{eef}")
    _mp = st.GetPrimAtPath("{prim}/link_{armname}_arm_6")
    _eT = ou.get_world_transform_matrix(_ep)
    _mT = ou.get_world_transform_matrix(_mp)
    _J = Gf.Matrix4d().SetRotate(_lr0)
    _J.SetTranslateOnly(_lp0)
    _Jw = _J * _eT                      # joint frame in world (row-vector convention)
    _rel1 = _Jw * _mT.GetInverse()      # joint frame in mount-link local
    _t1 = _rel1.ExtractTranslation()
    _q1 = _rel1.ExtractRotation().GetQuat()
    _j.CreateBody1Rel().SetTargets(["{prim}/link_{armname}_arm_6"])
    _j.CreateLocalPos1Attr().Set(Gf.Vec3f(_t1))
    _j.CreateLocalRot1Attr().Set(Gf.Quatf(_q1.GetReal(), Gf.Vec3f(_q1.GetImaginary())))
    _j.CreateBreakForceAttr().Set(3.4028235e38)
    _j.CreateBreakTorqueAttr().Set(3.4028235e38)
    _j.CreateExcludeFromArticulationAttr().Set(True)
    _j.CreateJointEnabledAttr().Set(True)
    _p = _j.GetPrim()
    for _ax, _lo, _hi in (("transX", 1.0, -1.0), ("transY", 1.0, -1.0), ("transZ", 0.0, 0.01),
                          ("rotX", -3.0, 3.0), ("rotY", -3.0, 3.0), ("rotZ", -3.0, 3.0)):
        _l = UsdPhysics.LimitAPI.Apply(_p, _ax)
        _l.CreateLowAttr(_lo); _l.CreateHighAttr(_hi)
    for _ax, _k, _d in (("transZ", 5000.0, 100.0), ("rotX", 100.0, 0.0),
                        ("rotY", 100.0, 0.0), ("rotZ", 10000.0, 0.0)):
        _dr = UsdPhysics.DriveAPI.Apply(_p, _ax)
        _dr.CreateStiffnessAttr(_k)
        if _d:
            _dr.CreateDampingAttr(_d)
    robot_schema.ApplyAttachmentPointAPI(_p)
    _p.GetAttribute("isaac:forwardAxis").Set("Z")
    _p.GetAttribute("isaac:clearanceOffset").Set(0.008)
_gp = "/World/SurfaceGripper"
_g = robot_schema.CreateSurfaceGripper(st, _gp)  # idempotent; re-assert attrs every setup
_g.GetRelationship("isaac:attachmentPoints").SetTargets([_jp])
_g.GetAttribute("isaac:maxGripDistance").Set(0.12)
_g.GetAttribute("isaac:shearForceLimit").Set(1.0e6)
_g.GetAttribute("isaac:coaxialForceLimit").Set(1.0e6)
_g.GetAttribute("isaac:retryInterval").Set(1.0)
print("surface gripper ready (authored stopped):", _jp, "->", _gp,
      "| parked body1 localPos1=(%.4f,%.4f,%.4f)" % (_t1[0], _t1[1], _t1[2]))
'''

_SG_CLOSE_TEMPLATE = '''
import isaacsim.robot.surface_gripper._surface_gripper as _sgmod
_sg = _sgmod.acquire_surface_gripper_interface()
_sg.set_write_to_usd(False)
_ok = _sg.close_gripper("/World/SurfaceGripper")
print("sg close ->", _ok, "| status:", _sg.get_gripper_status("/World/SurfaceGripper"),
      "| gripped:", _sg.get_gripped_objects("/World/SurfaceGripper"))
'''

_SG_OPEN_TEMPLATE = '''
import isaacsim.robot.surface_gripper._surface_gripper as _sgmod
_sg = _sgmod.acquire_surface_gripper_interface()
_sg.set_write_to_usd(False)
_ok = _sg.open_gripper("/World/SurfaceGripper")
print("sg open ->", _ok, "| status:", _sg.get_gripper_status("/World/SurfaceGripper"))
'''

_SG_STATE_TEMPLATE = '''
import isaacsim.robot.surface_gripper._surface_gripper as _sgmod
_sg = _sgmod.acquire_surface_gripper_interface()
print("SGSTATE status:", _sg.get_gripper_status("/World/SurfaceGripper"),
      "gripped:", _sg.get_gripped_objects("/World/SurfaceGripper"))
'''

# release: the fingers were already opened by the gripper template, so the friction-held fruit
# is let go and free-falls from the tongs into the basket (gripper held over the basket).
_RELEASE_TEMPLATE = '''
import omni.usd
st = omni.usd.get_context().get_stage()
print("released {name} -> fingers opened, real fruit drops into the basket")
'''
