"""Pose the imported RB-Y1 into its captured home/ready posture and frame it.

Sets each joint's drive target + initial state from robot/utils/homestate.json
(positions_rad; USD revolute units are degrees, prismatic meters) and plays the
timeline so the articulation holds the pose (base is pinned/kinematic).

Run via :8226:  isaac_client.py --bootstrap .../scene_pose_home.py "pose_home()"
"""
import json
import math

HOME = r"C:/path/to/projects/EgoDelta/robot/utils/homestate.json"
ROBOT = "/RBY1_A_v1_1"


def pose_home(home_path=HOME, robot=ROBOT, play=True, frame=True):
    import omni.usd
    import omni.timeline
    from pxr import Sdf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    data = json.load(open(home_path))
    hm = dict(zip(data["joint_names"], data["all_joints"]["positions_rad"]))

    root = robot + "/joints/"
    n = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root):
            continue
        jn = prim.GetName()
        if jn not in hm:
            continue
        tt = prim.GetTypeName()
        if tt == "PhysicsRevoluteJoint":
            val = math.degrees(hm[jn])
            keys = ("drive:angular:physics:targetPosition", "state:angular:physics:position")
        elif tt == "PhysicsPrismaticJoint":
            val = hm[jn]
            keys = ("drive:linear:physics:targetPosition", "state:linear:physics:position")
        else:
            continue
        for key in keys:
            a = prim.GetAttribute(key)
            if not a or not a.IsValid():
                a = prim.CreateAttribute(key, Sdf.ValueTypeNames.Float)
            a.Set(float(val))
        n += 1
    print("posed", n, "joints to home")

    if play:
        omni.timeline.get_timeline_interface().play()
        print("timeline playing -> robot holds home pose")

    if frame:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf
        vp = get_active_viewport()
        cam = stage.GetPrimAtPath(vp.camera_path)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(2.0, -2.4, 1.45), Gf.Vec3d(0.15, 0.0, 0.85), Gf.Vec3d(0, 0, 1))
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(view.GetInverse())
        print("framed on robot")
    return n
