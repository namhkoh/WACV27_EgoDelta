"""One clean pass: re-import the full RB-Y1, pose to home via drive targets + joint
state, un-hide the greenhouse splat, play, and frame the robot. Live-GUI friendly
(no Isaac Core SimulationContext / SingleArticulation, which fail inside a GUI).

Run:  isaac_client.py --bootstrap .../scene_setup_twin.py "setup()"
"""
import json
import math

# model.urdf imports fully on Isaac Sim 5.1; model_v1.1.urdf fails partway (right arm dropped).
URDF = r"C:/path/to/projects/EgoDelta/submodules/rby1-sdk/models/rby1a/urdf/model.urdf"
HOME = r"C:/path/to/projects/EgoDelta/robot/utils/homestate.json"
REV = (5000.0, 250.0, 1000.0)
PRI = (1000.0, 50.0, 200.0)
STRAY = ["/World/RBY1_A_v1_1", "/RBY1_A_v1_1", "/RBY1_A_v1_0", "/World/RBY1_A_v1_0",
         "/visuals", "/colliders", "/meshes", "/World/rby1"]

# finger-pad box collider in each finger-link local frame (= the visual mesh extent,
# which sits at the identity transform relative to the link). The URDF <capsule>
# collision tags do not survive import and a convex re-import corrupts the visual
# meshes, so we inject one analytic Cube per finger pad -- physics-only (invisible),
# so the gripper can physically grasp without touching the rendered geometry.
FINGER_LINKS = ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
PAD_CENTER = (0.005, 0.0, -0.0295)
PAD_HALF = (0.008, 0.016, 0.031)


def add_finger_colliders(stage, robot_prim):
    """Inject one invisible box collider into each gripper finger link. Idempotent."""
    from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
    made = 0
    for fn in FINGER_LINKS:
        link = "%s/%s" % (robot_prim, fn)
        if not stage.GetPrimAtPath(link).IsValid():
            continue
        box_path = link + "/collisions/pad_box"
        if stage.GetPrimAtPath(box_path).IsValid():
            continue  # already injected
        cube = UsdGeom.Cube.Define(stage, box_path)
        cube.GetSizeAttr().Set(2.0)
        x = UsdGeom.Xformable(cube)
        x.ClearXformOpOrder()
        x.AddTranslateOp().Set(Gf.Vec3d(*PAD_CENTER))
        x.AddScaleOp().Set(Gf.Vec3f(*PAD_HALF))
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr(0.003)
        pc.CreateRestOffsetAttr(0.0)
        made += 1
    print("finger colliders injected:", made)
    return made


def setup(show_greenhouse=True, pose=True):
    import omni.usd
    import omni.timeline
    import omni.kit.app
    import omni.kit.commands
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

    # clear any broken SimulationContext from earlier attempts
    try:
        from isaacsim.core.api import SimulationContext
        if SimulationContext.instance() is not None:
            SimulationContext.clear_instance()
            print("cleared stale SimulationContext")
    except Exception:
        pass

    st = omni.usd.get_context().get_stage()
    omni.timeline.get_timeline_interface().stop()
    for p in STRAY:
        if st.GetPrimAtPath(p).IsValid():
            st.RemovePrim(p)

    mgr = omni.kit.app.get_app().get_extension_manager()
    if not mgr.is_extension_enabled("isaacsim.asset.importer.urdf"):
        mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

    s, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False
    cfg.fix_base = True
    cfg.import_inertia_tensor = True
    cfg.distance_scale = 1.0
    cfg.convex_decomp = False
    cfg.collision_from_visuals = False
    s, model = omni.kit.commands.execute("URDFParseFile", urdf_path=URDF, import_config=cfg)
    if not s or model is None:
        raise RuntimeError("parse failed")
    omni.kit.commands.execute("URDFImportRobot", urdf_path=URDF, urdf_robot=model, import_config=cfg)

    # the robot is the top-level Xform named like the model (joints live under it)
    robot = None
    candidates = list(st.GetPseudoRoot().GetChildren())
    w = st.GetPrimAtPath("/World")
    if w.IsValid():
        candidates += list(w.GetChildren())
    for pr in candidates:
        if "rby1" in pr.GetName().lower():
            robot = pr.GetPath().pathString
            break
    if robot is None:
        raise RuntimeError("could not locate robot Xform")
    rp = st.GetPrimAtPath(robot)
    if not rp.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(rp)
    print("robot prim:", robot)

    joints = [pr for pr in st.Traverse()
              if str(pr.GetPath()).startswith(robot + "/joints/") and "Joint" in pr.GetTypeName()]
    print("imported joints:", len(joints))

    home = json.load(open(HOME))
    hm = dict(zip(home["joint_names"], home["all_joints"]["positions_rad"]))
    nposed = 0
    for pr in joints:
        nm = pr.GetName()
        tt = pr.GetTypeName()
        if tt == "PhysicsRevoluteJoint":
            dr, vals, tgt, inst = "drive:angular:physics", REV, math.degrees(hm.get(nm, 0.0)), "angular"
        elif tt == "PhysicsPrismaticJoint":
            dr, vals, tgt, inst = "drive:linear:physics", PRI, hm.get(nm, 0.0), "linear"
        else:
            continue
        for k, v in zip((dr + ":stiffness", dr + ":damping", dr + ":maxForce"), vals):
            a = pr.GetAttribute(k)
            if a and a.IsValid():
                a.Set(float(v))
        if pose and nm in hm:
            a = pr.GetAttribute(dr + ":targetPosition")
            if not a or not a.IsValid():
                a = pr.CreateAttribute(dr + ":targetPosition", Sdf.ValueTypeNames.Float)
            a.Set(float(tgt))
            try:
                js = PhysxSchema.JointStateAPI.Apply(pr, inst)
                js.CreatePositionAttr().Set(float(tgt))
                js.CreateVelocityAttr().Set(0.0)
            except Exception:
                pass
            nposed += 1
    print("configured drives; posed", nposed, "joints")

    # give the gripper fingers collision geometry so it can physically grasp fruit
    add_finger_colliders(st, robot)

    gh = st.GetPrimAtPath("/World/greenhouse")
    if gh.IsValid():
        img = UsdGeom.Imageable(gh)
        img.MakeVisible() if show_greenhouse else img.MakeInvisible()
        print("greenhouse visible:", show_greenhouse)

    if pose:
        omni.timeline.get_timeline_interface().play()
        print("playing -> settling to home pose")

    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    cam = st.GetPrimAtPath(vp.camera_path)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(2.2, -2.6, 1.5), Gf.Vec3d(0.1, 0.0, 0.85), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    print("framed on robot")
    return robot
