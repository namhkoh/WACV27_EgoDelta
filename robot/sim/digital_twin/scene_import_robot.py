"""Import the RB-Y1 (A-class) URDF into the LIVE Isaac Sim stage and frame it.

Inlines the importer config/drives/base-pin from robot/sim/rby1_sphere_scene.py
(that module can't be imported in a running Kit). Uses the two-step parse->import
flow (the combined command's get_articulation_root returns False on 5.1 even though
the robot imports fine); locates the articulation root via ArticulationRootAPI.

Run via :8226:  isaac_client.py --bootstrap .../scene_import_robot.py "import_robot()"
"""
from pathlib import Path

REPO = r"C:/path/to/projects/EgoDelta"
URDF = REPO + "/submodules/rby1-sdk/models/rby1a/urdf/model_v1.1.urdf"

REVOLUTE = (5000.0, 250.0, 1000.0)
PRISMATIC = (1000.0, 50.0, 200.0)
STRAY = ["/World/RBY1_A_v1_1", "/RBY1_A_v1_1", "/visuals", "/colliders", "/meshes", "/World/rby1"]


def _clean(stage):
    for p in STRAY:
        if stage.GetPrimAtPath(p).IsValid():
            stage.RemovePrim(p)


def _make_config():
    import omni.kit.commands
    s, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False
    cfg.fix_base = True
    cfg.import_inertia_tensor = True
    cfg.distance_scale = 1.0
    cfg.convex_decomp = False
    cfg.collision_from_visuals = False
    return cfg


def _find_robot_prim(stage):
    from pxr import UsdPhysics
    # 1) an existing articulation root anywhere in the stage
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            return p.GetPath().pathString
    # 2) top-level prim named like the robot or with a /joints child
    roots = list(stage.GetPseudoRoot().GetChildren())
    w = stage.GetPrimAtPath("/World")
    if w.IsValid():
        roots += list(w.GetChildren())
    for p in roots:
        pp = p.GetPath().pathString
        if "rby1" in p.GetName().lower() or stage.GetPrimAtPath(pp + "/joints").IsValid():
            return pp
    return None


def _configure_joint_drives(stage, robot_prim_path):
    root = robot_prim_path + "/joints/"
    n = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root):
            continue
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            drive, vals = "drive:angular:physics", REVOLUTE
        elif prim.GetTypeName() == "PhysicsPrismaticJoint":
            drive, vals = "drive:linear:physics", PRISMATIC
        else:
            continue
        for attr_name, value in zip((drive + ":stiffness", drive + ":damping", drive + ":maxForce"), vals):
            a = prim.GetAttribute(attr_name)
            if a and a.IsValid():
                a.Set(float(value))
        n += 1
    return n


def _pin_base(stage, base_link_path):
    from pxr import Sdf, UsdPhysics
    prim = stage.GetPrimAtPath(base_link_path)
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    for name, (tn, v) in {
        "physics:kinematicEnabled": (Sdf.ValueTypeNames.Bool, True),
        "physxRigidBody:disableGravity": (Sdf.ValueTypeNames.Bool, True),
        "physxRigidBody:lockedPosAxis": (Sdf.ValueTypeNames.Int, 7),
        "physxRigidBody:lockedRotAxis": (Sdf.ValueTypeNames.Int, 7),
    }.items():
        a = prim.GetAttribute(name)
        if not a or not a.IsValid():
            a = prim.CreateAttribute(name, tn)
        a.Set(v)


def import_robot(urdf=URDF, position=(0.0, 0.0, 0.0), yaw_deg=0.0, hide_greenhouse=True):
    import omni.kit.app
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools

    mgr = omni.kit.app.get_app().get_extension_manager()
    if not mgr.is_extension_enabled("isaacsim.asset.importer.urdf"):
        mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(stage, "/World")
    if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
        sc = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
        sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        sc.CreateGravityMagnitudeAttr().Set(9.81)
        physx = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
        # CCD must stay OFF: PhysX DISABLES friction inside CCD interactions, which silently
        # breaks the two-finger friction grasp (NVIDIA-documented; verified in the grasp
        # bench). Nothing in this scene moves fast enough to tunnel.
        physx.CreateEnableCCDAttr(False)
        physx.CreateEnableStabilizationAttr(True)
        physx.CreateSolverTypeAttr("TGS")
        # small-object contact hygiene for the 8cm fruit (NVIDIA small-object values): the
        # default frictionCorrelationDistance (0.025) collapses the fruit's two pad contacts
        # into ONE friction anchor, halving the grip's effective friction.
        physx.CreateFrictionOffsetThresholdAttr(0.01)
        physx.CreateFrictionCorrelationDistanceAttr(0.00625)
    if not stage.GetPrimAtPath("/World/GroundPlane").IsValid():
        PhysicsSchemaTools.addGroundPlane(stage, "/World/GroundPlane", "Z", 20.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.4, 0.4, 0.4))

    _clean(stage)
    cfg = _make_config()
    s, robot_model = omni.kit.commands.execute("URDFParseFile", urdf_path=str(urdf), import_config=cfg)
    if not s or robot_model is None:
        raise RuntimeError("URDF parse failed")
    omni.kit.commands.execute("URDFImportRobot", urdf_path=str(urdf), urdf_robot=robot_model, import_config=cfg)

    prim_path = _find_robot_prim(stage)
    if prim_path is None:
        raise RuntimeError("could not locate robot prim after import")
    rp0 = stage.GetPrimAtPath(prim_path)
    if not rp0.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(rp0)
    print("robot prim ->", prim_path)

    n_drives = _configure_joint_drives(stage, prim_path)
    _pin_base(stage, prim_path)
    print("configured", n_drives, "joint drives; base pinned")

    rp = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(rp)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    if yaw_deg:
        xf.AddRotateZOp().Set(float(yaw_deg))

    if hide_greenhouse:
        gh = stage.GetPrimAtPath("/World/greenhouse")
        if gh.IsValid():
            UsdGeom.Imageable(gh).MakeInvisible()

    n_links = sum(1 for p in stage.Traverse() if str(p.GetPath()).startswith(prim_path) and p.HasAPI(UsdPhysics.RigidBodyAPI))
    print("robot rigid-body links:", n_links)

    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = bbc.ComputeWorldBound(rp).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    c = (mn + mx) * 0.5
    d = (mx - mn).GetLength()
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    cam = stage.GetPrimAtPath(vp.camera_path)
    eye = Gf.Vec3d(c[0] + 1.4 * d, c[1] - 1.4 * d, c[2] + 0.6 * d)
    view = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(c[0], c[1], c[2]), Gf.Vec3d(0, 0, 1))
    xfc = UsdGeom.Xformable(cam)
    xfc.ClearXformOpOrder()
    xfc.AddTransformOp().Set(view.GetInverse())
    print("robot world bbox min", [round(x, 3) for x in mn], "max", [round(x, 3) for x in mx])
    return prim_path
