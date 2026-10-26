"""Author harvestable tomato rigid bodies + breakable stem joints from tomatoes.json.

Each tomato = a sphere with RigidBodyAPI + CollisionAPI + mass, held in place by a
UsdPhysics.FixedJoint to the world carrying breakForce/breakTorque (the 'stem'). A
grasp-and-pull whose constraint force exceeds breakForce irreversibly severs the
joint -> the fruit detaches. Rigid + breakable = RL-stable (no FEM).

Run via :8226:  isaac_client.py --bootstrap .../scene_add_tomatoes.py "add_tomatoes()"
"""
import json

TOMATOES = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/tomatoes.json"


def add_tomatoes(radius=0.04, mass=0.05, frame=True):
    # The stem is a COMPLIANT SPRING joint (see below), detached by the harvest pipeline by
    # zeroing its drive stiffness -- a pure drive-parameter write. Breakable/rigid stems are
    # unusable: PhysX breakForce trips on mere contact-pair formation at realistic values,
    # and releasing a rigid FixedJoint (disable/pause-wrapped/any form) dumps the stored
    # solver binding force into the force-limited finger, ejecting the fruit (all verified
    # by instrumented runs).
    # Matches the real testbed prop: "리얼 토마토 모형" 8cm x 7cm styrofoam/PVC
    # (single beefsteak tomato + green calyx). 8cm dia -> r=0.04; styrofoam -> ~50g.
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema

    st = omni.usd.get_context().get_stage()
    # clear the identification markers + any prior tomatoes
    for p in ("/World/tomato_markers", "/World/tomatoes"):
        if st.GetPrimAtPath(p).IsValid():
            st.RemovePrim(p)
    UsdGeom.Xform.Define(st, "/World/tomatoes")

    fruit = json.load(open(TOMATOES))
    for f in fruit:
        p = "/World/tomatoes/" + f["name"]
        r = float(radius) if radius else float(f["radius"])
        sph = UsdGeom.Sphere.Define(st, p)
        sph.CreateRadiusAttr(r)
        sph.CreateDisplayColorAttr([(0.88, 0.13, 0.07)])
        prim = sph.GetPrim()
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(f["x"], f["y"], f["z"]))
        # slightly oblate look (8cm wide, 7cm tall) -- visual scale on the prim;
        # collider stays a sphere of radius r for RL-stable contacts
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        # anti-fling contact stability for the two-finger friction grasp: cap the
        # depenetration kick + linear velocity (the violent fling came from PhysX
        # resolving finger/fruit overlap at the default high depenetration speed),
        # raise solver iterations, and tighten the contact/rest offsets so a gentle
        # close just touches the surface. Validated in the grasp bench.
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        # gentle contact dynamics for the friction grasp: a soft depenetration so re-enabling
        # the fruit collider against the closing pads does not kick it out. The linear velocity
        # cap MUST stay above the gripper's carry speed (~0.3 m/s): a lower cap means the fruit
        # physically cannot follow the moving hand and slides out of the tongs (verified -- a
        # 0.2 cap dropped the fruit the instant the carry started while the static grip held).
        # NVIDIA-standard depenetration (5.0 ships in every official manipulation demo):
        # starved values (0.1/0.01 tried) spread corrections into friction-immune creep.
        px.CreateMaxDepenetrationVelocityAttr(5.0)
        px.CreateMaxLinearVelocityAttr(0.8)
        # 32 position iterations (FrankaBrickStack) but 4 velocity iterations: the carry is
        # DYNAMIC (BrickStack's 1 is for static stacking) and friction convergence during
        # accelerations needs the velocity passes.
        px.CreateSolverPositionIterationCountAttr(32)
        px.CreateSolverVelocityIterationCountAttr(4)
        # sleep-friendly threshold: the SLEEPING state IS the vine attachment. A sleeping
        # dynamic body holds its pose like a kinematic one (ignores gravity) but becomes a
        # normal 50g body the instant a contact wakes it -- the perfect grasp handoff
        # (kinematic contact shoves the whole articulation back via depenetration against
        # infinite mass; joints snap/kick; support geometry wedges the arm -- all verified
        # dead ends). The pipeline puts the fruit to sleep explicitly after play.
        px.CreateSleepThresholdAttr(0.005)
        # rolling resistance: a RIGID sphere between two flat pads ROLLS out of the pinch
        # (static friction does not resist rolling; the torsional patch only covers spin
        # about the contact normal). A real tomato's skin flattens into a contact patch that
        # resists rolling; high angular damping is the rigid-body surrogate for that patch
        # (standard practice for sim grasping of spheres).
        px.CreateAngularDampingAttr(10.0)
        px.CreateMaxAngularVelocityAttr(4.0)
        pcx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        # 1mm contact offset: the OPEN tongs straddle the floating fruit at ~3.8mm/side,
        # and a 4mm offset band (2+2) generated pseudo-contacts that slowly shoved the
        # zero-g fruit 8cm out of the grasp during the settle (verified). 1+1 = 2mm band
        # stays silent at open clearance; contacts form only during the actual close.
        pcx.CreateContactOffsetAttr(0.001)
        pcx.CreateRestOffsetAttr(0.0)
        # torsional patch friction: a rigid SPHERE between two flat pads can spin/roll freely
        # about the contact normals (point contacts carry no twist resistance), so the fruit
        # works loose under carry accelerations even when linear friction holds. A torsional
        # patch radius gives each contact a finite frictional patch (like a real soft tomato
        # flattening against the tong) that resists that spin -- the canonical PhysX fix for
        # sphere-in-parallel-gripper grasps.
        pcx.CreateTorsionalPatchRadiusAttr(0.01)
        pcx.CreateMinTorsionalPatchRadiusAttr(0.003)

        # green calyx (sepal star) on top -- visual only, rides with the rigid body
        cal = UsdGeom.Sphere.Define(st, p + "/calyx")
        cal.CreateRadiusAttr(r * 0.42)
        cal.CreateDisplayColorAttr([(0.20, 0.42, 0.12)])
        cxf = UsdGeom.Xformable(cal.GetPrim())
        cxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, r * 0.92))
        cxf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.35))  # calyx is visual only, no collider

        # UNBREAKABLE stem (1e38), severed once by RemovePrim under SurfaceGripper cover.
        # Breakable thresholds are structurally unusable: the mid-play collider-enable
        # transient snapped 8.5N AND 60N stems outright (verified), and any breakable
        # value also risks snapping during the close. The harvest pipeline: enable
        # collider (safe vs 1e38) -> validated finger pinch -> Isaac SurfaceGripper
        # C++ attach (PxJoint::setActors, zero USD) -> RemovePrim this stem; the removal
        # window is covered by the pinch + the gripper's per-step revalidation.
        # excludeFromArticulation keeps the stem out of any articulation re-cook.
        from pxr import Sdf as _Sdf
        j2 = UsdPhysics.FixedJoint.Define(st, p + "_stem")
        j2.CreateBody1Rel().SetTargets([prim.GetPath()])
        j2.CreateLocalPos0Attr(Gf.Vec3f(f["x"], f["y"], f["z"]))
        j2.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j2.CreateBreakForceAttr(3.4028235e38)
        j2.CreateBreakTorqueAttr(3.4028235e38)
        j2.GetPrim().CreateAttribute("physics:excludeFromArticulation", _Sdf.ValueTypeNames.Bool).Set(True)
        print("  {} at [{:.3f},{:.3f},{:.3f}] r={} rigid stem (RemovePrim under SG cover)".format(
            f["name"], f["x"], f["y"], f["z"], r))
    print("authored", len(fruit), "harvestable tomatoes")

    if frame:
        import carb
        s = carb.settings.get_settings()
        # natural exposure (the splat is dimmed at the source via its NuRec ccm in
        # scene_add_greenhouse, so the white robot arm reads without crushing the scene).
        s.set("/rtx/post/histogram/enabled", False)
        s.set("/rtx/post/tonemap/fNumber", 6.0)
        s.set("/rtx/post/tonemap/filmIso", 100.0)
        s.set("/rtx/post/tonemap/cameraShutter", 70.0)
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        vp.camera_path = "/OmniverseKit_Persp"
        cam = st.GetPrimAtPath("/OmniverseKit_Persp")
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(-0.14, -1.1, 0.95), Gf.Vec3d(-0.14, 0.0, 0.89), Gf.Vec3d(0, 0, 1))
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(view.GetInverse())
        print("framed close on the two fruit")
