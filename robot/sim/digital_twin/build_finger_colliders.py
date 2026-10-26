"""Inject primitive box colliders into the RB-Y1 gripper finger links.

The URDF import produced visual-only finger links (the <capsule> collision tags
do not survive import), so the gripper passes through everything. Re-importing
with convex_decomp=True does add colliders but corrupts the visual mesh
rendering. This is the NON-DESTRUCTIVE path: open the known-good robot USD as a
standalone stage, drop one analytic Cube collider into each finger link's
pre-existing (empty) ``collisions`` Xform sized to the visual finger pad, then
export to a new file. The original USD is never modified; the visuals are never
touched.

Run inside Isaac Sim (pxr is available there); it operates on a separate stage
object so the live viewport is undisturbed.
"""
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

SRC = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/robot_home_v11.usd"
DST = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/robot_home_v11_fcoll.usd"

# finger-pad box in the finger-link local frame (mesh extent / identity rel-xform)
PAD_CENTER = Gf.Vec3d(0.005, 0.0, -0.0295)
PAD_HALF = Gf.Vec3f(0.008, 0.016, 0.031)
FINGERS = ["ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2"]

stage = Usd.Stage.Open(SRC)
made = 0
for fn in FINGERS:
    coll_scope = "/RBY1_A_v1_1/%s/collisions" % fn
    sc = stage.GetPrimAtPath(coll_scope)
    if not sc.IsValid():
        # create the collisions Xform if the importer didn't
        sc = UsdGeom.Xform.Define(stage, coll_scope).GetPrim()
    box_path = coll_scope + "/pad_box"
    cube = UsdGeom.Cube.Define(stage, box_path)
    cube.GetSizeAttr().Set(2.0)  # edge length 2 -> half-extent 1, scaled below
    x = UsdGeom.Xformable(cube)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(PAD_CENTER)
    x.AddScaleOp().Set(PAD_HALF)
    prim = cube.GetPrim()
    # make it physics-only: collider on, do not render the box
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")
    # tight contact so a gentle close just touches the fruit (anti-fling)
    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    pc.CreateContactOffsetAttr().Set(0.003)
    pc.CreateRestOffsetAttr().Set(0.0)
    made += 1
    print("collider:", box_path)

print("created", made, "finger colliders")
stage.GetRootLayer().Export(DST)
print("exported ->", DST)
