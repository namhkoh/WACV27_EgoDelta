"""Add a round, tapered basket (open top) at a drop-off point.

Tapered metal-pail shape: 40cm top diameter, 30cm bottom diameter, 40cm tall, grey. Built
as a ring of thin wall segments tilted to the taper (visual + static collision) plus a thin
bottom disk, so a harvested fruit dropped above it falls IN and is retained.

Run via :8226:  isaac_client.py --bootstrap .../scene_add_basket.py "add_basket(-0.82,-0.97)"
"""


def add_basket(cx, cy, *, radius_top=0.20, radius_bottom=0.15, height=0.40, segments=28,
               color=(0.55, 0.55, 0.58)):
    import math
    import omni.usd
    from pxr import UsdGeom, UsdPhysics, Gf

    st = omni.usd.get_context().get_stage()
    for path in ("/World/basket", "/World/basket_collision", "/World/bucket_splat"):
        if st.GetPrimAtPath(path).IsValid():
            st.RemovePrim(path)
    UsdGeom.Xform.Define(st, "/World/basket")

    # thin bottom disk at the bottom radius
    fl = UsdGeom.Cylinder.Define(st, "/World/basket/floor")
    fl.CreateRadiusAttr(float(radius_bottom * 0.97)); fl.CreateHeightAttr(0.012); fl.CreateAxisAttr("Z")
    fl.CreateDisplayColorAttr([color])
    fx = UsdGeom.Xformable(fl.GetPrim()); fx.ClearXformOpOrder()
    fx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.008))

    # tapered wall: ring of thin boxes leaning out at the top by the taper angle
    phi = math.atan2(radius_top - radius_bottom, height)
    cf, sf = math.cos(phi), math.sin(phi)
    mid_r = 0.5 * (radius_top + radius_bottom)
    chord = 2.0 * mid_r * math.sin(math.pi / segments)
    sx, sy, sz = 0.012, float(chord * 1.06), float(height / cf)
    for i in range(int(segments)):
        th = 2.0 * math.pi * i / segments
        ct, stt = math.cos(th), math.sin(th)
        z_ax = Gf.Vec3d(sf * ct, sf * stt, cf)          # box height axis, tilted outward at top
        y_ax = Gf.Vec3d(-stt, ct, 0.0)                  # tangential
        x_ax = Gf.Cross(y_ax, z_ax).GetNormalized()     # radial (thin)
        px, py, pz = cx + mid_r * ct, cy + mid_r * stt, 0.5 * height
        M = Gf.Matrix4d(
            x_ax[0] * sx, x_ax[1] * sx, x_ax[2] * sx, 0.0,
            y_ax[0] * sy, y_ax[1] * sy, y_ax[2] * sy, 0.0,
            z_ax[0] * sz, z_ax[1] * sz, z_ax[2] * sz, 0.0,
            px, py, pz, 1.0,
        )
        c = UsdGeom.Cube.Define(st, "/World/basket/wall%d" % i)
        c.CreateSizeAttr(1.0); c.CreateDisplayColorAttr([color])
        cxf = UsdGeom.Xformable(c.GetPrim()); cxf.ClearXformOpOrder(); cxf.AddTransformOp().Set(M)
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    print("basket top_d=%.2f bottom_d=%.2f h=%.2f at [%.3f, %.3f]" % (
        2 * radius_top, 2 * radius_bottom, height, cx, cy))
