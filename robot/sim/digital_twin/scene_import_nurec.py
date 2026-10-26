"""Reference a NuRec greenhouse .usdz into the live Isaac Sim stage and frame it.

Run inside Isaac Sim via the :8226 socket:
    isaac_client.py --bootstrap .../scene_import_nurec.py "import_nurec()"
"""
ASSET = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/greenhouse.usdz"
PRIM = "/World/greenhouse"


def import_nurec(asset=ASSET, prim_path=PRIM, frame=True):
    import omni.usd
    import omni.kit.app
    from pxr import Usd, UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()

    # enable any NuRec rendering extension that exists
    mgr = omni.kit.app.get_app().get_extension_manager()
    names = [e.get("name", "") for e in mgr.get_extensions()]
    nurec = [n for n in names if "nurec" in n.lower()]
    print("nurec extensions:", nurec)
    for n in nurec:
        if not mgr.is_extension_enabled(n):
            try:
                mgr.set_extension_enabled_immediate(n, True)
                print("enabled", n)
            except Exception as e:
                print("enable failed", n, repr(e))

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(asset)
        print("referenced", asset, "->", prim_path)
    else:
        print("prim already present:", prim_path)

    vols = [p.GetPath().pathString for p in stage.Traverse() if p.GetTypeName() == "Volume"]
    print("Volume prims in stage:", vols)

    if frame:
        up_axis = UsdGeom.GetStageUpAxis(stage)
        bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = bbc.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        center = (mn + mx) * 0.5
        diag = (mx - mn).GetLength()
        up = Gf.Vec3d(0, 1, 0) if up_axis == "Y" else Gf.Vec3d(0, 0, 1)
        eye = center + Gf.Vec3d(0.8 * diag, -0.8 * diag, 0.5 * diag)
        print("stage up:", up_axis, "| bbox center:", [round(c, 2) for c in center], "diag:", round(diag, 2))

        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        cam = stage.GetPrimAtPath(vp.camera_path)
        view = Gf.Matrix4d().SetLookAt(eye, center, up)
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(view.GetInverse())
        print("camera framed; eye:", [round(e, 2) for e in eye])

    return prim_path
