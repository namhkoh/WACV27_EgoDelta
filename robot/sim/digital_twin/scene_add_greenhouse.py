"""Add the leveled NuRec greenhouse splat into the CURRENT stage (e.g. the opened
robot_home.usd), applying the orient.json leveling and cropping to the dense core.

Run via :8226:  isaac_client.py --bootstrap .../scene_add_greenhouse.py "add_greenhouse()"
"""
import json

# no-tomato splat: the middle-gutter (T4) fruit are physics bodies, not splat blobs.
# notom2 = clean-pocket removal (kills the highlight/SH ghost the strict-red pass left).
USDZ = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/greenhouse_notom2.usdz"
# NuRec crop is an AABB in the gauss LOCAL frame; local +Y maps to world -Z, so local -Y is
# world UP. minBounds.y is therefore the world CEILING: at -0.5 it clipped world-z ~1.28m and
# cut ~19% of the splat off the top (the upper plant wall + structure). -1.47 raises the
# ceiling to world-z ~2.2m to show the full top while still excluding the sparse sky floaters
# above ~2.3m. maxBounds.y (set below) is the world FLOOR limit.
CROP_MIN_Y = -1.47
ORIENT = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/orient.json"
# the raw NuRec capture is over-bright (clipped near-white), which washes out the white robot
# arm rendered in front of it (the arm then looks like disconnected floating parts). Scale the
# splat's emissive RGB down at the source via its color-correction matrix so the background
# reads as a normal greenhouse and the robot stands out -- without darkening the robot/floor.
SPLAT_CCM_SCALE = 0.45


def add_greenhouse(prim_path="/World/greenhouse", crop=(2.5, 1.3, 2.5), frame=True):
    import omni.kit.app
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    # ensure the NuRec schema/render ext is on
    mgr = omni.kit.app.get_app().get_extension_manager()
    for ext in ("omni.usd.schema.omni_nurec_types",):
        if not mgr.is_extension_enabled(ext):
            try:
                mgr.set_extension_enabled_immediate(ext, True)
            except Exception:
                pass

    st = omni.usd.get_context().get_stage()
    if st.GetPrimAtPath(prim_path).IsValid():
        st.RemovePrim(prim_path)
    prim = st.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(USDZ)

    o = json.load(open(ORIENT))
    R, t = o["R"], o["t"]
    M = Gf.Matrix4d(
        R[0][0], R[1][0], R[2][0], 0.0,
        R[0][1], R[1][1], R[2][1], 0.0,
        R[0][2], R[1][2], R[2][2], 0.0,
        t[0], t[1], t[2], 1.0,
    )
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(M)

    g = st.GetPrimAtPath(prim_path + "/gauss/gauss")
    if g.IsValid():
        cx, cy, cz = crop
        # asymmetric min-y (CROP_MIN_Y) drops the messy foreground-floor splat in front of
        # the plant wall (where the robot + basket sit) -> clean ground, no clutter/floaters.
        g.GetAttribute("omni:nurec:crop:minBounds").Set(Gf.Vec3f(-cx, CROP_MIN_Y, -cz))
        g.GetAttribute("omni:nurec:crop:maxBounds").Set(Gf.Vec3f(cx, cy, cz))

    # dim the over-bright splat at the source (emissive color-correction matrix diagonal)
    ef = st.GetPrimAtPath(prim_path + "/gauss/gauss/emissive_color_field")
    if ef.IsValid():
        sc = SPLAT_CCM_SCALE
        for k, vv in (("omni:nurec:ccmR", (sc, 0, 0, 0)), ("omni:nurec:ccmG", (0, sc, 0, 0)),
                      ("omni:nurec:ccmB", (0, 0, sc, 0))):
            a = ef.GetAttribute(k)
            if a and a.IsValid():
                cur = a.Get()
                a.Set(type(cur)(*vv) if cur is not None else Gf.Vec4f(*vv))
        print("splat emissive dimmed to", sc, "via ccm")
    print("greenhouse referenced + leveled + cropped at", prim_path)

    if frame:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        cam = st.GetPrimAtPath(vp.camera_path)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(3.6, -3.6, 2.3), Gf.Vec3d(0.0, 0.0, 0.85), Gf.Vec3d(0, 0, 1))
        xfc = UsdGeom.Xformable(cam)
        xfc.ClearXformOpOrder()
        xfc.AddTransformOp().Set(view.GetInverse())
        print("framed to show robot + greenhouse")
