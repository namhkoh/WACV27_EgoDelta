"""Load RBY1-A in Isaac Sim with a fixed airborne sphere target.

Run with:
    D:\\isaac-sim\\python.bat robot\\sim\\rby1_sphere_scene.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "submodules" / "rby1-sdk" / "models" / "rby1a" / "urdf" / "model.urdf"
DEFAULT_HOME = REPO_ROOT / "robot" / "utils" / "homestate.json"
DEFAULT_MCP_EXT_FOLDER = Path("C:/path/to/user/Documents/isaac-sim-mcp")
REVOLUTE_DRIVE_STIFFNESS = 5000.0
REVOLUTE_DRIVE_DAMPING = 250.0
REVOLUTE_DRIVE_MAX_FORCE = 1000.0
PRISMATIC_DRIVE_STIFFNESS = 1000.0
PRISMATIC_DRIVE_DAMPING = 50.0
PRISMATIC_DRIVE_MAX_FORCE = 200.0


def parse_vec3(text: str, *, name: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must be formatted as x,y,z")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numeric values") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an Isaac Sim RBY1-A sphere-target scene.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--home-state", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--solution-json", type=Path, default=None)
    parser.add_argument(
        "--sphere-position-m",
        type=lambda value: parse_vec3(value, name="sphere position"),
        default=(0.58, 0.03, 0.90),
    )
    parser.add_argument("--sphere-radius-m", type=float, default=0.035)
    parser.add_argument(
        "--robot-position-m",
        type=lambda value: parse_vec3(value, name="robot position"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--save-usd", type=Path, default=None)
    parser.add_argument("--collisions", action="store_true",
                        help="convex-decompose visual meshes into colliders so the gripper can physically grasp")
    parser.add_argument("--frames", type=int, default=0, help="0 keeps the GUI open until closed.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable-mcp", action="store_true")
    parser.add_argument("--mcp-ext-folder", type=Path, default=DEFAULT_MCP_EXT_FOLDER)
    parser.add_argument("--play", action="store_true")
    return parser.parse_args()


def launch_app(args: argparse.Namespace) -> SimulationApp:
    extra_args = []
    if args.enable_mcp:
        extra_args = ["--ext-folder", str(args.mcp_ext_folder), "--enable", "isaac.sim.mcp_extension"]
    return SimulationApp({"renderer": "RaytracedLighting", "headless": bool(args.headless), "extra_args": extra_args})


def validate_paths(args: argparse.Namespace) -> None:
    if not args.urdf.exists():
        raise FileNotFoundError(f"URDF not found: {args.urdf}")
    if not args.home_state.exists():
        raise FileNotFoundError(f"home state not found: {args.home_state}")
    if args.solution_json is not None and not args.solution_json.exists():
        raise FileNotFoundError(f"IK solution not found: {args.solution_json}")
    if args.enable_mcp and not args.mcp_ext_folder.exists():
        raise FileNotFoundError(f"MCP extension folder not found: {args.mcp_ext_folder}")


def load_home_positions(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    names = data["joint_names"]
    values = data["all_joints"]["positions_rad"]
    return {name: float(value) for name, value in zip(names, values)}


def load_solution_positions(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("ok", False):
        raise ValueError(f"IK solution is not marked ok: {path}")
    names = data["joint_names"]
    values = data["q_arm_rad"]
    if len(names) != len(values):
        raise ValueError(f"IK solution joint_names/q_arm_rad length mismatch: {path}")
    non_arm_names = [name for name in names if not name.startswith(("left_arm_", "right_arm_"))]
    if non_arm_names:
        raise ValueError(f"IK solution may only contain arm joints, found: {non_arm_names}")
    return {name: float(value) for name, value in zip(names, values)}


def create_material(stage, path: str, color: tuple[float, float, float]):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_stage():
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
    physx.CreateEnableCCDAttr(True)
    physx.CreateEnableStabilizationAttr(True)
    physx.CreateSolverTypeAttr("TGS")
    PhysicsSchemaTools.addGroundPlane(
        stage,
        "/World/GroundPlane",
        "Z",
        20.0,
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.45, 0.45, 0.45),
    )

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(650.0)
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    distant.CreateIntensityAttr(450.0)
    distant.CreateAngleAttr(0.35)
    return stage


def import_rby1_urdf(urdf_path: Path, collisions: bool = False) -> str:
    import omni.kit.commands

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    import_config.merge_fixed_joints = False
    # convex-decompose the visual meshes into convex colliders so the (dynamic) articulation
    # links -- including the gripper fingers -- can physically contact/grasp objects. The v1.1
    # URDF's <capsule> <collision> tags don't import, so derive colliders from the visuals.
    import_config.convex_decomp = bool(collisions)
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.distance_scale = 1.0
    import_config.collision_from_visuals = bool(collisions)
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status:
        raise RuntimeError(f"URDF import failed: {urdf_path}")
    return str(prim_path)


def configure_joint_drives(stage, robot_prim_path: str) -> None:
    joint_root = f"{robot_prim_path}/joints/"
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(joint_root):
            continue
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            drive = "drive:angular:physics"
            values = {
                f"{drive}:stiffness": REVOLUTE_DRIVE_STIFFNESS,
                f"{drive}:damping": REVOLUTE_DRIVE_DAMPING,
                f"{drive}:maxForce": REVOLUTE_DRIVE_MAX_FORCE,
            }
        elif prim.GetTypeName() == "PhysicsPrismaticJoint":
            drive = "drive:linear:physics"
            values = {
                f"{drive}:stiffness": PRISMATIC_DRIVE_STIFFNESS,
                f"{drive}:damping": PRISMATIC_DRIVE_DAMPING,
                f"{drive}:maxForce": PRISMATIC_DRIVE_MAX_FORCE,
            }
        else:
            continue
        for attr_name, value in values.items():
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                attr.Set(float(value))


def pin_articulation_base(stage, base_link_path: str) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    prim = stage.GetPrimAtPath(base_link_path)
    if not prim.IsValid():
        raise RuntimeError(f"Imported base link is invalid: {base_link_path}")
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    values = {
        "physics:kinematicEnabled": (Sdf.ValueTypeNames.Bool, True),
        "physxRigidBody:disableGravity": (Sdf.ValueTypeNames.Bool, True),
        "physxRigidBody:lockedPosAxis": (Sdf.ValueTypeNames.Int, 7),
        "physxRigidBody:lockedRotAxis": (Sdf.ValueTypeNames.Int, 7),
        "physics:velocity": (Sdf.ValueTypeNames.Vector3f, Gf.Vec3f(0.0, 0.0, 0.0)),
        "physics:angularVelocity": (Sdf.ValueTypeNames.Vector3f, Gf.Vec3f(0.0, 0.0, 0.0)),
    }
    for name, (type_name, value) in values.items():
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, type_name)
        attr.Set(value)


def set_robot_position(stage, prim_path: str, position: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Imported robot prim is invalid: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))


def add_sphere_target(stage, position: tuple[float, float, float], radius: float) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    sphere = UsdGeom.Sphere.Define(stage, "/World/TargetSphere")
    sphere.CreateRadiusAttr(float(radius))
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    material = create_material(stage, "/World/Looks/TargetRed", (0.95, 0.05, 0.03))
    UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)


def add_camera(stage) -> None:
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(22.0)
    camera.AddTranslateOp().Set(Gf.Vec3d(1.8, -2.1, 1.25))
    camera.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 41.0))
    stage.GetRootLayer().defaultPrim = "World"
    stage.GetPrimAtPath("/World/Camera").CreateAttribute("omni:kit:viewport:active", Sdf.ValueTypeNames.Bool).Set(True)


def apply_joint_state(
    prim_path: str,
    home_positions: dict[str, float],
    solution_positions: dict[str, float],
) -> dict[str, list[str]]:
    import numpy as np
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation

    simulation = SimulationContext(stage_units_in_meters=1.0)
    simulation.initialize_physics()
    robot = SingleArticulation(prim_path=prim_path, name="rby1a")
    robot.initialize()
    dof_names = list(robot.dof_names)
    positions = np.array(robot.get_joint_positions(), dtype=np.float64)
    home_matched = []
    solution_matched = []
    for index, name in enumerate(dof_names):
        if name in home_positions:
            positions[index] = home_positions[name]
            home_matched.append(name)
    for index, name in enumerate(dof_names):
        if name in solution_positions:
            positions[index] = solution_positions[name]
            solution_matched.append(name)
    missing_solution_joints = sorted(set(solution_positions) - set(solution_matched))
    if missing_solution_joints:
        raise ValueError(f"IK solution joints not found in Isaac articulation: {missing_solution_joints}")
    robot.set_joint_positions(positions)
    robot.set_joint_velocities(np.zeros_like(positions))
    return {"home": home_matched, "solution": solution_matched}


def apply_collider_api(stage) -> int:
    """The URDF importer creates the convex-decomposed collider geometry under /colliders but
    does NOT mark it as a physical collider, so PhysX ignores it. Apply CollisionAPI +
    MeshCollisionAPI(convexHull) to every collider mesh so the links physically collide."""
    from pxr import UsdPhysics
    n = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh" and str(prim.GetPath()).startswith("/colliders"):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
            n += 1
    print(f"applied CollisionAPI(convexHull) to {n} collider meshes")
    return n


def save_stage(path: Path) -> None:
    import omni.usd

    path.parent.mkdir(parents=True, exist_ok=True)
    omni.usd.get_context().save_as_stage(str(path))


def run_frames(app: SimulationApp, frames: int) -> None:
    if frames > 0:
        for _ in range(frames):
            app.update()
        return
    while app.is_running():
        app.update()


def main() -> int:
    args = parse_args()
    validate_paths(args)
    app = launch_app(args)

    import omni.timeline

    stage = create_stage()
    prim_path = import_rby1_urdf(args.urdf, collisions=args.collisions)
    if args.collisions:
        apply_collider_api(stage)
    configure_joint_drives(stage, prim_path.rsplit("/", 1)[0])
    pin_articulation_base(stage, prim_path)
    set_robot_position(stage, prim_path, args.robot_position_m)
    add_sphere_target(stage, args.sphere_position_m, args.sphere_radius_m)
    add_camera(stage)
    applied_joints = apply_joint_state(
        prim_path,
        load_home_positions(args.home_state),
        load_solution_positions(args.solution_json),
    )
    if args.save_usd is not None:
        save_stage(args.save_usd)
    if args.play:
        omni.timeline.get_timeline_interface().play()
    print(
        json.dumps(
            {
                "robot_prim_path": prim_path,
                "urdf": str(args.urdf),
                "sphere_position_m": args.sphere_position_m,
                "sphere_radius_m": args.sphere_radius_m,
                "home_joints_applied": applied_joints["home"],
                "solution_joints_applied": applied_joints["solution"],
            },
            indent=2,
        ),
        flush=True,
    )
    run_frames(app, args.frames)
    if args.play:
        omni.timeline.get_timeline_interface().stop()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
