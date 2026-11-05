"""Load the scene only and expose ROS2 topics. No task or skill runs; physics waits for commands."""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGS = {
    "project_root": (os.environ.get("MOBILE_PANDA_ROOT", os.getcwd()), str, "repository root (contains mobile_panda_sim/)"),
    "backend": ("maniskill", str, "simulator: maniskill or mujoco"),
    "sim_python": (os.environ.get("MOBILE_PANDA_SIM_PYTHON", ""), str,
                   "interpreter with the backend's packages; empty = find a Conda environment that has them"),
    "scene": ("kitchen", str, "kitchen (home: kitchen + dining + living room) or tabletop (skill-test table)"),
    "base": ("wheels", str, "wheels (physical differential drive) or planar (virtual x/y/yaw joints)"),
    "seed": ("0", int, "episode seed (red cube yaw)"),
    "viewer": ("true", bool, "open the simulator's viewer window"),
    "start_x": ("nan", float, "initial chassis x in map (m); nan = scene default"),
    "start_y": ("nan", float, "initial chassis y in map (m); nan = scene default"),
    "start_yaw": ("nan", float, "initial chassis yaw (rad); nan = scene default"),
}


def nodes(context):
    base = LaunchConfiguration("base").perform(context)
    if base not in ("wheels", "planar"):
        raise RuntimeError("base must be wheels or planar")
    if LaunchConfiguration("scene").perform(context) not in ("kitchen", "tabletop"):
        raise RuntimeError("scene must be kitchen or tabletop")
    urdf = Path(get_package_share_directory("mobile_panda_description")) / f"urdf/mobile_panda_{base}.urdf"
    rviz_config = Path(get_package_share_directory("mobile_panda_bridge")) / "rviz/mobile_panda.rviz"
    parameters = {name: ParameterValue(LaunchConfiguration(name), value_type=kind) for name, (_, kind, _) in ARGS.items()}
    return [
        Node(package="mobile_panda_bridge", executable="bridge", name="mobile_panda_bridge",
             parameters=[parameters], output="screen"),
        Node(package="robot_state_publisher", executable="robot_state_publisher", name="robot_state_publisher",
             parameters=[{"robot_description": urdf.read_text()}],
             remappings=[("joint_states", "/mobile_panda/joint_states")], output="screen"),
        Node(package="rviz2", executable="rviz2", name="rviz2", arguments=["-d", str(rviz_config)],
             condition=IfCondition(LaunchConfiguration("rviz")), output="screen"),
    ]


def generate_launch_description():
    declarations = [DeclareLaunchArgument(name, default_value=default, description=text)
                    for name, (default, _, text) in ARGS.items()]
    declarations.append(DeclareLaunchArgument("rviz", default_value="false",
                                              description="open RViz with robot model, TF and camera images"))
    return LaunchDescription(declarations + [OpaqueFunction(function=nodes)])
