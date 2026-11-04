from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "mobile_panda_description"
DESCRIPTION = REPO / "ros2_ws/src" / PACKAGE
CHECKPOINTS = REPO / "skills/checkpoints"
OUTPUTS = REPO / "outputs"


def ros_urdf(mode):
    return DESCRIPTION / f"urdf/mobile_panda_{mode}.urdf"


def simulator_urdf(mode):
    """The ROS URDF with package:// URIs resolved to files, for SAPIEN's loader."""
    source = ros_urdf(mode)
    if not source.is_file():
        raise FileNotFoundError(f"{source} missing; run python scripts/build_description.py")
    text = source.read_text(encoding="utf-8").replace(f"package://{PACKAGE}/", DESCRIPTION.as_posix() + "/")
    target = OUTPUTS / "urdf" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return target
