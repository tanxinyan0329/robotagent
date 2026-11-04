"""Scene loads, cameras sit on their URDF frames, base and gripper respond."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import gymnasium as gym
    import torch
    import mani_skill  # noqa: F401
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False

UIDS = ["mobile_panda_wheels", "mobile_panda_planar"]


def test_scene():
    if not HAVE_SIM:
        import pytest
        pytest.skip("ManiSkill not installed in this Python; run python tests/test_scene.py in the Conda env")
    for uid in UIDS:
        check_scene(uid)


def check_scene(uid):
    import mobile_panda_sim.scenes  # noqa: F401
    from mani_skill.utils import sapien_utils
    from mobile_panda_sim.geometry import BASE_CAMERA_EYE, BASE_CAMERA_TARGET

    env = gym.make("MobilePandaTabletop-v1", robot_uids=uid, obs_mode="none", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cpu", reward_mode="none", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        u = env.unwrapped
        assert u.single_action_space.shape == (10,)
        links = u.agent.robot.links_map
        rel = links["panda_link0"].pose.inv() * links["base_camera_link"].pose
        ref = sapien_utils.look_at(BASE_CAMERA_EYE, BASE_CAMERA_TARGET)
        assert torch.allclose(rel.p, torch.tensor([BASE_CAMERA_EYE]), atol=1e-4)
        assert torch.allclose(rel.q.abs(), ref.q.abs(), atol=1e-4)
        images = u.capture_rgb()
        assert images["base_camera"].shape == (480, 640, 3) and images["hand_camera"].shape == (240, 320, 3)
        x0 = float(links["chassis_link"].pose.p[0, 0])
        for _ in range(40):
            action = u.agent.controller.from_action_dict(dict(
                arm=torch.zeros((1, 7)), gripper=torch.tensor([[-1.0]]),
                base=torch.tensor([u.agent.base_action(0.15, 0.0)], dtype=torch.float32)))
            env.step(action)
        assert float(links["chassis_link"].pose.p[0, 0]) - x0 > 0.2
        names = [j.name for j in u.agent.robot.get_active_joints()]
        assert float(u.agent.robot.get_qpos()[0, names.index("panda_finger_joint1")]) < 0.005
    finally:
        env.close()


if __name__ == "__main__":
    for uid in UIDS:
        check_scene(uid)
        print(f"{uid}: ok")
