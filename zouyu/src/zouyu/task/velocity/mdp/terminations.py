import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor


def illegal_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
    return torch.any(data.found, dim=-1)
