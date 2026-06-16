import os
import pathlib

import torch
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner


class AMPOnPolicyRunner(AmpOnPolicyRunner):
    """Zouyu AMP on-policy runner with ONNX export and metadata support."""

    env: RslRlVecEnvWrapper

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """Export the actor network to ONNX using the v5.3 MLPModel API.

        The exported model includes the obs normalizer so that the ONNX
        model expects raw observations directly.
        """
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            save_path,
            export_params=True,
            opset_version=18,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
        )
        # Move policy back to training device
        self.alg.get_policy().to(self.device)

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save model checkpoint and export ONNX policy with metadata."""
        super().save(path, infos)
        policy_path = os.path.dirname(path)
        filename = "policy.onnx"
        self.export_policy_to_onnx(policy_path, filename)

        run_name = (
            wandb.run.name
            if self.logger.writer is not None and self.logger.logger_type == "wandb" and wandb.run
            else "local"
        )
        onnx_path = os.path.join(policy_path, filename)
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)

        if self.logger.writer is not None and self.logger.logger_type in ["wandb"]:
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
