"""Add CV/residual decomposition to shared fixed-position captures."""
import numpy as np
import torch

from utils.experiment import ExperimentTracker


class ResidualExperimentTracker(ExperimentTracker):
    def capture_fixed_predictions(self, model, epoch, label=None):
        super().capture_fixed_predictions(model, epoch, label)
        x = torch.as_tensor(self.data_loader.eval_data[0][self.fixed_indices],
                            dtype=torch.float32, device=self.device)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                residual = model.residual_output(x)
                cv = model.cv_trajectory(x)
                k = self.cfg.model_params['num_gaussians']
                residual_mu = torch.stack((residual[..., :k], residual[..., k:2*k]), dim=-1)
            path = self.prediction_dir / f'{label or f"epoch_{epoch:04d}"}.npz'
            with np.load(path) as original:
                payload = {key: original[key] for key in original.files}
            payload.update(cv_trajectory=cv.cpu().numpy(),
                           residual_mu=residual_mu.cpu().numpy(),
                           raw_residual_output=residual.cpu().numpy())
            np.savez_compressed(path, **payload)
        finally:
            model.train(was_training)
