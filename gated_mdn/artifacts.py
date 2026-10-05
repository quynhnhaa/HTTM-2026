"""Gate state and unpenalized NLL stay explicit in all experiment artifacts."""
import json
import numpy as np
from utils.experiment import ExperimentTracker

class GatedTracker(ExperimentTracker):
    history_fields=ExperimentTracker.history_fields+['train_objective','validation_objective','l0_penalty',
        'lambda_l0','expected_active_components','deterministic_active_components']

    def capture_fixed_predictions(self,model,epoch,label=None):
        super().capture_fixed_predictions(model,epoch,label)
        path=self.prediction_dir/f'{label or f"epoch_{epoch:04d}"}.npz'
        with np.load(path) as d:payload={k:d[k] for k in d.files}
        gates=model.gate_report()
        payload.update(deterministic_gates=np.asarray(gates['deterministic_gates']),
                       gate_open_probabilities=np.asarray(gates['open_probabilities']),
                       active_indices=np.asarray(gates['active_indices'],dtype=np.int64),
                       expected_active_components=np.asarray(gates['expected_active_components']))
        np.savez_compressed(path,**payload)

    def checkpoint_payload(self,epoch,model,optimizer,scheduler,history):
        payload=super().checkpoint_payload(epoch,model,optimizer,scheduler,history)
        payload['architecture']='global_hard_concrete_mdn_v1';payload['gates']=model.gate_report()
        return payload

    def log_gates(self,epoch,model):
        with (self.run_dir/'gate_history.jsonl').open('a') as f:
            f.write(json.dumps({'epoch':epoch,**model.gate_report()})+'\n')
