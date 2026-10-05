import json
import numpy as np
from utils.experiment import ExperimentTracker

class BayesianTracker(ExperimentTracker):
    history_fields=ExperimentTracker.history_fields+[
        'train_objective','validation_plugin_nll','raw_kl','kl_per_observation',
        'kl_normalizer','effective_k_global_mass','effective_k_mean_conditional_mass',
        'mean_conditional_k_99','posterior_draws','posterior_seed']

    def inference_options(self):
        return {'mode':'posterior_predictive',
            'draws':int(self.params['posterior_draws']),
            'seed':int(self.params['posterior_seed'])}

    def checkpoint_payload(self,epoch,model,optimizer,scheduler,history):
        payload=super().checkpoint_payload(epoch,model,optimizer,scheduler,history)
        payload['architecture']='beta_stick_conditional_mdn_v2'
        payload['bayesian_weights']=model.weight_report()
        payload['evaluation_inference']=self.inference_options()
        return payload

    def capture_fixed_predictions(self,model,epoch,label=None):
        with model.inference(**self.inference_options()):
            super().capture_fixed_predictions(model,epoch,label)
            bank=model._posterior_bank.detach().cpu().numpy()
        path=self.prediction_dir/f'{label or f"epoch_{epoch:04d}"}.npz'
        with np.load(path) as d:payload={k:d[k] for k in d.files}
        payload.update(mean_global_weights=np.asarray(model.weight_report()['mean_global_weights']),
            posterior_weight_bank=bank,inference_mode='posterior_predictive_qmc',
            posterior_seed=np.int64(self.params['posterior_seed']),
            posterior_draws=np.int64(self.params['posterior_draws']))
        np.savez_compressed(path,**payload)

    def log_weights(self,epoch,model):
        with (self.run_dir/'weight_history.jsonl').open('a') as f:
            f.write(json.dumps({'epoch':epoch,**model.weight_report()})+'\n')

    def save_assessment(self,epoch,assessment):
        path=self.run_dir/'usage';path.mkdir(exist_ok=True)
        (path/f'epoch_{epoch:04d}.json').write_text(json.dumps({'epoch':epoch,'split':'full_validation',**assessment},indent=2)+'\n')
        with (self.run_dir/'usage_history.jsonl').open('a') as f:
            f.write(json.dumps({'epoch':epoch,'split':'full_validation',**assessment})+'\n')
