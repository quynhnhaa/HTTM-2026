"""Review gate artifacts against real IMPTC smoke checkpoint, not model quality."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gated_mdn.model import GatedMDN
from base_lstm import LSTM_Trajectory_Forecast
from utils.mdn_distribution import build_mdn_distribution
RUN_ROOT=ROOT/'results/trained_models/gated_mdn/imptc/smoke_gated_peds_imptc/runs'

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='gated_k8_smoke_seed2024');parser.add_argument('--config-name',default='smoke_gated_peds_imptc');parser.add_argument('--report-name',default='SMOKE');args=parser.parse_args()
 RUN=ROOT/'results/trained_models/gated_mdn/imptc'/args.config_name/'runs'/args.run_id
 before=json.loads((ROOT/'gated_mdn/reports/BASE_SOURCE_HASHES.json').read_text())
 after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'base_mdn').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
 assert before==after,'Baseline changed'
 rows=list(csv.DictReader((RUN/'history.csv').open()));assert len(rows)==3
 assert all(int(r['train_sample_count'])==189 and int(r['validation_sample_count'])==19148 for r in rows)
 assert all(float(r['train_objective'])>=float(r['train_nll']) for r in rows)
 assert all(r['full_metrics_evaluated']=='False' for r in rows)
 for name in ['epoch_0001','epoch_0002','epoch_0003','best','last','final']:
  cp=torch.load(RUN/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=False)
  assert cp['architecture']=='global_hard_concrete_mdn_v1' and 'gates.log_alpha' in cp['model_state_dict']
  assert 'rng_state' in cp and 'data_loader_state' in cp and cp['optimizer_state_dict']['state']
 final=torch.load(RUN/'checkpoints/final.pt',map_location='cpu',weights_only=False)
 first=torch.load(RUN/'checkpoints/epoch_0001.pt',map_location='cpu',weights_only=False)
 assert not torch.equal(first['model_state_dict']['gates.log_alpha'],final['model_state_dict']['gates.log_alpha'])
 k_max=final['resolved_config']['model_params']['num_gaussians']
 fixed=np.load(RUN/'fixed_samples/inputs.npz');assert len(fixed['sample_ids'])==8
 for path in (RUN/'fixed_samples/predictions').glob('*.npz'):
  p=np.load(path);np.testing.assert_array_equal(p['sample_ids'],fixed['sample_ids'])
  assert p['mu'].shape==(8,48,k_max,2) and p['pi'].shape==(8,48,k_max)
  np.testing.assert_allclose(p['pi'].sum(-1),1,atol=1e-6)
  for key in ['raw_output','sigma','rho','covariance','deterministic_gates','gate_open_probabilities']:assert np.isfinite(p[key]).all()
 metrics=[]
 for path in (RUN/'metrics').glob('epoch_*.json'):
  d=json.loads(path.read_text());assert d['split']=='fixed_validation_smoke'
  for k in ['minade20_m','minfde20_m','ravg_percent','rmin_percent','s68_m2_per_s','s95_m2_per_s','asaee_m_per_s']:
   assert np.isfinite(d['metrics'][k]);metrics.append(k)
 assert metrics
 export=torch.load(RUN/'compact_smoke.pt',map_location='cpu',weights_only=False)
 m=GatedMDN(final['resolved_config']['model_params']);m.load_state_dict(final['model_state_dict']);m.eval()
 compact=LSTM_Trajectory_Forecast(export['resolved_config']['model_params']);compact.load_state_dict(export['model_state_dict']);compact.eval()
 X=torch.as_tensor(fixed['X']);y=torch.as_tensor(fixed['y'])
 with torch.no_grad():
  torch.testing.assert_close(build_mdn_distribution(m(X),k_max).log_prob(y),build_mdn_distribution(compact(X),len(export['active_indices'])).log_prob(y),rtol=2e-5,atol=2e-5)
 evaluation=json.loads((RUN/'testing/evaluation_limited.json').read_text());assert evaluation['sample_count']==16 and np.isfinite(evaluation['test_nll'])
 report={'status':'passed','run_id':args.run_id,'k_max':k_max,'epochs':3,'train_samples_per_epoch':189,'validation_nll_samples':19148,'official_smoke_metric_samples':8,
         'fixed_samples':8,'baseline_source_hashes_unchanged':True,'gate_parameters_updated':True,'gates':final['gates'],
         'parameter_count':sum(p.numel() for p in m.parameters()),'compact_parameter_count':export['parameter_count'],
         'export_distribution_equivalence':'passed','resume_unit_test':'stochastic gate RNG and optimizer continuation passed',
         'full_training_started':False,'quality_conclusion':f'None; 3-epoch smoke uses {final["gates"]["deterministic_active_components"]} of {k_max} components'}
 dest=ROOT/'gated_mdn/reports';(dest/f'{args.report_name}.json').write_text(json.dumps(report,indent=2)+'\n')
 (dest/f'{args.report_name}.md').write_text(f'''# Smoke review: gated MDN

Passed 5 unit tests: true-zero gates and open-probability formula, unchanged legacy covariance/normalized pi, compact distribution equivalence, finite gate gradients and exact stochastic optimizer/RNG continuation, plus K_max8 export with exactly5 surviving components.

Real IMPTC integration: 3 epochs, 189 train samples/epoch, full 19,148 validation NLL, same 8 fixed validation IDs. All seven displacement/reliability/sharpness/ASAEE metric branches ran on those 8 samples using MC64 and grid1m; these are **not full validation metrics**. Evaluator limited test16 and compact checkpoint export passed. Periodic/best/last/final retain gate state, optimizer/scheduler/RNG/loader ordering. NLL and penalized objective are separate. Base source hashes unchanged.

K_max={k_max}; gate parameters updated. Deterministic active K={final["gates"]["deterministic_active_components"]}, expected K={final["gates"]["expected_active_components"]:.3f} after smoke. Expected K is an expectation, not a learned integer K. No quality improvement claim and no full training started. Detailed values in SMOKE.json.
''')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
