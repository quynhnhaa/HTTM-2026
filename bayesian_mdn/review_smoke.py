"""Read-only review of the predictive Bayesian smoke artifacts."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from bayesian_mdn.model import BayesianMDN

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='bayesian_k8_predictive_qmc_smoke_seed2024')
    parser.add_argument('--config-name',default='smoke_bayesian_peds_imptc')
    parser.add_argument('--report-name',default='PREDICTIVE_SMOKE')
    args=parser.parse_args()
    r=ROOT/'results/trained_models/bayesian_mdn/imptc'/args.config_name/'runs'/args.run_id
    final=torch.load(r/'checkpoints/final.pt',map_location='cpu',weights_only=False)
    assert final['architecture']=='beta_stick_conditional_mdn_v2'
    for name in ['initial','best','last','final','epoch_0001','epoch_0002','epoch_0003']:
        assert (r/f'checkpoints/{name}.pt').exists()
    for key in ['optimizer_state_dict','scheduler_state_dict','rng_state','data_loader_state']:
        assert key in final
    k=final['resolved_config']['model_params']['num_gaussians']
    files=list((r/'fixed_samples/predictions').glob('*.npz'));assert len(files)==6
    with np.load(r/'fixed_samples/inputs.npz') as d:
        inputs=d['X'].copy();sample_ids=d['sample_ids'].copy()
    for f in files:
        with np.load(f) as d:
            for key in ['raw_output','pi','mu','sigma','rho','covariance','mean_global_weights','posterior_weight_bank']:
                assert key in d.files,(f,key)
            assert all(np.isfinite(d[k]).all() for k in d.files if np.issubdtype(d[k].dtype,np.number))
            np.testing.assert_array_equal(d['sample_ids'],sample_ids)
            np.testing.assert_allclose(d['pi'].sum(-1),1,rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(d['posterior_weight_bank'].sum(-1),1,rtol=1e-5,atol=1e-6)
            assert str(d['inference_mode'])=='posterior_predictive_qmc'
            assert int(d['posterior_draws'])==64
            assert d['raw_output'].shape[-1]==6*k
            assert d['pi'].shape[-1]==k and d['posterior_weight_bank'].shape==(64,k)
    # Reproduce saved predictions from the stored MC bank rather than resampling.
    model=BayesianMDN(final['resolved_config']['model_params']).eval()
    model.load_state_dict(final['model_state_dict'])
    with np.load(r/'fixed_samples/predictions/final.npz') as d:
        with torch.no_grad():
            raw=model.ungated_output(torch.tensor(inputs))
            reproduced=model.output_with_weights(raw,torch.tensor(d['posterior_weight_bank']))
        np.testing.assert_allclose(reproduced.numpy(),d['raw_output'],rtol=2e-5,atol=2e-5)
    rows=list(csv.DictReader((r/'history.csv').open()));assert len(rows)==3
    weights=[json.loads(x) for x in (r/'weight_history.jsonl').read_text().splitlines()]
    assert [w['epoch'] for w in weights]==[0,1,2,3]
    usage=[json.loads(x) for x in (r/'usage_history.jsonl').read_text().splitlines()]
    assert [u['epoch'] for u in usage]==[0,1,2,3]
    for u in usage:
        assert u['sample_count']==19148 and u['positions']==19148*48
        np.testing.assert_allclose(np.asarray(u['mean_pi_by_horizon']).sum(-1),1,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(np.asarray(u['mean_responsibility_by_horizon']).sum(-1),1,rtol=1e-5,atol=1e-6)
        for data in u['conditional_k'].values():
            assert sum(data['histogram_all_positions'])==u['positions']
            assert np.all(np.asarray(data['histogram_by_horizon']).sum(-1)==u['sample_count'])
    metrics=list(csv.DictReader((r/'metrics/history.csv').open()));assert len(metrics)==3
    hashes=json.loads((ROOT/'gated_mdn/reports/BASE_SOURCE_HASHES.json').read_text())
    changes=[name for name,h in hashes.items() if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h]
    assert not changes,changes
    test=json.loads((r/'testing/evaluation_limited.json').read_text())
    assert test['sample_count']==16 and test['evaluation_inference']['mode']=='posterior_predictive'
    budget=json.loads((r/'posterior_mc_comparison.json').read_text())
    old=ROOT/'results/trained_models/bayesian_mdn/imptc/smoke_bayesian_peds_imptc/runs/bayesian_k8_smoke_seed2024/checkpoints/final.pt'
    unchanged=None
    if old.exists() and final['resolved_config']['model_params']['num_gaussians']==8:
        previous=torch.load(old,map_location='cpu',weights_only=False)
        unchanged=all(torch.equal(previous['model_state_dict'][k],v) for k,v in final['model_state_dict'].items())
    report={'run_dir':str(r),'epochs':3,'initial_weights':weights[0],'final_weights':weights[-1],
        'history':rows,'initial_usage':usage[0],'final_usage':usage[-1],
        'fixed_prediction_count':len(files),'fixed_sample_count':len(sample_ids),
        'unit_tests':7,'metric_epochs':len(metrics),'limited_test_samples':16,
        'baseline_hashes_unchanged':True,'training_state_equal_previous_smoke':unchanged,
        'mc_budget_comparison':budget,'k_max':final['resolved_config']['model_params']['num_gaussians'],
        'note':f"Initial K99={weights[0]['effective_k_global_mass']}; smoke is not evidence of optimal K or safe pruning."}
    (ROOT/'bayesian_mdn/reports'/f'{args.report_name}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['initial_usage','final_usage','history']},indent=2))

if __name__=='__main__':main()
