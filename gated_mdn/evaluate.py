"""Evaluate deterministic gates with the legacy repository metrics."""
import argparse,json,logging,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'base_mdn'))
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.experiment import set_global_seed
from eval import MDN_Forecaster
from gated_mdn.model import GatedMDN
from gated_mdn.train import validation_nll

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='gated_peds_imptc.json');p.add_argument('--run-id',required=True)
 p.add_argument('--limit',type=int);p.add_argument('--official',action='store_true');p.add_argument('--gpu',default='0');a=p.parse_args()
 if a.official and a.limit:raise ValueError('Official requires the full test split')
 if a.limit is not None and a.limit<1:raise ValueError('Positive limit required')
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;set_global_seed(2024);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 path=ROOT/'gated_mdn/configs/imptc'/a.config;cfg=ConfigLoader(str(path),'imptc',False,False,path.stem,'gated_mdn','testing')
 run=Path(cfg.result_path)/'runs'/a.run_id;cp=torch.load(run/'checkpoints/best.pt',map_location=device,weights_only=False)
 if cp.get('architecture')!='global_hard_concrete_mdn_v1' or cp['resolved_config']['model_params']!=cfg.model_params:raise ValueError('Incompatible checkpoint/config')
 model=GatedMDN(cfg.model_params).to(device);model.load_state_dict(cp['model_state_dict']);model.eval()
 loader=DataLoader(cfg);loader.load_test_data();X,y=loader.get_test_data()[:2];n=len(X) if a.limit is None else min(a.limit,len(X))
 d={'run_id':a.run_id,'epoch':cp['epoch'],'checkpoint':str(run/'checkpoints/best.pt'),'split':'test' if a.limit is None else 'test_limited',
 'sample_count':n,'seed':2024,'mdn_parameterization':{'mode':'legacy'},'evaluator_version':'base_mdn_legacy_bins',
 'test_params':cfg.test_params,'gates':model.gate_report(),'parameter_count':sum(p.numel() for p in model.parameters()),
 'test_nll':validation_nll(model,X[:n],y[:n],device,cfg.test_params['batch_size'],cfg.model_params['num_gaussians'])}
 if a.official:
  cfg.testing_path=str(run/'testing/official_evaluator');Path(cfg.testing_path).mkdir(parents=True,exist_ok=True)
  set_global_seed(2024);forecaster=MDN_Forecaster(cfg,model,loader,'testing',logging.getLogger('gated'),device)
  d['official_metrics']=forecaster.evaluate(epoch=cp['epoch'])
 dest=run/'testing';dest.mkdir(exist_ok=True);path=dest/('evaluation_limited.json' if a.limit else 'evaluation.json')
 path.write_text(json.dumps(d,indent=2)+'\n');print(path)
if __name__=='__main__':main()
