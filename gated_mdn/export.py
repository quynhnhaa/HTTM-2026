"""Export a deterministic pruned head as a plain legacy LSTM-MDN."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gated_mdn.model import GatedMDN

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 output=Path(a.output)
 if output.exists():raise FileExistsError(output)
 cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if cp.get('architecture')!='global_hard_concrete_mdn_v1':raise ValueError('Expected gated checkpoint')
 model=GatedMDN(cp['resolved_config']['model_params']);model.load_state_dict(cp['model_state_dict']);model.eval()
 compact,cfg,indices=model.compact()
 out={'architecture':'compact_legacy_lstm_mdn_v1','inference_only':True,'source_checkpoint':str(Path(a.checkpoint).resolve()),
      'epoch':cp['epoch'],'model_state_dict':compact.state_dict(),'resolved_config':{'model_params':cfg},
      'active_indices':indices,'gates':model.gate_report(),'parameter_count':sum(p.numel() for p in compact.parameters())}
 output.parent.mkdir(parents=True,exist_ok=True);torch.save(out,output);print(json.dumps({k:v for k,v in out.items() if k not in ['model_state_dict','resolved_config']},indent=2))
if __name__=='__main__':main()
