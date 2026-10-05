"""Read-only check: validation NLL before/after a split on a saved phase checkpoint.

Compares the old absolute split (delta = 0.05 m) with the relative split
(delta_rel * median source sigma). Uses train + validation data only.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

REPORT = ROOT / 'growing_mdn/reports/SPLIT_CHECK.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='growing_peds_imptc.json')
    parser.add_argument('--delta-rel', type=float, default=0.05)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from utils.config_loader import ConfigLoader
    from utils.data_loader import DataLoader
    from growing_mdn.model import build_model, grow_model
    from growing_mdn.selection import component_scores, pick_split, source_sigma_median
    from growing_mdn.train import validation_nll

    cfg = ConfigLoader(str(ROOT / 'growing_mdn/configs/imptc' / args.config), 'imptc', False, False,
                       Path(args.config).stem, 'growing_mdn', 'training')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    k = int(saved['num_gaussians'])
    model = build_model(cfg.model_params, k, device)
    model.load_state_dict(saved['model_state_dict'])
    loader = DataLoader(cfg)
    loader.load_train_data()
    loader.load_eval_data()
    Xv, yv = loader.eval_data[0], loader.eval_data[1]
    batch = cfg.test_params['batch_size']
    scores = component_scores(model, loader.train_data[0], loader.train_data[1], device, batch)
    source = pick_split(scores)
    before = validation_nll(model, Xv, yv, device, batch, k)
    sigma_med = source_sigma_median(model, loader.train_data[0], device, batch, source)
    offsets = args.delta_rel * sigma_med
    result = {'checkpoint': str(args.checkpoint), 'k_before': k, 'split_component': source,
              'validation_nll_before': before, 'delta_rel': args.delta_rel,
              'sigma_median_min': float(sigma_med.min()), 'sigma_median_max': float(sigma_med.max()),
              'offset_abs_min': float(offsets.min()), 'offset_abs_median': float(sorted(offsets.ravel())[offsets.size // 2]),
              'offset_abs_max': float(offsets.max())}
    for name, off in (('absolute_0.05', 0.05), ('relative', offsets)):
        grown, _ = grow_model(model, source, off)
        after = validation_nll(grown, Xv, yv, device, batch, k + 1)
        result[name] = {'validation_nll_after': after, 'delta_nll': after - before}
    print(json.dumps(result, indent=2))
    entries = json.loads(REPORT.read_text()) if REPORT.exists() else []
    entries = [e for e in entries if e['checkpoint'] != result['checkpoint']] + [result]
    REPORT.write_text(json.dumps(entries, indent=2) + '\n')


if __name__ == '__main__':
    main()
