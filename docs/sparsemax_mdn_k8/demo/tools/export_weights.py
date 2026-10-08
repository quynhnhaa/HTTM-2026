"""Bóc trọng số từ checkpoint .pt sang JSON cho demo web — KHÔNG cần PyTorch.

File .pt của PyTorch là một zip chứa pickle; ta đọc bằng ``zipfile`` và một
``Unpickler`` thay mọi lớp torch bằng stub, rồi dựng lại tensor từ storage thô.
Nhờ vậy script chạy được trên máy không cài torch.

    python docs/sparsemax_mdn_k8/demo/tools/export_weights.py

Đầu ra: docs/sparsemax_mdn_k8/demo/assets/weights.json, docs/sparsemax_mdn_k8/demo/assets/presets.json

Checkpoint sparsemax không nằm trong repo; script tự tải từ HuggingFace
(``huggingface_hub``). Thiếu gói đó thì chỉ bỏ qua mô hình sparsemax.
"""
import sys

# Console Windows mặc định cp1252, không in được tiếng Việt.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import io
import json
import os
import pickle
import zipfile

import numpy as np

def _repo_root(start):
    """Đi ngược lên cho tới khi thấy base_mdn/ — không phụ thuộc độ sâu thư mục."""
    p = os.path.abspath(start)
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, 'base_mdn')):
            return p
        p = os.path.dirname(p)
    raise SystemExit('không tìm thấy gốc repo (thư mục chứa base_mdn/)')


ROOT = _repo_root(__file__)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
HF_REPO = 'quinha10/HTTM-GK-2026'
SPARSEMAX_RUN = ('results/trained_models/sparsemax_mdn/imptc/sparsemax_k8_peds_imptc/'
                 'runs/sparsemax_k8_v2_seed2024/')

STORAGE_DTYPE = {
    'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16,
    'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
}


def load_checkpoint(path):
    """Đọc một checkpoint .pt mà không cần torch. Trả về (state_dict, checkpoint)."""
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split('/')[0]

    def stub(dtype):
        return type('Storage', (object,), {'dt': dtype})

    def persistent_load(pid):
        # ('storage', StorageType, key, location, numel)
        return pid[1].dt, pid[2], pid[4]

    def rebuild(storage, offset, size, stride, *_):
        dtype, key, _ = storage
        flat = np.frombuffer(zf.read(f'{prefix}/data/{key}'), dtype=dtype)
        n = int(np.prod(size)) if len(size) else 1
        return flat[offset:offset + n].reshape(size)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name in STORAGE_DTYPE:
                return stub(STORAGE_DTYPE[name])
            if name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
                return rebuild
            if name == '_rebuild_parameter':
                return lambda data, *_: data
            if module.startswith('torch'):
                return type(name, (object,), {})
            return super().find_class(module, name)

    up = Unpickler(io.BytesIO(zf.read(f'{prefix}/data.pkl')))
    up.persistent_load = persistent_load
    ck = up.load()
    return {k: np.asarray(v, np.float64) for k, v in ck['model'].items()}, ck


def quantise(a, digits=5):
    return [round(float(x), digits) for x in np.asarray(a).ravel()]


def entry(sd, K, epoch, metrics, extra=None):
    e = {
        'K': K, 'epoch': int(epoch),
        'params': int(sum(v.size for v in sd.values())),
        'metrics': metrics,
        'Wi': quantise(sd['lstm.weight_ih_l0']), 'Wh': quantise(sd['lstm.weight_hh_l0']),
        'bi': quantise(sd['lstm.bias_ih_l0']), 'bh': quantise(sd['lstm.bias_hh_l0']),
        'Wo': quantise(sd['fc.weight']), 'bo': quantise(sd['fc.bias']),
    }
    if extra:
        e.update(extra)
    return e


def fetch_sparsemax():
    """Tải run sparsemax từ HuggingFace. Trả về None nếu không tải được."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('  ! thiếu huggingface_hub — bỏ qua mô hình sparsemax')
        return None
    cache = os.path.join(OUT, '_hf')
    try:
        return {
            name: hf_hub_download(HF_REPO, SPARSEMAX_RUN + name,
                                  repo_type='dataset', local_dir=cache)
            for name in ('checkpoints/best.pt',
                         'evaluation/test_best_clampedpi.json',
                         'analysis/k_usage.json',
                         'fixed_samples/inputs.npz',
                         'fixed_samples/predictions/best.npz')
        }
    except Exception as exc:                                    # noqa: BLE001
        print(f'  ! không tải được sparsemax ({type(exc).__name__}) — bỏ qua')
        return None


def main():
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(ROOT, 'results/trained_models/base_mdn/imptc')
    softmax_runs = {
        1: f'{base}/m1_peds_imptc/runs/imptc_m1_seed2024/checkpoints/best.pt',
        2: f'{base}/m2_peds_imptc/runs/imptc_m2_seed2024/checkpoints/best.pt',
        3: f'{base}/default_peds_imptc/runs/imptc_baseline_seed2024/checkpoints/best.pt',
        5: f'{base}/m5_peds_imptc/runs/imptc_m5_seed2024/checkpoints/best.pt',
        8: f'{base}/m8_peds_imptc/runs/imptc_m8_seed2024/checkpoints/best.pt',
    }
    ablation = {r['num_gaussians']: r for r in json.load(open(
        os.path.join(ROOT, 'results/ablations/imptc_num_gaussians/ablation.json'),
        encoding='utf-8'))['results']}

    bundle = {'dt': 0.1, 'horizon': 48, 'input': 32, 'hidden': 8, 'models': {}}
    for K, path in softmax_runs.items():
        if not os.path.exists(path):
            print(f'  ! thiếu checkpoint M={K} — bỏ qua')
            continue
        sd, ck = load_checkpoint(path)
        a = ablation[K]
        bundle['models'][str(K)] = entry(sd, K, ck['epoch'], {
            'nll': round(a['test_nll'], 4),
            'ravg': round(a['ravg_percent'], 2), 'rmin': round(a['rmin_percent'], 2),
            's68': round(a['s68_m2_per_s'], 3), 's95': round(a['s95_m2_per_s'], 3),
            'ade': round(a['minade20_m'], 3), 'fde': round(a['minfde20_m'], 3),
        })
        print(f'  M={K}: epoch {ck["epoch"]}, {bundle["models"][str(K)]["params"]} tham số')

    paths = fetch_sparsemax()
    if paths:
        sd, ck = load_checkpoint(paths['checkpoints/best.pt'])
        ev = json.load(open(paths['evaluation/test_best_clampedpi.json'], encoding='utf-8'))
        ku = json.load(open(paths['analysis/k_usage.json'], encoding='utf-8'))
        om = ev['official_metrics']
        bundle['models']['sp'] = entry(sd, 8, ck['epoch'], {
            'nll': round(ev['exact_nll'], 4),
            'ravg': round(om['ravg_percent'], 2), 'rmin': round(om['rmin_percent'], 2),
            's68': round(om['s68_m2_per_s'], 3), 's95': round(om['s95_m2_per_s'], 3),
            'ade': round(om['minade20_m'], 3), 'fde': round(om['minfde20_m'], 3),
        }, {'sparsemax': True, 'run': 'sparsemax_k8_v2_seed2024'})
        bundle['kusage'] = {
            'hist': [ku['k_distribution']['histogram'][str(i)] for i in range(1, 9)],
            'mean': round(ku['k_distribution']['mean'], 4),
            'std': round(ku['k_distribution']['std'], 4),
            'test_mean': round(ev['support_size']['mean_support_size'], 4),
            'dead': ku['component_activity']['dead_components'],
            'full_frac': ev['support_size']['fraction_full_support'],
            'per_step': [round(x, 3) for x in ku['mean_k_per_forecast_step']],
            'spearman': round(
                ku['association_with_difficulty']['spearman_k_vs_weighted_sqrt_trace'], 3),
        }
        print(f'  sparsemax: epoch {ck["epoch"]}, arch={ck.get("architecture")}, '
              f'K trung bình {bundle["kusage"]["mean"]}')

    dst = os.path.join(OUT, 'weights.json')
    with io.open(dst, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(bundle, separators=(',', ':'), ensure_ascii=False))
    print(f'\nweights.json = {os.path.getsize(dst) / 1024:.0f} KB, '
          f'{len(bundle["models"])} mô hình')


if __name__ == '__main__':
    main()
