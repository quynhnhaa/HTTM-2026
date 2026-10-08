"""Sinh docs/sparsemax_mdn_k8/demo/assets/presets.json và hai file tham chiếu để verify.js đối chiếu.

Preset được chọn bằng QUY TẮC, không chọn tay: với mỗi ``movement_class``, lấy
mẫu có sai số (mode mạnh nhất, mốc 4,8 s) gần TRUNG VỊ của nhãn đó nhất. Thêm
hai ca khó cố định để demo cho thấy mô hình hỏng ở đâu.

    python docs/sparsemax_mdn_k8/demo/tools/make_presets.py
"""
import sys

# Console Windows mặc định cp1252, không in được tiếng Việt.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import io
import json
import os
import pickle

import numpy as np

from export_weights import load_checkpoint, SPARSEMAX_RUN

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
EVAL_PKL = os.path.join(ROOT, 'data/trajdata/ego/imptc/eval/ego_samples.pkl')
BASELINE = os.path.join(
    ROOT, 'results/trained_models/base_mdn/imptc/default_peds_imptc/'
          'runs/imptc_baseline_seed2024')

LABELS = {'straight': 'Đi thẳng', 'light_left': 'Rẽ trái nhẹ', 'light_right': 'Rẽ phải nhẹ',
          'strong_left': 'Rẽ trái gắt', 'strong_right': 'Rẽ phải gắt', 'standing': 'Đứng yên'}
HARD = [('imptc_0_00122_00127', 'Quay đầu ⚠'), ('imptc_0_00138_00123', 'Trái gắt ⚠')]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(sd, X, hidden=8):
    """LSTM(4→8) một lớp rồi Linear(8→48·6K). Thứ tự cổng PyTorch: i, f, g, o."""
    wi, wh = sd['lstm.weight_ih_l0'], sd['lstm.weight_hh_l0']
    bi, bh = sd['lstm.bias_ih_l0'], sd['lstm.bias_hh_l0']
    h = np.zeros((len(X), hidden))
    c = np.zeros((len(X), hidden))
    for t in range(X.shape[1]):
        g = X[:, t, :] @ wi.T + bi + h @ wh.T + bh
        i, f, gg, o = (g[:, :hidden], g[:, hidden:2 * hidden],
                       g[:, 2 * hidden:3 * hidden], g[:, 3 * hidden:])
        c = sigmoid(f) * c + sigmoid(i) * np.tanh(gg)
        h = sigmoid(o) * np.tanh(c)
    return (h @ sd['fc.weight'].T + sd['fc.bias']).reshape(len(X), 48, -1)


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sparsemax(z):
    zs = np.sort(z, -1)[..., ::-1]
    cs = np.cumsum(zs, -1)
    k = np.arange(1, z.shape[-1] + 1)
    kz = ((1 + k * zs) > cs).sum(-1, keepdims=True)
    return np.clip(z - (np.take_along_axis(cs, kz - 1, -1) - 1) / kz, 0, None)


def dump_reference(name, run_dir):
    """Chuyển file dự đoán .npz mà REPO đã lưu sang JSON cho verify.js.

    Quan trọng: đây là đầu ra do chính PyTorch sinh ra trên GPU, KHÔNG phải do
    script này tính lại. Nhờ vậy verify.js mới thực sự kiểm được bản JS, chứ
    không phải so numpy với numpy.
    """
    pred = np.load(os.path.join(run_dir, 'fixed_samples/predictions/best.npz'))
    ref = {
        'X': np.asarray(np.load(os.path.join(run_dir, 'fixed_samples/inputs.npz'))['X'], float).tolist(),
        'pi': pred['pi'].astype(float).tolist(),
        'mu': pred['mu'].astype(float).tolist(),
        'sigma': pred['sigma'].astype(float).tolist(),
    }
    path = os.path.join(OUT, name)
    with io.open(path, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(ref, separators=(',', ':')))
    print(f'  {name} ({os.path.getsize(path) / 1024:.0f} KB) — từ .npz của repo')


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = list(pickle.load(open(EVAL_PKL, 'rb')).values())
    by_source = {v['source']: v for v in samples}
    X = np.stack([np.asarray(v['X'], float) for v in samples])
    Y = np.stack([np.asarray(v['y'], float)[:, :2] for v in samples])
    klass = np.array([v['movement_class'] for v in samples])

    sd, _ = load_checkpoint(os.path.join(BASELINE, 'checkpoints/best.pt'))
    out = forward(sd, X)
    K, t = 3, 47
    pi = softmax(out[:, t, 5 * K:])
    top = pi.argmax(-1)
    err = np.hypot(np.take_along_axis(out[:, t, :K], top[:, None], 1)[:, 0] - Y[:, t, 0],
                   np.take_along_axis(out[:, t, K:2 * K], top[:, None], 1)[:, 0] - Y[:, t, 1])

    chosen = []
    for cls, label in LABELS.items():
        mask = klass == cls
        idx = np.where(mask)[0]
        # bỏ mẫu quá nhỏ để hình không thành một chấm (trừ 'standing')
        span = np.abs(Y[idx, t, 0]) + np.abs(Y[idx, t, 1])
        pool = idx[span > 3] if cls != 'standing' and (span > 3).any() else idx
        pick = pool[np.argmin(np.abs(err[pool] - np.median(err[mask])))]
        chosen.append((samples[pick], label, float(err[pick]), 'median'))
        print(f'  {label:14s} {cls:14s} sai {err[pick]:.2f} m '
              f'(trung vị nhãn {np.median(err[mask]):.2f} m)')
    for src, label in HARD:
        i = [j for j, v in enumerate(samples) if v['source'] == src][0]
        chosen.append((by_source[src], label, float(err[i]), 'hard'))
        print(f'  {label:14s} {klass[i]:14s} sai {err[i]:.2f} m  ← ca khó')

    presets = [{
        'name': label, 'label': v['movement_class'], 'source': v['source'],
        'avg_v': round(float(v['avg_velocity']), 3), 'err': round(e, 2), 'kind': kind,
        'past': [[round(a, 4), round(b, 4)] for a, b in np.asarray(v['X'], float)[:, :2]],
        'future': [[round(a, 4), round(b, 4)] for a, b in np.asarray(v['y'], float)[:, :2]],
    } for v, label, e, kind in chosen]

    path = os.path.join(OUT, 'presets.json')
    with io.open(path, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(presets, separators=(',', ':'), ensure_ascii=False))
    print(f'\npresets.json = {os.path.getsize(path) / 1024:.0f} KB, {len(presets)} mẫu\n')

    print('Tham chiếu để verify.js đối chiếu:')
    dump_reference('_ref_softmax.json', BASELINE)
    sp_run = os.path.join(OUT, '_hf', SPARSEMAX_RUN)
    if os.path.exists(os.path.join(sp_run, 'fixed_samples/predictions/best.npz')):
        dump_reference('_ref_sparsemax.json', sp_run)
    else:
        print('  ! thiếu fixed_samples của run sparsemax — chạy export_weights.py trước')


if __name__ == '__main__':
    main()
