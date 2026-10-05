"""Compare completed full test reports using the same stable evaluator."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    jobs = [('Baseline stable', 'stable_mdn', 'stable_peds_imptc', 'imptc_stable_seed2024_cont500'),
            ('Attention stable', 'stable_attention_mdn', 'stable_attention_peds_imptc', 'imptc_stable_attention_seed2024_cont500'),
            ('Residual MDN', 'residual_mdn', 'residual_peds_imptc', 'imptc_residual_seed2024_cont500')]
    reports = []
    for label, arch, config, run in jobs:
        path = ROOT / 'results/trained_models' / arch / 'imptc' / config / 'runs' / run / 'testing/evaluation.json'
        data = json.loads(path.read_text())
        assert data['split'] == 'test' and data['sample_count'] == 56694
        assert data['evaluator_version'] == 'stable_v2_ecdf_rng'
        reports.append({'label': label, 'source': str(path), 'data': data})
    baseline = reports[0]['data']
    for report in reports[1:]:
        for key in ('seed', 'mdn_parameterization', 'test_params', 'eval_metrics'):
            assert report['data'][key] == baseline[key], key
    keys = [('test_nll', 'Test NLL / vị trí ↓'), ('minade20_m', 'minADE20 (m) ↓'),
            ('minfde20_m', 'minFDE20 (m) ↓'), ('ravg_percent', 'Ravg (%) ↑'),
            ('rmin_percent', 'Rmin (%) ↑'), ('s68_m2_per_s', 'S68 (m²/s) ↓'),
            ('s95_m2_per_s', 'S95 (m²/s) ↓'), ('asaee_m_per_s', 'ASAEE (m/s) ↓'),
            ('parameter_count', 'Số tham số'), ('inference_ms_per_batch', 'Inference (ms/batch)')]
    def value(report, key):
        data = report['data']
        return float(data[key] if key in data else data['official_metrics'][key])
    rows = []
    text = '# So sánh test IMPTC: baseline stable, attention và residual\n\n'
    text += 'Toàn bộ 56.694 mẫu test, seed 2024, evaluator stable_v2_ecdf_rng. Best chọn theo full validation NLL.\n\n'
    text += '| Model | Best epoch |\n|---|---:|\n'
    for report in reports:
        text += f"| {report['label']} | {report['data']['epoch']} |\n"
    text += '\n| Chỉ số | Baseline stable | Attention stable | Residual MDN | Residual − baseline |\n|---|---:|---:|---:|---:|\n'
    for key, label in keys:
        values = [value(report, key) for report in reports]
        assert all(math.isfinite(v) for v in values)
        delta = values[2] - values[0]
        rows.append({'metric': key, 'baseline': values[0], 'attention': values[1], 'residual': values[2], 'delta_residual_minus_baseline': delta})
        text += f'| {label} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {delta:+.6f} |\n'
    text += '\n## Diễn giải và giới hạn\n\n'
    text += '- Residual giữ kiến trúc/số tham số baseline; chỉ cộng prior CV, vận tốc trung bình 5 bước cuối, vào mean.\n'
    text += '- Cả ba dùng covariance stable và lịch 2500 epoch + continuation500, Adam được giữ với LR continuation1e-5 đến1e-7.\n'
    text += '- NLL dùng 48 timestep; ADE/FDE repo dùng sáu mốc và sắp sample riêng mỗi timestep, không phải sampling quỹ đạo chung.\n'
    text += '- Đọc sharpness cùng reliability; chưa thể kết luận toàn diện chỉ bằng NLL/ADE/FDE.\n'
    text += '- Đây là một seed; không suy ra ý nghĩa thống kê. Thời gian evaluator chưa phải benchmark độc lập có warm-up.\n'
    text += '- Không so trực tiếp với legacy evaluator. Window velocity đã chốt trước run, không chọn bằng test.\n'
    text += '\nNguồn evaluation.json được ghi trong comparison.json.\n'
    out = ROOT / 'results/comparisons/stable_three_models'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps({'status': 'completed', 'seed': 2024,
        'split': 'test', 'sample_count': 56694, 'evaluator_version': 'stable_v2_ecdf_rng',
        'sources': [{'label': r['label'], 'path': r['source'], 'epoch': r['data']['epoch']} for r in reports],
        'results': rows}, indent=2) + '\n')
    (out / 'RESULTS.md').write_text(text)
    (ROOT / 'residual_mdn/reports/TEST_RESULTS.md').write_text(text)
    print('Comparison ready:', out / 'RESULTS.md')


if __name__ == '__main__':
    main()
