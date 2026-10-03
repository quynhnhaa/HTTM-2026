"""Build comparison artifacts only after both full test evaluations finish."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/comparisons/stable_baseline_vs_attention'


def main():
    reports = []
    for arch, config, run in [
        ('stable_mdn', 'stable_peds_imptc', 'imptc_stable_seed2024_cont500'),
        ('stable_attention_mdn', 'stable_attention_peds_imptc', 'imptc_stable_attention_seed2024_cont500'),
    ]:
        path = ROOT / 'results/trained_models' / arch / 'imptc' / config / 'runs' / run / 'testing/evaluation.json'
        data = json.loads(path.read_text())
        assert data['split'] == 'test' and data['sample_count'] == 56694
        assert data['evaluator_version'] == 'stable_v2_ecdf_rng'
        reports.append((path, data))
    baseline, attention = [r[1] for r in reports]
    for key in ('seed', 'mdn_parameterization', 'test_params', 'eval_metrics'):
        assert baseline[key] == attention[key], key
    metrics = [('test_nll', 'Test NLL / vị trí', '↓'),
               ('minade20_m', 'minADE20 (m)', '↓'),
               ('minfde20_m', 'minFDE20 (m)', '↓'),
               ('ravg_percent', 'Ravg (%)', '↑'),
               ('rmin_percent', 'Rmin (%)', '↑'),
               ('s68_m2_per_s', 'S68 (m²/s)', '↓'),
               ('s95_m2_per_s', 'S95 (m²/s)', '↓'),
               ('asaee_m_per_s', 'ASAEE (m/s)', '↓'),
               ('parameter_count', 'Số tham số', '↓'),
               ('inference_ms_per_batch', 'Inference (ms/batch)', '↓')]
    def value(report, key):
        return report[key] if key in report else report['official_metrics'][key]
    rows = []
    for key, label, direction in metrics:
        b, a = float(value(baseline, key)), float(value(attention, key))
        assert math.isfinite(b) and math.isfinite(a)
        rows.append({'metric': key, 'baseline': b, 'attention': a, 'delta_attention_minus_baseline': a-b})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'comparison.json').write_text(json.dumps({
        'status': 'completed', 'split': 'test', 'sample_count': 56694,
        'evaluator_version': 'stable_v2_ecdf_rng', 'seed': baseline['seed'],
        'baseline_source': str(reports[0][0]), 'attention_source': str(reports[1][0]),
        'baseline_best_epoch': baseline['epoch'], 'attention_best_epoch': attention['epoch'],
        'results': rows}, indent=2) + '\n')
    text = '# So sánh test: baseline stable và attention stable\n\n'
    text += f"Toàn bộ 56.694 mẫu test IMPTC; seed {baseline['seed']}; evaluator stable_v2_ecdf_rng.\n\n"
    text += f"Best checkpoint chọn bằng full validation: baseline epoch {baseline['epoch']}, attention epoch {attention['epoch']}.\n\n"
    text += '| Chỉ số | Hướng | Baseline stable | Attention stable | Attention − baseline |\n|---|---|---:|---:|---:|\n'
    for row, (_, label, direction) in zip(rows, metrics):
        text += f"| {label} | {direction} | {row['baseline']:.6f} | {row['attention']:.6f} | {row['delta_attention_minus_baseline']:+.6f} |\n"
    text += '\n## Giao thức và giới hạn\n\n'
    text += '- Sigma floor 0.01 m, rho limit 0.999, exp guard 1000 m ở cả hai.\n'
    text += '- NLL trung bình trên tất cả 48 future timestep; ADE/FDE evaluator dùng sáu mốc 0.8–4.8 s, K=20.\n'
    text += '- Evaluator sắp mẫu theo mật độ riêng tại từng timestep; không coi đây là 20 quỹ đạo có liên kết theo thời gian.\n'
    text += '- Reliability dùng empirical CDF đúng threshold; sharpness giữ phép tổng hợp local và diện tích grid đã sửa.\n'
    text += '- Cả hai đã train 2500 epoch rồi continuation 500 epoch, LR mới 1e-5 đến 1e-7 và giữ trạng thái Adam.\n'
    text += '- Đây là một seed; chưa có ablation tách đóng góp attention khỏi decoder. Không khẳng định ý nghĩa thống kê.\n'
    text += '- Sharpness cần đọc cùng reliability. Thời gian inference là trung bình batch của evaluator, chưa phải benchmark độc lập với warm-up.\n'
    text += '- Không so trực tiếp với metric legacy khác phiên bản evaluator. Không chọn checkpoint hoặc quyết định train tiếp bằng test.\n'
    text += '\nNguồn: hai file evaluation.json được ghi trong comparison.json.\n'
    (OUTPUT / 'RESULTS.md').write_text(text)
    print('Comparison ready:', OUTPUT / 'RESULTS.md')


if __name__ == '__main__':
    main()
