# Experimental Protocol: IMPTC LSTM-MDN Baseline

## 1. Mục đích và phạm vi

Tài liệu này định nghĩa protocol thực nghiệm trước khi thay đổi training pipeline hoặc chạy full training. Mục tiêu là tái tạo baseline IMPTC của official repository, đồng thời lưu đủ artifact để:

- kiểm tra khả năng hội tụ;
- đánh giá accuracy và uncertainty bằng metric chính thức;
- trực quan hóa từng Gaussian component;
- theo dõi cùng một validation sample qua nhiều epoch;
- chọn good, difficult/uncertain và failure cases theo tiêu chí có thể giải thích;
- tái tạo biểu đồ mà không phải train lại.

Protocol ưu tiên hành vi của official repository khi paper và code không thống nhất. Mọi khác biệt với paper phải được ghi trong báo cáo kết quả.

## 2. Trạng thái protocol

Trạng thái hiện tại: **đã chốt thiết kế, chưa triển khai**.

Các quyết định đã được xác nhận:

- baseline ưu tiên official repository;
- được sửa bug kỹ thuật có bằng chứng, nhưng không thay đổi thuật toán;
- công thức sharpness phải được kiểm chứng trước khi sửa;
- sử dụng 8 fixed samples từ validation set;
- checkpoint tại epoch 1, 5, 10, mỗi 100 epoch, best và final;
- history cơ bản được lưu mỗi epoch;
- full official metrics được tính mỗi 250 epoch và ở final;
- best checkpoint được chọn theo validation NLL.

Mọi thay đổi sau khi protocol được triển khai phải được ghi trong `run_manifest.json` và báo cáo thực nghiệm.

## 3. Câu hỏi nghiên cứu

### RQ1 - Khả năng học

LSTM-MDN có làm giảm train NLL và validation NLL trong quá trình training hay không?

### RQ2 - Độ chính xác dự báo

Chất lượng dự báo thay đổi thế nào theo epoch dựa trên metric displacement chính thức của repository, đặc biệt là `minADE20` và `minFDE20`?

### RQ3 - Chất lượng uncertainty

Các phân phối dự báo có reliable và sharp hay không, dựa trên `Ravg`, `Rmin`, `S68` và `S95`?

### RQ4 - Quá trình học của GMM

Với cùng một validation sample, các mixture weight, mean và covariance thay đổi thế nào từ epoch đầu đến best/final?

### RQ5 - Hành vi theo từng loại case

Model biểu hiện thế nào trên good, difficult/uncertain và failure cases?

## 4. Baseline được khóa

### 4.1 Dataset và split

- Dataset chính: IMPTC.
- Sử dụng dữ liệu đã preprocessing do official repository/paper cung cấp nếu có thể.
- Giữ nguyên train/eval/test split trong bộ dữ liệu đã preprocessing.
- Validation/eval set được dùng cho model selection và fixed-sample tracking.
- Test set chỉ được dùng sau khi hoàn tất training/model selection.
- Không chọn checkpoint hoặc điều chỉnh hyperparameter dựa trên test result.

Đường dẫn logic mặc định:

```text
data/trajdata/ego/imptc/train/ego_samples.pkl
data/trajdata/ego/imptc/eval/ego_samples.pkl
data/trajdata/ego/imptc/test/ego_samples.pkl
```

### 4.2 Input và target

```text
Input X shape:  [N, 32, 4]
Input features: [x, y, vx, vy]
Target y shape: [N, 48, 2]
Target:         [x, y]
Sampling rate:  10 Hz
Observed time:  3.2 s
Forecast time:  4.8 s
```

Tọa độ phải là human-egocentric theo preprocessing của repository. Không normalization hoặc augmentation bổ sung trong baseline.

### 4.3 Model

```text
LSTM input size:       4
LSTM hidden size:      8
LSTM layers:           1
Gaussian components:   3
Forecast steps:        48
Parameters/component:  6
Output shape:          [B, 48, 18]
```

MDN output layout với `K = 3`:

```text
output[..., 0:K]       = mu_x
output[..., K:2K]      = mu_y
output[..., 2K:3K]     = raw_sigma_x
output[..., 3K:4K]     = raw_sigma_y
output[..., 4K:5K]     = raw_rho
output[..., 5K:6K]     = mixture logits
```

Baseline decode theo official code:

```text
sigma_x = exp(raw_sigma_x)
sigma_y = exp(raw_sigma_y)
rho     = tanh(raw_rho)
pi      = softmax(mixture_logits, component_dimension)
```

Covariance của component `k`:

```text
Sigma_k = [[sigma_x^2,               rho * sigma_x * sigma_y],
           [rho * sigma_x * sigma_y,               sigma_y^2]]
```

### 4.4 Optimization

Giữ config official repository:

```text
Loss:                  mixture negative log-likelihood
Optimizer:             Adam
Initial learning rate: 1e-3
Scheduler:             linear decay
Final learning rate:   1e-7
Batch size:            4096
Epochs:                2500
Shuffle:               mỗi epoch
Train reduction:       0.5
Validation reduction:  0.5
Dynamic input horizon: false
```

Các điểm khác paper phải được báo cáo:

- paper ghi batch size 1024;
- paper không mô tả rõ việc giảm train/validation xuống 50%;
- paper mô tả activation của `sigma` và `rho` khác official code.

## 5. Reproducibility policy

### 5.1 Seed

Mỗi run phải có một `run_seed` duy nhất và được áp dụng cho:

- Python `random`;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA;
- DataLoader/subset selection;
- Monte Carlo evaluation nếu có thể tách generator.

Baseline đầu tiên dùng:

```text
run_seed = 2024
```

Nếu tài nguyên cho phép chạy nhiều seed, các seed bổ sung phải là run độc lập và không được ghi đè artifact của baseline.

### 5.2 Determinism

Run manifest phải ghi:

- deterministic algorithms có bật hay không;
- cuDNN deterministic/benchmark settings;
- phiên bản Python, PyTorch, CUDA runtime và driver;
- GPU name;
- operating system;
- git commit nếu repository có Git metadata;
- hash của config và dataset files.

Không được tuyên bố bitwise reproducibility nếu CUDA/backend không bảo đảm điều đó.

## 6. Kiểm tra dữ liệu trước training

Full training chỉ được bắt đầu khi data audit đạt các điều kiện sau:

1. Cả ba split đều đọc được.
2. Shape phù hợp `[N,32,4]` và `[N,48,2]`.
3. Không có `NaN` hoặc `Inf`.
4. Dtype có thể chuyển sang `float32` an toàn.
5. Vị trí quan sát cuối gần `(0,0)` theo ego-coordinate convention.
6. Velocity có giá trị hợp lý và không bị tráo với position.
7. Sample count được ghi cho từng split.
8. Thứ tự/key của sample có thể ánh xạ ổn định.
9. Dataset hash được lưu trong manifest.
10. Không có overlap ngoài ý muốn giữa train/eval/test nếu metadata cho phép kiểm tra.

Nếu sample count khác tài liệu tác giả, phải dừng trước full training và điều tra nguyên nhân.

## 7. Fixed-sample protocol

### 7.1 Nguồn và số lượng

- Chọn đúng 8 samples từ validation/eval set.
- Chọn trước full training.
- Không thay đổi sau khi đã xem prediction của model.
- Không chọn dựa trên model performance để tránh cherry-picking.

### 7.2 Phương pháp chọn

Phương pháp mặc định:

1. Liệt kê toàn bộ stable sample IDs theo thứ tự chuẩn.
2. Dùng generator riêng với `fixed_sample_seed = 2024`.
3. Lấy mẫu không hoàn lại 8 IDs.
4. Sắp xếp kết quả theo stable ID để artifact có thứ tự ổn định.

Nếu pickle không có ID gốc, tạo stable ID từ:

```text
split + source + original pickle key
```

Chỉ dùng array index nếu không còn metadata tốt hơn. Khi đó phải lưu cả index, source và checksum của `X`/`y`.

### 7.3 Dữ liệu cố định cần lưu

Cho mỗi sample:

- stable sample ID;
- original key/index;
- source;
- observed `X`;
- future ground truth `y`;
- reference position;
- rotation angle;
- checksum của `X` và `y`.

Tại mỗi checkpoint, chỉ prediction/GMM parameters được phép thay đổi.

## 8. Training schedule

### 8.1 History mỗi epoch

Lưu sau mọi epoch:

- epoch;
- train NLL;
- validation NLL;
- learning rate dùng trong epoch;
- epoch duration;
- số train samples thực tế;
- số validation samples thực tế;
- train subset fingerprint;
- validation subset fingerprint;
- GPU peak allocated memory nếu có;
- trạng thái finite/diverged.

### 8.2 Checkpoint schedule

Periodic checkpoint epochs:

```text
1
5
10
100, 200, 300, ..., 2500
```

Ngoài periodic checkpoint:

- `best.pt`: validation NLL thấp nhất;
- `final.pt`: trạng thái sau epoch cuối;
- `last.pt`: checkpoint phục hồi mới nhất, có thể cập nhật mỗi epoch.

Nếu epoch 2500 đồng thời là periodic và final, không cần nhân đôi model tensor; có thể dùng metadata/reference, nhưng tên logical artifact phải rõ ràng.

### 8.3 Best-model policy

Primary selection criterion:

```text
validation NLL thấp nhất
```

Quy tắc tie-break:

1. Validation NLL thấp hơn.
2. Nếu bằng nhau trong tolerance được khai báo, chọn epoch sớm hơn.

Reliability/sharpness không dùng trực tiếp để chọn best baseline vì chỉ được tính định kỳ và có Monte Carlo noise. Chúng vẫn phải được báo cáo tại best checkpoint sau training.

### 8.4 Full metric schedule

Chạy evaluation đầy đủ tại:

```text
250, 500, 750, ..., 2500
best checkpoint
final checkpoint
```

Các metric:

- `Ravg`;
- `Rmin`;
- reliability theo từng selected forecast horizon;
- `S68`;
- `S95`;
- `minADE20`;
- `minFDE20`;
- ASAEE/AEE nếu giữ official repository behavior;
- inference time theo protocol timing hợp lệ.

Nếu best hoặc final trùng một epoch đã evaluate, tái sử dụng artifact thay vì chạy Monte Carlo lại, trừ khi cần đánh giá với seed độc lập.

### 8.5 Fixed-sample inference schedule

Chạy 8 fixed samples tại mọi periodic checkpoint:

```text
1, 5, 10, 100, 200, ..., 2500, best, final
```

Lưu raw output và decoded GMM parameters cho đủ 48 future timesteps, không chỉ sáu evaluation horizons.

## 9. Artifact directory layout

Mỗi run có ID bất biến:

```text
YYYYMMDD-HHMMSS_imptc_baseline_seed2024
```

Không dùng timestamp làm nguồn thông tin duy nhất; `run_id` phải được lưu trong mọi artifact có cấu trúc.

Đề xuất cấu trúc:

```text
results/trained_models/base_mdn/imptc/default_peds_imptc/
└── runs/
    └── <run_id>/
        ├── run_manifest.json
        ├── resolved_config.json
        ├── environment.json
        ├── data_audit.json
        ├── history.csv
        ├── events.jsonl
        ├── checkpoints/
        │   ├── epoch_0001.pt
        │   ├── epoch_0005.pt
        │   ├── epoch_0010.pt
        │   ├── epoch_0100.pt
        │   ├── ...
        │   ├── best.pt
        │   ├── final.pt
        │   └── last.pt
        ├── fixed_samples/
        │   ├── manifest.json
        │   ├── inputs.npz
        │   └── predictions/
        │       ├── epoch_0001.npz
        │       ├── epoch_0005.npz
        │       ├── ...
        │       ├── best.npz
        │       └── final.npz
        ├── metrics/
        │   ├── history.csv
        │   ├── epoch_0250.json
        │   ├── ...
        │   ├── best.json
        │   └── final.json
        ├── per_sample/
        │   ├── validation_best.parquet
        │   └── test_final.parquet
        └── figures/
            ├── loss/
            ├── metrics/
            ├── trajectories/
            ├── gmm_components/
            ├── uncertainty/
            ├── learning_over_time/
            └── cases/
```

`figures/` là derived artifacts. Mọi hình phải có thể tái tạo từ structured artifacts mà không cần train lại.

## 10. Artifact schemas

### 10.1 `run_manifest.json`

```json
{
  "schema_version": "1.0",
  "run_id": "20260101-120000_imptc_baseline_seed2024",
  "status": "created|running|completed|failed|interrupted",
  "experiment_name": "imptc_official_repo_baseline",
  "dataset": "IMPTC",
  "split_policy": "official_preprocessed_split",
  "run_seed": 2024,
  "fixed_sample_seed": 2024,
  "started_at": "ISO-8601 timestamp",
  "finished_at": null,
  "best_epoch": null,
  "best_validation_nll": null,
  "final_epoch": null,
  "config_sha256": "...",
  "dataset_hashes": {},
  "code_version": {
    "git_commit": null,
    "dirty": null
  },
  "baseline_source": "official_repository",
  "paper_code_differences": [
    "batch_size: paper=1024, repository=4096",
    "sigma/rho activation differs from paper text",
    "train/eval data reduction=0.5 in repository config"
  ],
  "applied_bug_fixes": [],
  "notes": []
}
```

### 10.2 `resolved_config.json`

Lưu snapshot config sau khi resolve path và override, gồm:

- model parameters;
- training parameters;
- evaluation parameters;
- metric flags;
- resolved data paths;
- resolved result path;
- checkpoint/metric schedules;
- seed policy.

Không lưu secret hoặc token nếu có.

### 10.3 `environment.json`

```json
{
  "python": "3.12.x",
  "pytorch": "...",
  "cuda_runtime": "...",
  "cuda_driver": "...",
  "cudnn": "...",
  "gpu_name": "NVIDIA GeForce RTX 5060 Ti",
  "gpu_count": 1,
  "cpu": "...",
  "ram_bytes": 0,
  "os": "...",
  "pip_freeze_sha256": "...",
  "deterministic_algorithms": false,
  "cudnn_deterministic": false,
  "cudnn_benchmark": false
}
```

### 10.4 `data_audit.json`

Cho mỗi split:

```json
{
  "train": {
    "path": "...",
    "sha256": "...",
    "sample_count": 0,
    "x_shape": [0, 32, 4],
    "y_shape": [0, 48, 2],
    "x_dtype": "...",
    "y_dtype": "...",
    "nan_count": 0,
    "inf_count": 0,
    "source_counts": {},
    "position_summary": {},
    "velocity_summary": {}
  },
  "eval": {},
  "test": {}
}
```

### 10.5 `history.csv`

Một row cho mỗi epoch:

```text
run_id
epoch
train_nll
validation_nll
learning_rate
duration_seconds
train_sample_count
validation_sample_count
train_subset_sha256
validation_subset_sha256
gpu_peak_memory_bytes
is_best
checkpoint_saved
full_metrics_evaluated
finite
timestamp_utc
```

Giá trị số phải được lưu ở full precision hợp lý, không chỉ giá trị đã làm tròn để in console.

### 10.6 `events.jsonl`

Mỗi dòng là một JSON event:

```json
{
  "timestamp_utc": "...",
  "run_id": "...",
  "event": "run_started|epoch_completed|checkpoint_saved|evaluation_completed|warning|run_failed|run_completed",
  "epoch": 1,
  "payload": {}
}
```

File này dùng cho audit/debug; `history.csv` vẫn là nguồn chính để vẽ loss.

### 10.7 Checkpoint `.pt`

Checkpoint phải chứa:

```python
{
    "schema_version": "1.0",
    "run_id": str,
    "epoch": int,
    "global_step": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "train_history": dict,
    "best_epoch": int | None,
    "best_validation_nll": float | None,
    "resolved_config": dict,
    "rng_state": {
        "python": object,
        "numpy": object,
        "torch_cpu": tensor,
        "torch_cuda": list[tensor]
    },
    "data_subset_state": dict,
    "created_at": str
}
```

`last.pt` phải được ghi an toàn bằng temporary file rồi atomic rename để tránh checkpoint hỏng nếu process bị dừng.

### 10.8 Fixed-sample `manifest.json`

```json
{
  "schema_version": "1.0",
  "split": "eval",
  "selection_method": "seeded_random_without_replacement",
  "selection_seed": 2024,
  "dataset_sha256": "...",
  "samples": [
    {
      "slot": 0,
      "sample_id": "eval:<source>:<key>",
      "original_key": "...",
      "array_index": 0,
      "source": "...",
      "x_sha256": "...",
      "y_sha256": "..."
    }
  ]
}
```

### 10.9 Fixed-sample `inputs.npz`

Arrays:

```text
sample_ids         string [8]
X                  float32 [8, 32, 4]
y                  float32 [8, 48, 2]
reference_position float32 [8, ...]
rotation_angle     float32 [8, ...]
source             string [8]
```

### 10.10 Fixed-sample prediction `.npz`

Mỗi checkpoint lưu:

```text
sample_ids       string  [8]
epoch            int64   scalar
raw_output       float32 [8, 48, 18]
pi               float32 [8, 48, 3]
mu               float32 [8, 48, 3, 2]
sigma            float32 [8, 48, 3, 2]
rho              float32 [8, 48, 3]
covariance       float32 [8, 48, 3, 2, 2]
component_mode   float32 [8, 48, 3, 2]
mixture_samples  optional; không lưu mặc định
```

`component_mode` của Gaussian chính là mean; trường này có thể bỏ nếu gây trùng dữ liệu. Mixture mode tổng thể không được gọi là component mean và cần metadata riêng nếu được xấp xỉ trên grid.

Mỗi file cần metadata:

```text
schema_version
run_id
checkpoint_name
decode_version
dtype
```

### 10.11 Metric snapshot JSON

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "split": "eval|test",
  "checkpoint": "epoch_0250|best|final",
  "epoch": 250,
  "evaluation_seed": 2024,
  "sample_count": 0,
  "test_horizons": [7, 15, 23, 31, 39, 47],
  "times_seconds": [0.8, 1.6, 2.4, 3.2, 4.0, 4.8],
  "num_k_samples": 20,
  "confidence_mc_samples": 1000,
  "metrics": {
    "ravg_percent": 0.0,
    "rmin_percent": 0.0,
    "reliability_by_horizon_percent": [],
    "s68_m2_per_s": 0.0,
    "s95_m2_per_s": 0.0,
    "minade20_m": 0.0,
    "minfde20_m": 0.0,
    "asaee_m_per_s": 0.0,
    "inference_ms_per_batch": 0.0,
    "inference_ms_per_sample": 0.0
  },
  "sharpness_formula_version": "official_unverified|verified_original|corrected",
  "evaluation_implementation_version": "..."
}
```

### 10.12 Per-sample table

Per-sample artifact phục vụ case selection:

```text
run_id
split
checkpoint
epoch
sample_id
source
ade_selected_horizons_m
fde_m
minade20_m
minfde20_m
nll
confidence_level_gt_mean
confidence_level_gt_final
uncertainty_area_68_final_m2
uncertainty_area_95_final_m2
mixture_entropy_final
selected_case_label
```

Parquet được ưu tiên để giữ kiểu dữ liệu; CSV có thể xuất thêm cho việc đọc thủ công.

## 11. Metric definitions và versioning

Mỗi metric phải gắn với:

- implementation version;
- danh sách horizon;
- Monte Carlo seed;
- số samples;
- aggregation rule;
- split;
- checkpoint.

Không được ghi chung `ADE` nếu đó là ADE trên sáu selected horizons. Tên báo cáo nên là:

```text
minADE20 over official selected horizons
minFDE20 at 4.8 s
```

Nếu bổ sung metric trên đủ 48 timestep, phải đặt tên khác và không thay thế metric official.

## 12. Bug-fix protocol

### 12.1 Được phép sửa

Các sửa đổi đã được cho phép về nguyên tắc:

- evaluation mixture softmax phải dùng component dimension;
- loại bỏ hoặc sửa dummy forward sai shape trong testing;
- GPU timing phải synchronize đúng;
- logger phải hoạt động khi logging tắt;
- resume phải restore scheduler và RNG states.

### 12.2 Điều kiện trước khi sửa

Mỗi bug cần có:

1. Test tái hiện lỗi trên code cũ.
2. Giải thích expected behavior.
3. Patch nhỏ nhất có thể.
4. Test chứng minh lỗi đã được sửa.
5. Ghi vào manifest/changelog của run.

### 12.3 Không được trộn với thay đổi thuật toán

Không tự ý thay:

- LSTM architecture;
- loss;
- optimizer;
- LR schedule;
- Gaussian count;
- preprocessing;
- sampling strategy của metric, trừ bug đã chứng minh;
- official test horizons.

## 13. Sharpness verification protocol

Không sửa sharpness trước khi hoàn thành các bước:

1. Viết test với confidence mask có diện tích grid biết trước.
2. Xác nhận `mesh_range_x/y` biểu diễn half-range hay full width.
3. So sánh số grid point và physical cell area.
4. Kiểm tra sai số discretization tại resolution 0.1 m.
5. Chạy official implementation trên một distribution đơn giản có diện tích confidence ellipse tính được.
6. Nếu có pretrained model, đối chiếu kết quả gần với paper.
7. Lập kết luận một trong ba trạng thái:
   - `official_unverified`;
   - `verified_original`;
   - `corrected`.

Nếu phải sửa, lưu cả official value và corrected value trong một validation run nhỏ để giải thích tác động. Baseline report phải nêu rõ phiên bản được dùng.

## 14. Visualization protocol

### 14.1 Loss curves

- Dùng `history.csv`.
- Vẽ train NLL và validation NLL cùng trục.
- Đánh dấu best epoch và checkpoint epochs.
- Không smooth đường chính; nếu thêm smoothed curve, phải giữ raw curve.

### 14.2 Metrics over epoch

- Dùng `metrics/history.csv`.
- Vẽ riêng metric có đơn vị/thang đo khác nhau.
- Reliability không ghép chung trục với displacement error nếu gây hiểu nhầm.

### 14.3 Past + ground truth + prediction

- Dùng ego coordinates.
- Past, ground truth và prediction có style cố định giữa các hình.
- Ghi rõ prediction là mixture mode, expected value hay sampled trajectory.
- Không gọi component mean là predicted trajectory nếu chưa định nghĩa aggregation.

### 14.4 GMM components

Tại timestep được chọn, hiển thị:

- mỗi `pi_k`;
- `mu_k`;
- covariance ellipse 68% và/hoặc 95%;
- ground-truth point;
- aggregate mixture contour nếu hữu ích.

Ellipse phải được tính từ eigenvalue/eigenvector của covariance và quantile đúng của chi-square hai chiều, không dùng quy tắc 1D một cách trực tiếp.

### 14.5 Uncertainty through time

- Dùng output thật tại nhiều horizons.
- Giữ cùng axis limits khi so sánh.
- Không giả định uncertainty tăng đơn điệu.
- Có thể hiển thị cả confidence region của mixture và covariance component.

### 14.6 Same sample over epochs

- Dùng fixed-sample artifacts.
- Cùng sample ID, input, target, axis limits và style.
- Panel đề xuất: epoch 1, 5, 10, một epoch giữa, best và final.

## 15. Case-selection protocol

Case selection chỉ thực hiện sau khi model/checkpoint đã được khóa.

### 15.1 Good case

Ứng viên có:

- displacement error thấp;
- ground truth nằm trong confidence region hợp lý;
- uncertainty không quá rộng.

### 15.2 Difficult/uncertain case

Ứng viên có một hoặc nhiều đặc điểm:

- mixture entropy cao;
- nhiều component có weight đáng kể;
- confidence area lớn;
- ground truth vẫn được distribution bao phủ.

### 15.3 Failure case

Ứng viên có:

- displacement error cao;
- ground truth ngoài confidence region mong đợi;
- model overconfident nhưng sai;
- hoặc distribution đặt mass vào hướng không phù hợp.

Không chọn chỉ bằng quan sát hình. Trước tiên lọc bằng per-sample table, sau đó kiểm tra định tính. Lưu cả tiêu chí và sample ID.

## 16. Inference timing protocol

Để đo GPU đúng:

1. Warm-up nhiều iteration.
2. Đồng bộ CUDA trước khi bắt đầu.
3. Đồng bộ CUDA sau inference.
4. Đo nhiều iteration.
5. Báo cáo median và percentile, không chỉ mean.
6. Ghi batch size, dtype, device và bao gồm/không bao gồm post-processing.

Timing model và post-processing phải được báo cáo riêng, tương tự cách paper phân tách.

## 17. Preflight smoke test

Trước full training, chạy một smoke test tách biệt với baseline chính:

- số samples nhỏ;
- 2-3 epoch;
- ít nhất một periodic checkpoint;
- một full metric evaluation nhỏ;
- đủ 8 fixed-sample artifacts;
- resume từ `last.pt` ít nhất một epoch;
- tạo thử cả sáu nhóm visualization.

Smoke test đạt khi:

- mọi artifact đọc lại được;
- shape/schema đúng;
- loss hữu hạn;
- checkpoint resume đúng epoch/LR;
- fixed inputs có checksum không đổi;
- decoded covariance positive definite trong tolerance;
- `sum(pi)=1` trên component dimension;
- figure có thể tái tạo mà không gọi lại training.

Smoke-test artifacts phải nằm ở run riêng và không được dùng làm baseline result.

## 18. Full-run acceptance criteria

Một baseline run chỉ được đánh dấu `completed` khi:

1. Hoàn tất 2500 epoch hoặc có lý do dừng được ghi rõ.
2. `history.csv` có row liên tục cho mọi epoch đã chạy.
3. Có checkpoint 1, 5, 10, mỗi 100 epoch, best và final.
4. Có metric snapshots theo schedule.
5. Có prediction cho đủ 8 fixed samples tại mọi checkpoint yêu cầu.
6. Best checkpoint được xác định đúng theo validation NLL.
7. Test evaluation chỉ chạy sau model selection.
8. Không có NaN/Inf không được giải thích.
9. Manifest ở trạng thái `completed` và có hashes.
10. Sáu nhóm visualization có thể sinh từ artifact.

## 19. Thứ tự triển khai sau khi protocol được duyệt

1. Audit dataset IMPTC thực tế.
2. Viết test tái hiện các bug đã phát hiện.
3. Sửa bug theo protocol.
4. Thêm seed và run manifest.
5. Thêm structured history.
6. Thêm checkpoint/resume schema.
7. Thêm fixed-sample manifest và prediction capture.
8. Thêm metric snapshot/per-sample artifacts.
9. Kiểm chứng sharpness.
10. Chạy preflight smoke test.
11. Review toàn bộ artifact.
12. Chỉ sau đó mới chạy full baseline.

## 20. Những việc chưa được protocol này cho phép

- Chưa cho phép bắt đầu full training.
- Chưa cho phép thay architecture hoặc loss.
- Chưa chốt sửa công thức sharpness.
- Chưa cho phép dùng test set để chọn model.
- Chưa cho phép thêm context, map hoặc social interaction vào baseline.
- Chưa cho phép gọi kết quả là paper-exact reproduction nếu các khác biệt chưa được giải quyết và báo cáo.
