# Thiết lập cục bộ cho MDN Trajectory Forecasting

Tài liệu này dành cho môi trường của nhóm và không thay đổi `README.md` gốc.

## 1. Môi trường Python

Repo sử dụng virtual environment tại `.venv` với Python 3.12 và PyTorch có CUDA.

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Cài lại môi trường khi cần:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-local.txt
```

Không cần cài CUDA Toolkit (`nvcc`) riêng vì PyTorch wheel mang theo CUDA runtime. Máy vẫn cần NVIDIA driver tương thích.

## 2. Bố trí dữ liệu

Mặc định, dữ liệu được đọc từ `data/trajdata` bên trong repo. Ví dụ IMPTC:

```text
data/trajdata/ego/imptc/train/ego_samples.pkl
data/trajdata/ego/imptc/eval/ego_samples.pkl
data/trajdata/ego/imptc/test/ego_samples.pkl
```

Có thể đặt dữ liệu ở nơi khác mà không sửa JSON:

```bash
export MDN_DATA_ROOT=/duong/dan/toi/trajdata
```

Kết quả mặc định được ghi vào `results/trained_models`. Có thể đổi bằng:

```bash
export MDN_RESULT_ROOT=/duong/dan/toi/trained_models
```

Nếu bật trực quan hóa trên bản đồ, map mặc định nằm dưới `data/imptc` và
`data/ind`. Có thể đặt thư mục gốc khác bằng `MDN_MAP_ROOT`.

## 3. Chạy train và test

Các lệnh có thể chạy từ bất kỳ thư mục hiện hành nào:

```bash
source .venv/bin/activate
python base_mdn/train.py --target imptc --configs default_peds_imptc.json --gpu 0 --log --print
python base_mdn/testing.py --target imptc --configs default_peds_imptc.json --gpu 0 --log --print
```

Khi test, pipeline mặc định dùng checkpoint `best.pt` được chọn theo validation
NLL. Nếu config chỉ có một run thì run đó được nhận diện tự động. Khi có nhiều
run, chỉ rõ run để tránh chọn nhầm:

```bash
.venv/bin/python base_mdn/testing.py \
  --target imptc --configs default_peds_imptc.json --gpu 0 \
  --run-id imptc_baseline_seed2024 --log --print
```

Chỉ dùng `--checkpoint /duong/dan/checkpoint.pt` khi chủ động muốn đánh giá một
checkpoint khác như `final.pt` hoặc checkpoint legacy. Giá trị này ghi đè
`--run-id` và `MDN_RUN_ID`.

Trước khi train, cần đặt đủ các file `ego_samples.pkl` đúng cấu trúc hoặc khai báo `MDN_DATA_ROOT`.

## 4. Kiểm chứng trước khi train baseline

Chạy unit tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Chạy smoke test cho checkpoint, history và output của 8 fixed samples:

```bash
MDN_RUN_ID=smoke_protocol_v1 \
  .venv/bin/python base_mdn/train.py \
  --target imptc --configs smoke_peds_imptc.json --gpu 0 --log --print
```

Chạy smoke test cực nhỏ cho toàn bộ metric pipeline:

```bash
MDN_RUN_ID=smoke_metrics_v1 \
  .venv/bin/python base_mdn/train.py \
  --target imptc --configs metric_smoke_peds_imptc.json --gpu 0 --log --print
```

Hai file `smoke_*.json` chỉ dùng để kiểm tra kỹ thuật. Không dùng metric từ các
run này làm kết quả thực nghiệm hoặc so sánh với paper.

Mỗi run có artifact riêng tại:

```text
results/trained_models/base_mdn/imptc/<config>/runs/<run_id>/
├── run_manifest.json
├── resolved_config.json
├── environment.json
├── history.csv
├── checkpoints/
├── metrics/
└── fixed_samples/
    ├── inputs.npz
    └── predictions/
```

Các prediction `.npz` lưu cả output thô và các tham số đã decode:
`pi`, `mu`, `sigma`, `rho`, `covariance`.

## 5. Baseline chính thức và resume

Không chạy lệnh dưới đây trước khi nhóm duyệt artifact smoke, dung lượng đĩa và
thời gian dự kiến. Config baseline vẫn là `default_peds_imptc.json`.

```bash
MDN_RUN_ID=imptc_baseline_seed2024 \
  .venv/bin/python base_mdn/train.py \
  --target imptc --configs default_peds_imptc.json --gpu 0 --log --print
```

Khi resume, giữ nguyên cả `MDN_RUN_ID` và file config ban đầu, rồi chỉ rõ
checkpoint. Có `MDN_RESUME_CHECKPOINT` là đủ để bật chế độ resume; không cần sửa
`resume_training` trong JSON:

```bash
MDN_RUN_ID=imptc_baseline_seed2024 \
MDN_RESUME_CHECKPOINT="$PWD/results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024/checkpoints/last.pt" \
  .venv/bin/python base_mdn/train.py \
  --target imptc --configs default_peds_imptc.json --gpu 0 --log --print
```

Resume tiếp tục từ epoch hoàn tất gần nhất và khôi phục model, Adam optimizer,
learning-rate scheduler, RNG, best validation state, history và thứ tự train data.
Nếu tiến trình bị ngắt giữa epoch, epoch dở dang sẽ chạy lại từ đầu. `training.log`
được nối tiếp thay vì bị xóa.

Checkpoint cũ được tạo trước bản sửa resume không có `data_loader_state` vẫn có
thể dùng, nhưng chương trình sẽ cảnh báo rằng thứ tự dữ liệu không thể được tái
lập chính xác. Checkpoint của full baseline mới sẽ có trạng thái này.

## 6. Tài liệu kiểm chứng

- Dataset audit: `reports/data_audit/IMPTC_DATA_AUDIT.md`
- Sharpness verification: `reports/bug_fixes/SHARPNESS_VERIFICATION.md`
- Instrumentation verification: `reports/bug_fixes/PIPELINE_INSTRUMENTATION_VERIFICATION.md`
- Experimental protocol và artifact schema: `EXPERIMENT_PROTOCOL.md`

## 7. Sinh bộ hình sau training

Script chỉ đọc artifact đã lưu, không train hoặc chạy model lại:

```bash
.venv/bin/python base_mdn/visualize_experiment.py
```

Mặc định hình được ghi vào thư mục `figures/` của run
`imptc_baseline_seed2024`. Có thể chọn run, thư mục đầu ra, fixed sample và
future timestep khác:

```bash
.venv/bin/python base_mdn/visualize_experiment.py \
  --run-dir /duong/dan/toi/run \
  --output-dir /duong/dan/toi/figures \
  --sample-id 'eval:imptc_0_00093_00225:4103' \
  --future-step 48
```

Xem toàn bộ tùy chọn:

```bash
.venv/bin/python base_mdn/visualize_experiment.py --help
```

## 8. So sánh validation-best M=1 và M=3

Sau khi cả hai run hoàn tất, đánh giá hai `best.pt` trên toàn bộ IMPTC test set:

```bash
.venv/bin/python base_mdn/compare_m1_m3.py --gpu 0
```

Script xác minh hai resolved config chỉ khác `model_params.num_gaussians`, đặt
lại cùng evaluation seed trước mỗi model và lưu kết quả tại:

```text
results/comparisons/imptc_m1_vs_m3/
├── comparison.json
├── comparison.csv
├── config_difference.json
├── metric_comparison.png
├── m1/
└── m3/
```

Có thể dùng `--m1-run`, `--m3-run` và `--output-dir` để chọn artifact khác.

## 9. Tạo Demo 2 — uncertainty theo forecast horizon

Script chỉ đọc prediction thật đã lưu tại `best.pt` của 8 fixed samples, không
train hoặc chạy model lại:

```bash
.venv/bin/python base_mdn/create_demo2_horizon_uncertainty.py
```

Mặc định script vẽ bốn mốc `+1.0`, `+2.0`, `+3.0`, `+4.8` giây, dùng toàn bộ
GMM `M=3`, vùng 68%/95% theo đúng Monte Carlo confidence-set implementation và
lưu PNG/PDF/SVG cùng bảng diện tích tại:

```text
results/trained_models/base_mdn/imptc/default_peds_imptc/
runs/imptc_baseline_seed2024/figures/demo2/
```

Trên từng panel, `A68(t)`/`A95(t)` là diện tích của sample và horizon đang vẽ,
không phải global sharpness score `S68`/`S95`.

## 10. Tạo Demo 1 — M=1 so với M=3 trên cùng một mẫu

Mục 8 cho ra **số** (biểu đồ cột metric). Demo 1 cho thấy trên cùng một
trajectory rằng `M=1` tạo một vùng Gaussian, còn `M=3` có thể tạo vùng mật độ
khác hình dạng và tập trung hơn. Không suy ra hành vi của toàn bộ test set từ
một hình minh họa.

Script chỉ đọc prediction đã lưu tại `best.pt` của 8 fixed samples, không train
hay chạy lại model:

```bash
.venv/bin/python base_mdn/create_demo1_m1_vs_m3.py
```

Tám mẫu này thuộc tập **validation**. Hai panel dùng đúng cùng một mẫu và một
future timestep. Hình là ví dụ minh họa; các metric định lượng trên toàn bộ
test set nằm ở mục 8.

Cách chọn mẫu theo đúng `EXPERIMENT_PROTOCOL.md` mục 15.2 (difficult/uncertain
case), **không chọn bằng mắt**:

- mixture entropy cao → `perplexity = exp(H)`;
- ít nhất **hai** thành phần có trọng số `>= --min-weight` (mặc định 0.15);
- các thành phần **tách nhau hơn 2 sigma** — đây là ngưỡng để hỗn hợp hai
  Gaussian trọng số bằng nhau thực sự có hai đỉnh, nên nó là điều kiện cứng
  chứ không phải một số hạng mềm.
- Trong các mẫu đạt tiêu chí trên, chỉ chọn mẫu có diện tích vùng 95% của
  `M=1` lớn hơn `M=3` ít nhất 5%. Nếu không có mẫu nào như vậy, script dừng
  thay vì tạo hình minh họa trái với kết quả thực tế.

Toàn bộ bảng điểm per-sample được ghi ra đĩa để lựa chọn tái tạo được.

Tuỳ chọn hay dùng:

```bash
.venv/bin/python base_mdn/create_demo1_m1_vs_m3.py \
  --sample-id <id> \        # ép một mẫu cụ thể
  --time 3.0 \              # cố định mốc thời gian, mặc định quét cả horizon
  --min-weight 0.10         # nới ngưỡng nếu không mẫu nào đạt
```

Kết quả tại `results/comparisons/imptc_m1_vs_m3/demo1/`:

```text
demo1_m1_vs_m3.{png,pdf,svg}
demo1_selection.json      # tiêu chí, mẫu được chọn, A68/A95, tỉ lệ A95 M1/M3
demo1_candidates.csv      # điểm của cả 8 mẫu
```

Hai panel dùng **cùng giới hạn trục** và **cùng seed Monte Carlo**, nếu không thì
chênh lệch diện tích không đọc được từ hình và có thể do nhiễu lấy mẫu.
Vùng tin cậy của hình được đo trên một lưới cục bộ đủ mịn để thấy Gaussian
rộng vài centimet; độ phân giải này chỉ phục vụ hình và không thay đổi model.
`A68`/`A95` là diện tích của một sample tại một timestep, không phải metric
`S68`/`S95` tổng hợp của test set.

Script cảnh báo khi:

- không mẫu nào đạt tiêu chí → dừng kèm hướng dẫn, thay vì vẽ hai panel giống nhau;
- vùng tin cậy **chạm biên mesh** → diện tích báo cáo chỉ là chặn dưới;
- `A95(M=1) / A95(M=3) < 1.05` → hình sẽ không thuyết phục, nên đổi mẫu.

## 11. Ablation số thành phần hỗn hợp M

Paper viết `we set the number of Gaussians to three` cho mọi dataset, nhưng config
chính thức dùng `M=3` cho IMPTC và `M=5` cho ETH/UCY, và **không báo cáo ablation
nào**. Mục này đo đúng đại lượng đó.

Các config `m2`, `m5`, `m8` chỉ khác `default_peds_imptc.json` ở `num_gaussians`.

Train ba giá trị còn thiếu (`M=1` và `M=3` đã có từ mục 5 và 8):

```bash
MDN_RUN_ID=imptc_m2_seed2024 .venv/bin/python base_mdn/train.py \
  --target imptc --configs m2_peds_imptc.json --gpu 0 --log --print
MDN_RUN_ID=imptc_m5_seed2024 .venv/bin/python base_mdn/train.py \
  --target imptc --configs m5_peds_imptc.json --gpu 0 --log --print
MDN_RUN_ID=imptc_m8_seed2024 .venv/bin/python base_mdn/train.py \
  --target imptc --configs m8_peds_imptc.json --gpu 0 --log --print
```

Đánh giá toàn bộ sweep và vẽ đường cong:

```bash
.venv/bin/python base_mdn/run_m_ablation.py --gpu 0
```

Script xác minh mọi run **chỉ khác `num_gaussians`** trước khi so sánh, đặt lại
cùng evaluation seed trước mỗi model, và lưu tại
`results/ablations/imptc_num_gaussians/`:

```text
num_gaussians_ablation.{png,pdf,svg}
ablation.json
ablation.csv
```

Vẽ đường cong một phần khi chưa train đủ:

```bash
.venv/bin/python base_mdn/run_m_ablation.py --gpu 0 --only 1 3 --skip-missing
```

Hình gồm hai tầng: trên là `R_avg`/`R_min` kèm hai ngưỡng 95% và 90% và vạch đánh
dấu `M=3` của config chính thức; dưới là `S95` và `minADE20` để thấy đánh đổi khi
tăng `M`.

Mỗi cấu hình chỉ chạy **một seed**, nên đọc **xu hướng** chứ không đọc con số
tuyệt đối. Nếu còn thời gian, chạy thêm seed rồi lấy trung bình.
## 12. Tạo Demo 3 — calibration plot

Script dùng `best.pt` của baseline IMPTC `M=3`, chạy đúng confidence-level
Monte Carlo của repository trên toàn bộ test set và không train lại model:

```bash
.venv/bin/python base_mdn/create_demo3_calibration.py --gpu 0
```

Kết quả được lưu trong `figures/demo3/` của run baseline, gồm PNG/PDF/SVG,
confidence levels, hai CSV, `summary.json` và `demo3_notes.md`. Các đường biểu
diễn các horizon `+0.8s` đến `+4.8s`; hình ghi `Ravg` và `Rmin` theo đúng cách
binning của evaluator gốc.

Chỉ để smoke test kỹ thuật, không dùng kết quả này trên slide:

```bash
.venv/bin/python base_mdn/create_demo3_calibration.py \
  --max-test-samples 64 --output-dir /tmp/demo3-smoke
```
