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
