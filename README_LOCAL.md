# Thiết lập cục bộ cho MDN Trajectory Forecasting

Tài liệu này hướng dẫn chạy lại bài nộp của nhóm và không thay đổi `README.md` gốc.

**Phạm vi của bài nộp:** dữ liệu IMPTC, baseline LSTM-MDN với K = 3 (`base_mdn/`) và cải tiến sparsemax với K_max = 8 (`sparsemax_mdn/`). 

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

Không cần cài CUDA Toolkit (`nvcc`) riêng vì PyTorch wheel mang theo CUDA runtime. Máy vẫn cần NVIDIA driver tương thích. Các lệnh dưới đây chạy được trên CPU bằng cách dùng `--gpu -1` (chậm hơn nhiều).

## 2. Dữ liệu và kết quả đi kèm

Thư mục `data/` và `results/` **không nằm trong git** (đã `.gitignore`); chúng được nộp riêng và phải đặt ở thư mục gốc của repo.

**Dữ liệu IMPTC** (mặc định đọc từ `data/trajdata` bên trong repo):

```text
data/trajdata/ego/imptc/train/ego_samples.pkl
data/trajdata/ego/imptc/eval/ego_samples.pkl
data/trajdata/ego/imptc/test/ego_samples.pkl
```

**Kết quả đã huấn luyện**, cần có để chạy lại phân tích và sinh hình:

| Thư mục | Nội dung |
|---|---|
| `results/trained_models/base_mdn/imptc/default_peds_imptc/` | Baseline K = 3, run `imptc_baseline_seed2024` |
| `results/trained_models/sparsemax_mdn/imptc/sparsemax_k8_peds_imptc/` | Sparsemax K_max = 8, run `sparsemax_k8_v2_seed2024` |
| `results/trained_models/sparsemax_mdn/baseline_persample/K3/` | Số liệu từng mẫu của baseline K = 3 trên tập test |
| `results/ablations/imptc_num_gaussians/ablation.json` | Số liệu test của baseline K = 3 được tài liệu trích dẫn |

Có thể đặt dữ liệu và kết quả ở nơi khác mà không sửa JSON:

```bash
export MDN_DATA_ROOT=/duong/dan/toi/trajdata
export MDN_RESULT_ROOT=/duong/dan/toi/trained_models
```

## 3. Kiểm chứng nhanh (không cần dữ liệu hay GPU)

```bash
# 67 unit test của sparsemax_mdn
.venv/bin/python -m unittest discover -s sparsemax_mdn/tests -t .
# base_mdn không bị sửa so với bản chụp hash
.venv/bin/python -m sparsemax_mdn.base_hashes --check
```

## 4. Baseline K = 3 (`base_mdn/`)

Các lệnh chạy từ thư mục gốc repo. Cấu hình baseline là `base_mdn/configs/imptc/default_peds_imptc.json` (K = 3, LSTM hidden 8, 2500 epoch, seed 2024). Run `imptc_baseline_seed2024` **đã có sẵn trong `results/`**, không cần huấn luyện lại; chỉ chạy lại khi muốn tái lập, và phải dùng run ID mới vì run cũ đã tồn tại.

Smoke test nhỏ (chỉ kiểm tra kỹ thuật, không dùng metric của nó làm kết quả):

```bash
MDN_RUN_ID=smoke_protocol_v1 \
  .venv/bin/python base_mdn/train.py \
  --target imptc --configs smoke_peds_imptc.json --gpu 0 --log --print
```

Huấn luyện đầy đủ (chỉ khi cần tái lập):

```bash
MDN_RUN_ID=imptc_baseline_rerun_seed2024 \
  .venv/bin/python base_mdn/train.py \
  --target imptc --configs default_peds_imptc.json --gpu 0 --log --print
```

Đánh giá trên tập test bằng checkpoint `best.pt` (chọn theo validation NLL). Khi config có nhiều run, chỉ rõ run để tránh chọn nhầm:

```bash
.venv/bin/python base_mdn/testing.py \
  --target imptc --configs default_peds_imptc.json --gpu 0 \
  --run-id imptc_baseline_seed2024 --log --print
```

Chỉ dùng `--checkpoint /duong/dan/checkpoint.pt` khi chủ động muốn đánh giá một checkpoint khác như `final.pt`. Giá trị này ghi đè `--run-id` và `MDN_RUN_ID`.

Resume: giữ nguyên `MDN_RUN_ID` và file config ban đầu, rồi chỉ rõ checkpoint (có `MDN_RESUME_CHECKPOINT` là đủ để bật chế độ resume):

```bash
MDN_RUN_ID=imptc_baseline_rerun_seed2024 \
MDN_RESUME_CHECKPOINT="$PWD/results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_rerun_seed2024/checkpoints/last.pt" \
  .venv/bin/python base_mdn/train.py \
  --target imptc --configs default_peds_imptc.json --gpu 0 --log --print
```

Resume tiếp tục từ epoch hoàn tất gần nhất và khôi phục model, optimizer, scheduler, RNG, best validation state, history và thứ tự dữ liệu train; epoch dở dang chạy lại từ đầu.

Vẽ đồ thị NLL train/validation của baseline (chỉ đọc `history.csv`, không train hay đánh giá):

```bash
.venv/bin/python base_mdn/plot_training_history.py
```

Mỗi run có artifact riêng:

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

Các file prediction `.npz` lưu cả đầu ra thô và các tham số đã giải mã: `pi`, `mu`, `sigma`, `rho`, `covariance`.

## 5. Sparsemax K_max = 8 (`sparsemax_mdn/`)

Cải tiến chỉ thay softmax của trọng số π bằng sparsemax (K_max = 8); mọi thứ khác giữ như baseline. Cấu hình mặc định là `sparsemax_mdn/configs/imptc/sparsemax_k8_peds_imptc.json`; run đã huấn luyện là `sparsemax_k8_v2_seed2024`. Mô tả chi tiết: [sparsemax_mdn/README.md](sparsemax_mdn/README.md).

```bash
# smoke (CPU) rồi review; --full chỉ chạy được khi reports/SMOKE.json có status passed
.venv/bin/python -m sparsemax_mdn.train --smoke --gpu -1 --run-id sparsemax_smoke_seed2024
.venv/bin/python -m sparsemax_mdn.review_smoke

# huấn luyện đầy đủ (chỉ khi cần tái lập; dùng run ID mới)
.venv/bin/python -m sparsemax_mdn.train --full --gpu 0 --run-id sparsemax_k8_rerun_seed2024

# phân tích K(x, t) trên validation (không dùng test)
.venv/bin/python -m sparsemax_mdn.analyze_k --run-id sparsemax_k8_v2_seed2024 \
  --config sparsemax_k8_peds_imptc --gpu -1

# đánh giá validation của một checkpoint
.venv/bin/python -m sparsemax_mdn.evaluate --run-id sparsemax_k8_v2_seed2024 \
  --config sparsemax_k8_peds_imptc --split validation --limit 500 --gpu -1
```

Đánh giá trên **tập test chỉ được chạy một lần cho mỗi run** (cần `--confirm-test-once`; file `test_evaluation_started.lock` được tạo trước khi đọc dữ liệu và chặn lần chạy thứ hai). Run `sparsemax_k8_v2_seed2024` đã được đánh giá test một lần, kết quả ở `results/trained_models/sparsemax_mdn/imptc/sparsemax_k8_peds_imptc/runs/sparsemax_k8_v2_seed2024/evaluation/`. Không chạy lại lệnh này trên run đó:

```bash
.venv/bin/python -m sparsemax_mdn.evaluate --run-id ID --config CONFIG --split test --confirm-test-once --gpu 0
```

Số liệu sharpness từng mẫu của baseline K = 3 (dùng cùng bộ đánh giá để so sánh đuôi với sparsemax) đã có ở `results/trained_models/sparsemax_mdn/baseline_persample/K3/`. Lệnh tạo trên validation (test cần `--confirm-test-baseline` và cũng chỉ chạy một lần cho mỗi nhãn):

```bash
.venv/bin/python -m sparsemax_mdn.baseline_persample \
  --baseline-run results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024 \
  --label K3 --split validation --limit 300 --gpu 0
```

## 6. Công thức MDN

Cả baseline và sparsemax dùng công thức legacy cho tham số của mỗi Gaussian hai chiều:

```text
sigma = exp(raw_sigma),   rho = tanh(raw_rho)
Sigma = [[sigma_x^2,                rho * sigma_x * sigma_y],
         [rho * sigma_x * sigma_y,  sigma_y^2             ]]
```

Đầu ra thô của mạng có dạng `[B, 48, 6K]`, chia thành sáu khối theo thứ tự `mu_x, mu_y, log sigma_x, log sigma_y, rho_pre, pi_logit`, mỗi khối K số. Baseline chuẩn hóa `pi` bằng softmax; sparsemax thay bằng sparsemax. Mỗi bước dự báo là một hỗn hợp Gaussian hai chiều riêng; code không định nghĩa một hỗn hợp chung cho cả quỹ đạo.
