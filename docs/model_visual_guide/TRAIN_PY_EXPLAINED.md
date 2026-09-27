# Đọc hiểu `base_mdn/train.py`

## Vai trò của file

`train.py` là entry point lắp các thành phần đã học lại với nhau:

```text
ConfigLoader
+ DataLoader
+ LSTM_Trajectory_Forecast
+ NLL_MDN_loss
+ Adam
+ LinearLR scheduler
+ ExperimentTracker
+ MDN_Trainer
```

File này chủ yếu chuẩn bị mọi thành phần. Vòng lặp qua epoch và batch nằm trong `base_mdn/mdn.py`.

Luồng tổng thể:

```text
Đọc CLI arguments
-> chọn JSON config
-> tạo ConfigLoader
-> gọi training(cfg, gpu_id)
-> chọn GPU và đặt seed
-> load train/eval data
-> tạo logger
-> tạo model, Adam, scheduler, loss và tracker
-> nếu resume: khôi phục checkpoint
-> tạo MDN_Trainer
-> trainer.train()
-> lưu final checkpoint
```

## 1. Các import chuẩn

```python
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import logging
import sys
```

- `os`: biến môi trường, đường dẫn và kiểm tra file;
- `torch`: chọn device và model tensor;
- `torch.optim`: Adam optimizer;
- `torch.optim.lr_scheduler`: learning-rate scheduler;
- `logging`: ghi `training.log`;
- `sys`: kết thúc chương trình.

## 2. Import các thành phần của repository

```python
from base_lstm import LSTM_Trajectory_Forecast, NLL_MDN_loss
from mdn import MDN_Trainer
from utils.config_loader import ConfigLoader
from utils.helper import config_parser, count_model_parameters
from utils.data_loader import DataLoader
from utils.experiment import ExperimentTracker, restore_checkpoint, set_global_seed
from termcolor import colored
```

Vai trò:

```text
LSTM_Trajectory_Forecast -> LSTM + Linear
NLL_MDN_loss             -> GMM negative log-likelihood
MDN_Trainer              -> vòng epoch/batch thật sự
ConfigLoader             -> đọc JSON và tạo đường dẫn
config_parser            -> đọc command-line arguments
count_model_parameters   -> đếm trainable parameters
DataLoader               -> load X, y và metadata
ExperimentTracker        -> history, fixed samples, checkpoint, metrics
restore_checkpoint       -> khôi phục trạng thái run
set_global_seed          -> đặt Python/NumPy/PyTorch seeds
colored                  -> màu chữ terminal
```

## 3. Hàm `training()`

```python
def training(cfg, gpu_id):
```

Hàm nhận:

```text
cfg    -> đối tượng ConfigLoader đã đọc JSON
gpu_id -> GPU do CLI chọn
```

Mỗi lần gọi `training()` xử lý một config.

## 4. In thông báo bắt đầu

```python
print(colored(
    f"Starting training: {cfg.name} on GPU: {gpu_id}",
    'green'
))
```

Chỉ phục vụ terminal, không ảnh hưởng model.

## 5. Chọn GPU hiển thị cho CUDA

```python
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
```

Ví dụ `gpu_id="1"` làm process chỉ nhìn thấy GPU vật lý số 1. Bên trong process, GPU đó thường trở thành CUDA device đầu tiên.

Comment nhắc rằng biến này phải được đặt trước CUDA API call đầu tiên. `torch` đã được import, nhưng việc import riêng nó chưa nhất thiết khởi tạo CUDA context; dòng này được đặt trước `torch.cuda.is_available()` và trước khi chuyển model lên GPU.

## 6. Đặt random seed

```python
seed = int(cfg.experiment_params.get('seed', 2024))
set_global_seed(seed)
```

Config hiện tại có:

```json
"seed": 2024
```

Nếu config không khai báo, fallback cũng là 2024.

`set_global_seed()` đặt:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Seed ảnh hưởng:

- Xavier initialization của model;
- shuffle và random reduction của DataLoader;
- NumPy random operations;
- PyTorch sampling;
- CUDA random state nếu có GPU.

Đặt seed trước khi tạo model là cần thiết để initial weights có thể tái lập.

Seed giúp reproducibility nhưng không tự động bảo đảm mọi CUDA operation luôn bit-for-bit deterministic trên mọi phần cứng/thư viện.

## 7. Kiểm tra yêu cầu resume

```python
resume_path = os.environ.get('MDN_RESUME_CHECKPOINT')
resume_requested = (
    bool(resume_path)
    or cfg.train_params['resume_training']
)
```

Có hai cách yêu cầu resume.

### Qua biến môi trường

```bash
MDN_RESUME_CHECKPOINT=/path/to/checkpoint.pt
```

### Qua JSON

```json
"resume_training": true
```

Nếu một trong hai được bật, `resume_requested=True`.

Đường dẫn trong biến môi trường được ưu tiên nếu có.

## 8. Tạo DataLoader

```python
data_loader = DataLoader(cfg=cfg)
```

Ở dòng này custom DataLoader mới lưu config và khởi tạo các container. Dữ liệu pickle chưa được đọc cho tới hai dòng tiếp theo.

## 9. Load train và eval data

```python
data_loader.load_train_data()
data_loader.load_eval_data()
```

Training chỉ load:

```text
train split
eval/validation split
```

Nó không load test split. Test data chỉ nên dùng sau training trong `testing.py`.

Sau khi load:

```text
train X [189595,32,4]
train y [189595,48,2]

eval X  [19148,32,4]
eval y  [19148,48,2]
```

Reduction 50% chưa xảy ra ở đây. Nó xảy ra mỗi khi trainer gọi `get_train_data()` hoặc `get_eval_data()`.

## 10. Chuẩn bị logger

```python
train_logger = None
log_file_handler = None
```

Ban đầu chưa có logger hoặc file handler.

Nếu:

```python
if cfg.with_log:
```

thì log file là:

```python
log_file_path = os.path.join(
    cfg.evaluation_path,
    'training.log'
)
```

Tức nằm dưới:

```text
<run result>/evaluation/training.log
```

## 11. Tạo logger

```python
train_logger = logging.getLogger('training')
train_logger.setLevel(logging.INFO)
```

Logger có tên `training` và ghi từ mức `INFO` trở lên.

File mode:

```python
logging.FileHandler(
    log_file_path,
    mode='a' if resume_requested else 'w'
)
```

Ý nghĩa:

```text
resume -> append vào log cũ
train mới -> ghi lại file từ đầu
```

Format:

```python
'%(asctime)s - %(message)s'
```

Mỗi dòng log có timestamp.

Nếu logging bị tắt:

```python
log_file_path = None
```

## 12. Trạng thái bắt đầu

```python
epoch = 1
history = []
```

Train mới bắt đầu ở epoch 1 và chưa có lịch sử.

Nếu resume, hai giá trị này sẽ được thay bằng checkpoint state.

## 13. Chọn device

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Nếu CUDA khả dụng:

```text
device = cuda
```

Nếu không:

```text
device = cpu
```

`CUDA_VISIBLE_DEVICES` quyết định GPU vật lý nào có thể nhìn thấy; dòng này quyết định dùng CUDA hay CPU.

## 14. Tạo model

```python
model = LSTM_Trajectory_Forecast(
    cfg=cfg.model_params
).to(device)
```

Các bước:

```text
cfg.model_params
-> tạo LSTM hidden size 8
-> tạo Linear 8 -> 864
-> Xavier initialize
-> chuyển parameters sang CPU/GPU đã chọn
```

Model baseline có 8.224 trainable parameters.

## 15. Tạo Adam optimizer

```python
optimizer = optim.Adam(
    params=model.parameters(),
    lr=cfg.train_params['lr_default']
)
```

Config:

```text
lr_default = 1e-3 = 0,001
```

`model.parameters()` gồm:

- LSTM input-to-hidden weights;
- LSTM hidden-to-hidden weights;
- LSTM biases;
- Linear weights;
- Linear bias.

Adam giữ thêm optimizer state cho mỗi parameter, điển hình là moving averages của gradient và squared gradient.

Mỗi batch trong `mdn.py` sẽ gọi:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 16. Tạo LinearLR scheduler

```python
scheduler = lr_scheduler.LinearLR(
    optimizer,
    start_factor=cfg.train_params['lr_start_factor'],
    end_factor=cfg.train_params['lr_end_factor'],
    total_iters=cfg.train_params['train_epochs']
)
```

Config:

```text
base learning rate = 1e-3
start_factor       = 1,0
end_factor         = 0,0001
total_iters        = 2500
```

Learning rate đi từ khoảng:

```text
1e-3 x 1,0 = 1e-3
```

đến:

```text
1e-3 x 0,0001 = 1e-7
```

theo lịch tuyến tính qua 2.500 scheduler steps.

Trong `mdn.py`, `scheduler.step()` được gọi một lần sau phần train batches của mỗi epoch. Vì vậy scheduler thay đổi learning rate theo epoch, không theo batch.

## 17. Đóng gói loss function

```python
loss_fn = lambda output, target: NLL_MDN_loss(
    output=output,
    target=target,
    num_gaussians=cfg.model_params['num_gaussians']
)
```

`NLL_MDN_loss` cần ba đối số, nhưng trainer chỉ muốn gọi:

```python
self.loss_fn(outputs, targets)
```

Lambda giữ cố định:

```text
num_gaussians = 3
```

theo active config.

Loss vẫn là:

```text
-mean log p(y | X)
```

Không có ADE, FDE, reliability hoặc sharpness trong training loss.

## 18. Tạo ExperimentTracker

```python
tracker = ExperimentTracker(
    cfg=cfg,
    data_loader=data_loader,
    device=device
)
```

Tracker là instrumentation được thêm để bảo tồn artifact mà không thay đổi kiến trúc/loss baseline.

Nó chuẩn bị các nội dung như:

- run identifier;
- resolved config;
- environment information;
- event log;
- epoch history;
- fixed validation samples;
- prediction artifacts;
- best/last/periodic/final checkpoint;
- metric snapshots.

Tracker được tạo sau khi train/eval data đã load, vì nó cần full eval arrays và sample keys để chuẩn bị fixed samples.

## 19. Nhánh resume

```python
if resume_requested:
```

Nếu có path từ biến môi trường, dùng path đó. Nếu không, fallback:

```python
resume_path = resume_path or os.path.join(
    cfg.checkpoint_path,
    "model_final.pt"
)
```

Vậy `resume_training=true` mà không chỉ path sẽ tìm legacy final checkpoint tại:

```text
<checkpoint_path>/model_final.pt
```

## 20. Kiểm tra checkpoint tồn tại

```python
if os.path.exists(resume_path):
```

Chỉ khi file có thật, code mới restore.

Nếu không tồn tại, code ghi lỗi và:

```python
return -1
```

Training không tự bắt đầu lại từ đầu khi người dùng đã yêu cầu resume nhưng checkpoint bị thiếu.

## 21. Restore checkpoint

```python
checkpoint = restore_checkpoint(
    resume_path,
    model,
    optimizer,
    scheduler,
    map_location=device
)
```

`restore_checkpoint()` khôi phục:

```text
model weights
Adam optimizer state
LinearLR scheduler state, nếu có
RNG state, nếu có
```

`map_location=device` đưa serialized tensors về active CPU/GPU.

Khôi phục optimizer state quan trọng vì Adam không chỉ có model weights; nó còn giữ moving averages tích lũy từ các batch trước.

Khôi phục scheduler giữ learning rate tiếp tục đúng vị trí, thay vì quay lại lịch epoch đầu.

Khôi phục RNG giúp random shuffle, subset selection và sampling tiếp tục gần đúng run không bị gián đoạn.

## 22. Kiểm tra model config khi resume

```python
saved_model_params = checkpoint.get(
    'resolved_config', {}
).get('model_params')
```

Nếu checkpoint có lưu model params và chúng khác active config:

```python
if (
    saved_model_params is not None
    and saved_model_params != cfg.model_params
):
    raise ValueError(...)
```

Điều này ngăn các tình huống như:

```text
checkpoint K=3
nhưng config hiện tại K=1
```

hoặc hidden size/forecast horizon khác nhau.

Nếu checkpoint cũ không có `resolved_config`, kiểm tra này được bỏ qua để giữ tương thích.

## 23. Khôi phục DataLoader order

```python
data_state_restored = data_loader.load_state_dict(
    checkpoint.get('data_loader_state')
)
```

DataLoader shuffle train arrays in-place. Vì vậy để resume order-exact, phải khôi phục thứ tự in-memory của `X,y`, không chỉ RNG.

Nếu checkpoint không có loader state:

```text
continuation vẫn chạy được
nhưng sample ordering không hoàn toàn tương đương run liên tục
```

Code ghi warning thay vì dừng.

## 24. Khôi phục history và epoch

```python
history = checkpoint.get('train_history', [])
epoch = int(checkpoint['epoch']) + 1
```

Ví dụ checkpoint lưu epoch 100:

```text
resume bắt đầu từ epoch 101
```

History cũ được truyền lại cho trainer để tiếp tục biểu đồ/log thay vì tạo một lịch sử mới từ rỗng.

## 25. Khôi phục best-model state

```python
tracker.best_epoch = checkpoint.get('best_epoch')
tracker.best_validation_nll = checkpoint.get(
    'best_validation_nll',
    float('inf')
)
```

Nếu không khôi phục, tracker có thể coi một checkpoint sau resume là `best` dù validation NLL tệ hơn best trước đó.

`float('inf')` là fallback cho checkpoint cũ không có trường này.

## 26. Ghi nhận sự kiện resume

```python
tracker._write_manifest(
    'running',
    resumed_from=os.path.abspath(resume_path)
)
```

Manifest được cập nhật thành trạng thái running và ghi nguồn checkpoint.

```python
tracker.event(
    'run_resumed',
    checkpoint['epoch'],
    {
        'checkpoint': os.path.abspath(resume_path),
        'data_loader_state_restored': data_state_restored,
    }
)
```

Event log ghi:

- checkpoint nào được dùng;
- resume từ epoch nào;
- DataLoader order có khôi phục thành công không.

## 27. Train from scratch

```python
else:
```

Nếu không yêu cầu resume, model dùng Xavier initialization vừa tạo, Adam/scheduler ở trạng thái mới và `epoch=1`.

Phần này chủ yếu ghi thông tin config vào logger.

Có một chi tiết code:

```python
if cfg.with_print: (colored(...))
```

Dòng này tạo chuỗi có màu nhưng không gọi `print()`, nên thông báo `Start training from scratch...` không thực sự được in ra terminal ở nhánh này. Logging vẫn hoạt động nếu `with_log=True`. Đây là hành vi code hiện tại, không ảnh hưởng model training.

## 28. Tạo `MDN_Trainer`

```python
trainer = MDN_Trainer(
    cfg=cfg,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epoch=epoch,
    loss_hist=history,
    logger=train_logger,
    tracker=tracker
)
```

Trainer nhận toàn bộ dependency đã chuẩn bị:

```text
config
model
loss
Adam
LinearLR
device
starting epoch
old/new history
logger
experiment tracker
```

`train.py` không dùng global variables để trainer tự tìm các thành phần này; nó truyền chúng rõ ràng vào constructor.

## 29. Bắt đầu vòng training thật

```python
final_epoch, history = trainer.train(
    data_loader=data_loader
)
```

Từ đây quyền điều khiển chuyển sang `mdn.py`.

`trainer.train()` sẽ thực hiện:

```text
for epoch
  -> get random 50% train data
  -> for batch
       -> tensor -> device
       -> forward
       -> NLL
       -> backward
       -> Adam update
  -> scheduler step
  -> validation NLL
  -> log/checkpoint/full metrics
```

Nó trả:

```text
final_epoch
updated history
```

Chi tiết và cách tính `final_epoch` phải đọc trong `mdn.py`.

## 30. Lưu final model theo cơ chế legacy

```python
trainer.save(
    epoch=final_epoch,
    diverged=False,
    final=True
)
```

Hàm `MDN_Trainer.save()` lưu:

```text
<cfg.checkpoint_path>/model_final.pt
```

Checkpoint này phục vụ cơ chế upstream/legacy.

## 31. Lưu final checkpoint theo ExperimentTracker

```python
tracker.save_checkpoint(
    'final',
    final_epoch,
    model,
    optimizer,
    scheduler,
    history
)
```

Tracker lưu final checkpoint phong phú hơn trong cấu trúc run artifact, gồm thêm trạng thái phục vụ reproducibility và analysis.

Hai dòng save không hoàn toàn trùng vai trò:

```text
trainer.save()          -> legacy model_final.pt
tracker.save_checkpoint -> instrumented run checkpoint
```

## 32. Đánh dấu run hoàn thành

```python
tracker.complete(final_epoch)
```

Tracker cập nhật manifest:

```text
status = completed
final_epoch
finished_at
```

và ghi event `run_completed`.

## 33. Báo số parameters

```python
count_model_parameters(model=model)
```

Helper thực hiện:

```python
sum(p.numel() for p in model.parameters())
```

Với config hiện tại kết quả là:

```text
8.224 parameters
```

## 34. Đóng logger

```python
if cfg.with_log:
    log_file_handler.close()
    train_logger.removeHandler(log_file_handler)
```

Việc remove handler quan trọng nếu process chạy nhiều configs. Nếu giữ handler cũ, log của config sau có thể bị ghi lặp vào file trước.

## 35. Entry point

```python
if __name__ == "__main__":
```

Khối này chỉ chạy khi gọi trực tiếp:

```bash
python base_mdn/train.py ...
```

Nếu `train.py` được import như module, khối này không tự chạy.

## 36. Các giá trị cố định cho entry point

```python
model_arch = 'base_mdn'
type = 'training'
```

Các giá trị này được truyền vào `ConfigLoader` để đặt metadata và result path.

Tên biến `type` trùng với built-in Python `type`, nhưng trong phạm vi này nó chỉ là chuỗi metadata và không làm thay đổi training.

## 37. Parse CLI arguments

```python
parser = config_parser()
args = parser.parse_args()
```

`config_parser()` định nghĩa:

```text
-t / --target      mặc định imptc
-c / --configs     mặc định default_peds_imptc.json
-l / --log         bật log
-p / --print       bật terminal output
-g / --gpu         mặc định "0"
--run-id            chủ yếu dùng testing
--checkpoint        chủ yếu dùng testing
```

Helper gọi:

```python
p.set_defaults(log=True)
p.set_defaults(print=True)
```

nên logging và printing mặc định đang bật.

Ví dụ lệnh:

```bash
python base_mdn/train.py \
  --target imptc \
  --configs default_peds_imptc.json \
  --gpu 0
```

## 38. Chọn GPU ID

```python
if args.gpu:
    gpu_id = args.gpu
else:
    gpu_id = 0
```

`args.gpu` mặc định là chuỗi `"0"`, là truthy, nên thông thường `gpu_id="0"`.

Nếu giá trị rỗng/không có, fallback là integer `0`; sau đó vẫn được chuyển thành string khi đặt biến môi trường.

## 39. Tìm project root và config directory

```python
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
```

`train.py` nằm trong `base_mdn`, nên đi lên hai cấp cho project root.

```python
config_dir = os.path.join(
    project_root,
    model_arch,
    'configs',
    args.target
)
```

Với IMPTC:

```text
<project>/base_mdn/configs/imptc
```

Cách này không phụ thuộc current working directory của terminal.

## 40. Chạy tất cả config

```python
if args.configs == 'all':
```

Code duyệt mọi file `.json` trong config directory và tạo một `ConfigLoader` cho mỗi file.

Tên config:

```python
name = conf[:-5]
```

để bỏ suffix `.json`.

Lưu ý thư mục config có thể chứa cả default, smoke, metric-smoke, M1 và fixed-sample JSON. Tùy nội dung thư mục, `--configs all` có thể không đồng nghĩa chỉ chạy các baseline chính; cần xem danh sách file trước một long run.

## 41. Chạy một config

```python
else:
    configs = [ConfigLoader(
        config_path=os.path.join(config_dir, args.configs),
        target=args.target,
        with_log=args.log,
        with_print=args.print,
        name=args.configs[:-5],
        model_arch=model_arch,
        type=type
    )]
```

Đây là flow thường dùng cho baseline IMPTC.

`ConfigLoader` ngay lúc khởi tạo sẽ:

- đọc JSON;
- resolve data/result paths;
- tạo reliability bins/colors;
- tạo result directory structure.

## 42. Chạy lần lượt các config

```python
for cfg in configs:
    training(cfg=cfg, gpu_id=gpu_id)
```

Nếu có nhiều config, chúng chạy tuần tự trong cùng process, không chạy song song.

Sau khi hoàn thành:

```python
sys.exit()
```

## 43. Những gì `train.py` chưa làm

`train.py` không trực tiếp chứa:

- vòng `for epoch`;
- vòng `for batch`;
- chuyển NumPy batch thành tensor;
- dynamic input horizon;
- gọi `loss.backward()`;
- gọi `optimizer.step()`;
- tính validation NLL;
- quyết định best checkpoint;
- gọi full evaluation theo lịch.

Các phần này nằm trong `mdn.py` và `experiment.py`.

## 44. Bản đồ đối tượng sau khi setup

```text
cfg
 |
 +-> DataLoader
 |     +-> full train arrays
 |     `-> full eval arrays
 |
 +-> LSTM_Trajectory_Forecast
 |     +-> LSTM
 |     `-> Linear
 |
 +-> Adam(model.parameters)
 |
 +-> LinearLR(Adam)
 |
 +-> NLL_MDN_loss(num_gaussians=3)
 |
 `-> ExperimentTracker
       +-> fixed samples
       +-> history
       `-> checkpoints/artifacts

Tất cả được truyền vào MDN_Trainer
```

## 45. Các dòng quan trọng nhất

```text
set_global_seed(seed)
-> đặt nguồn random trước initialization

load_train_data() / load_eval_data()
-> đưa full arrays vào RAM

LSTM_Trajectory_Forecast(...).to(device)
-> tạo và chuyển model

optim.Adam(model.parameters(), lr=1e-3)
-> tạo optimizer

LinearLR(...)
-> lịch giảm learning rate theo epoch

loss_fn = lambda ... NLL_MDN_loss(..., K=3)
-> khóa cách tính loss

ExperimentTracker(...)
-> chuẩn bị artifact/reproducibility

restore_checkpoint(...)
-> khôi phục model/Adam/scheduler/RNG

MDN_Trainer(...)
-> nhận toàn bộ thành phần

trainer.train(data_loader)
-> bắt đầu epoch/batch loop thật
```

## Kết luận

`train.py` không phải nơi định nghĩa toán học của model và cũng không chứa batch loop. Nó là file orchestration:

```text
chọn config
-> chuẩn bị môi trường
-> load data
-> tạo model
-> tạo optimizer/scheduler/loss
-> khôi phục checkpoint nếu cần
-> giao toàn bộ cho MDN_Trainer
-> lưu final artifacts
```

File cần đọc tiếp theo là `base_mdn/mdn.py`, nơi ta sẽ thấy chính xác một epoch và một batch được chạy như thế nào.

