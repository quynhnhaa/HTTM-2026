# Đọc hiểu `base_mdn/base_lstm.py`

## Vai trò của file

`base_lstm.py` định nghĩa hai thành phần:

```text
LSTM_Trajectory_Forecast
→ kiến trúc model và forward pass

NLL_MDN_loss
→ dựng GMM và tính training loss
```

Luồng tổng thể:

```text
X [B,32,4]
→ LSTM
→ lstm_out [B,32,8]
→ lấy hidden output cuối h_32 [B,8]
→ Linear [B,864]
→ reshape [B,48,18]
→ dựng 48 GMM
→ so với target [B,48,2]
→ một scalar NLL loss
```

Hình minh họa liên quan:

![LSTM, Linear và loss](./lstm_linear_loss_flow.png)

## 1. Các import

```python
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions as dist

from utils.mdn_distribution import build_mdn_distribution
```

- `torch`: tensor và reshape;
- `torch.nn`: `Module`, `LSTM`, `Linear`;
- `torch.nn.init`: khởi tạo weights và bias;
- `torch.distributions`: hiện được import nhưng không còn gọi trực tiếp trong file;
- `build_mdn_distribution`: biến raw output thành một GMM hợp lệ.

## 2. Class model

```python
class LSTM_Trajectory_Forecast(nn.Module):
```

Kế thừa `nn.Module` cho phép PyTorch:

- đăng ký trainable parameters;
- chuyển model giữa CPU/GPU;
- tự động tính gradient;
- lưu và load `state_dict`;
- sử dụng `train()` và `eval()`.

## 3. Hyperparameter

```python
self.lstm_input_shape = cfg['lstm_input_shape']
self.lstm_hidden_size = cfg['lstm_hidden_size']
self.output_size = cfg['num_gaussians'] * cfg['output_factor']
self.forecast_horizon = cfg['forecast_horizon']
self.lstm_num_layers = cfg['lstm_num_layers']
```

Với config IMPTC hiện tại:

```text
lstm_input_shape = 4
lstm_hidden_size = 8
num_gaussians = 3
output_factor = 6
output_size = 3 x 6 = 18
forecast_horizon = 48
lstm_num_layers = 1
```

Input feature đã được xác nhận từ dữ liệu:

```text
[x, y, vx, vy]
```

## 4. LSTM encoder

```python
self.lstm = nn.LSTM(
    input_size=self.lstm_input_shape,
    hidden_size=self.lstm_hidden_size,
    num_layers=self.lstm_num_layers,
    batch_first=True
)
```

Thay giá trị cấu hình:

```python
nn.LSTM(
    input_size=4,
    hidden_size=8,
    num_layers=1,
    batch_first=True
)
```

`batch_first=True` nghĩa là tensor có thứ tự:

```text
[batch, time, feature]
```

Input:

```text
[B,32,4]
```

LSTM dùng cùng một bộ weights ở mọi timestep:

```text
x_1  -> h_1, C_1
x_2  -> h_2, C_2
...
x_32 -> h_32, C_32
```

Đây không phải 32 LSTM layer khác nhau. Đó là một LSTM được unroll qua 32 timestep.

Mặc dù comment trong code ghi `stacked LSTM`, baseline hiện có `num_layers=1`, nên chỉ có một LSTM layer.

## 5. Linear layer

```python
self.fc = nn.Linear(
    in_features=self.lstm_hidden_size,
    out_features=self.output_size * self.forecast_horizon
)
```

Với config hiện tại:

```text
Linear: 8 -> 18 x 48 = 864
```

LSTM chỉ tạo vector cuối:

```text
h_32 [B,8]
```

Nhưng model cần sinh:

```text
48 timestep x 3 Gaussian x 6 raw values = 864 values
```

Linear học phép biến đổi:

```text
z = h_32 W^T + b
```

Shape:

```text
h_32 [B,8]
W    [864,8]
b    [864]
z    [B,864]
```

## 6. Số trainable parameters

### LSTM

Với input size 4 và hidden size 8:

```text
weight_ih [32,4] = 128
weight_hh [32,8] = 256
bias_ih   [32]   = 32
bias_hh   [32]   = 32

Tổng LSTM = 448
```

Kích thước 32 đến từ bốn nhóm cổng:

```text
4 x hidden_size = 4 x 8 = 32
```

### Linear

```text
weights = 864 x 8 = 6.912
bias    = 864

Tổng Linear = 7.776
```

### Toàn model

```text
448 + 7.776 = 8.224 trainable parameters
```

Khoảng 94,6% parameters nằm trong Linear và 5,4% nằm trong LSTM.

## 7. Khởi tạo weights

LSTM weights:

```python
for name, param in self.lstm.named_parameters():
    if 'weight' in name:
        init.xavier_uniform_(param)
    elif 'bias' in name:
        init.zeros_(param)
```

Linear:

```python
init.xavier_uniform_(self.fc.weight)
init.zeros_(self.fc.bias)
```

Tóm lại:

```text
weights -> Xavier uniform
bias    -> 0
```

Code không đặt forget-gate bias thành 1. Mọi bias của LSTM đều bắt đầu từ 0.

Seed được đặt trước khi model được tạo trong `train.py`, giúp initialization có thể tái lập.

## 8. Forward qua LSTM

```python
lstm_out, _ = self.lstm(x)
```

PyTorch thực tế trả:

```python
lstm_out, (h_n, c_n)
```

Code dùng `_` để bỏ `h_n` và `c_n`, chỉ giữ `lstm_out`.

Shape:

```text
input:    [B,32,4]
lstm_out: [B,32,8]
```

Trong đó:

```text
lstm_out[:,0,:]  = h_1
lstm_out[:,1,:]  = h_2
...
lstm_out[:,31,:] = h_32
```

## 9. Chỉ lấy output cuối

```python
lstm_out[:, -1, :]
```

Ý nghĩa:

```text
:   -> tất cả sample trong batch
-1  -> timestep cuối
:   -> toàn bộ 8 hidden features
```

Kết quả:

```text
h_32 [B,8]
```

Các output `h_1 ... h_31` không được nối trực tiếp vào Linear, nhưng chúng đã ảnh hưởng đến `h_32` qua recurrence của LSTM.

Do đó phần LSTM là:

```text
many-to-one encoder
```

## 10. Linear tạo toàn bộ tương lai cùng lúc

```python
output = self.fc(lstm_out[:, -1, :])
```

Shape:

```text
[B,8] -> [B,864]
```

Linear chạy một lần và sinh raw parameters cho toàn bộ 48 timestep.

Không có:

- LSTM decoder;
- vòng lặp dự đoán 48 lần;
- đưa prediction của timestep trước vào timestep sau;
- autoregressive future decoding.

Cách mô tả phù hợp nhất:

```text
many-to-one LSTM encoder
+ one-shot multi-horizon probabilistic output
```

## 11. Reshape

```python
output = torch.reshape(
    output,
    (-1, self.forecast_horizon, self.output_size)
)
```

Với config hiện tại:

```text
[B,864] -> [B,48,18]
```

`-1` để PyTorch tự suy ra batch size.

Reshape không có weights và không học tham số. Nó chỉ thay đổi cách diễn giải tensor.

## 12. Ý nghĩa output `[B,48,18]`

```text
B  = số sample trong batch
48 = số future timestep
18 = raw MDN values tại mỗi timestep
```

Mười tám giá trị tại mỗi timestep gồm:

```text
3 mu_x
3 mu_y
3 raw_sigma_x
3 raw_sigma_y
3 raw_rho
3 raw_pi
```

Đây chưa phải tọa độ dự đoán duy nhất và cũng chưa phải distribution hợp lệ. File `mdn_distribution.py` sẽ áp dụng:

```text
raw_sigma -> exp
raw_rho   -> tanh
raw_pi    -> softmax
```

## 13. Hàm loss

```python
def NLL_MDN_loss(output, target, num_gaussians):
```

Input:

```text
output        [B,48,18]
target        [B,48,2]
num_gaussians = 3
```

`output` là raw prediction của model. `target` là tọa độ future ground truth `[x,y]`.

## 14. Dựng GMM

```python
mixture = build_mdn_distribution(
    output,
    num_gaussians
)
```

Hàm này thực hiện:

```text
raw output [B,48,18]
-> mu, sigma, rho, pi
-> covariance matrices
-> 3 Gaussian hai chiều tại mỗi timestep
-> một GMM tại mỗi timestep
```

Tức là:

```text
48 timestep
x
1 GMM/timestep
x
3 Gaussian/GMM
```

## 15. Ground truth trong loss

```python
mixture.log_prob(target)
```

Với mỗi sample `b` và timestep `t`, code tính:

```text
log p_(b,t)(y_(b,t))
```

Shape:

```text
target:                    [B,48,2]
mixture.log_prob(target):  [B,48]
```

Ground truth không đi vào LSTM. Nó chỉ được dùng sau khi model đã dự đoán để tính loss.

## 16. Loss dùng toàn bộ mixture

Tại một timestep:

```text
p_t(y_t)
= pi_1 N_1(y_t)
+ pi_2 N_2(y_t)
+ pi_3 N_3(y_t)
```

Loss không chọn Gaussian gần ground truth nhất và không tính `min` giữa ba component.

NLL:

```text
NLL_t = -log p_t(y_t)
```

## 17. Lấy trung bình thành scalar

```python
params_loss = -mixture.log_prob(target).mean()
```

Trước `.mean()`:

```text
[B,48]
```

Sau `.mean()`:

```text
một scalar
```

Công thức:

```text
Loss
= -(1/(B x 48))
  sum_b sum_t log p_(b,t)(y_(b,t))
```

Một batch chỉ gọi:

```text
loss.backward() một lần
optimizer.step() một lần
```

Gradient của tất cả sample và 48 timestep cùng đóng góp vào lần cập nhật này.

## 18. Vì sao có dấu âm?

Mục tiêu xác suất là:

```text
maximize p(y|X)
```

Tương đương:

```text
maximize log p(y|X)
```

Optimizer của PyTorch giảm objective, nên ta đổi thành:

```text
minimize -log p(y|X)
```

## 19. NLL có bắt buộc dương không?

Không. Đây là probability density liên tục, không phải xác suất rời rạc. Density có thể lớn hơn 1 khi distribution rất tập trung.

Nếu:

```text
p(y) > 1
```

thì:

```text
-log p(y) < 0
```

Vì vậy điều cần quan sát là NLL có giảm và ổn định hay không, không phải nó có luôn dương hoặc tiến về 0 hay không.

## 20. Cờ `diverged`

```python
try:
    mixture = build_mdn_distribution(...)
except:
    return None, True
```

Nếu dựng distribution phát sinh exception, hàm trả:

```text
loss = None
diverged = True
```

Trainer sẽ dừng và lưu trạng thái.

Đây là bare `except`, nên nó bắt mọi exception chứ không chỉ lỗi numerical divergence. Lỗi shape hoặc lỗi lập trình cũng có thể bị báo chung là `diverged`.

Nếu thành công:

```python
return loss, False
```

## 21. Gradient đi qua model thế nào?

Sau:

```python
loss.backward()
```

gradient đi theo chiều ngược:

```text
scalar NLL
-> mixture log probability
-> pi, mu, sigma, rho
-> raw output [B,48,18]
-> Linear
-> h_32
-> LSTM qua 32 timestep
```

Mặc dù Linear chỉ nhận `h_32`, gradient vẫn đi ngược qua recurrence:

```text
h_32 -> h_31 -> ... -> h_1
```

Vì vậy toàn bộ 32 observed timestep đều có thể ảnh hưởng đến việc học.

## 22. Loss không chứa metric đánh giá

Training loss chỉ là:

```text
NLL
```

Nó không cộng thêm:

```text
ADE
FDE
reliability
sharpness 68%
sharpness 95%
ASAEE
```

Các giá trị trên là evaluation metrics, không được gọi `backward()`.

## 23. Test xác nhận implementation

`tests/test_pipeline.py` kiểm tra:

```text
model input:       [2,32,4]
raw output:        [2,48,18]
pi:                [2,48,3]
mu:                [2,48,3,2]
covariance:        [2,48,3,2,2]
```

Test cũng xác nhận:

```text
sum(pi, dim=-1) = 1
covariance có eigenvalue dương
training và evaluation dùng cùng distribution builder
NLL_MDN_loss = -distribution.log_prob(target).mean()
```

## 24. Những gì không nằm trong file này

`base_lstm.py` không chứa:

- load data;
- chia batch;
- vòng lặp epoch;
- Adam optimizer;
- learning-rate scheduler;
- validation loop;
- checkpoint;
- resume;
- sampling trajectory;
- ADE/FDE;
- reliability và sharpness;
- visualization.

Các phần này nằm trong `train.py`, `mdn.py`, `experiment.py` và `eval.py`.

## 25. Các dòng code quan trọng nhất

```text
self.lstm = nn.LSTM(...)
→ tạo sequence encoder

self.fc = nn.Linear(8, 864)
→ biến h_32 thành raw parameters cho toàn bộ tương lai

lstm_out, _ = self.lstm(x)
→ xử lý 32 observed timestep

self.fc(lstm_out[:, -1, :])
→ chỉ lấy hidden output cuối

reshape(..., 48, 18)
→ tổ chức raw output theo future timestep

build_mdn_distribution(...)
→ raw values thành GMM

-mixture.log_prob(target).mean()
→ scalar NLL loss
```

## Kết luận

Kiến trúc trong `base_lstm.py` là:

```text
32 observed timesteps [x,y,vx,vy]
-> một LSTM layer, hidden size 8
-> lấy h_32
-> Linear 8 -> 864
-> reshape 48 x 18
-> một GMM ba Gaussian tại mỗi future timestep
-> NLL với future ground truth [x,y]
```

Đây là `many-to-one LSTM encoder` kết hợp với `one-shot multi-horizon MDN output`, không phải LSTM encoder-decoder many-to-many.

---

# Đọc từng dòng `base_mdn/utils/mdn_distribution.py`

Phần này nối tiếp raw output `[B,48,18]` từ `base_lstm.py` và giải thích cách repository biến nó thành GMM.

Ta dùng ví dụ cố định:

```text
B = 2 sample
T = 48 future timestep
K = 3 Gaussian

output.shape = [2,48,18]
```

Tại mỗi sample và future timestep, model có một vector 18 raw values. Mục tiêu của file là:

```text
18 raw values
-> tham số của 3 Gaussian
-> covariance matrix
-> một GMM hợp lệ
```

## 26. Mô tả và import

```python
"""Shared MDN parameter decoding used by training, evaluation and artifacts."""

import torch
import torch.distributions as dist
```

Cùng một cách decode được dùng trong training, evaluation và artifact capture. Điều này tránh việc các phần khác nhau hiểu sai thứ tự 18 raw values.

`torch` cung cấp:

```text
exp
tanh
softmax
stack
zeros
```

`torch.distributions` cung cấp:

```text
MultivariateNormal
Categorical
MixtureSameFamily
```

Các phép biến đổi đều hỗ trợ automatic differentiation, nên gradient có thể đi ngược về Linear và LSTM.

## 27. Hàm `decode_mdn_output()`

```python
def decode_mdn_output(output, num_gaussians):
```

Hàm nhận:

```text
output        [B,48,18]
num_gaussians = 3
```

Nó không nhận ground truth và không tính loss. Nhiệm vụ là:

```text
raw output
-> pi, mu, sigma, rho, covariance
```

## 28. Layout của 18 raw values

Tại một sample `b` và timestep `t`:

```text
output[b,t,:]
= [
    mu_x1, mu_x2, mu_x3,
    mu_y1, mu_y2, mu_y3,
    raw_sigma_x1, raw_sigma_x2, raw_sigma_x3,
    raw_sigma_y1, raw_sigma_y2, raw_sigma_y3,
    raw_rho1, raw_rho2, raw_rho3,
    raw_pi1, raw_pi2, raw_pi3
  ]
```

| Index | Nội dung |
|---|---|
| `0:3` | ba `mu_x` |
| `3:6` | ba `mu_y` |
| `6:9` | ba `raw_sigma_x` |
| `9:12` | ba `raw_sigma_y` |
| `12:15` | ba `raw_rho` |
| `15:18` | ba `raw_pi` |

Repository nhóm theo loại tham số, không sắp sáu tham số của Gaussian 1 rồi sáu tham số của Gaussian 2.

## 29. Lấy `mu_x`

```python
mu_x = output[..., :num_gaussians]
```

Với `K=3`:

```python
mu_x = output[..., 0:3]
```

Dấu `...` giữ nguyên tất cả chiều phía trước. Shape:

```text
output [B,48,18]
mu_x   [B,48,3]
```

Ba giá trị cuối là tọa độ x của tâm ba Gaussian.

## 30. Lấy `mu_y`

```python
mu_y = output[..., num_gaussians:2 * num_gaussians]
```

Với `K=3`:

```python
mu_y = output[..., 3:6]
```

Shape:

```text
mu_y [B,48,3]
```

Sau này ghép lại:

```text
Gaussian 1 mean = [mu_x1,mu_y1]
Gaussian 2 mean = [mu_x2,mu_y2]
Gaussian 3 mean = [mu_x3,mu_y3]
```

`mu` không cần activation vì tọa độ có thể nhận mọi giá trị thực.

## 31. Lấy và biến đổi `sigma_x`

```python
sigma_x = torch.exp(
    output[..., 2 * num_gaussians:3 * num_gaussians]
)
```

Với `K=3`:

```python
sigma_x = torch.exp(output[..., 6:9])
```

Linear có thể sinh mọi số thực, nhưng standard deviation phải dương. `exp` bảo đảm:

```text
sigma_x > 0
```

Ví dụ:

```text
raw = -2 -> sigma = exp(-2) ~= 0,135
raw =  0 -> sigma = 1
raw =  1 -> sigma ~= 2,718
```

Shape:

```text
sigma_x [B,48,3]
```

## 32. Lấy và biến đổi `sigma_y`

```python
sigma_y = torch.exp(
    output[..., 3 * num_gaussians:4 * num_gaussians]
)
```

Với `K=3`:

```python
sigma_y = torch.exp(output[..., 9:12])
```

Shape:

```text
sigma_y [B,48,3]
```

Ý nghĩa:

```text
sigma_x -> độ phân tán theo x
sigma_y -> độ phân tán theo y
```

Ví dụ `sigma_x` lớn và `sigma_y` nhỏ tạo ellipse dài theo x và hẹp theo y.

## 33. Lấy correlation `rho`

```python
rho = torch.tanh(
    output[..., 4 * num_gaussians:5 * num_gaussians]
)
```

Với `K=3`:

```python
rho = torch.tanh(output[..., 12:15])
```

Correlation hợp lệ cần:

```text
-1 < rho < 1
```

`tanh` ánh xạ mọi raw value vào khoảng này. Shape:

```text
rho [B,48,3]
```

`rho=0` nghĩa là không có correlation tuyến tính x-y. `rho` dương hoặc âm làm covariance ellipse nghiêng theo các hướng khác nhau.

## 34. Lấy mixture weights `pi`

```python
pi = torch.softmax(
    output[..., 5 * num_gaussians:],
    dim=-1
)
```

Với `K=3`:

```python
pi = torch.softmax(output[..., 15:18], dim=-1)
```

Softmax chạy trên chiều cuối gồm ba Gaussian và bảo đảm:

```text
pi_k >= 0
pi_1 + pi_2 + pi_3 = 1
```

Ví dụ:

```text
raw_pi = [2,1,0]
pi ~= [0,665; 0,245; 0,090]
```

Shape:

```text
pi [B,48,3]
```

Tới đây ta có:

```text
mu_x    [B,48,3]
mu_y    [B,48,3]
sigma_x [B,48,3]
sigma_y [B,48,3]
rho     [B,48,3]
pi      [B,48,3]
```

## 35. Ghép mean hai chiều

```python
mu = torch.stack([mu_x, mu_y], dim=-1)
```

Trước stack:

```text
mu_x [B,48,3]
mu_y [B,48,3]
```

Sau stack:

```text
mu [B,48,3,2]
```

Tại một timestep:

```text
mu[b,t,:,:]
= [
    [mu_x1,mu_y1],
    [mu_x2,mu_y2],
    [mu_x3,mu_y3]
  ]
```

## 36. Ghép standard deviation

```python
sigma = torch.stack([sigma_x, sigma_y], dim=-1)
```

Kết quả:

```text
sigma [B,48,3,2]
```

Tại một timestep:

```text
[
  [sigma_x1,sigma_y1],
  [sigma_x2,sigma_y2],
  [sigma_x3,sigma_y3]
]
```

Biến `sigma` được trả về cho artifact và visualization. `MultivariateNormal` bên dưới dùng covariance matrix.

## 37. Tạo tensor covariance rỗng

```python
covariance = torch.zeros(
    *mu.shape[:-1],
    2,
    2,
    device=output.device,
    dtype=output.dtype
)
```

Vì:

```text
mu.shape      = [B,48,3,2]
mu.shape[:-1] = [B,48,3]
```

nên:

```text
covariance.shape = [B,48,3,2,2]
```

Ý nghĩa:

```text
B sample
x 48 timestep
x 3 Gaussian
x ma trận covariance 2x2
```

`device=output.device` giữ covariance trên cùng CPU/GPU với output. `dtype=output.dtype` giữ cùng kiểu số, thường là `float32` khi training.

## 38. Điền covariance matrix

Mục tiêu là:

```text
Sigma =
[[sigma_x^2,               rho sigma_x sigma_y],
 [rho sigma_x sigma_y,               sigma_y^2]]
```

Variance theo x:

```python
covariance[..., 0, 0] = sigma_x.square()
```

Covariance x-y:

```python
covariance[..., 0, 1] = rho * sigma_x * sigma_y
```

Ma trận phải đối xứng:

```python
covariance[..., 1, 0] = covariance[..., 0, 1]
```

Variance theo y:

```python
covariance[..., 1, 1] = sigma_y.square()
```

Ví dụ:

```text
sigma_x = 2
sigma_y = 1
rho = 0,5

Sigma = [[4,1],
         [1,1]]
```

Gaussian này rộng hơn theo x và có correlation dương.

Determinant:

```text
det(Sigma) = sigma_x^2 sigma_y^2 (1-rho^2)
```

Vì `sigma_x,sigma_y>0` và `|rho|<1`, covariance positive definite về mặt toán học. Raw sigma cực đoan vẫn có thể gây overflow/underflow trong floating-point vì code không clamp trước `exp`.

## 39. Trả các tham số đã decode

```python
return {
    'pi': pi,
    'mu': mu,
    'sigma': sigma,
    'rho': rho,
    'covariance': covariance,
}
```

Shape:

```text
pi         [B,48,3]
mu         [B,48,3,2]
sigma      [B,48,3,2]
rho        [B,48,3]
covariance [B,48,3,2,2]
```

Các giá trị được dùng để dựng distribution, lưu artifact và vẽ covariance ellipse.

## 40. Hàm `build_mdn_distribution()`

```python
def build_mdn_distribution(output, num_gaussians):
```

Hàm nhận raw output `[B,48,18]` và trả một đối tượng PyTorch distribution.

## 41. Decode raw output

```python
params = decode_mdn_output(output, num_gaussians)
```

Sau dòng này:

```text
params['pi']         [B,48,3]
params['mu']         [B,48,3,2]
params['covariance'] [B,48,3,2,2]
```

## 42. Tạo ba Gaussian hai chiều

```python
components = dist.MultivariateNormal(
    params['mu'],
    params['covariance']
)
```

PyTorch hiểu:

```text
components.batch_shape = [B,48,3]
components.event_shape = [2]
```

Các batch dimensions là:

```text
B sample
x 48 timestep
x 3 Gaussian
```

Event dimension `[2]` là một vector tọa độ `[x,y]`. Vì là Multivariate Normal, x và y có thể tương quan thông qua `rho`.

## 43. Tạo categorical distribution

```python
mixture = dist.Categorical(probs=params['pi'])
```

Input:

```text
pi [B,48,3]
```

PyTorch hiểu ba giá trị cuối là xác suất chọn ba component:

```text
mixture.batch_shape = [B,48]
```

Ví dụ:

```text
pi = [0,6; 0,3; 0,1]
```

nghĩa là component 1 có weight 60%, component 2 có 30%, component 3 có 10%.

## 44. Ghép thành GMM

```python
return dist.MixtureSameFamily(mixture, components)
```

`MixtureSameFamily` ghép:

```text
Categorical weights
+
3 MultivariateNormal components
```

thành một GMM hai chiều.

Sau khi ghép:

```text
GMM.batch_shape = [B,48]
GMM.event_shape = [2]
```

Tức là mỗi sample và mỗi future timestep có một GMM trên tọa độ `[x,y]`.

Chiều component không còn xuất hiện bên ngoài vì ba component đã được trộn thành một mixture.

## 45. `log_prob(target)`

Ground truth có shape:

```text
target [B,48,2]
```

Khi gọi:

```python
mixture.log_prob(target)
```

kết quả có shape:

```text
[B,48]
```

Mỗi phần tử là:

```text
log[
    pi_1 N_1(y_(b,t))
  + pi_2 N_2(y_(b,t))
  + pi_3 N_3(y_(b,t))
]
```

Nó không trả ba loss riêng và không chọn component gần nhất.

## 46. `sample()`

Ví dụ:

```python
samples = mixture.sample(
    sample_shape=torch.Size([20])
)
```

Shape:

```text
[20,B,48,2]
```

Ý nghĩa:

```text
20 lần lấy mẫu
x B input samples
x 48 future timestep
x tọa độ [x,y]
```

Tại mỗi `[b,t]`, PyTorch dùng `pi` để chọn component rồi lấy một tọa độ từ Gaussian được chọn.

Distribution được định nghĩa riêng theo từng future timestep. `MixtureSameFamily` không tạo một joint mixture duy nhất cho toàn bộ chuỗi 48 bước.

## 47. Ví dụ raw values đều bằng 0

Giả sử tại một timestep, Linear sinh:

```text
mu_x raw    = [0,0,0]
mu_y raw    = [0,0,0]
raw_sigma_x = [0,0,0]
raw_sigma_y = [0,0,0]
raw_rho     = [0,0,0]
raw_pi      = [0,0,0]
```

Sau decode:

```text
mu_x = [0,0,0]
mu_y = [0,0,0]

sigma_x = exp([0,0,0]) = [1,1,1]
sigma_y = [1,1,1]

rho = tanh([0,0,0]) = [0,0,0]

pi = softmax([0,0,0])
   = [1/3,1/3,1/3]
```

Mỗi covariance:

```text
Sigma = [[1,0],
         [0,1]]
```

Ba Gaussian lúc này giống hệt nhau, nên toàn GMM tương đương một standard bivariate Gaussian tại origin.

Training sẽ điều chỉnh LSTM và Linear để tạo mean, covariance và weight khác nhau.

## 48. Bản đồ shape hoàn chỉnh

```text
Raw output [B,48,18]
     |
     |-- 0:3   -> mu_x       [B,48,3]
     |-- 3:6   -> mu_y       [B,48,3]
     |-- 6:9   -> sigma_x    [B,48,3]
     |-- 9:12  -> sigma_y    [B,48,3]
     |-- 12:15 -> rho        [B,48,3]
     `-- 15:18 -> pi         [B,48,3]

mu_x + mu_y
-> mu [B,48,3,2]

sigma_x + sigma_y + rho
-> covariance [B,48,3,2,2]

mu + covariance
-> MultivariateNormal
   batch_shape [B,48,3]
   event_shape [2]

pi
-> Categorical
   batch_shape [B,48]

Categorical + MultivariateNormal
-> MixtureSameFamily
   batch_shape [B,48]
   event_shape [2]
```

## 49. Kết luận phần MDN distribution

`mdn_distribution.py` không chứa trainable weights. Nó chỉ biến raw neural-network output thành một probability distribution hợp lệ:

```text
Linear raw values
-> decode
-> Gaussian components
-> mixture weights
-> GMM
```

Trainable weights nằm trong LSTM và Linear. Tuy nhiên gradient đi xuyên qua `exp`, `tanh`, `softmax`, covariance construction và `log_prob`, nên LSTM và Linear học cách sinh GMM tốt hơn.
