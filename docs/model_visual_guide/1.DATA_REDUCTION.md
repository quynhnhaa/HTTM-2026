# Data reduction 50% trong quá trình training

## Kết luận ngắn gọn

Với cấu hình hiện tại:

```json
"train_data_reduction": 0.5,
"eval_data_reduction": 0.5
```

mỗi epoch chỉ train trên khoảng 50% tổng số sample của train set và chỉ tính validation NLL trên khoảng 50% eval set.

50% còn lại không được sử dụng trong epoch hiện tại, nhưng không bị loại vĩnh viễn. Sang epoch tiếp theo, code chọn lại một subset ngẫu nhiên mới từ toàn bộ split.

## Train reduction

IMPTC train split có:

```text
189.595 sample
```

Số sample được chọn trong mỗi epoch:

```text
int(189.595 x 0.5) = 94.797 sample
```

Khoảng 94.798 sample còn lại không tham gia epoch đó.

Luồng thực tế:

```text
Toàn bộ 189.595 train samples
        |
        v
Shuffle toàn bộ X và y
        |
        v
Chọn ngẫu nhiên 94.797 sample không trùng nhau
        |
        v
Chia thành các batch
        |
        v
Forward -> NLL -> backward -> optimizer.step()
```

## Subset có cố định qua các epoch không?

Không. `get_train_data()` được gọi ở đầu mỗi epoch và sinh một tập index ngẫu nhiên mới.

Ví dụ với 10 sample:

```text
Train set: A B C D E F G H I J

Epoch 1 dùng: A C D H J
Epoch 2 dùng: B C E F I
Epoch 3 dùng: A D F G H
```

Các subset có thể chồng lặp:

- một sample có thể xuất hiện trong nhiều epoch liên tiếp;
- một sample có thể bị bỏ qua trong một vài epoch;
- sample bị bỏ qua có thể được chọn ở epoch sau;
- code không bảo đảm mọi sample phải xuất hiện một lần trước khi sample cũ được chọn lại.

Với xác suất chọn xấp xỉ 50% mỗi epoch, xác suất một sample không được chọn trong `E` epoch liên tiếp xấp xỉ:

```text
P(không được chọn trong E epoch) = 0.5^E
```

Ví dụ:

```text
1 epoch:  50%
2 epoch:  25%
5 epoch:  3,125%
10 epoch: 0,0977%
```

Với 2.500 epoch, mỗi sample có số lần xuất hiện kỳ vọng:

```text
2.500 x 0.5 = 1.250 epoch
```

Đây là kỳ vọng xác suất, không phải lịch chọn cứng cho từng sample.

## Số batch và số lần cập nhật mỗi epoch

Batch size hiện tại:

```json
"batch_size": 4096
```

Khi dùng 50% train data:

```text
ceil(94.797 / 4.096) = 24 batch/epoch
```

Cụ thể:

```text
23 batch đầy x 4.096 = 94.208 sample
1 batch cuối              = 589 sample
```

Mỗi batch thực hiện:

```text
optimizer.zero_grad()
-> model(inputs)
-> tính một scalar NLL loss
-> loss.backward()
-> optimizer.step()
```

Do đó một epoch hiện có khoảng:

```text
24 lần forward
24 lần backward
24 lần cập nhật tham số
```

Nếu dùng toàn bộ train set:

```text
ceil(189.595 / 4.096) = 47 batch/epoch
```

Reduction 50% làm số batch mỗi epoch giảm gần một nửa.

## Shuffle khác reduction như thế nào?

Shuffle chỉ thay đổi thứ tự, không bỏ sample:

```text
[A,B,C,D] -> [C,A,D,B]
```

Reduction chỉ lấy một phần tập dữ liệu:

```text
[A,B,C,D] -> [A,D]
```

Implementation hiện tại thực hiện:

```text
shuffle toàn bộ train data
-> chọn random 50%
-> trả X và y của subset
```

`X` và `y` luôn được lấy bằng cùng một danh sách index, nên input vẫn khớp với ground truth tương ứng.

## Khái niệm epoch trong repository này

Định nghĩa thông thường:

> Một epoch là một lần model đi qua toàn bộ training set.

Nhưng trong config hiện tại:

> Một epoch là một vòng train trên random subset bằng khoảng 50% training set.

Vì vậy 2.500 epoch không tương đương 2.500 full passes qua toàn bộ train set.

Nếu chỉ quy đổi theo số lượt sample:

```text
2.500 x 0.5 = 1.250 full-dataset-equivalent epochs
```

Đây chỉ là quy đổi theo số lượt sample. Nó không có nghĩa quá trình tối ưu giống hệt việc chạy 1.250 epoch trên toàn bộ data, vì subset và thứ tự sample thay đổi ngẫu nhiên.

## Eval reduction

IMPTC eval split có:

```text
19.148 sample
```

Mỗi lần gọi `get_eval_data()` với reduction 50% sẽ chọn:

```text
int(19.148 x 0.5) = 9.574 sample
```

Validation NLL trong một epoch vì vậy chỉ được tính trên 9.574 sample ngẫu nhiên.

Với batch size 4.096:

```text
ceil(9.574 / 4.096) = 3 validation batch
```

Validation không gọi `backward()` và không cập nhật model.

Do subset eval có thể thay đổi giữa các epoch, validation NLL có thể dao động do cả hai nguyên nhân:

```text
model thay đổi
+
validation subset thay đổi
```

Ngoài ra, nếu `get_eval_data()` được gọi lại để chạy full evaluation trong cùng epoch, lần gọi mới có thể chọn một random subset khác với subset vừa dùng để tính validation NLL.

## Test set có reduction không?

Không. `get_test_data()` trả toàn bộ test split:

```text
56.694 sample
```

Nó không shuffle và không áp dụng `eval_data_reduction`.

## Fixed validation samples có bị reduction ảnh hưởng không?

Không. `ExperimentTracker` lấy fixed samples trực tiếp từ full `eval_data` đã load và xác định chúng bằng pickle key/checksum.

Vì vậy:

```text
random eval subset dùng tính validation NLL
```

khác với:

```text
8 fixed validation samples dùng so sánh checkpoint
```

Các fixed samples giữ nguyên `X` và ground truth `y` qua các checkpoint.

## Code liên quan

Config:

```json
"train_data_reduction": 0.5,
"eval_data_reduction": 0.5
```

Chọn train subset trong `base_mdn/utils/data_loader.py`:

```python
n = int(len(self.train_data[0]) * self.train_data_reduction)
random_reduced_indices = generate_unique_randoms(
    count=n,
    min_value=0,
    max_value=len(self.train_data[0]) - 1
)
X = np.take(self.train_data[0], random_reduced_indices, axis=0)
y = np.take(self.train_data[1], random_reduced_indices, axis=0)
```

Chọn eval subset:

```python
n = int(len(self.eval_data[0]) * self.eval_data_reduction)
random_reduced_indices = generate_unique_randoms(
    count=n,
    min_value=0,
    max_value=len(self.eval_data[0]) - 1
)
```

Hàm `generate_unique_randoms()` sử dụng một `set`, nên index trong cùng một subset không bị trùng.

## Điều không nên tự suy diễn

Từ code, có thể khẳng định reduction làm giảm số sample và số batch mỗi epoch. Tuy nhiên, nếu paper hoặc upstream documentation không nêu rõ, không nên khẳng định chắc chắn tác giả chọn `0.5` chỉ để tăng tốc hoặc vì một lý do khoa học cụ thể.

Đây là hyperparameter của baseline hiện tại. Không nên tự đổi thành `1.0` nếu mục tiêu đang là tái lập baseline, trừ khi thay đổi đó được xác định rõ là một thí nghiệm bổ sung.

