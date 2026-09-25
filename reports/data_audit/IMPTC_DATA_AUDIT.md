# IMPTC Dataset Audit

## Kết luận

**PASS WITH METADATA CAVEAT**

Bộ IMPTC đã preprocessing từ liên kết chính thức của upstream đã được tải và đặt tại:

```text
data/trajdata/ego/imptc/
```

Ba split khớp README, có shape đồng nhất, không chứa `NaN`/`Inf`, tuân thủ ego-coordinate convention và có velocity nhất quán với sai phân position. Repository `DataLoader` đọc được dữ liệu thực tế.

Caveat duy nhất: metadata không chứa raw sequence identity đủ rõ để kết luận sequence-level leakage. Exact trajectory overlap giữa các split bằng 0, nhưng tên `source/id` có namespace lặp và không thể tự nó chứng minh leakage.

## Nguồn và tính toàn vẹn

- Nguồn: archive dữ liệu đã preprocessing được liên kết trong upstream `README.md`.
- SHA-256 archive: `30e779acf36122a52fd1fdfa1474c32ac9d51e5099512f8a7eaf7a0227e325be`.
- Archive chứa ETH/UCY, IMPTC, inD, nuScenes và Waymo; chỉ IMPTC được giải nén cho experiment hiện tại.

| Split | File size | SHA-256 |
|---|---:|---|
| Train | 522,910,756 bytes | `c0cc18426d1fb428820682a390d132184065b86a4a58a18b2254078a0feba1f0` |
| Eval | 52,763,298 bytes | `a9377b87476ea230b97bd7ce333d1a36518842602af6e6c9cd0f594c2c8d1f4c` |
| Test | 156,259,575 bytes | `75aa900e762a0ce33b052a798db94caefd0abae2830203c72c76fb4076a41b12` |

## Sample count và shape

| Split | Sample thực tế | README | Raw `X` | Raw `y` | Loader target |
|---|---:|---:|---|---|---|
| Train | 189,595 | 189,595 | `(32,4)` | `(48,4)` | `(48,2)` |
| Eval | 19,148 | 19,148 | `(32,4)` | `(48,4)` | `(48,2)` |
| Test | 56,694 | 56,694 | `(32,4)` | `(48,4)` | `(48,2)` |

Tất cả raw arrays là `float64`. Training code chuyển batch sang `float32` trước khi đưa lên device.

Raw `y` có `[x,y,vx,vy]`, nhưng `DataLoader` chủ động lấy `y[..., :2]`; do đó target training là future position `[x,y]` đúng với MDN output.

## Cấu trúc sample

Mỗi sample có đầy đủ:

```text
X
y
reference_position
rotation_angle
source
id
shift
class
movement_class
avg_velocity
```

Shape metadata:

- `reference_position`: `(1,2)`;
- `rotation_angle`: scalar;
- pickle key: integer;
- `source`: string duy nhất trong từng split.

## Numerical integrity

| Kiểm tra | Train | Eval | Test |
|---|---:|---:|---:|
| NaN trong X/y position | 0 | 0 | 0 |
| Inf trong X/y position | 0 | 0 | 0 |
| Exact duplicate trong split | 0 | 0 | 0 |
| Last observed point trong 1 mm của origin | 189,595 | 19,148 | 56,694 |
| Velocity recompute max error | `8.88e-16` | `4.44e-16` | `8.88e-16` |

Last observed position không bằng chính xác `(0,0)` vì preprocessing thêm noise rất nhỏ. Norm trung bình khoảng `7.6e-5 m`, và maximum nhỏ hơn `1.42e-4 m`; toàn bộ sample nằm trong 1 mm của origin.

Velocity ở cả `X` và raw `y` khớp với sai phân position tại 10 Hz tới sai số floating-point.

## Movement classes

| Movement | Train | Eval | Test |
|---|---:|---:|---:|
| Strong left | 18,641 | 1,858 | 5,612 |
| Light left | 36,649 | 4,201 | 11,356 |
| Straight | 41,995 | 4,888 | 13,311 |
| Light right | 36,670 | 3,945 | 11,602 |
| Strong right | 18,093 | 1,515 | 5,740 |
| Standing | 37,547 | 2,741 | 9,073 |

Dataset có đủ các dạng chuyển động phù hợp để sau này chọn fixed samples và phân tích good/uncertain/failure cases. Fixed samples vẫn phải được chọn không dựa trên prediction.

## Split-overlap audit

Exact hash của `X + future position`:

| Cặp split | Exact duplicate trajectories |
|---|---:|
| Train–Eval | 0 |
| Train–Test | 0 |
| Eval–Test | 0 |

Tên `source` có trùng giữa split và track-like token trước suffix cuối cũng có trùng:

| Cặp split | Source strings trùng | Track-like tokens trùng |
|---|---:|---:|
| Train–Eval | 2,734 | 65 |
| Train–Test | 5,661 | 123 |
| Eval–Test | 2,032 | 48 |

Điều này chưa phải bằng chứng leakage: `id/source` được tái sử dụng trong namespace riêng của từng split và không chứa raw sequence identity rõ ràng. Để chứng minh sequence-level separation tuyệt đối cần metadata split từ quá trình preprocessing hoặc raw IMPTC sequence identifiers.

Theo protocol baseline, ta sử dụng official preprocessed split và ghi caveat này trong báo cáo.

## DataLoader integration

`ConfigLoader` resolve đúng file eval và official `DataLoader` trả về:

```text
X:                  (19148, 32, 4)
y:                  (19148, 48, 2)
reference_position: (19148, 1, 2)
rotation_angle:     (19148,)
source:             (19148,)
```

Kết quả: **PASS**.

## Quyết định readiness

Dataset đủ điều kiện cho:

- chọn 8 fixed validation samples;
- test logging/checkpoint pipeline;
- smoke test ngắn.

Dataset audit không cho phép bắt đầu full training ngay. Theo experimental protocol, trước full run vẫn phải:

1. triển khai và test bug fixes;
2. triển khai seed/history/checkpoint/resume;
3. lưu fixed-sample GMM artifacts;
4. kiểm chứng sharpness;
5. chạy preflight smoke test và review artifact.

Machine-readable summary nằm trong `imptc_data_audit.json` cùng thư mục.
