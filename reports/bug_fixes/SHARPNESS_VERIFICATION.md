# Sharpness Area Verification

## Kết luận

Official code cũ quy đổi tỷ lệ confidence mask sang diện tích bằng:

```text
mesh_range_x * mesh_range_y
```

Nhưng `build_mesh_grid()` tạo tọa độ:

```text
x ∈ [-mesh_range_x, +mesh_range_x]
y ∈ [-mesh_range_y, +mesh_range_y]
```

Với config IMPTC, miền grid là:

```text
x ∈ [-18,18] → width 36 m
y ∈ [-18,18] → height 36 m
```

Diện tích vật lý đầy đủ phải là:

```text
(2 × 18) × (2 × 18) = 1296 m²
```

Code cũ trả về `18 × 18 = 324 m²` cho full mask, thấp hơn đúng hệ số 4.

## Bug fix

Phép quy đổi được sửa thành:

```text
(2 * mesh_range_x) * (2 * mesh_range_y)
```

Không thay đổi cách tạo confidence level, Monte Carlo sampling hoặc cách aggregate sharpness theo thời gian. Đây là sửa lỗi đơn vị hình học, không phải thay đổi model hoặc loss.

## Reporting rule

Kết quả sharpness sinh bởi code đã sửa phải được ghi là `corrected`. Không so sánh trực tiếp với con số paper mà không chú thích rằng official repository cũ thiếu hệ số 4 trong phép quy đổi area.
