# demo — chạy mô hình thật trong trình duyệt

Mở `index.html` bằng cách **nhấp đúp**. Không cần máy chủ, không cần cài gì.

Vẽ 3,2 giây quá khứ của một người đi bộ bằng chuột (hoặc chọn một mẫu thật).
LSTM + MDN đã huấn luyện trên IMPTC chạy ngay trong trang và trả về một **hỗn hợp
Gaussian** cho từng mốc trong 4,8 giây tương lai.

| | |
|---|---|
| Mô hình nhúng sẵn | **6** — softmax M = 1, 2, 3, 5, 8 và **sparsemax K≤8 đã huấn luyện** |
| Tổng tham số | 3 040 → 21 184 |
| Dung lượng trang | 644 KB, **toàn bộ dữ liệu nhúng trong file** |
| Phụ thuộc ngoài | chỉ font Google Fonts |

---

## Vì sao chạy được trong trình duyệt

Mô hình rất nhỏ. Baseline M = 3 chỉ có **8 224 tham số**:

```
lstm.weight_ih_l0   (32, 4)      128
lstm.weight_hh_l0   (32, 8)      256
lstm.bias_ih_l0     (32,)         32
lstm.bias_hh_l0     (32,)         32
fc.weight           (864, 8)    6912
fc.bias             (864,)       864
```

Toàn bộ forward pass viết lại bằng JavaScript trong `src/core.js`, khoảng 25 dòng.
Không dùng ONNX, không dùng TensorFlow.js.

---

## Kiểm chứng — phần quan trọng nhất

`tools/verify.js` đối chiếu bản JavaScript với **file dự đoán `.npz` do chính
PyTorch sinh ra trên GPU** và repo đã lưu (8 mẫu cố định × 48 mốc × K Gaussian):

```
base M=3 (softmax)
   mu     lệch tối đa 15.7 mm
   sigma  lệch tối đa 10.7 mm
   ✓ ĐẠT

sparsemax K≤8 (đã huấn luyện)
   mu     lệch tối đa 11.9 mm
   số 0 thật: JS 1662 · repo 1662  ✓ khớp
   K trung bình: JS 3.67 · repo 3.67
   ✓ ĐẠT
```

Chênh ~15 mm trên giá trị tới 6,25 m (**0,25 %**) là do repo chạy GPU ở độ chính
xác **TF32** (~1e−3 tương đối); bản JavaScript dùng float64 nên thực ra *chính xác
hơn*. Ở tỷ lệ vẽ 50 px/m thì 15 mm ≈ **0,8 pixel**.

Riêng với sparsemax, số lượng **số 0 thật sự** khớp tuyệt đối (1 662 / 3 072) —
đây là bằng chứng mạnh nhất rằng phép chiếu lên simplex được dựng lại đúng.

---

## Quy ước tiền xử lý (đã kiểm trên dữ liệu gốc)

| | |
|---|---|
| Hệ toạ độ | ego — gốc tại **bước 32**, hướng đi trùng **trục +y**, `−x` là trái |
| Vận tốc | `v[t] = (p[t] − p[t−1]) × 10`, `v[0]` sao chép từ `v[1]` — **khớp 0,00e+00** |
| Góc `φ` | lấy từ dịch chuyển **24 bước cuối** (khớp dữ liệu gốc nhất: lệch trung vị 0,54°) |

⚠️ Công thức `φ` chính xác của tác giả **không truy được** — script tiền xử lý không
có trong repo. Nhưng đã kiểm được điều quan trọng nhất: `φ` **không** dùng thông tin
tương lai (giả thuyết đó lệch 12,51°, tệ nhất trong các giả thuyết thử) → **không
có rò rỉ nhãn**.

---

## Mẫu thật trong demo

Sáu mẫu đầu chọn bằng **quy tắc**, không chọn tay: với mỗi `movement_class`, lấy
mẫu có sai số gần **trung vị** của nhãn đó nhất.

| Nhãn | Sai số mẫu | Trung vị nhãn (19 148 mẫu eval) |
|---|---|---|
| `standing` | 0,04 m | 0,04 m |
| `straight` | 0,67 m | 0,67 m |
| `light_left` / `light_right` | 1,30 / 1,28 m | 1,30 / 1,28 m |
| `strong_left` / `strong_right` | 2,39 / 2,17 m | 2,39 / 2,17 m |
| | | **tất cả: 0,88 m** |

Hai nút **viền nét đứt** là ca khó cố ý giữ lại (phân vị > 90 %):

- **Quay đầu** (sai 9,03 m) — người này đi lên tới `y = +3,91 m` ở giây 2,5 rồi quay
  lại tới `y = −1,19 m`. Mô hình bám sát 2 giây đầu rồi cứ ngoại suy thẳng. Hành vi
  quay đầu chỉ chiếm **0,82 %** dữ liệu và chỉ đến từ **90 / 1 881** người.
- **Trái gắt** (sai 5,50 m) — phân vị 94 % trong nhãn của nó.

Vì cú quay đầu chỉ rộng **0,48 m** theo chiều ngang nên đường đi và đường về gần
như chồng khít; xem các **mốc 1s–4s** trên đường nét đứt để thấy nó đi lên rồi tụt xuống.

---

## Hai chế độ sparsemax — đừng nhầm

**Nút `SPARSEMAX K≤8`** nạp checkpoint **đã huấn luyện thật**
(`sparsemax_k8_v2_seed2024`, epoch 1961, `architecture: sparsemax_mdn_v1`).
Khi chọn nó, hai nút softmax/sparsemax **tự khoá** — mô hình này train bằng
sparsemax, ép softmax lên nó là sai.

**Nút `sparsemax`** khi đang ở M = 1…8 thì khác hẳn: nó áp phép chiếu lên logit của
mô hình *softmax*, chỉ để minh hoạ cơ chế. Cách đó cho K trung bình **2,15**; mô
hình train thật được **3,27**.

---

## Dựng lại từ đầu

```bash
python tools/export_weights.py   # checkpoint .pt  -> assets/weights.json
python tools/make_presets.py     # dữ liệu eval    -> assets/presets.json + tham chiếu
python tools/build.py            # ghép            -> index.html
node   tools/verify.js           # đối chiếu với .npz của repo
```

`export_weights.py` đọc file `.pt` **không cần PyTorch** — file `.pt` là một zip
chứa pickle, script đọc bằng `zipfile` và một `Unpickler` thay mọi lớp torch bằng
stub. Checkpoint sparsemax không nằm trong repo nên script tự tải từ HuggingFace
(`quinha10/HTTM-GK-2026`); thiếu `huggingface_hub` thì chỉ bỏ qua mô hình đó.

### Cấu trúc

```
demo/
├── index.html          ← mở cái này (đã nhúng sẵn dữ liệu)
├── src/
│   ├── core.js           forward pass + sparsemax + biến đổi ego  (dùng chung cho trang và verify)
│   └── shell.html        khung trang; tools/build.py chèn core.js và JSON vào
├── data/
│   ├── weights.json      trọng số 6 mô hình (599 KB)
│   └── presets.json      8 mẫu thật kèm sự thật (11 KB)
└── tools/
    ├── export_weights.py
    ├── make_presets.py
    ├── build.py
    └── verify.js
```

Sửa giao diện thì sửa `src/shell.html`, sửa phần tính toán thì sửa `src/core.js`,
rồi chạy lại `build.py`. **Đừng sửa thẳng `index.html`** — nó là file sinh ra.

---

## Giới hạn

- Elip là **đường mức giải tích** của từng Gaussian, *không* phải vùng tin cậy
  Monte-Carlo mà `eval.py` dùng để tính S68 / S95 (lưới 36 × 36 m ở bước 0,1 m,
  chạy cho từng mẫu — quá nặng cho trình duyệt).
- Chỉ suy luận, không huấn luyện.
- Chỉ 8 mẫu, không nhúng cả 265 437 mẫu.
- Với đường **tự vẽ**, `φ` do demo tự xác định từ nét vẽ nên có thể lệch chút so
  với quy ước của tác giả; mũi tên nét đứt cho thấy hướng mà mô hình nhận ra.

## Nguồn số liệu

Chỉ số trong bảng bên phải đọc trực tiếp từ
`results/ablations/imptc_num_gaussians/ablation.json` (softmax) và
`.../sparsemax_k8_v2_seed2024/evaluation/test_best_clampedpi.json` (sparsemax,
toàn bộ 56 694 mẫu test). Phân bố K lấy từ `analysis/k_usage.json`
(19 148 mẫu validation).
