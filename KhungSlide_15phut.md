# Tài liệu nhóm — Mô hình hoá chuỗi thời gian nhiều chiều bằng GMM + LSTM

> **Đề bài:** Nhóm 3 SV tìm hiểu lý thuyết và trình bày demo một ứng dụng của **mô hình hoá chuỗi thời gian nhiều chiều dựa trên GMM kết hợp với một mô hình chuỗi thời gian** (vd: HMM, DNN,…).
>
> **Paper nền:** Hetzel, Reichert, Doll, Sick — *Reliable Probabilistic Human Trajectory Prediction for Autonomous Applications*, arXiv:**2410.06905v2** [cs.CV], 10/2024, 17 trang.
> **Code paper:** `https://github.com/kav-institute/mdn_trajectory_forecasting`

**Tổng thời lượng trình bày: 15 phút · 16 slide · 3 người × 5 phút.** Q&A tính riêng.

---

## Cách dùng tài liệu này

Tài liệu **tự chứa** — không cần đọc thêm file nào khác.

| Bạn là | Đọc bắt buộc | Đọc khi cần |
|---|---|---|
| **SV1** (lý thuyết GMM) | I.2 slide 1–6 · **II.2, II.3, II.4, II.5** | III.B (câu hỏi nhóm A, B, C) |
| **SV2** (mô hình & demo) | I.2 slide 7–10 · I.3 · **II.1, II.6** | III.A (code) · III.B (nhóm B, D) |
| **SV3** (đánh giá & đóng góp) | I.2 slide 11–16 · I.3 · I.5 · **II.7, II.8, II.9** | III.A (code) · III.B (nhóm D, E) |
| **Cả nhóm** | I.1 · I.4 · I.7 | III.C, III.D |

**Quy ước ký hiệu nguồn dùng xuyên suốt:**

| Ký hiệu | Nghĩa |
|---|---|
| 📄 | **Có trong paper** — kèm vị trí (Sec./Eq./Fig./Table/trang) |
| 🔧 | **Bổ sung của nhóm** — diễn giải, ví dụ, tái dựng; **không** có trong paper |
| ⚠️ | **Paper mơ hồ hoặc thiếu** — cẩn thận khi bảo vệ |

---

## Mục lục

**PHẦN I — KHUNG TRÌNH BÀY**
- [I.0 — Đối chiếu với dàn 12 ý ban đầu](#i0--đối-chiếu-với-dàn-12-ý-ban-đầu)
- [I.1 — Bảng tổng quan 16 slide](#i1--bảng-tổng-quan-16-slide)
- [I.2 — Chi tiết từng slide](#i2--chi-tiết-từng-slide)
- [I.3 — Ba màn demo: đặc tả kỹ thuật](#i3--ba-màn-demo-đặc-tả-kỹ-thuật)
- [I.4 — Phân công và tiến độ](#i4--phân-công-và-tiến-độ)
- [I.5 — Đóng góp riêng: đặc tả thí nghiệm](#i5--đóng-góp-riêng-đặc-tả-thí-nghiệm)
- [I.6 — Slide dự phòng](#i6--slide-dự-phòng)
- [I.7 — Checklist và phương án cháy giờ](#i7--checklist-và-phương-án-cháy-giờ)

**PHẦN II — LÝ THUYẾT NỀN**
- [II.1 — Bài toán và năm yêu cầu](#ii1--bài-toán-và-năm-yêu-cầu)
- [II.2 — Ký hiệu, Input/Output, Horizon](#ii2--ký-hiệu-inputoutput-horizon)
- [II.3 — Vì sao một quỹ đạo là chưa đủ](#ii3--vì-sao-một-quỹ-đạo-là-chưa-đủ)
- [II.4 — GMM và Eq. (1)](#ii4--gmm-và-eq-1)
- [II.5 — NLL, multi-modality và cầu nối EM](#ii5--nll-multi-modality-và-cầu-nối-em)
- [II.6 — LSTM, MDN head và kiến trúc](#ii6--lstm-mdn-head-và-kiến-trúc)
- [II.7 — Reliability và Calibration](#ii7--reliability-và-calibration)
- [II.8 — Kết quả thực nghiệm của paper](#ii8--kết-quả-thực-nghiệm-của-paper)
- [II.9 — Hạn chế: cái nào thật, cái nào sửa được](#ii9--hạn-chế-cái-nào-thật-cái-nào-sửa-được)

**PHẦN III — PHỤ LỤC**
- [III.A — Code tham khảo](#iiia--code-tham-khảo)
- [III.B — Ngân hàng câu hỏi](#iiib--ngân-hàng-câu-hỏi)
- [III.C — Tra cứu công thức, hình, bảng của paper](#iiic--tra-cứu-công-thức-hình-bảng-của-paper)
- [III.D — Thuật ngữ](#iiid--thuật-ngữ)

---
---

# PHẦN I — KHUNG TRÌNH BÀY

## I.0 — Đối chiếu với dàn 12 ý ban đầu

**Thay đổi cốt lõi: đổi trật tự kể chuyện.** Dàn 12 ý đi theo **thứ tự đọc paper**; khung slide đi theo **thứ tự lập luận**:

```
vấn đề → vì sao cần GMM → GMM là gì → ghép với LSTM → DEMO → kiểm chứng → đóng góp riêng
```

| Dàn 12 ý | Vào slide | Trạng thái |
|---|---|---|
| 1. Bài toán & mô hình | 1, 2 | ✅ Giữ |
| 2. I/O + chuỗi TG nhiều chiều | 2 (rút gọn) | 🔽 I/O chi tiết → dự phòng |
| 3. Vì sao 1 quỹ đạo chưa đủ | 3 | 🔼 **Đề cao** — slide bản lề dẫn vào GMM |
| 4. LSTM làm gì | 7 | ✅ Giữ, nén |
| 5. GMM & tại sao cần | 4 | 🔼 **Đề cao** — slide trung tâm |
| 6. MDN & quan hệ 3 tầng | 4 + 8 | 🔀 Gộp vào Eq. (1) và kiến trúc |
| 7. Công thức & NLL | 4, 5 | ✅ Giữ nguyên trọng lượng |
| 8. Reliability & Calibration | 11, 12 | 🔽 Nén mạnh (~25% → ~13%) |
| 9. Kiến trúc hoàn chỉnh | 8 | ✅ Giữ |
| 10. Dataset, preprocessing, code | — | ❌ Cắt → dự phòng |
| 11. Experiment, metric, kết quả | 14 | ❌ Cắt 6 thí nghiệm, giữ 1 con số |
| 12. Đóng góp, hạn chế, câu hỏi | 16 + dự phòng | 🔽 Câu hỏi → slide dự phòng |

**Ba thứ hoàn toàn mới:**

| Mới | Slide | Vì sao cần |
|---|---|---|
| **Cầu nối MDN ↔ EM** | 6 | Nối với Bài 5 — khiến bài thuộc về **môn học này** |
| **3 màn demo** | 9, 10, 13 | Đề bài **bắt buộc** "trình bày demo"; paper không có |
| **Ablation `M`** | 15 | Đóng góp riêng của nhóm |

**Kiểm tra phủ đề bài:**

| Yêu cầu đề bài | Slide đáp ứng |
|---|---|
| Chuỗi thời gian **nhiều chiều** | 2, 4, 10 |
| Dựa trên **GMM** | 4, 5, 6, 9, 15 |
| **Kết hợp** mô hình chuỗi thời gian | 7, 8 |
| **Demo** một ứng dụng | 9, 10, 13 |
| **Lý thuyết** | 2–6, 11, 12 |

---

## I.1 — Bảng tổng quan 16 slide

| # | Slide | Người | Phút | Loại |
|---|---|---|---|---|
| 1 | Bìa + Đề bài ↔ Paper | SV1 | 0.5 | Mở đầu |
| 2 | Bài toán & vì sao nhiều chiều | SV1 | 1.0 | Lý thuyết |
| 3 | ⭐ Vì sao 1 quỹ đạo chưa đủ | SV1 | 1.0 | Lý thuyết |
| 4 | ⭐ **Eq. (1) — GMM** | SV1 | 1.25 | Lý thuyết |
| 5 | Eq. (2) — NLL & multi-modality | SV1 | 1.0 | Lý thuyết |
| 6 | ⭐⭐ **Cầu nối MDN ↔ EM** | SV1 | 1.25 | Lý thuyết |
| 7 | LSTM làm gì | SV2 | 1.0 | Mô hình |
| 8 | ⭐ Kiến trúc end-to-end | SV2 | 1.25 | Mô hình |
| 9 | 🎬 ⭐ **DEMO 1 — `M=1` vs `M=3`** | SV2 | 1.5 | Demo |
| 10 | 🎬 **DEMO 2 — vùng giãn nở** | SV2 | 1.25 | Demo |
| 11 | ⭐ ADE/FDE chưa đủ + Calibration | SV3 | 1.0 | Đánh giá |
| 12 | Reliability & Sharpness | SV3 | 1.0 | Đánh giá |
| 13 | 🎬 **DEMO 3 — calibration plot** | SV3 | 0.75 | Demo |
| 14 | ⭐ Kết quả paper | SV3 | 0.75 | Đánh giá |
| 15 | ⭐⭐ **Đóng góp riêng — ablation `M`** | SV3 | 1.0 | Đóng góp |
| 16 | Kết luận | SV3 | 0.5 | Kết |

**Tổng: 15.0 phút** · SV1 = 6.0 · SV2 = 5.0 · SV3 = 5.0

> ⚠️ SV1 hơn 1 phút vì gánh slide bìa. Muốn cân đúng 5/5/5 thì để **SV3 đọc slide 1** rồi mời SV1.

---

## I.2 — Chi tiết từng slide

### Slide 1 — Bìa + Đề bài ↔ Paper · SV1 · 0.5 phút

**Mục tiêu:** trong 30 giây cho thấy nhóm bám **đúng** yêu cầu.

**Nội dung nói:** tên đề tài, 3 thành viên, paper gốc. Chỉ vào bảng: từng yêu cầu của đề bài đều có chỗ tương ứng.

**Hình cần vẽ — sơ đồ pipeline (giữa slide, to):**

```
Trajectory quá khứ ──► LSTM ──► MDN ──► Gaussian Mixture ──► dự đoán + uncertainty
   (chuỗi TG          (mô hình     (đầu ra      (GMM)            (vùng 68%/95%)
    nhiều chiều)       chuỗi TG)    xác suất)
```

**Bảng nhỏ bên dưới:**

| Yêu cầu đề bài | Trong paper |
|---|---|
| Chuỗi TG nhiều chiều | ✅ Quỹ đạo `[x, y, vx, vy]` theo thời gian |
| GMM | ✅ Hỗn hợp 3 Gaussian 2 biến |
| Kết hợp mô hình chuỗi TG | ✅ LSTM (loại "DNN") |
| Demo ứng dụng | ✅ Xe tự hành — người đi bộ |

**Câu chốt:** *"Kiến trúc của paper chính là bản dịch từng chữ của đề bài."*

---

### Slide 2 — Bài toán & vì sao chuỗi TG nhiều chiều · SV1 · 1.0 phút

📖 *Lý thuyết: [II.1](#ii1--bài-toán-và-năm-yêu-cầu), [II.2.3](#ii23--ba-tầng-nhiều-chiều)*

**Nội dung nói:** Xe tự hành chia đường với **VRU** (người đi bộ, xe đạp). Xe có thể chia sẻ đường đi của mình, nhưng quỹ đạo người thì **phải đoán**. Đây là chuỗi thời gian nhiều chiều ở **3 tầng**.

**Hình cần vẽ — 3 ô ngang:**

```
① KHÔNG GIAN              ② THỜI GIAN               ③ PHÂN BỐ
  mỗi mốc = [x, y]          48 bước đầu ra            nhiều nhánh khả năng
  x, y TƯƠNG QUAN           bất định TĂNG DẦN         (trái / thẳng / dừng)
       ↓                          ↓                          ↓
  cần tham số ρ            cần mô hình chuỗi         cần HỖN HỢP
                                                      (không phải 1 Gaussian)
```

**Công thức cần hiện:**

```
ᵂp_t = [ᵂx_t , ᵂy_t]          ← p = position (MỘT điểm)

ᵂT_in;t = { ᵂp_{t+h} | h ∈ H_in }     H_in = {0, −1, …, −n}    ← INPUT,  3.2 s
```

**Câu chốt:** *"Ba tầng này quyết định mọi lựa chọn kiến trúc phía sau: `ρ` cho tầng 1, LSTM cho tầng 2, GMM cho tầng 3."*

---

### Slide 3 — ⭐ Vì sao dự đoán 1 quỹ đạo là chưa đủ · SV1 · 1.0 phút

📖 *Lý thuyết: [II.3](#ii3--vì-sao-một-quỹ-đạo-là-chưa-đủ)*

**Mục tiêu:** **slide bản lề** — lý do tồn tại của GMM trong bài này.

**Nội dung nói:** Người đứng mép vỉa hè có thể **băng qua**, **đi dọc**, hoặc **đứng lại**. Huấn luyện bằng MSE → nghiệm tối ưu là **kỳ vọng có điều kiện** → rơi vào **giữa lòng đường**, nơi gần như chắc chắn không ai tới.

**Hình cần vẽ:**

```
      Thực tế (3 khả năng)                 Dự đoán bằng MSE
   ●━━━━━━━━━►  băng qua   40%
   ●━━━━━━━━━►  đi dọc     40%      ──►        ✗  ← trung bình
   ●            đứng lại   20%                     KHÔNG khớp
                                                   khả năng nào
```

**Công thức cần hiện:**

```
L_MSE = ‖p̂ − p‖²      ⟹   nghiệm tối ưu:  p̂* = E[ p | quá khứ ]
                                                └──────┬──────┘
                                            trung bình các khả năng
                                            = điểm giữa lòng đường
```

**Câu chốt:** *"Muốn tránh điều này, đầu ra phải là một **phân bố**, không phải một tọa độ. Và phân bố đó phải **đa mode**."*

---

### Slide 4 — ⭐ Eq. (1): GMM là đầu ra của mô hình · SV1 · 1.25 phút

📖 *Lý thuyết: [II.4](#ii4--gmm-và-eq-1)*

**Mục tiêu:** slide trung tâm của cả bài — đây chính là "GMM" mà đề bài yêu cầu.

**Công thức chính (to, giữa slide, có mũi tên chú thích):**

```
  𝒟_{t+h}( p_{t+h} )  =   Σ_{m ∈ M}   c_{m:t+h}   ·   ℱ_{m;t+h}( p_{t+h} )
  └────────┬────────┘     └────┬───┘  └────┬────┘      └────────┬────────┘
    mật độ tổng hợp        tổng qua      trọng số           mật độ Gaussian
     tại điểm p            3 mode      của mode m          2 biến của mode m
                                      (softmax, Σ=1)
```

**Công thức phụ — mỗi mode có 6 tham số:**

```
              1                    ⎡   −1    ⎛ (x−μx)²   (y−μy)²   2ρ(x−μx)(y−μy) ⎞ ⎤
ℱ_m(x,y) = ─────────────── · exp  ⎢ ─────── ⎜ ─────── + ─────── − ─────────────── ⎟ ⎥
           2π σx σy √(1−ρ²)        ⎣ 2(1−ρ²) ⎝   σx²       σy²         σx σy       ⎠ ⎦
```

**Hình cần vẽ — 3 ellipse minh họa vai trò tham số:**

```
     μ = tâm              σ = bán trục           ρ = độ nghiêng
    ╭─────╮              ╭───────────╮              ╭─╮
    │  ●  │      vs      │     ●     │     vs      ╱  ╱
    ╰─────╯              ╰───────────╯            ╰─╯
  "đi tới đâu"        "bất định bao nhiêu"    "x, y tương quan ra sao"
```

**Nội dung nói:**
- Với **mỗi** bước thời gian tương lai, mô hình xuất một hỗn hợp 3 Gaussian 2 biến.
- ⚠️ `𝒟(p)` là **mật độ** (đơn vị 1/m²), **có thể lớn hơn 1** — không phải xác suất.
- Bỏ dấu tổng đi (`M = 1`) thì quay về Social-LSTM, **không xử lý được đa mode**.

**Câu chốt:** *"Đây là GMM — nhưng khác Bài 5 ở chỗ tham số không cố định, mà được mạng dự đoán từ quá khứ."*

---

### Slide 5 — Eq. (2): NLL và vì sao sinh multi-modality · SV1 · 1.0 phút

📖 *Lý thuyết: [II.5.1](#ii51--hàm-mất-mát-eq-2), [II.5.2](#ii52--vì-sao-nll-sinh-multi-modality)*

**Công thức chính:**

```
L_NLL  =  − Σ_{h ∈ H_fc}  log 𝒟_{t+h}( p_{t+h} )
                              ↑              ↑
                        OUTPUT mô hình    NHÃN (ground truth)
```

**Công thức đối chiếu (phần đắt giá nhất slide):**

```
     log ( Σ_m c_m ℱ_m )          vs         Σ_m log ( c_m ℱ_m )
     └────────┬────────┘                     └────────┬────────┘
     chỉ cần MỘT mode giải                   MỌI mode phải giải
     thích tốt → loss thấp                   thích MỌI điểm
              ↓                                       ↓
     mode TỰ PHÂN CÔNG                        MODE COLLAPSE
     (không cần nhãn mode)                    (tất cả chụm một chỗ)
```

**Nội dung nói:**
- Tối thiểu NLL ≡ cực đại likelihood → đây là **MLE**, đúng như Bài 5.
- Với 1000 mẫu gồm 400 rẽ trái / 400 đi thẳng / 200 dừng, trọng số `c` **tự hội tụ** về ≈ 0.4 / 0.4 / 0.2 mà **không ai gán nhãn mode**. 📄 Paper gọi đây là *"semi-supervised"*.

**Câu chốt:** *"Vị trí của dấu log so với dấu tổng chính là thứ tạo ra tính đa mode."*

---

### Slide 6 — ⭐⭐ Cầu nối MDN ↔ EM · SV1 · 1.25 phút

📖 *Lý thuyết: [II.5.3](#ii53--cầu-nối-mdn--em)*

**Mục tiêu:** slide **không được cắt**. Nối paper với Bài 5 và chặn trước câu hỏi *"sao không dùng EM?"*.

**Bảng so sánh (nửa trên slide):**

| | **GMM cổ điển — Bài 5** | **MDN — paper này** |
|---|---|---|
| Mô hình hóa | `p(x)` — vô điều kiện | `p(y \| x)` — **có điều kiện** |
| Tham số `π, μ, Σ` | **hằng số**, ước lượng bằng EM | **hàm của input**, do LSTM dự đoán |
| Tối ưu | **EM**: E-step / M-step | **Gradient descent** (ADAM) |
| Chọn `K` / `M` | BIC, AIC | cố định `M = 3` |
| Nguyên lý | **MLE** | **Cũng là MLE** |

**Công thức chốt (nửa dưới slide, to):**

```
            ∂L                                  c_m · ℱ_m(p)
           ────  ∝  − γ_m · ( … ) ,     γ_m = ─────────────────
           ∂θ_m                               Σ_k c_k · ℱ_k(p)
                                                    ║
                                                    ║  CHÍNH LÀ
                                                    ▼
                                    responsibility  r_nk  của E-step
                                    (công thức 11.17, Bài 5)
```

**Câu chốt:** *"Huấn luyện MDN bằng gradient descent chính là chạy EM ở dạng mềm và liên tục — mỗi mode được cập nhật theo đúng tỉ lệ trách nhiệm của nó."*

---

### Slide 7 — LSTM làm gì trong mô hình · SV2 · 1.0 phút

📖 *Lý thuyết: [II.6.1](#ii61--lstm-làm-gì-và-vì-sao-chọn-lstm)*

**Hình cần vẽ:**

```
   ᵉp_{t−n} … ᵉp_{t−1}  ᵉp_t   (+ vận tốc)
        │       │        │
        ▼       ▼        ▼
   ╔═══════════════════════════╗
   ║   Stacked LSTM (8 lớp)    ║   ← nhận độ dài input BẤT KỲ
   ╚═══════════════════════════╝
                │
                ▼  hidden state h_t
           ┌─────────┐
           │ MDN Head│
           └─────────┘
```

**Bảng — 3 lý do chọn LSTM:**

| Lý do | Bằng chứng trong paper |
|---|---|
| **Độ dài input động** — người vừa ra khỏi vùng khuất chỉ có vài frame | Chỉ **0.5 s** quan sát đã đạt gần trọn hiệu năng (Table 4) |
| **Nhẹ** | **0.4 ms** — so với MID **12 giây** (Table 7) |
| **Chạy được trên nhúng** | 2.4 ms cho **128 dự đoán song song** trên board **10 W** (Table 5) |

**Câu chốt:** *"LSTM nén toàn bộ lịch sử chuyển động thành một vector — rồi MDN dịch vector đó thành tham số GMM."*

---

### Slide 8 — ⭐ Kiến trúc end-to-end · SV2 · 1.25 phút

📖 *Lý thuyết: [II.6.3](#ii63--kiến-trúc-end-to-end)*

**Hình cần vẽ — sơ đồ 6 khối, chiếm gần cả slide:**

```
      ᵂT_in;t  (tọa độ thế giới, độ dài thay đổi)
          │
          ▼
 ①  EGO TRANSFORM   xoay theo hướng đi φ_t + thêm vận tốc
          │              → chống bias vị trí, tăng khả năng chuyển miền
          ▼
 ②  STACKED LSTM    8 lớp → hidden state h_t
          │
          ▼
 ③  MDN HEAD        σ = exp(o)+1+ε  (>0)
          │         ρ = tanh(o)·ε   (∈ −1,1)
          │         c = softmax     (Σ=1)
          ▼
 ④  GAUSSIAN MIXTURE     𝒟_{t+h} = Σ_m c_m · ℱ_m        ← Eq. (1)
          │
    ┌─────┴─────────────────────┐
    ▼                           ▼
 ⑤ LOSS = NLL            ⑥ HẬU XỬ LÝ
   Eq. (2)                  lấy mẫu → vùng 68% / 95%
   [chỉ khi HUẤN LUYỆN]     [khi CHẠY THẬT]
```

**Nội dung nói:**
- Nhấn mạnh nhánh đôi ở cuối: **cùng một `𝒟`**, lúc train rẽ vào loss, lúc chạy thật rẽ vào vùng tin cậy.
- Nhắc nhanh vì sao `σ` phải có sàn `+1+ε`: nếu `σ → 0` thì `−log 𝒟 → −∞`, loss phân kỳ.

**Câu chốt:** *"Đây là toàn bộ mô hình — một encoder chuỗi, một head xác suất, không dùng bản đồ, không mô hình hóa tương tác."*

---

### Slide 9 — 🎬 ⭐ DEMO 1: `M = 1` vs `M = 3` · SV2 · 1.5 phút

📐 *Đặc tả: [I.3](#i3--ba-màn-demo-đặc-tả-kỹ-thuật)*

**Mục tiêu:** chứng minh **bằng mắt** vì sao đề bài cần GMM chứ không phải một Gaussian.

**Hình — hai ảnh contour cạnh nhau, cùng một input:**

```
        M = 1  (Gaussian đơn)              M = 3  (hỗn hợp)
     ┌───────────────────────┐          ┌───────────────────────┐
     │                       │          │    ╭───╮              │
     │     ╭───────────╮     │          │    │ A │      ╭───╮   │
     │     │     ●     │     │          │    ╰───╯      │ B │   │
     │     ╰───────────╯     │          │               ╰───╯   │
     └───────────────────────┘          └───────────────────────┘
      một ellipse phủ CẢ                  hai vùng TÁCH RỜI,
      khoảng giữa — nơi                   khoảng giữa để TRỐNG
      không ai đi
```

**Nội dung nói:** cùng một quỹ đạo đầu vào, chỉ khác số mode. Với `M = 1`, vùng 95% buộc phải phủ cả khoảng giữa hai nhánh. Với `M = 3`, vùng tách rời — đúng thực tế "rẽ trái **hoặc** đi thẳng, không đi vào giữa".

**Câu chốt:** *"Với xe tự hành, khác biệt này là sống còn: nếu đường đi hoạch định nằm đúng khoảng trống ở giữa, Gaussian đơn báo động giả, còn hỗn hợp cho phép đi qua an toàn."*

---

### Slide 10 — 🎬 DEMO 2: vùng tin cậy giãn nở · SV2 · 1.25 phút

**Hình — 4 khung ngang (hoặc GIF):**

```
   t+1s          t+2s          t+3s          t+4.8s
  ╭────╮        ╭──────╮      ╭────────╮    ╭──────────╮
  │ ⬤ │        │  ⬤   │      │   ⬤    │    │    ⬤     │
  ╰────╯        ╰──────╯      ╰────────╯    ╰──────────╯
   nhỏ                                        rộng hơn nhiều

  ─── quá khứ (đỏ)   ─── ground truth (đen)
  ▨ vùng 68% (trong)   ▨ vùng 95% (ngoài)
```

**Nội dung nói:** cùng một dự đoán, xem ở các mốc thời gian khác nhau. Vùng **giãn nở theo horizon** — đây chính là "chuỗi thời gian nhiều chiều" nhìn thấy được. Ghi `S68` / `S95` dưới mỗi khung.

**Câu chốt:** *"Bất định tăng dần theo thời gian — một dự đoán điểm không bao giờ nói được điều này."*

---

### Slide 11 — ⭐ ADE/FDE chưa đủ + Calibration là gì · SV3 · 1.0 phút

📖 *Lý thuyết: [II.7.1](#ii71--ba-khái-niệm-hay-bị-nhầm), [II.7.2](#ii72--calibration-là-gì)*

**Bảng (nửa trên):**

| Đo cái gì | Metric | Đủ chưa |
|---|---|---|
| Gần vị trí thật bao nhiêu mét | ADE, FDE | ❌ Không nói gì về độ tin cậy |
| Mật độ khớp dữ liệu tới đâu | NLL | ❌ NLL thấp **không** đảm bảo hiệu chuẩn |
| **Xác suất công bố có đúng tần suất thực tế** | **Reliability** | ✅ |

**Nội dung nói:**
- `minADE20` = sinh 20 quỹ đạo, lấy cái gần ground truth nhất. Nhưng **lúc xe chạy thật không có ground truth** để chọn.
- **Calibration:** *"Nếu mô hình nói 90%, thì đúng 90% số lần điều đó phải xảy ra."*
- Ví dụ: người dự báo nói "70% mưa" 100 lần → mưa ~70 lần là **calibrated**; mưa 30 lần là **quá tự tin**.

**Hình (nửa dưới) — hậu quả 2 kiểu lệch:**

```
   QUÁ TỰ TIN                        QUÁ DÈ DẶT
   nói 95% → thực tế 60%             nói 95% → thực tế 99.9%
   vùng QUÁ HẸP                      vùng QUÁ RỘNG
   → xe tưởng đường thoáng           → xe thấy đâu cũng nguy hiểm
   → ⚠️ VA CHẠM                       → phanh gấp liên tục, vô dụng
```

---

### Slide 12 — Reliability & Sharpness · SV3 · 1.0 phút

📖 *Lý thuyết: [II.7.4](#ii74--hai-metric-định-lượng-eq-5-6), [II.7.5](#ii75--sharpness-và-vì-sao-cần-cả-hai)*

**Công thức:**

```
R_avg = 1 − ────────────── · Σ_h Σ_α | (1−α) − f_o(1−α) |          ← Eq. (6)
             |H_fc| · |A|

R_min = 1 −  max_h max_α  | (1−α) − f_o(1−α) |                      ← Eq. (5)

     điểm hoàn hảo = 1      ngưỡng paper đặt:  R_avg ≥ 0.95,  R_min ≥ 0.90
```

**Nội dung nói:** `f_o(1−α)` = tần suất ground truth **thực sự** rơi vào vùng tin cậy mức `1−α`. Lý tưởng: `f_o(1−α) = 1−α` với mọi mức → vẽ ra **đường chéo**. `R_min` là **trường hợp xấu nhất** — quan trọng hơn cho an toàn.

**Hình — vì sao cần CẢ HAI metric:**

```
   Reliability CAO + Sharpness LỚN   →  đúng nhưng VÔ DỤNG
                                        (vùng phủ cả bản đồ thì luôn đúng)
   Reliability THẤP + Sharpness NHỎ  →  sắc nét nhưng NGUY HIỂM
   Reliability CAO + Sharpness NHỎ   →  ✅ MỤC TIÊU
```

**Câu chốt:** *"Sharpness (`S68`, `S95`) là diện tích vùng tin cậy. Mục tiêu là vùng nhỏ nhất có thể **với điều kiện** vẫn đáng tin."*

---

### Slide 13 — 🎬 DEMO 3: Calibration plot · SV3 · 0.75 phút

**Hình:**

```
  f_o  1.0 ┤                          ╱
 (quan     │                     ╱ ╱       ← đường chéo = LÝ TƯỞNG
  sát)     │                ╱  ╱
       0.5 ┤           ╱  ╱ ·········      ← dưới đường chéo = QUÁ TỰ TIN
           │      ╱  ╱ ····
           │ ╱  ╱····
       0.0 ┼──────────────────────────
           0.0        0.5        1.0
                 1 − α  (mức tin cậy công bố)

           R_avg = __._%     R_min = __._%      ← ghi thẳng lên hình
```

**Câu chốt:** *"Đường cong nằm dưới đường chéo nghĩa là mô hình quá tự tin — và đó là điều nguy hiểm nhất với xe tự hành."*

---

### Slide 14 — ⭐ Kết quả paper: ADE tốt ≠ Reliability tốt · SV3 · 0.75 phút

📖 *Lý thuyết: [II.8.3](#ii83--table-6--so-sánh-reliability-với-sota-)*

**Bảng rút gọn — chỉ 3 cột, đừng copy cả Table 6:**

| Mô hình | ADE ↓ | `R_min` ↑ |
|---|---|---|
| **Trajectron++** | **0.30** (tốt) | **30.9%** 🔴 |
| MID | 0.21 (rất tốt) | 65.9% |
| FlowChain | 0.29 | 62.2% |
| **Paper này** | 0.26 | **72.5%** |
| *Ngưỡng đáng tin* | — | *≥ 90%* |

**Nội dung nói:** nhóm tác giả **tự cài metric Reliability vào codebase của 4 phương pháp khác** rồi đo lại. Trajectron++ ADE tốt nhưng kịch bản xấu nhất lệch **69 điểm phần trăm**. **Không mô hình nào đạt ngưỡng** — kể cả chính paper.

**Câu chốt:** *"Đây là phát hiện chính: đo đúng khoảng cách không có nghĩa là đo đúng độ tin cậy."*

---

### Slide 15 — ⭐⭐ Đóng góp riêng: ablation số mode `M` · SV3 · 1.0 phút

📐 *Đặc tả: [I.5](#i5--đóng-góp-riêng-đặc-tả-thí-nghiệm)*

**Bố cục 3 ô ngang:**

```
┌─ LỖ HỔNG ──────────┐  ┌─ THIẾT LẬP ────────┐  ┌─ KẾT QUẢ ──────────┐
│ Paper cố định M = 3 │  │ M ∈ {1,2,3,5,8}    │  │  [đồ thị]          │
│ chỉ bằng MỘT trích  │  │ ETH/UCY            │  │  trục ngang: M     │
│ dẫn — KHÔNG kiểm    │  │ giữ nguyên mọi     │  │  trái:  R_avg,R_min│
│ chứng.              │  │ siêu tham số khác  │  │  phải: S95, ADE    │
│                     │  │                    │  │                    │
│ Nhưng chọn số thành │  │ Đo: R_avg, R_min,  │  │  → có bão hòa?     │
│ phần là CÂU HỎI     │  │     S68, S95,      │  │  → M=3 tối ưu?     │
│ KINH ĐIỂN của GMM   │  │     ADE, FDE       │  │                    │
│ (Bài 5: BIC/AIC)    │  │                    │  │                    │
└─────────────────────┘  └────────────────────┘  └────────────────────┘
```

**Nội dung nói:** paper lập luận *"một Gaussian không đủ"* bằng trích dẫn, rồi chọn 3 cũng bằng trích dẫn — **không thử gì cả**. Nếu `M` lớn hơn cho `R_min` tốt hơn, kết luận *"không mô hình nào đủ tin cậy"* cần xét lại.

⚠️ **Nói rõ giới hạn:** chỉ 1 dataset, 1 seed → kết quả mang tính **chỉ dấu**.

**Câu chốt:** *"Đây là câu hỏi chọn `K` của GMM cổ điển, đặt lại trong bối cảnh GMM có điều kiện."*

---

### Slide 16 — Kết luận · SV3 · 0.5 phút

**Ba dòng, không hơn:**

```
1. MÔ HÌNH
   GMM có điều kiện (MDN) + LSTM → mô hình hóa chuỗi thời gian nhiều chiều,
   xuất phân bố xác suất thay vì tọa độ.

2. PHÁT HIỆN CHÍNH CỦA PAPER
   ADE/FDE tốt KHÔNG kéo theo độ tin cậy tốt.
   Không mô hình nào trong 5 mô hình đạt ngưỡng an toàn.

3. HẠN CHẾ
   • Không kiểm chứng số mode M  → nhóm đã thử (slide 15)
   • Không công bố N trong ước lượng Monte Carlo
   • Không ràng buộc nhất quán thời gian giữa các mode
```

---

## I.3 — Ba màn demo: đặc tả kỹ thuật

### 🎬 Demo 1 — `M = 1` vs `M = 3`

| Hạng mục | Chi tiết |
|---|---|
| **Mục tiêu** | Chứng minh trực quan vì sao cần **hỗn hợp**, không phải một Gaussian |
| **Chuẩn bị** | Train 2 mô hình: `M = 1` và `M = 3`, **giữ nguyên** mọi siêu tham số khác |
| **Dữ liệu** | ETH/UCY, tập test |
| **Chọn mẫu** | ⚠️ **Quan trọng nhất** — chọn quỹ đạo ở **điểm rẽ**. Duyệt tập test, chọn mẫu mà `M=3` cho 2 mode có trọng số `c` gần bằng nhau (vd 0.45 / 0.40) |
| **Vẽ** | 2 subplot cạnh nhau, **cùng scale trục**. Mỗi cái: contour 68% + 95%, quỹ đạo quá khứ (đỏ), ground truth (đen) |
| **Ghi trên hình** | `S95` của từng mô hình → cho thấy `M=1` có vùng **to hơn** |
| **Rủi ro** | Nếu mẫu không đủ đa mode, hai hình sẽ giống nhau → **mất cả màn diễn**. Duyệt vài chục mẫu, chọn cái rõ nhất |

### 🎬 Demo 2 — Vùng tin cậy giãn nở theo horizon

| Hạng mục | Chi tiết |
|---|---|
| **Mục tiêu** | Cho thấy chiều thời gian và bất định tăng dần |
| **Chuẩn bị** | Dùng mô hình `M = 3` đã train ở Demo 1 |
| **Vẽ** | 4 khung tại `t+1s`, `t+2s`, `t+3s`, `t+4.8s` (hoặc GIF). **Cùng scale trục** cho cả 4 |
| **Lớp vẽ** | Quá khứ (đỏ) · ground truth (đen) · vùng 68% (đậm) · vùng 95% (nhạt) |
| **Ghi trên hình** | `S68` và `S95` dưới mỗi khung |
| **Cách tính vùng** | Lấy `N` mẫu từ GMM → tính mật độ tại từng mẫu → ngưỡng theo phân vị 68%/95% → vẽ contour (Eq. 3, 4) |
| **Mẹo** | Nếu làm GIF: 1.5 s/khung, lặp. Nhúng dạng GIF hoặc video |

### 🎬 Demo 3 — Calibration plot

| Hạng mục | Chi tiết |
|---|---|
| **Mục tiêu** | Định lượng độ tin cậy của GMM nhóm train được |
| **Chuẩn bị** | Với **mọi** cặp (phân bố dự đoán, ground truth) trong tập test: tính `1−α(p)` theo Eq. (3) |
| **Vẽ** | Trục ngang `1−α` ∈ [0,1], trục dọc `f_o`. Vài đường cho vài horizon + đường chéo nét đứt |
| **Ghi trên hình** | `R_avg`, `R_min` theo Eq. (5), (6) với `A = {0.01, …, 0.99}` |
| **Lưu ý** | ⚠️ Kết quả phụ thuộc `N` (số mẫu Monte Carlo). Dùng `N ≥ 1000`. Nếu có thời gian, vẽ thêm đường với vài `N` khác nhau — **đây chính là một hạn chế của paper mà nhóm chỉ ra được** |

---

## I.4 — Phân công và tiến độ

| | Phụ trách | Slide | Phút | Sản phẩm phải làm |
|---|---|---|---|---|
| **SV1** | Lý thuyết GMM — Eq. (1), NLL/MLE, multi-modality, cầu nối EM | 1–6 | 6.0 | Nội dung lý thuyết + hình ellipse minh họa |
| **SV2** | Mô hình & cài đặt — LSTM, MDN head, kiến trúc | 7–10 | 5.0 | **Code chạy được** + Demo 1 + Demo 2 |
| **SV3** | Đánh giá + đóng góp + kết | 11–16 | 5.0 | Cài metric Reliability/Sharpness + Demo 3 + chạy ablation |

**Thứ tự phụ thuộc:**

```
SV2 (code + train M=1, M=3)
        │
        ├──────────────► Demo 1, Demo 2   [SV2]
        │
        └──► SV3 (cài metric R, S) ──────► Demo 3   [SV3]
                    │
                    └──► train M ∈ {1,2,3,5,8} ──► Đóng góp riêng   [cả nhóm]
```

⚠️ **SV2 là nút cổ chai — phải bắt đầu sớm nhất.** SV3 không làm được gì cho tới khi có mô hình. SV1 làm độc lập, chạy song song.

**Năm mốc:**

| Mốc | Việc | Ai |
|---|---|---|
| 1 | Chạy được code paper trên ETH/UCY, train xong `M=3` | SV2 |
| 2 | Train xong `M=1`; chọn mẫu, dựng Demo 1 và Demo 2 | SV2 |
| 3 | Cài xong metric `R_avg`/`R_min`/`S68`/`S95`; dựng Demo 3 | SV3 |
| 4 | Train `M ∈ {2, 5, 8}`; vẽ đồ thị ablation | SV3 + SV2 |
| 5 | Hoàn thiện slide; **chạy thử bấm giờ** ít nhất 2 lần | Cả nhóm |

---

## I.5 — Đóng góp riêng: đặc tả thí nghiệm

**Tên:** *Ảnh hưởng của số thành phần hỗn hợp `M` tới độ tin cậy và độ sắc nét*

### Vì sao chọn thí nghiệm này

| Lý do | Giải thích |
|---|---|
| **Đúng trọng tâm đề bài** | Chọn số thành phần là **câu hỏi kinh điển của GMM** — Bài 5 giải bằng BIC/AIC |
| **Lấp lỗ hổng thật** | Paper cố định `M = 3` chỉ bằng một trích dẫn [17], **không có ablation** |
| **Rẻ** | ETH/UCY rất nhỏ, mô hình siêu nhẹ (8 lớp LSTM) |
| **Tái sử dụng** | `M = 1` cho luôn **Demo 1** — không tốn thêm công |
| **Đụng vào kết luận chính** | Nếu `M` lớn cho `R_min` tốt hơn, câu *"no model achieves suitable reliability"* lung lay |

### Thiết lập

| Hạng mục | Giá trị |
|---|---|
| **Biến thay đổi** | `M ∈ {1, 2, 3, 5, 8}` |
| **Giữ nguyên** | 8 lớp LSTM · ADAM · lr `1e−3` → `1e−7` · batch · số epoch · seed |
| **Dữ liệu** | ETH/UCY (nếu kịp, thêm 1 subset nữa) |
| **Đo** | `R_avg`, `R_min`, `S68`, `S95`, `minADE20`, `minFDE20` |
| **Số lần chạy** | Tối thiểu 1 seed. **Nếu kịp thì 3 seed** rồi lấy trung bình — mạnh hơn nhiều, tốn thêm rất ít |

### Đồ thị trình bày

```
  R (%)                                              S95 (m²/s)
   100 ┤                                                  ┤ 4
       │      ╭────●─────────●──────────●                 │
    90 ┤     ╱          R_avg                             ┤ 3
       │    ╱                                             │
    80 ┤   ●      ╭──●────────●──────────●                ┤ 2
       │  ╱      ╱       R_min                            │
    70 ┤ ●      ●                                         ┤ 1
       │     ╲──────────────────────────  S95 (nét đứt)   │
    60 ┼───┬───┬───┬───┬───┬───────────────────────────── ┼ 0
        M=1   2   3   5   8
```

### Ba kịch bản kết luận — chuẩn bị sẵn cả ba

| Nếu kết quả cho thấy | Kết luận sẽ là |
|---|---|
| `M` tăng → `R_min` **tăng rõ** | Paper chọn `M = 3` **chưa tối ưu**; kết luận *"no model achieves suitable reliability"* cần xét lại |
| `R_min` **bão hòa** từ `M = 3` | **Xác nhận bằng thực nghiệm** điều paper chỉ khẳng định bằng trích dẫn. Thêm mode chỉ tốn tài nguyên |
| `M` lớn → `R` **xấu đi** | Dấu hiệu **overfit / mode collapse** — nối thẳng với vấn đề chọn `K` trong GMM cổ điển |

> ✅ **Cả ba kịch bản đều là kết quả đáng báo cáo.** Thí nghiệm này an toàn: không có khả năng "chạy xong không có gì để nói".

⚠️ **Giới hạn phải nói rõ trong slide:** chỉ **một dataset nhỏ**, **một seed** → kết quả mang tính **chỉ dấu**. Nêu thẳng điều này tốt hơn là để thầy chỉ ra.

---

## I.6 — Slide dự phòng

Đặt **sau slide 16**, không trình, chỉ mở khi bị hỏi.

| Slide dự phòng | Dùng khi bị hỏi | 📖 Lý thuyết |
|---|---|---|
| Input / Nhãn / Output — 3 vai trò | *"`T_gt` là input hay output?"* | [II.2.4](#ii24--ba-vai-trò-input--nhãn--output) |
| Observation vs Forecast horizon | *"3.2 s và 4.8 s là gì?"* | [II.2.2](#ii22--observation-horizon-và-forecast-horizon) |
| Ego coordinate transform | *"Sao phải xoay tọa độ?"* | [II.2.5](#ii25--ego-coordinate-transform) |
| Hàm kích hoạt MDN | *"Sao `σ` phải có sàn `+1+ε`?"* | [II.6.2](#ii62--mdn-head-và-hàm-kích-hoạt) |
| GMM hợp lệ — tổ hợp lồi | *"Sao bắt buộc softmax?"* | [II.4.3](#ii43--vì-sao-eq-1-vẫn-là-phân-bố-hợp-lệ) |
| 4 công thức — 4 vai trò | *"Eq. (2) có phải output không?"* | [II.4.5](#ii45--bốn-công-thức-bốn-vai-trò) |
| CL & Confidence Set, HDR | *"Vùng 95% tính thế nào?"* | [II.7.3](#ii73--confidence-level-và-confidence-set-eq-3-4) |
| Ví dụ số `𝒟(p) = 0.338` | *"Cho ví dụ cụ thể được không?"* | [II.4.4](#ii44--ví-dụ-số-cụ-thể) |
| Dataset & cấu hình huấn luyện | *"Train trên gì, bao lâu?"* | [II.6.4](#ii64--dữ-liệu-và-cấu-hình-huấn-luyện) |
| Tốc độ & transferability | *"Nhanh cỡ nào? Chuyển miền được không?"* | [II.8.4](#ii84--tốc-độ-và-transferability) |
| 3 lỗi hay mắc khi đọc Eq. (1) | *"3 mode có phải 3 quỹ đạo không?"* | [II.4.6](#ii46--ba-lỗi-hay-mắc-khi-đọc-eq-1) |
| Hạn chế đầy đủ | *"Paper thiếu sót gì?"* | [II.9](#ii9--hạn-chế-cái-nào-thật-cái-nào-sửa-được) |
| Bộ câu hỏi đã chuẩn bị | mọi câu khác | [III.B](#iiib--ngân-hàng-câu-hỏi) |

---

## I.7 — Checklist và phương án cháy giờ

### Checklist trước buổi trình bày

- [ ] Chạy thử **bấm giờ** ít nhất 2 lần, cả nhóm liên tục
- [ ] Demo đã **xuất ra ảnh/GIF tĩnh** nhúng vào slide — **không chạy code live**
- [ ] Mẫu Demo 1 đã chọn kỹ, hai hình **khác nhau rõ rệt**
- [ ] Slide dự phòng đã để sau slide 16
- [ ] Mỗi người biết **câu chốt** của slide mình
- [ ] Biết chuyển tiếp giữa 3 người (ai nói câu gì khi bàn giao)
- [ ] Có bản PDF dự phòng phòng lỗi phần mềm

### Nếu cháy giờ — thứ tự bỏ

Bỏ lần lượt **từ trên xuống**, không nhảy cóc:

```
1️⃣  Slide 10 (DEMO 2)        → vẫn còn 2 demo, đủ đáp ứng "trình bày demo"
2️⃣  Slide 14 (kết quả paper)  → gộp 1 câu vào slide 13
3️⃣  Slide 2 (bài toán)        → gộp vào slide 1
────────────── VẠCH ĐỎ — dưới đây KHÔNG bỏ ──────────────
    Slide 4  (Eq. 1)          → đây LÀ đề bài
    Slide 6  (cầu nối EM)     → đây là môn học
    Slide 9  (DEMO 1)         → đây là "demo" và là lý lẽ cho GMM
    Slide 15 (đóng góp riêng) → đây là điểm cộng
```

### Nếu thừa giờ

1. Mở slide dự phòng **CL & Confidence Set** — cách dựng vùng 68%/95%
2. Mở slide dự phòng **ví dụ số** `𝒟(p) = 0.338` — mode gần chiếm 99%
3. Mở **tốc độ** — 0.4 ms vs MID 12 giây

---
---

# PHẦN II — LÝ THUYẾT NỀN

## II.1 — Bài toán và năm yêu cầu

📍 *Abstract (tr.1) · Sec. 1.1 (tr.1–2) · Sec. 1.2 (tr.2–3)*

### Bài toán

📄 **Human Trajectory Prediction (HTP)** — dự đoán vị trí tương lai của người đi bộ, phục vụ **xe tự hành (AV)** và **máy tự hành (AM)**.

Paper **không** giải bài toán theo nghĩa "dự đoán chính xác hơn". Bài toán thật sự:

> **Làm sao dự đoán quỹ đạo kèm độ bất định mà độ bất định đó ĐÁNG TIN về mặt thống kê, đồng thời đủ nhẹ để chạy real-time trên phần cứng nhúng?**

📄 Câu chốt — **Sec. 1.1, tr.2**:
> *"If a prediction states that a pedestrian has a 90% probability of being within a specified area, the prediction should occur 90% of the time it is made. Otherwise, downstream tasks like path-planing cannot trust these predictions."*

### Vì sao đây là vấn đề

| Thực trạng | Vấn đề | 📍 |
|---|---|---|
| Đánh giá bằng **Best-of-N ADE/FDE** | Chạy thật **không có ground truth** để chọn "cái tốt nhất" | Sec. 1.1, tr.2 |
| Dùng **NLL / KDE-NLL** | *"A low NLL does not necessarily mean that a model's distribution is reliable"* | Sec. 2, tr.5 |
| Mạng nơ-ron **miscalibrated** | Ví von hiệu ứng **Dunning-Kruger** [13]; dẫn Guo et al. [7] ICML 2017 | Sec. 1.1, tr.2 |
| Chuẩn input **3.2 giây** | *"An initial latency of 3.2s to the first prediction is unsuitable for AV applications"* | Sec. 1.2, tr.3 |
| Chỉ test **ETH/UCY** | Schoeller [31]: 5 subset tương quan; **CVM** đơn giản đạt ADE/FDE ngang SOTA | Sec. 2, tr.5 |

### Năm yêu cầu định hình thiết kế

📍 *Sec. 1.2, tr.2–3*

| Yêu cầu 📄 | Nội dung | Quyết định thiết kế 🔧 |
|---|---|---|
| **Uncertainty Modeling** | Xuất vùng tin cậy, không chỉ một điểm | MDN → phân bố xác suất |
| **Reliability & Confidence** | Độ bất định phải đúng thống kê | Metric `R_avg`, `R_min` |
| **Usability** | Vùng **sắc nét** mà **vẫn** đáng tin | Metric `S68`, `S95` |
| **Real-World Applicability** | Dữ liệu thưa, input động, real-time, nhiều VRU | LSTM + mô hình siêu nhẹ |
| **Transferability** | Không bias địa điểm; ≥ 3 dataset khác domain | Ego coordinate + 4 dataset |

---

## II.2 — Ký hiệu, Input/Output, Horizon

📍 *Sec. 3, tr.6 "Human Egocentric Coordinate Transformation"*

### II.2.1 — Bảng ký hiệu

```
        ┌── hệ tọa độ: w = world (thế giới), e = ego (gắn với người)
        │
        ᵂ p _t   =  [ ᵂx_t , ᵂy_t ]
          │  │         │      │
          │  │         └──────┴── hai thành phần tọa độ
          │  └── thời điểm
          └── position = VỊ TRÍ (MỘT điểm)

        ᵂ T _in ; t
          │  │    │
          │  │    └── "quy chiếu về thời điểm t"
          │  └── in = input   |   gt = ground truth
          └── Trajectory = QUỸ ĐẠO (một TẬP các điểm p)
```

| Ký hiệu 📄 | Đọc là | Là gì |
|---|---|---|
| **`p`** | **position** | **một điểm** `[x, y]` |
| **`T`** | **Trajectory** | **một tập điểm** — quỹ đạo |
| `ᵂ` / `ᵉ` (mũ trái) | world / ego | hệ tọa độ |
| `t` | time | thời điểm "hiện tại" của mẫu |
| `h` | horizon offset | độ lệch thời gian **tương đối** so với `t` |
| `;t` | "tại mốc `t`" | quỹ đạo cắt ra quanh thời điểm `t` |

🔧 **Quan hệ `p` ↔ `T`:** `p` là phần tử, `T` là tập:

```
ᵂT_in;t = { ᵂp_t , ᵂp_{t−1} , … , ᵂp_{t−n} }
```

🔧 **Ý nghĩa `;t`:** cùng một track sinh ra **nhiều mẫu** tùy chọn `t` — đây là cơ chế 📄 *"Longer tracks are split into subsamples"* (Sec. 4.1, tr.8).

### II.2.2 — Observation horizon và Forecast horizon

```
          QUÁ KHỨ                  HIỆN TẠI              TƯƠNG LAI
  ├────────────────────────────────────┼──────────────────────────────────┤
  h=−n   …   h=−2   h=−1            h=0            h=1   h=2   …   h=m
  └──────────────────────────────────────┘          └────────────────────┘
        H_in = {0, −1, −2, …, −n}                 H_fc = {1, 2, …, m}
        OBSERVATION HORIZON                       FORECAST HORIZON
              3.2 s                                      4.8 s
           32 bước @10Hz                              48 bước @10Hz
```

| | **`H_in`** | **`H_fc`** |
|---|---|---|
| Nghĩa | Nhìn **lùi** vào quá khứ | Nhìn **tới** tương lai |
| Định nghĩa 📄 | `{0, −1, −2, …, −n}` | `{1, 2, …, m}` |
| Dấu `h` | 0 và **âm** | **dương** |
| Độ dài 📄 | **3.2 s** | **4.8 s** |
| Cố định? | ❌ **KHÔNG** — thay đổi được (Table 4) | ✅ Có |

⚠️ **Hai chi tiết dễ bỏ sót:**
1. **`h = 0` thuộc `H_in`** → **vị trí hiện tại cũng là đầu vào**. `H_in` có `n+1` phần tử.
2. Paper dùng **lẫn hai tên**: *"input horizon"* (Sec. 3 tr.6) và *"observation horizon"* (Sec. 4.3 tr.11) — cùng là `H_in`.

### II.2.3 — Ba tầng nhiều chiều

🔧 *(Paper không có mục riêng; đây là tổng hợp từ các mảnh rải rác.)*

| Tầng | Nội dung | 📄 Bằng chứng |
|---|---|---|
| **① Không gian** | Mỗi mốc là vector `[x, y]`, hai trục **tương quan** | Mô hình dùng tham số `ρ` (Sec. 3, tr.6) |
| **② Thời gian** | Xuất **đồng thời 48 phân bố**, bất định tăng dần | `H_fc = {1,…,m}`, loss tổng qua mọi `h` (Eq. 2) |
| **③ Phân bố** | Tại một mốc, tương lai **tách nhiều nhánh** | *"a single normal distribution … is not sufficient"* (Sec. 3, tr.6) |

🔧 Tầng thứ tư (ít khi kể riêng): **độ dài chuỗi input biến đổi** — 📄 *"occlusions, dynamic observation times, sensor errors"* (Sec. 1.2, tr.3).

### II.2.4 — Ba vai trò: input ≠ nhãn ≠ output

> ⚠️ **Chỗ cực kỳ dễ nhầm.** `ᵂT_in;t` và `ᵂT_gt;t` **đều đến từ dữ liệu**, nhưng **chỉ MỘT cái đi vào mô hình**.

**① INPUT** 📄 — đi vào mô hình:
```
ᵂT_in;t = { ᵂp_{t+h} | h ∈ H_in },     H_in = {0, −1, …, −n}
```

**② NHÃN (ground truth)** 📄 — **KHÔNG** đi vào mô hình, chỉ dùng trong loss:
```
ᵂT_gt;t = { ᵂp_{t+h} | h ∈ H_fc },     H_fc = {1, 2, …, m}
```
📄 *(Sec. 3, tr.6)*: *"We use the future trajectory `ᵂT_gt;t` … **as ground truth**."*

**③ OUTPUT** 📄 — thứ mô hình **thực sự sinh ra**: 48 phân bố `𝒟_{t+h}` (Eq. 1), **không phải tọa độ**.

```
                    ┌─────────────────────┐
                    │  Track đã ghi       │
                    └──────────┬──────────┘
                    cắt tại t  │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      ① ᵂT_in;t                          ② ᵂT_gt;t
              │                                 │
              ▼                                 │  ⛔ KHÔNG vào mô hình
    ╔═════════════════╗                         │
    ║  LSTM + MDN     ║                         │
    ╚═════════════════╝                         │
              │                                 │
              ▼  ③ 𝒟_{t+h}                      │
              └────────────┬────────────────────┘
                           ▼
                  ╔════════════════════╗
                  ║  HÀM LOSS  Eq.(2)  ║  ← chỉ ở ĐÂY ② và ③ gặp nhau
                  ╚════════════════════╝
```

| | `ᵂT_in;t` | `ᵂT_gt;t` | `𝒟_{t+h}` |
|---|---|---|---|
| Vai trò | **x** (đầu vào) | **y** (nhãn) | **ŷ** (đầu ra) |
| Vào **mô hình**? | ✅ | ❌ | — |
| Vào **loss**? | ❌ | ✅ | ✅ |
| Lúc **chạy thật**? | ✅ | ❌ **không tồn tại** | ✅ |

🔧 **Ví von:** giống bài thi — đề bài (`T_in`) đưa cho thí sinh, đáp án (`T_gt`) **giấu đi**, chỉ lấy ra khi chấm.

⚠️ **Điểm mấu chốt của cả paper:** ground truth **chỉ tồn tại trong dataset**. Khi xe chạy thật, tương lai chưa xảy ra → **không tính được ADE/FDE**. Đó là lý do phải có Reliability.

🔧 **Chống nhầm:** mô hình **non-autoregressive** — xuất cả 48 phân bố **cùng lúc**, **không có teacher forcing**.

⚠️ Paper nói phép ego transform áp cho **cả hai** — nhưng đó là **tiền xử lý nhãn** để cùng hệ tọa độ với output, **không phải** đưa nhãn vào mô hình.

### II.2.5 — Ego coordinate transform

📄 *Sec. 3, tr.6* — `φ_t` là **hướng di chuyển** của VRU tại `t`; phép biến đổi áp cho cả `ᵂT_in;t` và `ᵂT_gt;t`.

⚠️ Paper **không viết ma trận xoay ra**. 🔧 Dạng chuẩn:

```
ᵉp_{t+h} = R(−φ_t) · ( ᵂp_{t+h} − ᵂp_t )

          ⎡  cos φ_t   sin φ_t ⎤
R(−φ_t) = ⎢                    ⎥
          ⎣ −sin φ_t   cos φ_t ⎦
```

📄 **Ba lý do** *(Sec. 3, tr.6: "solve location-based circumstances, help to avoid a systematic bias, and increase transferability")*:
1. Hai người ở hai góc giao lộ khác nhau, hành vi giống nhau → sau chuẩn hóa thành **cùng một mẫu**
2. Không học thuộc "ở tọa độ (12, 45) thì hay đi sang phải"
3. Kiểm chứng ở Table 3 — mô hình train trên inD vẫn đạt `R_avg` 94.8% trên Waymo

### II.2.6 — I/O ở mức code

> ⚠️ **Paper KHÔNG công bố shape tensor.** 🔧 Tái dựng từ `H_in`/`H_fc`, tần số, `M = 3`, batch 1024.

```
INPUT :  x      (B, T_in, 4)      [ᵉx, ᵉy, ᵉvx, ᵉvy]   T_in thay đổi được
         ⚠️ paper chỉ nói "positional information + velocity", không nói F = 4

TRUNG GIAN: h   (B, H)            ⚠️ paper nói "8 Hidden Layers" = SỐ LỚP,
                                     không công bố số unit → H không xác định được

OUTPUT:  mu     (B, 48, 3, 2)     tâm   [μx, μy]
         sigma  (B, 48, 3, 2)     [σx, σy]  > 0, có sàn
         rho    (B, 48, 3)        ∈ (−1, 1) ngặt
         c      (B, 48, 3)        softmax, Σ = 1
                                  → tổng B × 48 × 3 × 6 = B × 864 con số
```

---

## II.3 — Vì sao một quỹ đạo là chưa đủ

📍 *Sec. 1.1 (tr.2) · Sec. 1.2 (tr.2) · Sec. 2 (tr.4, tr.5) · Sec. 3 (tr.7)*

### 1. Tương lai đa mode — trung bình là vị trí vô nghĩa

📄 **Sec. 2, tr.4** — *"Social-LSTM [1] and Social-STGCNN [25] use unimodal Gaussian distributions … but **cannot handle multimodality**."*

🔧 Người đứng mép vỉa hè: **băng qua** / **đi dọc** / **đứng lại**. Huấn luyện MSE → nghiệm tối ưu là kỳ vọng có điều kiện `E[p | quá khứ]` → rơi vào **giữa lòng đường**.

### 2. Không có "một quỹ đạo đúng" để đo sai số

📄 **Sec. 1.1, tr.2** — *"Human movements are in-deterministic by nature, and movement paths could be changed quickly."*

### 3. Path planning cần VÙNG, không cần ĐIỂM

📄 **Sec. 3, tr.7** — *"CLs enable the estimate of the probability of a collision if the planned path of an autonomous system is given."*

### 4. Best-of-N che giấu 19 dự đoán tồi

📄 **Sec. 2, tr.5** — *"BoN ADE/FDE cannot measure a model's distribution reliability because they **do not consider all generated samples**."*
📄 **Sec. 1.1, tr.2** — *"for deterministic evaluation at execution time, **no ground truth data is available**."*

### 5. Dự đoán điểm không mang thông tin về mức tin cậy

🔧 Hai tình huống rủi ro hoàn toàn khác nhau nhưng dự đoán điểm cho **kết quả giống hệt**:

| Tình huống | Dự đoán điểm | Dự đoán phân bố |
|---|---|---|
| Đi thẳng đều trên vỉa hè | (5.2, 1.1) | vùng 95% = 0.8 m² → **xe đi qua được** |
| Đứng phân vân ở mép đường | (5.2, 1.1) | vùng 95% = 12 m² → **xe phải giảm tốc** |

---

## II.4 — GMM và Eq. (1)

📍 *Sec. 3, tr.6 · **Eq. (1)**, tr.6 · Fig. 4, tr.11*

### II.4.1 — Bóc tách từng thành phần

```
     ┌─ 𝒟 hoa mỹ: ký hiệu cho PHÂN BỐ
     │   ┌─ bước thời gian tương lai (h ∈ H_fc)
     │   │        ┌─ điểm được đánh giá (in ĐẬM = vector 2 chiều)
     ▼   ▼        ▼
    𝒟_{t+h} ( p_{t+h} )  =   Σ_{m ∈ M}   c_{m:t+h}   ·   ℱ_{m;t+h} ( p_{t+h} )
    └──────────────────┘     └────────┘  └─────────┘      └──────────────────┘
          VẾ TRÁI              TỔNG       TRỌNG SỐ           MẬT ĐỘ GAUSSIAN
       mật độ tổng hợp      qua 3 mode   của mode m          của riêng mode m
```

| Ký hiệu | Là gì | Ràng buộc | 📍 |
|---|---|---|---|
| **𝒟** | Phân bố — hàm mật độ trên ℝ² | `≥ 0`, tích phân **= 1** | Eq. (1), tr.6 |
| **`t+h`** | Bước thời gian tương lai | `h ∈ H_fc = {1,…,48}` | Sec. 3, tr.6 |
| **p** (đậm) | **Vector 2 chiều** `[x, y]` | `p ∈ ℝ²` | Sec. 3, tr.6 |
| **Σ_{m∈M}** | Cộng qua **tất cả mode** | `\|M\| = 3` | Sec. 4.1, tr.9 |
| **`c_{m:t+h}`** | **Trọng số** = xác suất mode `m` | `> 0`, **`Σ_m c_m = 1`** | Sec. 3, tr.6 |
| **ℱ** | Mật độ Gaussian **hai biến** | `≥ 0`, tích phân **= 1** | Eq. (1), tr.6 |

**① Vế trái** — mật độ mô hình gán cho điểm `p` tại bước `t+h`.
⚠️ Là **mật độ**, đơn vị **1/m²**, **có thể > 1**. Muốn ra xác suất phải **tích phân trên một vùng**.

**② Dấu tổng** — chỗ biến Gaussian đơn thành **hỗn hợp**. Bỏ đi (`M = 1`) thì quay về Social-LSTM.

**③ Trọng số `c`** — *"nhánh này có xác suất bao nhiêu?"*
⚠️ Chú ý **hai chỉ số**: `c` phụ thuộc **cả mode `m` lẫn thời điểm `t+h`** → tỉ lệ giữa các nhánh **thay đổi theo horizon**.

**④ Mật độ `ℱ`** — hình dạng của riêng nhánh `m`:

```
                       1                    ⎡   −1    ⎛ (x−μx)²   (y−μy)²   2ρ(x−μx)(y−μy) ⎞ ⎤
ℱ_m(x,y) = ──────────────────────── · exp  ⎢ ─────── ⎜ ─────── + ─────── − ─────────────── ⎟ ⎥
            2π σx σy √(1 − ρ²)              ⎣ 2(1−ρ²) ⎝   σx²       σy²         σx σy       ⎠ ⎦
            └──────────────────────┘        └──────────────────────────────────────────────────┘
               HỆ SỐ CHUẨN HÓA                        PHẦN MŨ (dạng toàn phương)
```

| Tham số | Vai trò hình học |
|---|---|
| `μx, μy` | **tâm** ellipse — nhánh này dự đoán tới đâu |
| `σx, σy` | **bán trục** — mức bất định theo mỗi hướng |
| `ρ` | **độ nghiêng** — x, y tương quan ra sao |

### II.4.2 — Hàm `𝒟` vs giá trị `𝒟(p)` vs tham số

| Viết là | Là gì | Kiểu | Ai tạo ra |
|---|---|---|---|
| `𝒟_{t+h}` (**không** ngoặc) | **phân bố** — hàm trên ℝ² | **hàm** | mô hình (gián tiếp) |
| `𝒟_{t+h}(p)` (**có** ngoặc) | **mật độ tại `p`** | **một số** ≥ 0 | tính khi thay `p` vào |
| `{c, μ, σ, ρ}` | **tham số** | **mảng số** | mô hình (trực tiếp) |

```
  Mạng nơ-ron THỰC SỰ xuất:  tham số {c_m, μ_m, σ_m, ρ_m}   (864 số)
                                      ▼
  Tham số ĐỊNH NGHĨA:        𝒟_{t+h}   ← Eq.(1) ← "OUTPUT" ở mức khái niệm
                                      ▼
  Thay p vào thì được:       𝒟_{t+h}(p)  — MỘT SỐ
```

⚠️ Eq. (1) viết sẵn đối số `p_{t+h}` vì **dọn đường cho Eq. (2)** in ngay bên cạnh. Bản chất `𝒟(·)` nhận **điểm bất kỳ** — nhờ vậy mới tính được Eq. (3), (4).

### II.4.3 — Vì sao Eq. (1) vẫn là phân bố hợp lệ

Vì là **tổ hợp lồi** của các phân bố hợp lệ:

```
∫∫ 𝒟(p) dp  =  Σ_m c_m ∫∫ ℱ_m(p) dp  =  Σ_m c_m · 1  =  1  ✅
                        └─────┬─────┘      └────┬────┘
                   mỗi Gaussian tích phân = 1  softmax
```

🔧 Đây là lý do **bắt buộc** softmax cho `c`. Nếu `Σ c_m ≠ 1` thì Eq. (1) **không còn là hàm mật độ**, và Eq. (2), (3), (4) sụp đổ theo.

### II.4.4 — Ví dụ số cụ thể

🔧 Giả sử tại `t+2s` mô hình xuất (tọa độ ego, mét):

| mode | `c_m` | `μ` | `σ` | `ρ` | diễn giải |
|---|---|---|---|---|---|
| 1 | **0.5** | (3.0, 0.2) | (0.6, 0.4) | +0.1 | đi thẳng |
| 2 | **0.3** | (2.0, 1.8) | (0.8, 0.7) | −0.2 | rẽ trái |
| 3 | **0.2** | (0.5, 0.1) | (0.3, 0.3) | 0.0 | dừng lại |

Tính mật độ tại `p = (3.0, 0.2)` — đúng tâm mode 1:

```
ℱ₁(p) = 1 / (2π · 0.6 · 0.4 · √(1−0.1²)) · exp(0)   = 0.6665   (p trùng tâm μ₁)
ℱ₂(p) = 0.2901 · exp(−2.940)                         = 0.0153   (cách tâm khá xa)
ℱ₃(p) = 1.7684 · exp(−34.8)                          ≈ 0.0000   (rất xa)
──────────────────────────────────────────────────────────────
𝒟(p) = 0.5×0.6665 + 0.3×0.0153 + 0.2×0.0000  =  0.338     ← Eq. (1)
```

Nếu `p` **là ground truth**, đóng góp vào loss Eq. (2): `−log(0.338) = 1.085`

🔧 **Đọc gì:** mode **gần** áp đảo (0.333/0.338 ≈ **99%**); mode **xa** bị `exp` dập tắt theo cấp số mũ. Đây là cơ chế khiến mode **tự phân công**.

### II.4.5 — Bốn công thức, bốn vai trò

⚠️ Eq. (1) và Eq. (2) được paper in **cạnh nhau trên cùng một dòng** cuối tr.6 → rất dễ tưởng cả hai đều là "đầu ra".

| Eq. | Công thức | Vai trò | Kết quả | Có lúc chạy thật? |
|---|---|---|---|---|
| **(1)** | `𝒟 = Σ_m c_m ℱ_m` | 🟢 **OUTPUT** | một **hàm** | ✅ Có |
| **(2)** | `L = −Σ_h log 𝒟(p)` | 🔴 **LOSS** | một **số** | ❌ cần ground truth |
| **(3)** | `1−α(p) = (1/N)Σ[𝒟(z)≥𝒟(p)]` | 🔵 hậu xử lý | số ∈ [0,1] | ✅ Có |
| **(4)** | `Ω(1−α) = {p : 1−α(p) ≥ 1−α}` | 🔵 hậu xử lý | một **tập** | ✅ Có |

```
        Eq.(1) ──┬──► Eq.(2)  [HUẤN LUYỆN]  ──► gradient
       OUTPUT    │
                 └──► Eq.(3) ──► Eq.(4)  [SUY LUẬN] ──► vùng 68%/95%
```

### II.4.6 — Ba lỗi hay mắc khi đọc Eq. (1)

| Lỗi | Vì sao sai |
|---|---|
| *"`𝒟(p)` là xác suất người đó ở tại `p`"* | Là **mật độ** (1/m²), có thể > 1. Xác suất của **một điểm** trong phân bố liên tục luôn **= 0** |
| *"Eq. (1) cho ra một quỹ đạo"* | Cho ra **một phân bố cho MỘT bước**. Cả horizon cần **48 lần** áp Eq. (1) |
| *"3 mode = 3 quỹ đạo dự đoán"* | 3 mode là **3 nhánh tại mỗi bước**. Mode 1 ở `t+1s` **không nhất thiết** nối với mode 1 ở `t+2s` |

⚠️ **Lỗi thứ ba là hạn chế thật của mô hình**, không chỉ hiểu nhầm — paper **không mô hình hóa nhất quán thời gian giữa các mode** và **không bàn tới**.

### II.4.7 — Vì sao là 3 mode?

📄 **Sec. 4.1, tr.9** — *"we set the number of Gaussians to three, which is reasonable for human motion [17]."*

⚠️ **Paper không có ablation cho `M`.** Đây chính là lỗ hổng nhóm nhắm vào ([I.5](#i5--đóng-góp-riêng-đặc-tả-thí-nghiệm)).

---

## II.5 — NLL, multi-modality và cầu nối EM

📍 *Eq. (2), tr.6*

### II.5.1 — Hàm mất mát Eq. (2)

```
L_NLL  =  − Σ_{h ∈ H_fc}  log 𝒟_{t+h}( p_{t+h} )
```

📄 *(Sec. 3, tr.6)*: *"we use the Negative Log-Likelihood (NLL) over the mixture distribution and **average the loss over each forecast horizon**."*

⚠️ Công thức in `−Σ_h` nhưng phần chữ nói "average" → thực tế có chia `|H_fc|`. **Thiếu nhất quán nhỏ**, không ảnh hưởng kết quả (hằng số nhân không đổi vị trí cực tiểu).

🔧 Dạng đầy đủ:
```
L_NLL = − Σ_{h}  log ( Σ_{m=1..3}  c_{m:t+h} · 𝒩(p_{t+h} ; μ_{m;t+h}, Σ_{m;t+h}) )
```

🔧 **Trực giác:** `𝒟(p_gt)` là mật độ mô hình gán cho **vị trí thật**. Gán cao → loss thấp.

### II.5.2 — Vì sao NLL sinh multi-modality

Chú ý vị trí **log** so với **tổng**: `log(Σ_m …)` chứ **không phải** `Σ_m log(…)`.

- `Σ_m log(…)`: mọi mode bị buộc giải thích mọi điểm → **mode collapse**
- `log(Σ_m …)`: chỉ cần **ÍT NHẤT MỘT** mode giải thích tốt → các mode khác **tự do** phục vụ mẫu khác

🔧 Với 1000 mẫu gồm 400 rẽ trái / 400 đi thẳng / 200 dừng, mô hình **tự phân công** và `c` hội tụ về ≈ 0.4/0.4/0.2 — **không cần nhãn mode**.

📄 Khớp chính xác với từ *"semi-supervised"* (Sec. 3, tr.6).

🔧 **So với MSE:** `L_MSE = ‖p̂ − p‖²` có nghiệm tối ưu là **kỳ vọng** → điểm giữa lòng đường.

### II.5.3 — Cầu nối MDN ↔ EM

> ⚠️ **Paper KHÔNG nhắc EM.** Đây là cầu nối do nhóm thêm để liên hệ Bài 5.

| | **GMM cổ điển (EM — Bài 5)** | **MDN (paper này)** |
|---|---|---|
| Mô hình hóa | `p(x)` — vô điều kiện | `p(y \| x)` — **có điều kiện** |
| Tham số `π, μ, Σ` | **hằng số**, ước lượng từ tập cố định | **hàm của input**, mạng dự đoán |
| Cách tối ưu | **EM**: E-step (`r_nk`) / M-step (Weighted MLE) | **Gradient descent** (ADAM) |
| Chọn `K` / `M` | BIC, AIC | cố định `M = 3` |
| Nguyên lý | **MLE** | **Cũng là MLE** |

Đạo hàm NLL theo tham số mode `m`:

```
            ∂L                                  c_m · ℱ_m(p)
           ────  ∝  − γ_m · ( … ) ,     γ_m = ─────────────────
           ∂θ_m                               Σ_k c_k · ℱ_k(p)
                                                    ║  CHÍNH LÀ
                                                    ▼
                                    responsibility  r_nk  (công thức 11.17)
```

→ Huấn luyện MDN bằng gradient descent ≈ chạy **EM "mềm" và liên tục**: mỗi bước, mỗi mode được cập nhật **theo tỉ lệ trách nhiệm** của nó.

---

## II.6 — LSTM, MDN head và kiến trúc

📍 *Sec. 3, tr.6 · Fig. 1, tr.7 · Sec. 4.1, tr.9*

### II.6.1 — LSTM làm gì và vì sao chọn LSTM

📄 **Vai trò: encoder.** LSTM nén toàn bộ lịch sử chuyển động thành **một vector trạng thái ẩn** chứa hướng, tốc độ, gia tốc, mức độ đều đặn/lưỡng lự.

**Bốn lý do chọn LSTM:**

| Lý do | 📄 Bằng chứng |
|---|---|
| **① Độ dài input ĐỘNG** (lý do chính) | *"We use a stacked LSTM to cope with **dynamic input horizons**"* (Sec. 3, tr.6). Kiểm chứng: Table 4 — chỉ **0.5 s** đã đạt gần trọn hiệu năng. Đối chứng [34]: các mô hình khác **suy giảm tới 3×** |
| **② Nhẹ và nhanh** | Table 5: 0.3–1.2 ms cho 128 dự đoán song song. Table 7: MID **12 giây**, mô hình này **0.4 ms** |
| **③ Chạy trên nhúng** | Table 5: 2.4 ms trên Jetson TX2 (**10 W**); 4.0 ms khi **chỉ dùng CPU ARM** |
| **④ Không cần attention** 🔧 | Paper **cố ý** không dùng bản đồ/tương tác → input chỉ ≤ 32 bước của **một** agent |

📄 **"Stacked" = 8 hidden layers**, chọn bằng **grid search** (Sec. 4.1, tr.9).

### II.6.2 — MDN head và hàm kích hoạt

📄 **Sec. 3, tr.6 (nguyên văn):**
> *"we use **σ(o_σ) = exp(o_σ) + 1 + ε_σ** and **ρ(o_σ) = tanh(o_σ)·ε_ρ** with σ greater than zero, and ρ within ]−1, 1[ … For the weights of the components c_{m:t+h}, we use **soft-max** activation."*

⚠️ Hai công thức này **KHÔNG được đánh số Eq.** — trích dẫn phải ghi "Sec. 3, tr.6".

| Tham số | Ràng buộc | Hàm kích hoạt 📄 | Vì sao 🔧 |
|---|---|---|---|
| `σx, σy` | **> 0** | `exp(o) + 1 + ε_σ` | `exp` luôn dương; `+1+ε` đặt **sàn** tránh `σ→0` gây NLL phân kỳ về −∞ |
| `ρ` | ∈ **(−1, 1)** | `tanh(o) · ε_ρ` | `tanh ∈ (−1,1)`; nhân `ε_ρ` để **không chạm ±1** (Σ suy biến) |
| `c_m` | **> 0**, **Σ = 1** | `softmax` | định nghĩa phân bố rời rạc trên mode |

⚠️ **Kích thước đầu ra — chỗ paper mơ hồ.** 📄 Sec. 4.1, tr.9: *"a fixed output size of **six parameters** per forecast horizon"* nhưng cũng nói `M = 3`.

| Cách đọc | Tham số / horizon | Nhận xét |
|---|---|---|
| (A) 6 tham số **mỗi mode**, 3 mode | 6 × 3 = **18** | Hợp lý: 1 Gaussian 2 biến cần 5 tham số + 1 trọng số = 6 |
| (B) 6 tham số **tổng** | **6** | Mâu thuẫn với `M = 3` |

🔧 **Cách đọc (A) gần như chắc chắn đúng.** Nếu bị hỏi: *"6 tham số cho mỗi thành phần Gaussian; với 3 mode là 18 mỗi bước. Paper diễn đạt tắt ở chỗ này."*

### II.6.3 — Kiến trúc end-to-end

> 🔧 Sơ đồ chi tiết hóa. 📄 **Fig. 1 (tr.7)** đơn giản hơn nhiều — chỉ vẽ lưới LSTM nối vào "MDN Head".

```
      ᵂT_in;t
          ▼
 ① EGO TRANSFORM      📄 Sec. 3, tr.6  (không có trong Fig. 1)
          ▼
 ② STACKED LSTM       📄 Fig. 1 tr.7 · 8 lớp (Sec. 4.1 tr.9)
          ▼
 ③ MDN HEAD           📄 Sec. 3, tr.6 (hàm kích hoạt, không đánh số Eq.)
          ▼
 ④ GAUSSIAN MIXTURE   📄 Eq. (1), tr.6
          │
    ┌─────┴─────┐
    ▼           ▼
 ⑤ NLL      ⑥ HẬU XỬ LÝ
 Eq.(2)       Eq.(3)(4) → vùng 68%/95% → PATH PLANNING
```

📄 **Những gì paper CỐ Ý KHÔNG dùng** *(Sec. 5, tr.14: "we solely use past trajectories, excluding context information")*:

| Không dùng | Lý do | 📍 |
|---|---|---|
| Bản đồ / HD map | *"outdated mapping data"*; tăng tài nguyên | Sec. 1.2 tr.3; Sec. 5 tr.14 |
| Tương tác xã hội | Tăng chi phí; ưu tiên tốc độ | Sec. 5, tr.14 |
| Transformer / Diffusion / GAN | Quá nặng — MID mất **12 s** | Table 7, tr.13 |

### II.6.4 — Dữ liệu và cấu hình huấn luyện

📍 *Sec. 4.1, tr.8–9 · Table 1, tr.8*

| | nuScenes | Waymo | inD | IMPTC | ETH/UCY |
|---|---|---|---|---|---|
| **Type** | vehicle | vehicle | drone | infra | infra |
| **Sample rate gốc** | 2.0 Hz | 10.0 Hz | 25.0 Hz | 25.0 Hz | 2.5 Hz |
| **Sau resample** | 10.0 Hz | 10.0 Hz | 10.0 Hz | 10.0 Hz | 2.5 Hz |
| **Quy mô** | 1 K scene | > 100 K scene | 33 phiên | 270 chuỗi | 5 địa điểm |

📄 **Preprocessing** *(Sec. 4.1, tr.8)*:
1. **Cubic Spline** resample về **10 Hz** (ETH/UCY giữ 2.5 Hz)
2. Chỉ giữ track **dài ≥ (input + output)**; dài hơn thì **cắt thành mẫu con**
3. **Ego transform** cho cả input và ground truth
4. Thêm **vận tốc** vào đặc trưng

📄 **Cấu hình** *(Sec. 4.1, tr.9)*: 8 lớp LSTM (grid search) · `M = 3` · ADAM · lr `1e−3` → giảm tuyến tính về `1e−7` · batch **1024** · **2500 epoch** · xáo trộn mỗi epoch · RTX 3090 + Ryzen 5900X.

---

## II.7 — Reliability và Calibration

📍 *Sec. 1.2 (tr.2–3) · Sec. 2 (tr.5–6) · Sec. 4.2 (tr.9) · Fig. 3 (tr.10) · Fig. 5 (tr.14)*

### II.7.1 — Ba khái niệm hay bị nhầm

| Khái niệm | Trả lời câu hỏi | Metric | Đủ? | 📍 |
|---|---|---|---|---|
| **Accuracy** | Gần vị trí thật bao nhiêu mét? | ADE, FDE | ❌ | Table 6, tr.13 |
| **Likelihood / fit** | Mật độ khớp dữ liệu tới đâu? | NLL, KDE-NLL | ❌ | Sec. 2, tr.5 |
| **Calibration / Reliability** | Xác suất công bố có đúng tần suất? | **`R_avg`, `R_min`** | ✅ | Sec. 4.2, tr.9 |

📄 **Sec. 2, tr.5** — *"**a low NLL does not necessarily mean that a model's distribution is reliable**. It only measures how well its density fits the ground truth data."*

📄 Paper cũng phê phán **AMD** (Average Mahalanobis Distance) và **AMV** (Average Maximum Eigenvalue) của Social-Implicit — vẫn không đo được calibration.

### II.7.2 — Calibration là gì

> 🔧 **Một mô hình xác suất ĐÁNG TIN khi: trong tất cả những lần nó nói "90%", đúng 90% số lần đó sự kiện thực sự xảy ra.**

📄 Cơ sở — **Sec. 4.2, tr.9**: *"The observed relative frequency of occurrences f_o(1−α) should match 1−α."*

🔧 Ví dụ: người dự báo nói "70% mưa" 100 lần → mưa ~70 lần thì **calibrated**; 30 lần thì **overconfident**; 95 lần thì **underconfident**.

📄 **Sec. 1.2, tr.3** — *"If a model produces under- or overconfident predictions, it poses a significant risk."*

```
Overconfident (quá tự tin)              Underconfident (quá dè dặt)
nói 95% → thực tế chỉ 60%               nói 95% → thực tế 99.9%
vùng QUÁ HẸP                            vùng QUÁ RỘNG
→ xe tưởng đường thoáng                 → xe thấy đâu cũng nguy hiểm
→ ⚠️ VA CHẠM                             → phanh gấp liên tục, vô dụng
```

### II.7.3 — Confidence Level và Confidence Set (Eq. 3, 4)

📄 **Sec. 3, tr.7** — *"we search for points z ∈ ℝ², which have a probability density level **greater than or equal to** the density 𝒟(p) … we draw **N random samples** Z ∼ 𝒟."*

```
                 1        ⎧ 1,  nếu 𝒟(z) ≥ 𝒟(p)
1 − α(p)  =  ─── · Σ      ⎨                              ← Eq. (3), tr.7
                 N   z∈Z  ⎩ 0,  ngược lại

Ω(1 − α)  =  { p ∈ ℝ²  :  1 − α(p)  ≥  1 − α }           ← Eq. (4), tr.7
```

📄 **Fig. 2 (tr.7)** minh họa: chấm **cam** = `p`; marker **xanh lá** = mẫu có `𝒟(z) ≥ 𝒟(p)`; **xanh dương** = ngược lại. Tỉ lệ xanh lá chính là `1−α(p)`.

⚠️ **Paper KHÔNG công bố `N`.**

🔧 Tên chuẩn trong thống kê: **Highest Density Region (HDR)** — paper không dùng thuật ngữ này.

🔧 **Tính chất quan trọng:** vì định nghĩa bằng **ngưỡng mật độ**, với GMM đa mode `Ω` có thể **rời rạc nhiều mảnh** — ưu thế mà ellipse Gaussian đơn không có. 📄 Fig. 4 (tr.11) cho thấy `Ω` thực tế: *"Inner contours represent the 68%, and outer contours the 95% CL"*.

### II.7.4 — Hai metric định lượng (Eq. 5, 6)

📄 **Sec. 4.2, tr.9** — quét qua **mọi** `α ∈ A = {0.01, 0.02, …, 0.99}` và **mọi** `h ∈ H_fc`:

```
R_min = Γ̂ = 1 −  max_{h} max_{α}  | (1−α) − f_{o;t+h}(1−α) |         ← Eq. (5)

                        1
R_avg = Γ̄ = 1 − ───────────────── · Σ_{h} Σ_{α} | (1−α) − f_{o;t+h}(1−α) |   ← Eq. (6)
                  |H_fc| · |A|
```

📄 **Ngưỡng (Sec. 4.2, tr.9, nguyên văn):** *"the **perfect score is 1** … the **worst score is 0**. A thoroughly reliable system has an Average Reliability Γ̄ of **≥ 0.95** and a Minimum reliability score Γ̂ of **≥ 0.90**."*

⚠️ Ngưỡng 0.95/0.90 **do chính tác giả đặt ra**, không dẫn từ chuẩn an toàn nào.

📄 **Fig. 3 (tr.10)** — calibration plot cho 4 dataset. Chú giải màu: `t+0.8s` **xanh dương**, `t+1.6s` **cam**, `t+2.4s` **xanh lá**, `t+3.2s` **đỏ**, `t+4.0s` **tím**, `t+4.8s` **nâu**, lý tưởng = **nét đứt đen**.

### II.7.5 — Sharpness và vì sao cần cả hai

📄 **Sec. 4.2, tr.9** — *"we define the **Sharpness κ(1−α) in m²/s** as the volumetric measure of the confidence set Ω(1−α) … to report specific Sharpness levels called **S95** and **S68**."*

⚠️ **Đơn vị `m²/s` không được giải thích.** 🔧 Diễn giải hợp lý nhất: diện tích **chuẩn hóa theo forecast horizon**. Nên **thừa nhận đây là thiếu sót trình bày** thay vì đoán chắc.

🔧 **Vì sao Reliability một mình chưa đủ:** dự đoán vùng khổng lồ phủ cả bản đồ → ground truth **luôn** nằm trong → calibration hoàn hảo nhưng **vô dụng**.

📄 **Sec. 1.2, tr.3** — mục tiêu là *"as little uncertainty as possible **while meeting the reliability condition**"*.

```
   Reliability CAO + Sharpness LỚN  →  đúng nhưng vô dụng
   Reliability THẤP + Sharpness NHỎ →  sắc nét nhưng NGUY HIỂM
   Reliability CAO + Sharpness NHỎ  →  ✅ MỤC TIÊU
```

### II.7.6 — ADE/FDE tính thế nào nếu output là phân bố?

⚠️ Mô hình **không xuất quỹ đạo**. Phải **lấy mẫu** trước:

```
𝒟_{t+h} ──[lấy 20 mẫu]──► 20 quỹ đạo ──[so với ᵂT_gt;t]──► minADE20 / minFDE20
                                            ↑
                                    lấy cái GẦN NHẤT trong 20
```

🔧 Đó là ý nghĩa **"20"** trong `minADE20`. Và là điều paper phê phán: muốn biết **cái nào tốt nhất** thì **phải có ground truth** — thứ không tồn tại lúc chạy thật.

---

## II.8 — Kết quả thực nghiệm của paper

### II.8.1 — Bốn nhóm metric

| Metric | Đo gì | Tốt là | 📍 |
|---|---|---|---|
| **`R_avg`** | Sai lệch hiệu chuẩn trung bình | **cao**, ≥ 95% | Eq. (6), tr.9 |
| **`R_min`** | Sai lệch **trường hợp xấu nhất** | **cao**, ≥ 90% | Eq. (5), tr.9 |
| **`S68` / `S95`** | Diện tích vùng tin cậy 68%/95% | **thấp** | Sec. 4.2, tr.9 |
| **`minADE20` / `minFDE20`** | Sai số L2 trung bình / cuối, best-of-20 | **thấp** | Sec. 4.2, tr.9 |

### II.8.2 — Table 2: Multi-Train / Single-Test

📄 Train trên **gộp cả 4 dataset**, test riêng từng bộ *(Sec. 4.3, tr.10)*.

| Score | IMPTC | inD | nuScenes | Waymo | **Avg** |
|---|---|---|---|---|---|
| `R_avg` (%) | 96.3 | 96.2 | 95.3 | 94.6 | **95.6** ✅ |
| `R_min` (%) | 87.9 | 90.9 | 84.4 | 88.8 | **88.0** ⚠️ |
| `S68` (m²/s) | 0.4 | 0.3 | 0.4 | 0.5 | **0.4** |
| `S95` (m²/s) | 2.0 | 1.6 | 2.1 | 1.8 | **1.9** |
| `minADE20` (m) | 0.30 | 0.27 | 0.27 | 0.26 | **0.28** |
| `minFDE20` (m) | 0.63 | 0.53 | 0.50 | 0.52 | **0.55** |

📄 Paper thừa nhận: *"The minimum scores are **slightly below 90%, which leaves room for improvement**."*

### II.8.3 — Table 6: So sánh Reliability với SOTA ⭐

📄 **Bảng quan trọng nhất của paper.** Nhóm tác giả *"implemented the Reliability and Sharpness evaluation into named frameworks"* (tr.12) — tự cài metric vào codebase của 4 phương pháp khác.

| Subset | Trajectron++ | Social-Implicit | MID | FlowChain | **Ours** |
|---|---|---|---|---|---|
| ETH | 58.3/21.4 | 82.0/56.6 | 89.9/**75.6** | 85.7/67.6 | **91.9**/67.4 |
| Hotel | 64.5/29.5 | 70.1/52.0 | 89.0/55.1 | 85.7/67.6 | **90.7/78.1** |
| Univ | 66.8/42.4 | 89.3/75.4 | 84.3/73.8 | 87.5/69.0 | **94.7/77.8** |
| Zara1 | 61.1/21.8 | **90.7/74.0** | 81.3/69.5 | 89.0/55.1 | 89.4/72.8 |
| Zara2 | 73.9/39.4 | 81.3/54.3 | 78.7/55.3 | 80.3/47.8 | **88.6/66.1** |
| **Avg** | **64.9/30.9** | 82.7/62.5 | 84.6/65.9 | 86.5/62.2 | **91.1/72.5** |

*(`R_avg`/`R_min`, %)*

📄 **Nguyên văn (tr.12):** *"**None of the tested models can achieve suitable Reliability scores.** … It must be noted that **ADE/FDE performance does not directly translate into good Reliability results**."*

🔧 **Ba kết luận:**
1. **KHÔNG mô hình nào đạt ngưỡng 95/90** — kể cả chính paper (91.1/72.5)
2. **Trajectron++ `R_min` chỉ 30.9%** — lệch ~69 điểm phần trăm ở kịch bản xấu nhất, dù ADE/FDE (0.30/0.51) rất tốt
3. ⭐ **ADE/FDE tốt KHÔNG kéo theo Reliability tốt**

### II.8.4 — Tốc độ và transferability

**Table 5 (tr.12) — inference, batch 128:**

| | Nền tảng | Inference | Post-proc | **Tổng** |
|---|---|---|---|---|
| **GPU** | Desktop (RTX 3060) | 0.3 ms | 0.4 ms | **0.7 ms** |
| | Nhúng (Jetson TX2, 10 W) | 1.2 ms | 1.2 ms | **2.4 ms** |
| **CPU** | Desktop | 1.2 ms | 0.4 ms | **1.6 ms** |
| | Nhúng ARM | 2.8 ms | 1.2 ms | **4.0 ms** |

📄 *"Using a GPU speeds up the inferencing by 2-4x, but **having one is optional**."*

**Table 7 (tr.13) — so ADE/FDE và tốc độ:**

| | CVM | Social-LSTM | Trajectron++ | MID | GATraj | FlowChain | **Ours** |
|---|---|---|---|---|---|---|---|
| **Avg ADE/FDE** | 0.28/0.56 | 0.45/0.72 | 0.30/0.51 | 0.21/0.38 | **0.17/0.29** | 0.29/0.52 | 0.26/0.50 |
| **Reliability calib.** | ✕ | ✕ | ✕ | ✕ | ✕ | ✕ | **✓** |
| **Inference** | 1.8 ms | 1.8 s | 29 ms | **12 s** | 10 ms | 34 ms | **0.4 ms** ⚡ |

🔍 🔧 **CVM (vận tốc hằng) đạt 0.28/0.56 — ngang Trajectron++!** Bằng chứng ngay trong paper cho phê phán về ETH/UCY.

**Table 3 (tr.11) — transferability:** train một dataset, test cả bốn. Nhìn chung **chuyển miền tốt** (`R_avg` 94.6–97.0%). Ngoại lệ: mô hình train riêng trên IMPTC tụt `R_min` xuống 71.2% / 70.2% — 📄 paper giải thích *"training has stopped too early"*.

**Table 4 (tr.12) — input horizon động:**

| `H_in` (ms) | `R_avg` | `R_min` | `minADE20` |
|---|---|---|---|
| 100 | 86.7 | 68.4 | 0.42 |
| **500** | **92.9** | **84.3** | **0.38** |
| 1000 | 94.3 | 86.3 | 0.30 |
| 2000 | 95.2 | 87.5 | 0.28 |

🔑 📄 **Phát hiện quan trọng (tr.11):** bỏ **vận tốc** khỏi input → *"highly reduced results"*. ⚠️ Paper **không đưa bảng số** cho thí nghiệm này.

---

## II.9 — Hạn chế: cái nào thật, cái nào sửa được

### Ba nhóm khác nhau

| Nhóm | Nghĩa | Gồm |
|---|---|---|
| 🔴 **Hạn chế khoa học thật** | Ảnh hưởng **độ tin cậy của kết luận** | #1, #2, #3, #4, #5 |
| 🟡 **Thiếu sót trình bày** | Khó tái lập, **nhưng kết luận vẫn đứng** | #6, #7 |
| 🔵 **Trade-off có chủ đích** | **Không phải lỗi** — lựa chọn được tuyên bố rõ | #8, #9, #10 |

### 🔴 Hạn chế thật, xếp theo mức nghiêm trọng

**🥇 #1 — Không công bố `N`, không phân tích sai số Monte Carlo**

Không chỉ thiếu thông tin, mà là **thiên lệch thống kê có hệ thống**:

```
R_min = 1 − max qua 99 mức α × 48 horizon = max qua 4 752 ước lượng nhiễu
                                             └──────────┬──────────┘
                              lấy MAX của nhiều đại lượng nhiễu
                              → luôn tóm phải đuôi nhiễu xấu nhất
                              → độ lệch bị THỔI PHỒNG
                              → R_min bị BÁO CÁO THẤP HƠN thực tế
```

Thêm nữa, `1−α(p)` chỉ nhận `N+1` giá trị rời rạc. `N` nhỏ → **không đủ phân giải** để khớp lưới `A = {0.01,…,0.99}`.
→ Con số `R_min` ở **Table 2 và Table 6** — bằng chứng cho luận điểm trung tâm — phụ thuộc tham số **không công bố**.

**🥈 #2 — Không ablation cho `M = 3`**

Đánh thẳng vào kết luận chính. Paper bác `M = 1` bằng trích dẫn, chọn `M = 3` bằng trích dẫn, **không thử gì cả**. Nếu `M = 8` cho `R_min = 93%`, câu *"none of the tested models achieves suitable Reliability"* **không còn đúng cho chính họ**.

**🥉 #3 — Không ràng buộc nhất quán thời gian giữa các mode**

Eq. (1) áp **độc lập từng `h`**. Mức nghiêm trọng tùy mục đích:

| Dùng để | Ảnh hưởng |
|---|---|
| Path planning (mục tiêu chính) | 🟢 Nhẹ — chỉ cần vùng chiếm dụng từng mốc |
| Reliability/Sharpness | 🟢 Nhẹ — cũng tính theo từng `h` |
| **ADE/FDE (Table 6, 7)** | 🔴 **Nặng** — lấy mẫu độc lập cho quỹ đạo "nhảy cóc", phi vật lý. Paper **không nói** cách lấy mẫu |

**#4 và #5 — hai ablation bị thiếu**
- **Bỏ vận tốc:** khẳng định mạnh (*"velocity is an essential input feature"*) nhưng chứng cứ chỉ là *"highly reduced"*, **không một con số**
- **Ego transform:** nói *"increase transferability"*, chứng cứ chỉ gián tiếp qua Table 3

**#Bonus — `R_min` 88% < ngưỡng 90%.** Là hạn chế thật nhưng paper **tự thừa nhận thẳng** — đây là kết quả trung thực, không phải lỗi che giấu.

### 🟡 Chỉ là lỗi trình bày (đừng gọi là hạn chế khoa học)

| # | Vấn đề | Vì sao không nghiêm trọng |
|---|---|---|
| **#6** | Đơn vị `m²/s` không giải thích | Mọi con số tính **cùng một cách** → **so sánh vẫn hoàn toàn hợp lệ** |
| **#7** | Eq. (2) thiếu `1/\|H_fc\|`; "six parameters" mơ hồ | Hằng số nhân **không đổi vị trí cực tiểu** → không ảnh hưởng huấn luyện |

### 🔵 Trade-off có chủ đích (không phải lỗi)

| # | | Lý do |
|---|---|---|
| **#8** | Không dùng ngữ cảnh (bản đồ, tương tác) | 📄 Tuyên bố rõ ở Sec. 5 tr.14 — đánh đổi lấy 0.4 ms / 10 W |
| **#9** | Không vượt SOTA ADE/FDE | Hệ quả trực tiếp của #8 |
| **#10** | Không đánh giá multi-agent (JADE/JFDE) | Nằm ngoài phạm vi tuyên bố |

### Cái nào nhóm tự cải thiện được

Xếp theo **giá trị / công sức**:

| # | Việc làm | Cần gì | Chi phí | Giá trị |
|---|---|---|---|---|
| **#1** | Vẽ `R_avg`, `R_min` theo `N ∈ {50, 100, 500, 1000, 5000}` | **Không cần train lại** — chỉ hậu xử lý | ⭐ Rẻ nhất | 🔥🔥🔥 |
| **#6** | Đọc repo xem Sharpness chia cho gì | 15 phút | ⭐ | 🔥 |
| **#2** | Train `M ∈ {1,2,3,5,8}` trên ETH/UCY | ETH/UCY rất nhỏ | ⭐⭐ | 🔥🔥🔥 |
| **#4** | Train có/không vận tốc, lập **bảng số** | 2 lần train | ⭐⭐ | 🔥🔥 |
| **#5** | Train có/không ego transform | 2 lần train | ⭐⭐ | 🔥🔥 |

> 👉 Nhóm đã chọn **#2** làm đóng góp chính ([I.5](#i5--đóng-góp-riêng-đặc-tả-thí-nghiệm)). Nếu còn thời gian, **#1 gần như miễn phí** — chỉ chạy lại bước hậu xử lý với các `N` khác nhau.

### Một hướng paper bỏ lỡ — đáng nêu nếu bị hỏi

📄 Paper trích **Guo et al. [7]** *(On calibration of modern neural networks, ICML 2017)* để **nêu vấn đề** miscalibration — nhưng **không dùng giải pháp của chính bài đó**: **temperature scaling**, hiệu chuẩn hậu kỳ trên tập validation, **không cần train lại**, chỉ học một tham số vô hướng:

```
c_m  ←  softmax( logits / T )      ← hiệu chỉnh trọng số mode
σ    ←  σ · s                       ← hiệu chỉnh độ rộng
```

🔧 *"Paper trích Guo et al. để chẩn đoán bệnh nhưng không kê đơn thuốc của chính Guo et al."*

⚠️ **Chưa kiểm chứng** — temperature scaling cải thiện calibration trung bình rất tốt, nhưng `R_min` là **max sai lệch** nên khó hơn. Phải thử mới biết; kết quả âm tính cũng là kết quả đáng báo cáo.

### Cái không thể giải quyết

| Vấn đề | Vì sao |
|---|---|
| **Ngưỡng 95%/90% không có cơ sở** | Cần **cơ quan chuẩn hóa** (ISO 26262 / ISO 21448). Hiện **chưa tồn tại** chuẩn nào cho calibration trong HTP |
| **Không dùng ngữ cảnh / thua SOTA ADE** | Thêm bản đồ + graph attention = **đổi sang bài toán khác**, phá vỡ mục tiêu 0.4 ms / 10 W |
| **Đánh đổi Reliability ↔ Sharpness** | Giới hạn **lý thuyết thông tin**. Chỉ dịch được điểm cân bằng, không xóa được nó |

---
---

# PHẦN III — PHỤ LỤC

## III.A — Code tham khảo

> ⚠️ **CẢNH BÁO NGUỒN:** các đoạn mã dưới đây là 🔧 **tái dựng từ mô tả trong paper** — **KHÔNG trích từ repo**. Khi cần con số chính xác (hidden size, `N`, thứ tự chiều tensor), phải đọc `github.com/kav-institute/mdn_trajectory_forecasting`.

### A.1 — MDN head

Dựa trên 📄 hàm kích hoạt Sec. 3 tr.6 + `M=3`, "6 parameters" Sec. 4.1 tr.9.

```python
import torch
import torch.nn as nn

class MDNHead(nn.Module):
    """Xuất tham số GMM: M mode x 6 tham số cho mỗi forecast horizon.
    Hàm kích hoạt theo đúng Sec. 3, tr.6 của paper."""
    def __init__(self, hidden_dim, n_horizon, n_modes=3,
                 eps_sigma=1e-3, eps_rho=0.999):
        super().__init__()
        self.n_horizon, self.n_modes = n_horizon, n_modes
        self.eps_sigma, self.eps_rho = eps_sigma, eps_rho
        # 6 tham số / mode: mu_x, mu_y, o_sx, o_sy, o_rho, o_c
        self.fc = nn.Linear(hidden_dim, n_horizon * n_modes * 6)

    def forward(self, h):                      # h: (B, hidden_dim)
        o = self.fc(h).view(-1, self.n_horizon, self.n_modes, 6)

        mu    = o[..., 0:2]                                        # tự do
        sigma = torch.exp(o[..., 2:4]) + 1.0 + self.eps_sigma      # > 0, CÓ SÀN
        rho   = torch.tanh(o[..., 4]) * self.eps_rho               # ngặt trong (-1,1)
        c     = torch.softmax(o[..., 5], dim=-1)                   # sum = 1
        return mu, sigma, rho, c
```

### A.2 — NLL loss (Eq. 2)

```python
def mdn_nll(mu, sigma, rho, c, target):
    """Eq. (2), tr.6:  L = -SUM_h log D_{t+h}(p_{t+h})
    target: (B, H, 2) — ground truth trong tọa độ ego."""
    t  = target.unsqueeze(2)                              # (B, H, 1, 2)
    dx = (t[..., 0] - mu[..., 0]) / sigma[..., 0]
    dy = (t[..., 1] - mu[..., 1]) / sigma[..., 1]
    one_m_r2 = 1.0 - rho ** 2

    z        = dx**2 + dy**2 - 2 * rho * dx * dy
    log_norm = -torch.log(2 * torch.pi * sigma[..., 0] * sigma[..., 1]
                          * torch.sqrt(one_m_r2))
    log_F    = log_norm - z / (2 * one_m_r2)              # log N(p; mu, Sigma)

    # Eq. (1): log( SUM_m c_m * F_m ) — logsumexp để ổn định số học
    log_D = torch.logsumexp(torch.log(c) + log_F, dim=-1) # (B, H)
    return -log_D.mean()     # "average the loss over each forecast horizon"
```

🔧 `logsumexp` **bắt buộc**: đây chính là `log(Σ …)` trong Eq. (2). Tính `log` sau `sum` trực tiếp sẽ **underflow** (mật độ có thể cỡ `1e−300` → `log(0) = −∞` → NaN).

### A.3 — Ego coordinate transform

```python
import numpy as np

def to_ego(traj_world, t_idx):
    """Sec. 3, tr.6: tịnh tiến về p_t, xoay -phi_t.
    traj_world: (T, 2). phi_t = hướng di chuyển tại t_idx."""
    origin = traj_world[t_idx]
    d   = traj_world[t_idx] - traj_world[t_idx - 1]
    phi = np.arctan2(d[1], d[0])
    R   = np.array([[ np.cos(phi), np.sin(phi)],
                    [-np.sin(phi), np.cos(phi)]])          # R(-phi)
    return (traj_world - origin) @ R.T
```

⚠️ Cách ước lượng `φ_t` từ bước cuối là 🔧 suy luận — **paper không nói `φ_t` tính thế nào** (có thể dùng smoothing nhiều frame).

### A.4 — Confidence Level (Eq. 3) — dùng cho Demo 2, Demo 3

```python
def confidence_level(gmm, p, n_samples=1000):
    """Eq. (3), tr.7:  1-alpha(p) = (1/N) SUM_z [ D(z) >= D(p) ]
    Fig. 2 minh họa: mẫu xanh lá = D(z)>=D(p), xanh dương = ngược lại."""
    z = gmm.sample(n_samples)              # paper KHÔNG công bố N
    return float(np.mean(gmm.pdf(z) >= gmm.pdf(p)))
```

### A.5 — Reliability metric (Eq. 5, 6) — dùng cho Demo 3 và ablation

```python
def reliability_scores(cl_per_horizon, alphas=np.arange(0.01, 1.00, 0.01)):
    """Eq. (5)(6), tr.9.  alphas = A = {0.01, ..., 0.99}
    cl_per_horizon: dict {h: mảng 1-alpha(p_gt) của mọi mẫu ở horizon h}"""
    devs = []
    for h, cls in cl_per_horizon.items():
        for a in alphas:
            expected = 1.0 - a
            observed = np.mean(cls <= expected)   # f_o(1-alpha)
            devs.append(abs(expected - observed))
    devs = np.array(devs)
    R_min = 1.0 - devs.max()      # Eq. (5) — Gamma_hat
    R_avg = 1.0 - devs.mean()     # Eq. (6) — Gamma_bar
    return R_avg, R_min
```

---

## III.B — Ngân hàng câu hỏi

> Mỗi câu trả lời đều dẫn 📍 vị trí trong paper để trích khi cần.

### Nhóm A — Bài toán và động lực

**A1. Tại sao ADE/FDE không đủ? Nó là chuẩn của lĩnh vực mà?**
> ADE/FDE đo **khoảng cách**, không đo **độ tin cậy**. Ba vấn đề: (1) `minADE20` cần ground truth để chọn giả thuyết tốt nhất — chạy thật không có 📍 *Sec. 1.1, tr.2*; (2) bỏ qua 19/20 dự đoán 📍 *Sec. 2, tr.5*; (3) bằng chứng trực tiếp: Trajectron++ ADE 0.30 nhưng `R_min` 30.9% 📍 **Table 6, tr.13**.

**A2. Vì sao paper phê phán ETH/UCY?**
> Dẫn Schoeller [31]: (1) 5 subset **tương quan** → LOO không đo được transferability; (2) chủ yếu quỹ đạo **thẳng ngang/dọc**, nên **CVM đạt 0.28/0.56 — ngang Trajectron++** 📍 **Table 7, tr.13**.

**A3. Reliability tốt mà Sharpness tệ thì sao?**
> Reliability cao + Sharpness lớn = đúng nhưng vô dụng. Ngược lại = sắc nét nhưng **nguy hiểm**. Paper phát biểu mục tiêu là *"as little uncertainty as possible **while meeting the reliability condition**"* 📍 *Sec. 1.2, tr.3*.

### Nhóm B — Mô hình

**B1. Tại sao LSTM mà không phải Transformer?**
> (1) **Độ dài input động** 📍 *Sec. 3, tr.6*, kiểm chứng 📍 **Table 4**; (2) nhẹ — 0.4 ms vs 12 s của MID 📍 **Table 7**; (3) chạy trên Jetson TX2 10 W 📍 **Table 5**; (4) input chỉ 32 bước của một agent, không có ngữ cảnh để attend.

**B2. Tại sao 3 Gaussian? Sao không 5, không 10?**
> 📍 *Sec. 4.1, tr.9*: *"reasonable for human motion [17]"*. ⚠️ **Đây là điểm yếu — paper KHÔNG có ablation.** Nên thừa nhận, rồi nói: **nhóm đã thử** (slide 15). Nhiều mode hơn tăng khả năng biểu diễn nhưng dễ mode collapse và tăng tham số tuyến tính theo `M`.

**B3. MDN, GMM, LSTM khác nhau thế nào?**
> **Không phải 3 mô-đun ngang hàng.** LSTM = **bộ mã hóa**, MDN = **lớp đầu ra + hàm kích hoạt**, GMM = **đối tượng toán học** mà tham số đó định nghĩa 📍 **Eq. (1)**. Ví von: người quan sát / người phiên dịch / bản dự báo.

**B4. Tại sao σ dùng `exp(o)+1+ε` mà không chỉ `exp(o)`?**
> 📍 *Sec. 3, tr.6* (không đánh số Eq.). Để đặt **sàn cứng cho σ**. Nếu `σ → 0`, mật độ → ∞, `−log 𝒟 → −∞`, loss phân kỳ. Tương tự `tanh(o)·ε_ρ` giữ `ρ` không chạm ±1 để Σ không suy biến.

**B5. Mô hình xuất bao nhiêu tham số mỗi bước?**
> ⚠️ **Paper viết mơ hồ**: 📍 *tr.9* nói *"six parameters per forecast horizon"* nhưng cũng nói `M = 3`. Đúng là **6 tham số mỗi thành phần Gaussian** → **18 mỗi bước**. Trả lời: *"Paper diễn đạt tắt ở chỗ này."*

### Nhóm C — Toán học

**C1. Vì sao NLL tạo multi-modality mà MSE thì không?**
> Nằm ở vị trí **log** so với **tổng** trong 📍 **Eq. (2)**: `log(Σ_m c_m ℱ_m)` — chỉ cần **MỘT** mode giải thích tốt → mode tự phân công. MSE có nghiệm tối ưu là **kỳ vọng có điều kiện** → với "rẽ trái hoặc rẽ phải", trung bình rơi vào **giữa lòng đường**. Khớp với từ *"semi-supervised"* 📍 *Sec. 3, tr.6*.

**C2. Quan hệ giữa MDN và EM?**
> ⚠️ **Paper không nhắc EM.** Cả hai là **MLE trên hỗn hợp Gaussian**: EM ước lượng `π, μ, Σ` là **hằng số**; MDN **dự đoán** chúng như **hàm của input**. Gradient của Eq. (2) chứa `γ_m = c_m ℱ_m / Σ_k c_k ℱ_k` — chính là **responsibility `r_nk`** (11.17). Gọi là **"soft EM" bằng gradient descent**.

**C3. Confidence Set khác gì ellipse 2σ?**
> Ellipse 2σ chỉ đúng với **Gaussian đơn**. Paper dùng định nghĩa theo **ngưỡng mật độ** 📍 **Eq. (4)**, ước lượng Monte Carlo 📍 **Eq. (3)**. Ưu điểm: vùng có thể **rời rạc nhiều mảnh** — biểu diễn đúng "rẽ trái HOẶC rẽ phải".

**C4. Vì sao code phải dùng `logsumexp`?**
> Vì 📍 **Eq. (2)** là `log(Σ_m c_m ℱ_m)`. Mật độ có thể rất nhỏ (`1e−300`) → cộng trực tiếp rồi lấy log sẽ **underflow về 0** → `log(0) = −∞` → NaN.

**C5. `𝒟(p)` có thể lớn hơn 1 không?**
> **Có.** Đó là **mật độ** (đơn vị 1/m²), không phải xác suất. Xác suất của **một điểm** trong phân bố liên tục luôn **bằng 0**. Muốn ra xác suất phải **tích phân trên một vùng** — chính là Eq. (4).

### Nhóm D — Thực nghiệm

**D1. Vì sao dùng 4 dataset chứ không chỉ ETH/UCY?**
> Để test **transferability thật**. 📍 **Table 1** cho thấy 4 dataset đại diện **ba kiểu quan sát**: trên xe, drone, hạ tầng. 📍 *Sec. 1.2, tr.3* nêu yêu cầu *"three or more different data sets from different domains"*.

**D2. Kết quả nào quan trọng nhất?**
> 📍 **Table 6, tr.13**: (a) **không mô hình nào** đạt ngưỡng; (b) Trajectron++ `R_min` chỉ **30.9%** dù ADE tốt; (c) **ADE/FDE tốt không kéo theo Reliability tốt** — 📍 kết luận nguyên văn *Sec. 4.3, tr.12*.

**D3. Vì sao vận tốc quan trọng đến thế?**
> 📍 *Sec. 4.3, tr.11*: bỏ vận tốc → *"highly reduced results"*; Trajectron++ — mô hình duy nhất khác chịu được input động — cũng là mô hình duy nhất khác dùng vận tốc. Với input 0.1–0.5 s, chuỗi vị trí quá ít thông tin; vận tốc cung cấp **trực tiếp** thông tin bậc một. ⚠️ Paper **không đưa bảng số**.

**D4. Vì sao mô hình train trên IMPTC kém trên inD và nuScenes?**
> 📍 *Sec. 4.3, tr.10*: *"**the training has stopped too early**"*. Bổ sung: IMPTC là dataset một giao lộ, ít đa dạng hơn Waymo (>100 K scene).

### Nhóm E — Câu hỏi khó / phản biện

**E1. Đóng góp chỉ là ghép LSTM cũ với MDN cũ, có gì mới?**
> **Kiến trúc không mới** (MDN từ 1994, ref [2]) — paper cũng không giấu. Đóng góp thật ở **phương pháp luận đánh giá**: chuẩn hóa metric 📍 **Eq. (5)(6)**, và **tự cài metric vào codebase của 4 SOTA** 📍 **Table 6**. Chưa ai làm trước đó.

**E2. Chính mô hình của họ cũng không đạt ngưỡng — có tự mâu thuẫn không?**
> Không mâu thuẫn, nhưng là **hạn chế thật**. 📍 Paper thừa nhận thẳng *(tr.10)*. Ý đồ là **nêu vấn đề**: nếu cả 5 mô hình tốt nhất đều không đạt ngưỡng an toàn, đó chính là thông điệp.

**E3. Thêm bản đồ và tương tác thì tốt hơn — sao không làm?**
> **Đánh đổi có chủ đích.** Ưu tiên chạy trên nền nhúng 10 W với 128 agent song song 📍 **Table 5**. Bản đồ cũng *"không phải lúc nào cũng có"* hoặc *"outdated"* 📍 *Sec. 1.2, tr.3*. Paper để là future work.

**E4. Metric Reliability có thể bị "gian lận" không?**
> Riêng Reliability thì **có** — vùng cực rộng luôn calibrated. Đó chính là lý do paper **bắt buộc** báo cáo Sharpness cùng lúc. Cặp (Reliability, Sharpness) tạo ràng buộc hai chiều không thể gian lận đồng thời.

**E5. Sharpness đơn vị m²/s nghĩa là gì?**
> ⚠️ 📍 *Sec. 4.2, tr.9* chỉ viết *"in m²/s as the volumetric measure"* — **không giải thích**. Diễn giải hợp lý nhất: diện tích **chuẩn hóa theo forecast horizon**. Nên **thừa nhận đây là thiếu sót trình bày** thay vì đoán chắc.

**E6. Nếu triển khai thật, con số nào là "deal-breaker"?**
> `R_min` 📍 **Eq. (5)**. Vì nó là **trường hợp xấu nhất**, mà an toàn xe tự hành quyết định bởi trường hợp xấu nhất. `R_min = 88%` nghĩa là tồn tại ít nhất một mức tin cậy và một horizon nơi mô hình lệch 12 điểm phần trăm.

**E7. Paper có chỗ nào mơ hồ hoặc thiếu không?**
> **Bốn chỗ**: (1) *"six parameters per forecast horizon"* mâu thuẫn với `M=3` 📍 tr.9; (2) **đơn vị m²/s** không giải thích 📍 tr.9; (3) **`N` trong Eq. (3)** không công bố 📍 tr.7; (4) Eq. (2) viết `−Σ_h` nhưng phần chữ nói *"average"* 📍 tr.6.

**E8. Mode 1 ở `t+1s` có nối với mode 1 ở `t+2s` không?**
> ⚠️ **Không, và paper KHÔNG bàn tới.** Eq. (1) áp **độc lập từng `h`** — **không có ràng buộc nhất quán thời gian**. Hệ quả: 3 mode **không phải** 3 quỹ đạo hoàn chỉnh. Đây là hệ quả của thiết kế **non-autoregressive** (đổi lấy tốc độ 0.4 ms). Là **hạn chế thật chưa được nêu trong paper**.

**E9. Nhóm có thể cải thiện gì nếu có thêm thời gian?**
> Ba việc theo thứ tự giá trị/công sức: (1) **phân tích độ nhạy theo `N`** — không cần train lại, lấp lỗ hổng nặng nhất; (2) **thử temperature scaling** của Guo et al. [7] — paper trích bài này để nêu vấn đề nhưng không dùng giải pháp của nó; (3) **ablation ego transform và vận tốc** — lập bảng số cho hai claim mà paper chỉ khẳng định bằng lời.

---

## III.C — Tra cứu công thức, hình, bảng của paper

### Công thức

| Eq. | Nội dung | Vai trò | Trang | 📖 Mục |
|---|---|---|---|---|
| **(1)** | `𝒟_{t+h}(p) = Σ_m c_{m:t+h} · ℱ_{m;t+h}(p)` | 🟢 **OUTPUT** | 6 | [II.4](#ii4--gmm-và-eq-1) |
| **(2)** | `L_NLL = −Σ_h log 𝒟_{t+h}(p_{t+h})` | 🔴 **LOSS** | 6 | [II.5.1](#ii51--hàm-mất-mát-eq-2) |
| **(3)** | `1−α(p) = (1/N)·Σ_z [𝒟(z) ≥ 𝒟(p)]` | 🔵 hậu xử lý | 7 | [II.7.3](#ii73--confidence-level-và-confidence-set-eq-3-4) |
| **(4)** | `Ω(1−α) = {p : 1−α(p) ≥ 1−α}` | 🔵 hậu xử lý | 7 | [II.7.3](#ii73--confidence-level-và-confidence-set-eq-3-4) |
| **(5)** | `Γ̂ = R_min` | 🟣 metric | 9 | [II.7.4](#ii74--hai-metric-định-lượng-eq-5-6) |
| **(6)** | `Γ̄ = R_avg` | 🟣 metric | 9 | [II.7.4](#ii74--hai-metric-định-lượng-eq-5-6) |

⚠️ Hàm kích hoạt `σ = exp(o)+1+ε_σ` và `ρ = tanh(o)·ε_ρ` ở **Sec. 3 tr.6** nhưng **KHÔNG đánh số**.

### Hình

| Fig. | Nội dung | Trang | 📖 Mục |
|---|---|---|---|
| **1** | Kiến trúc mạng: stacked LSTM + MDN Head | 7 | [II.6.1](#ii61--lstm-làm-gì-và-vì-sao-chọn-lstm) |
| **2** | Minh họa ước lượng CL bằng Monte Carlo | 7–8 | [II.7.3](#ii73--confidence-level-và-confidence-set-eq-3-4) |
| **3** | Calibration plot 4 dataset | 10 | [II.7.4](#ii74--hai-metric-định-lượng-eq-5-6) |
| **4** | Ví dụ dự đoán IMPTC (vùng 68%/95%) | 11 | [II.7.3](#ii73--confidence-level-và-confidence-set-eq-3-4) |
| **5** | Calibration plot so sánh 5 phương pháp | 14 | [II.8.3](#ii83--table-6--so-sánh-reliability-với-sota-) |

### Bảng

| Table | Nội dung | Trang | 📖 Mục |
|---|---|---|---|
| **1** | Key facts 5 dataset | 8 | [II.6.4](#ii64--dữ-liệu-và-cấu-hình-huấn-luyện) |
| **2** | Multi-Train / Single-Test | 10 | [II.8.2](#ii82--table-2-multi-train--single-test) |
| **3** | Cross-validation transferability | 11 | [II.8.4](#ii84--tốc-độ-và-transferability) |
| **4** | Input horizon scaling | 12 | [II.8.4](#ii84--tốc-độ-và-transferability) |
| **5** | So sánh tốc độ inference | 12 | [II.8.4](#ii84--tốc-độ-và-transferability) |
| **6** | So sánh **Reliability** với SOTA ⭐ | 13 | [II.8.3](#ii83--table-6--so-sánh-reliability-với-sota-) |
| **7** | So sánh **ADE/FDE** + tốc độ | 13 | [II.8.4](#ii84--tốc-độ-và-transferability) |

### Tài liệu tham khảo hay được nhắc

| Ref | Công trình | Vai trò |
|---|---|---|
| **[2]** | Bishop — *Mixture Density Networks* (1994) | Nguồn gốc MDN head |
| **[40]** | — | Nguồn gốc metric Reliability, định nghĩa CL |
| **[39]** | — | Chứng minh 1 Gaussian không đủ; Sharpness |
| **[31]** | Schoeller | Phê phán ETH/UCY; mô hình CVM |
| **[30]** | Trajectron++ | Đối thủ chính, đại diện CVAE |
| **[7]** | Guo et al. — *On calibration of modern NN* (ICML'17) | Cơ sở vấn đề miscalibration |
| **[17]** | — | Căn cứ duy nhất cho `M = 3` |

---

## III.D — Thuật ngữ

| Thuật ngữ | Ý nghĩa |
|---|---|
| **HTP** | Human Trajectory Prediction |
| **VRU** | Vulnerable Road User — người đi bộ, xe đạp |
| **AV / AM** | Autonomous Vehicle / Autonomous Machine |
| **MDN** | Mixture Density Network — mạng xuất tham số của một hỗn hợp phân bố |
| **GMM / GM** | Gaussian Mixture (Model) |
| **LSTM** | Long Short-Term Memory |
| **ADE / FDE** | Average / Final Displacement Error |
| **BoN (min·₂₀)** | Best-of-N — lấy giả thuyết tốt nhất trong N, **che giấu** N−1 cái còn lại |
| **NLL** | Negative Log-Likelihood — đo độ khớp mật độ, **không** đo hiệu chuẩn |
| **AMD / AMV** | Average Mahalanobis Distance / Average Maximum Eigenvalue |
| **CL** | Confidence Level `1−α` |
| **Ω(1−α)** | Confidence Set — tập/vùng tin cậy |
| **HDR** | Highest Density Region — tên chuẩn của khái niệm Eq. (4); **paper không dùng** |
| **Calibration** | Hiệu chuẩn — xác suất công bố khớp tần suất thực tế |
| **Reliability** (`R_avg`, `R_min`) | Độ đáng tin — sai lệch hiệu chuẩn trung bình / xấu nhất |
| **Sharpness** (`S68`, `S95`) | Diện tích vùng tin cậy 68% / 95% |
| **LOO** | Leave-One-Out — chuẩn đánh giá của ETH/UCY |
| **CVM** | Constant Velocity Model — baseline "tầm thường" nhưng ngang SOTA trên ETH/UCY |
| **CVAE** | Conditional Variational Auto-Encoder (Trajectron++) |
| **Ego coordinate** | Hệ tọa độ gắn với người: gốc tại vị trí hiện tại, trục x theo hướng đi |
| **`H_in` / `H_fc`** | Observation horizon / Forecast horizon |
| **`φ_t`** | Hướng di chuyển của VRU tại `t` |
| **`γ_m` / `r_nk`** | Responsibility — tỉ lệ trách nhiệm của mode `m` với một điểm dữ liệu |
