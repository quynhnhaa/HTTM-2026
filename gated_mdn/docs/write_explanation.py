"""Vietnamese guide grounded in the implementation and saved teaching figures."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 snapshot=json.loads((HERE/'gate_snapshot.json').read_text());last=snapshot['records'][-1]
 text=r'''# Model học số Gaussian K như thế nào?

> Tài liệu giải thích variant `gated_mdn` đang được triển khai trong project. Các hình 01–08 và 10 là minh họa cơ chế, không phải kết quả thí nghiệm. Hình 09 dùng snapshot log training thật, ghi rõ epoch. Không thay đổi `base_mdn` hay gián đoạn training để tạo tài liệu này.

## 1. Ý tưởng trong một câu

**Bắt đầu với tối đa 8 Gaussian, thêm công tắc học được cho từng Gaussian, rồi train dự báo và công tắc cùng nhau. Sau training, đếm công tắc còn mở để biết số component hoạt động.**

Đây không phải chạy riêng K=1,2,…,8 rồi chọn model có kết quả tốt nhất. Một model có head 8 component được train với một objective có sparsity. Giá trị K hoạt động có thể là bất kỳ số nguyên từ **1 đến 8**.

Ta chưa biết run hiện tại sẽ giữ bao nhiêu component. Nếu giữ 5, đó là 5 component hoạt động dưới cấu hình run này, không phải chứng minh K=5 tối ưu toàn cục.

![8 ứng viên và ví dụ còn5](images/02_survivors_example.png)

Hình này giả định các gate `[1,0.8,0,0.65,0,0.5,0.9,0]`, nên có 5 component còn mở. **Không phải run hiện tại đã học ra 5.**

## 2. K trong base_mdn nghĩa là gì?

Ở mỗi timestep tương lai, MDN xuất một phân phối hỗn hợp Gaussian 2D:

$$
p(p_h\mid X)=\sum_{k=0}^{K-1}\pi_{h,k}(X)\,\mathcal N(p_h;\mu_{h,k}(X),\Sigma_{h,k}(X)).
$$

- `X`: lịch sử quan sát của người đi bộ, shape `[32,4]`, với x,y,vx,vy.
- `h`: một trong 48 timestep dự báo, cách nhau0.1s.
- `k`: chỉ số component Gaussian ở timestep đó.
- `pi`: trọng số, không âm và tổng bằng1.
- `mu`: tâm Gaussian 2D.
- `Sigma`: covariance, xác định độ rộng/hướng tương quan.

**Một Gaussian là một component phân phối vị trí, không phải một người đi bộ hay một quỹ đạo chắc chắn.** Code cho bivariate mixture marginal theo timestep. Component k cố định qua48 timestep cũng chưa được chứng minh là cùng một hành vi hoặc một mode quỹ đạo joint.

`base_mdn` mặc định K=3. Nó học pi/mu/sigma/rho, nhưng vẫn giữ3 slot Gaussian. `gated_mdn` bắt đầu với8 slot và thêm cơ chế để giảm subset hoạt động.

![Pipeline có gate](images/01_architecture.png)

LSTM và head dự báo vẫn được đọc từ baseline. Variant giữ sigma=`exp(raw)` và rho=`tanh(raw)` legacy; hướng này không tự thay công thức covariance bằng stable.

## 3. Gate là gì? Có khác pi không?

Gate là hệ số `g_k` trong đoạn `[0,1]`:

| Gate | Ý nghĩa |
|---|---|
| `g_k=1` | Không giảm trọng số component bằng gate |
| `0<g_k<1` | Component vẫn hoạt động, nhưng logit được điều chỉnh |
| `g_k=0` | Component bị tắt; pi sau gate bằng0 trong implementation float32 |

**Gate và pi là hai thứ khác nhau.** Pi của MDN phụ thuộc người đi bộ và horizon. Gate ở variant này dùng chung trên mọi người đi bộ và48 timestep.

Ví dụ: component2 có gate dương nhưng pi rất nhỏ trên một mẫu đứng yên. Nó vẫn có thể hữu ích trên một mẫu rẽ, nên không tự bị xóa chỉ vì pi nhỏ ở mẫu đầu tiên.

![Gate global và pi local](images/07_global_vs_local.png)

Chúng ta học một **subset component global**. Không thiết kế K khác nhau cho mỗi pedestrian. Tất cả mẫu dùng cùng subset ở evaluation của một checkpoint, còn pi và Gaussian của subset đó vẫn thay đổi theo input/horizon.

## 4. Vì sao có8 component nhưng chỉ7 gate được học?

Component0 có gate luôn bằng1. Bảy component còn lại có tham số gate học được.

Nếu cho tắt cả8, có thể gặp mixture không còn component nào: tổng trọng số bằng0, không còn distribution hợp lệ. Giữ một component là cách bảo đảm ít nhất1 component tồn tại.

Component0 không có nghĩa là "đi thẳng" hoặc component quan trọng nhất. Đây là một slot được bảo vệ để mixture không rỗng, là quyết định thiết kế cho variant này.

```text
K_max = 8
1 protected component + 7 learned component gates
=> 1 <= deterministic active K <= 8
```

## 5. Sau gate, trọng số mixture tính thế nào?

Head vẫn xuất logits `a_{h,k}(X)`. Nếu chưa có gate:

$$
\pi^{base}_{h,k}=\frac{e^{a_{h,k}}}{\sum_j e^{a_{h,j}}}.
$$

Có gate thì:

$$
\pi^{gated}_{h,k}=\frac{g_k e^{a_{h,k}}}{\sum_j g_j e^{a_{h,j}}}.
$$

Code tính tương đương bằng `softmax(logits + log(gate))`. Gate0 dùng sentinel -1e9 thay cho log(0), tạo pi0 trong float32 và vẫn lưu raw logits hữu hạn. Một component được bảo vệ bảo đảm có slot hoạt động; không gán pi cho component đã tắt để bù tổng.

Ví dụ nhỏ với3 component:

```text
Pi trước gate       = [0.20, 0.30, 0.50]
Gate                = [1.00, 0.00, 0.50]
Nhân pi × gate      = [0.20, 0.00, 0.25]
Chuẩn hóa tổng0.45  = [0.4444, 0.0000, 0.5556]
```

![Chuẩn hóa pi](images/06_weight_normalization.png)

Ảnh hưởng trực tiếp của gate nằm ở **trọng số mixture**. Nó không trực tiếp cộng vào mu hay nhân sigma. Nhưng vì train chung, gradient có thể làm mu/sigma/rho và LSTM thay đổi gián tiếp.

![Ví dụ mật độ GMM khi tắt một component](images/10_toy_gmm.png)

Hình density này là distribution giả định để giảng giải. Việc một vùng density biến mất không có nghĩa ta xóa observed trajectory hoặc ground truth.

## 6. Model có học trực tiếp một biến nguyên K không?

**Không.** Một con số nguyên như K=5 khó cập nhật trực tiếp bằng gradient. Chúng ta học7 tham số thực tên `log_alpha`, mỗi tham số điều khiển một gate.

```text
Tham số model được học
├── LSTM weights
├── MDN head weights
└── gate log_alpha[7]

Sau đó tính gate -> đếm gate dương -> K hoạt động
```

`log_alpha` ở đây là tham số gate, **không phải mixture weight alpha/pi** thường dùng trong paper MDN.

Gate cần vừa nhận gradient vừa có thể bằng0 chính xác. Vì vậy implementation dùng **hard-concrete**, thay vì chỉ đặt gate=`sigmoid(log_alpha)` (sigmoid hữu hạn không tạo0 chính xác trong lý thuyết).

## 7. Hard-concrete tạo công tắc bằng0 như thế nào?

Trong mỗi train batch, với từng gate học được:

**Bước A — lấy số ngẫu nhiên:**

$$u_k\sim Uniform(0,1).$$

Code tránh endpoint bằng clamp `u` vào `[1e-6,1-1e-6]`.

**Bước B — tạo giá trị mềm:**

$$s_k=\mathrm{sigmoid}\left(\frac{\log u_k-\log(1-u_k)+\log\alpha_k}{\beta}\right).$$

**Bước C — kéo dãn vượt đoạn0–1:**

$$\bar s_k=s_k(\zeta-\gamma)+\gamma.$$

**Bước D — cắt về0–1:**

$$g_k=\mathrm{clamp}(\bar s_k,0,1).$$

Cấu hình hiện tại: beta=2/3, gamma=-0.1, zeta=1.1. Vì đoạn kéo dãn vượt0 và1, một vùng các giá trị u bị cắt thành **0 thật sự**, một vùng thành1, phần giữa vẫn mềm.

![Các bước hard-concrete](images/04_hard_concrete.png)

Random không phải quyết định bật/tắt tùy tiện mãi mãi. `log_alpha` được optimizer cập nhật, làm xác suất mở/đóng thay đổi. Train lấy **một vector7 gate mỗi batch**, dùng chung vector đó trong batch và48 timestep.

Một gate bằng0 ở batch này chưa bị xóa khỏi model: batch sau nó có thể mở lại. Hard clipping làm gradient NLL qua gate bị chặn ở phần đã clamp; penalty xác suất mở vẫn có gradient theo log_alpha. Mô hình không đảm bảo mọi gate đóng sớm đều phục hồi tốt, nên cần theo dõi collapse/underfitting.

Nguồn cơ chế: [Louizos, Welling, Kingma — Learning Sparse Neural Networks through L0 Regularization, ICLR2018](https://arxiv.org/abs/1712.01312). Gate component GMM là adaptation của project này, không phải paper đó đã chứng minh hiệu quả cho IMPTC hoặc đã đề xuất chính xác variant này.

## 8. Vì sao model muốn tắt bớt Gaussian?

Nếu chỉ giảm NLL, model có thể giữ tất cả8 slot. Chúng ta thêm chi phí sử dụng component:

$$\mathcal L=\mathrm{NLL}+\lambda\,\mathbb E[K_{active}].$$

NLL trong batch vẫn theo baseline:

$$\mathrm{NLL}=-\frac{1}{B\times48}\sum_{b=1}^{B}\sum_{h=1}^{48}\log p(y_{b,h}\mid X_b,g).$$

**Không cộng log likelihood riêng của component thành NLL.** `p` là density của cả mixture, đã cộng weighted Gaussian; đó cũng không phải một mixture joint trên toàn48 bước.

Xác suất gate nonzero có công thức:

$$q_k=P(g_k>0)=\mathrm{sigmoid}\left(\log\alpha_k-\beta\log\frac{-\gamma}{\zeta}\right).$$

Số component hoạt động kỳ vọng:

$$\mathbb E[K_{active}]=1+\sum_{k=1}^{7}q_k.$$

Số1 đến từ component được bảo vệ. Chi phí này là **expected L0 penalty**, vì L0 đếm số phần tử nonzero; không phải L1=sum giá trị gate. Train dùng một draw gate để ước lượng phần NLL stochastic của batch, còn expected active K được tính bằng công thức, không đếm0/1 sample để làm penalty không khả vi.

Cấu hình hiện tại lambda=0.01:

- Phần NLL khuyến khích distribution dự báo tốt ở GT.
- Phần penalty tạo áp lực giảm xác suất sử dụng component.
- Backprop cập nhật cả LSTM, MDN và log_alpha bằng cùng Adam.

![Vòng học chung](images/03_learning_loop.png)

Nếu giảm một gate giúp bớt penalty nhưng làm NLL tệ đi nhiều, gradient tổng có thể ưu tiên giữ gate. Nếu component ít cần thiết, penalty có thể giúp đóng gate. Đây là cơ chế khuyến khích, **không phải thuật toán kiểm tra từng component rồi đảm bảo loại đúng component dư**.

Lambda là sức ép sparsity, không phải số component muốn giữ. Lambda nhỏ có thể giữ8; lớn có thể đóng quá nhiều, làm NLL/displacement/coverage kém. Chưa biết lambda0.01 phù hợp hay không. Không đổi lambda giữa run để ép ra một K mong muốn.

## 9. Ba con số rất dễ bị nhầm

| Đại lượng | Ý nghĩa | Có nguyên không? |
|---|---|---|
| `K_max=8` | Số slot/head component ban đầu | Có, cố định |
| `expected_active_components` | 1+sum xác suất gate stochastic nonzero | Không bắt buộc |
| `deterministic_active_components` | Đếm gate evaluation dương | Có, từ1 đến8 |

Ví dụ giả định q của7 gate đều0.95: expected K=1+7×0.95=7.65. **Không phải model chọn K=7.65 hoặc làm tròn thành8 để kết luận K.**

Tại evaluation, code không lấy random u. Nó dùng:

$$\hat g_k=\mathrm{clamp}(\mathrm{sigmoid}(\log\alpha_k)(\zeta-\gamma)+\gamma,0,1).$$

Rồi đếm:

$$K_{eval}=1+\sum_{k=1}^{7}\mathbf1[\hat g_k>0].$$

Không dùng `pi>0.1`, không làm tròn expected K, không giữ top5 theo ý người dùng.

![Gate và xác suất mở](images/05_probability_vs_gate.png)

Với gamma=-0.1, zeta=1.1, gate deterministic bằng0 khi `log_alpha <= log(0.1/1.1) ≈ -2.398`. Ngưỡng này đến từ phép stretch/clamp đã chọn, không phải threshold pi tùy ý sau khi xem test. Tại ngưỡng đó, xác suất stochastic gate mở vẫn có thể dương: expected K và eval K do đó thực sự khác nhau.

## 10. Ví dụ nếu model còn5 component

Giả sử sau training gate xác định là:

```text
Component index:   0    1    2    3    4    5    6    7
Gate:           1.0  0.8  0.0  0.65 0.0  0.5  0.9  0.0
Active:          có   có   tắt   có   tắt   có   có   tắt

Active indices = [0,1,3,5,6]
K_eval = 5
```

Đây là loại3 slot trong cùng một run8-slot. Không có bước train riêng model K5 trong ví dụ này.

Nếu còn8: run chưa prune component ở checkpoint đó. Không được diễn giải ngay rằng "K8 tối ưu"; có thể penalty yếu, NLL muốn giữ component, gate chưa hội tụ hoặc minimum cục bộ. Nếu còn1: sparsity mạnh, nhưng cũng có thể underfit. Chất lượng phải được evaluate, không đánh giá chỉ bằng số K nhỏ.

## 11. Vì sao gate đổi mà model chưa nhỏ đi?

Trong training, head vẫn tạo đầy đủ `48×6×8` outputs. Tắt gate chỉ làm pi0; các hàng weights/head chưa bị xóa.

Khi **compact export**:

1. Đọc gate deterministic của checkpoint.
2. Giữ các hàng mean/sigma/rho/logits của component active ở mỗi horizon.
3. Cộng `log(gate)` của component sống vào bias logits để giữ weighting đã học.
4. Export một LSTM–MDN thông thường với K bằng số component còn sống.

![Train, eval, export](images/08_train_eval_export.png)

Chúng ta đã test head8 có thể export còn5 và density/log_prob tương đương trong sai số floating point. **Unit test đó gán gate thủ công để kiểm tra export; không chứng minh training đã tìm ra K5.**

| Model/head | Số tham số |
|---|---:|
| Base K3, hidden8 | 8.224 |
| Gated K_max8, hidden8 | 21.191 |
| Compact nếu active5 | 13.408 |
| Compact nếu active3 | 8.224 |

Gated head8 không rẻ hơn baseK3 khi training. Có active5 sẽ nhỏ hơn head8 ban đầu, nhưng vẫn lớn hơn baseK3. Chưa benchmark tốc độ compact.

## 12. Checkpoint nào quyết định K cuối cùng?

Variant train objective có penalty, nhưng **best checkpoint hiện chọn bằng unpenalized NLL trên full validation**, với deterministic gates.

```text
Mỗi epoch
  train stochastic NLL + L0 penalty
  -> validation deterministic NLL
  -> cập nhật best nếu validation NLL giảm
  -> lưu gate state / active indices của checkpoint

K ở best = số deterministic gate dương trong best checkpoint
K ở final = số deterministic gate dương ở epoch cuối
Hai K này có thể khác nhau.
```

Không tự lấy checkpoint K nhỏ nhất dù NLL kém. Vì best selection không dùng penalty, checkpoint best có thể còn8 trong khi final sparse hơn. Đây là lựa chọn protocol đã ghi rõ, không đảm bảo best là model nhỏ nhất.

Full validation19.148 mẫu và sample-weighted epoch NLL của variant khác random50%/equal-batch average ở run base cũ. Công thức NLL trong batch và preprocessing/covariance vẫn giữ baseline. Khi báo cáo so sánh, phải nêu khác biệt selection này. Đối chứng ungated K8 cùng protocol sẽ giúp tách tác động tăng capacity và gating; chưa train đối chứng đó.

## 13. Điều gì đang xảy ra trong run thật?

<!-- SNAPSHOT_SUMMARY -->

![Snapshot thật của gate](images/09_actual_gate_snapshot.png)

Hình này chụp log tại thời điểm tạo tài liệu, không tự cập nhật và không phải kết quả cuối. Gate dương tăng ở giai đoạn đầu cho thấy các component vẫn được giữ; không nên vẽ một đường K8→K5 giả định rồi gọi đó là kết quả thực nghiệm.

Run vẫn có thể tiếp tục; để xem thông tin mới nhất, đọc `gate_history.jsonl` và `history.csv` của run. Expected K có thể tăng hoặc giảm, không có điều kiện ép giảm đơn điệu.

## 14. Giới hạn cần trình bày với thầy

- Đây là cơ chế học sparsity/component activity, không phải phép chứng minh số Gaussian đúng của dữ liệu.
- K_max và lambda vẫn là lựa chọn thiết kế; dùng validation để chọn nếu có bước chọn cấu hình, không chọn bằng test. Không có phương pháp này tự loại bỏ mọi siêu tham số.
- Gate và logits cùng điều khiển pi: logits có thể bù một phần sự giảm gate mềm. Vì vậy gate nhỏ không tự chứng minh component vô dụng và penalty không đảm bảo pruning. Hard zeros/global sharing/regularization tạo thêm cấu trúc, nhưng không giải quyết mọi vấn đề identifiability.
- Label Gaussian có tính hoán vị. Gate global tắt slot dùng chung; slot đó không mang một nghĩa hành vi cố định qua toàn bộ dữ liệu hoặc horizon.
- Legacy sigma/rho giữ nguyên nên numerical divergence/covariance collapse vẫn có thể xảy ra, kể cả khi component gate đóng vì decoder chưa export vẫn tính covariance các slot.
- Sparse hơn chưa chắc NLL, ADE/FDE, reliability hoặc sharpness tốt hơn. Gating cũng không được đảm bảo giải quyết các outlier S68 đã phân tích trước.
- Run đang thực hiện một seed; cần kết quả hoàn chỉnh, so sánh và đối chứng trước khi nói phương pháp hiệu quả. Không gọi adaptation này là phương pháp hoàn toàn mới của lĩnh vực.

Cách mô tả đóng góp chính xác hơn:

> “Chúng tôi bổ sung global hard-concrete component gates và expected L0 regularization vào LSTM–MDN, nhằm học một subset Gaussian hoạt động trong quá trình training; sau đó đánh giá trade-off giữa chất lượng dự báo và độ phức tạp model.”

Tránh viết: “Thuật toán tự tìm K tối ưu từ1 tới8 và chắc chắn cải thiện mọi metric.”

## 15. Đọc code và artifact ở đâu?

| Nội dung | File |
|---|---|
| Lấy gate, xác suất mở, expected K, điều chỉnh logits, compact export | [model.py](../model.py) |
| Objective, train/eval mode, best selection và RNG | [train.py](../train.py) |
| Log gate và checkpoint/fixed captures | [artifacts.py](../artifacts.py) |
| Cấu hình K_max8/lambda0.01 | [gated_peds_imptc.json](../configs/imptc/gated_peds_imptc.json) |
| Evaluator dùng deterministic gates | [evaluate.py](../evaluate.py) |
| Export inference-only | [export.py](../export.py) |
| Test đúng5 survivors từ head8 và distribution equivalence | [test_model.py](../tests/test_model.py) |
| Smoke review | [SMOKE.md](../reports/SMOKE.md) |
| Run hiện tại/tmux | [RUN_K8.md](../reports/RUN_K8.md) |
| Snapshot dùng trong hình09 | [gate_snapshot.json](gate_snapshot.json) |
| Script tạo10 hình | [create_explanation_figures.py](create_explanation_figures.py) |

Artifacts thật của run:

```text
results/trained_models/gated_mdn/imptc/gated_peds_imptc/
  runs/gated_k8_seed2024_tmux/
    history.csv                    # NLL, objective, penalty, expected K, eval K
    gate_history.jsonl             # gate vectors/probabilities/active indices
    checkpoints/best.pt            # best theo validation NLL
    checkpoints/last.pt             # epoch hoàn tất mới nhất
    checkpoints/final.pt            # chỉ có sau khi hoàn tất training
    fixed_samples/predictions/      # cùng8 mẫu, decoded GMM + gate state
```

Theo dõi phiên đã chạy bằng `tmux attach -t gated_k8`. Tài liệu này không yêu cầu tạo run khác.

Để tái tạo tài liệu/hình với snapshot mới:

```bash
.venv/bin/python gated_mdn/docs/create_explanation_figures.py
.venv/bin/python gated_mdn/docs/write_explanation.py
```

Lệnh đầu ghi lại snapshot/hình09 mới từ log hiện có; không đổi training. Các hình minh họa khác có dữ liệu cố định.
'''
 summary=(f"Snapshot run **{snapshot['run_id']}** đến epoch **{last['epoch']}**:\n\n"
          f"- `K_max = 8`.\n- `deterministic_active_components = {last['deterministic_active_components']}`.\n"
          f"- `expected_active_components = {last['expected_active_components']:.4f}`.\n"
          f"- Gate xác định: `{[round(x,4) for x in last['deterministic_gates']]}`.\n\n"
          "**Ở snapshot này chưa có component bị tắt tại evaluation. Đây chưa phải kết quả học ra K nhỏ hơn8.**")
 text=text.replace('<!-- SNAPSHOT_SUMMARY -->',summary)
 # Raw string uses escaped backslashes for TeX; write single slashes to Markdown.
 text=text.replace('\\\\','\\')
 (HERE/'GIAI_THICH_HOC_K.md').write_text(text+'\n')
 print(HERE/'GIAI_THICH_HOC_K.md')
if __name__=='__main__':main()
