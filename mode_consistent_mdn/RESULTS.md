# Kết quả IMPTC: MDN nhất quán theo quỹ đạo và baseline M3

## Thiết lập

- Cùng dữ liệu IMPTC, 3 Gaussian, seed huấn luyện 2024 và 2500 epoch.
- Baseline M3: checkpoint tốt nhất epoch 1961, 8224 tham số.
- Mode-consistent MDN: checkpoint tốt nhất epoch 2473, 6955 tham số (ít hơn 1269, tương đương 15,4%).
- Bộ test có 56.694 mẫu. Đánh giá bổ sung lấy 20 mẫu dự đoán cho mỗi quan sát, với seed 2024.

## So sánh bổ sung trên toàn bộ test

| Chỉ số | Baseline M3 | Mode-consistent MDN | Cách đọc |
| --- | ---: | ---: | --- |
| minADE20 (m) | 0,518 | **0,476** | Thấp hơn tốt hơn; cải thiện khoảng 8,1% trong phép lấy mẫu này. |
| minFDE20 (m) | **0,604** | 0,615 | Bản cải tiến kém hơn khoảng 1,8%. |
| NLL biên mỗi vị trí | **-1,092** | -0,989 | Thấp hơn tốt hơn; bản cải tiến kém hơn. |

Những số trên được tính bằng `evaluate.py` theo cùng công thức khoảng cách và cùng 6 mốc dự báo, nhưng **quy tắc lấy mẫu khác nhau theo đúng mô hình**: baseline lấy Gaussian độc lập tại từng thời điểm, còn bản cải tiến chọn một mode cho toàn quỹ đạo. Vì vậy, cải thiện minADE phản ánh cả mô hình lẫn quy tắc sinh quỹ đạo; không thể quy riêng cho một yếu tố.

NLL chung của cả quỹ đạo chia cho 48 bước là -1,433 ở bản cải tiến. Không so sánh con số này với NLL biên -1,092 của baseline vì hai đại lượng mô tả hai phân phối khác nhau.

## Bộ đánh giá chính thức của repo

Evaluator gốc đọc GMM 2D **biên tại từng thời điểm** ở cả hai checkpoint. Đây là bảng dùng để đối chiếu trực tiếp với baseline đã công bố trong repo; nó không đo được tính nhất quán của mode trên cả quỹ đạo.

| Chỉ số test | Baseline M3 | Mode-consistent MDN | Nhận xét |
| --- | ---: | ---: | --- |
| minADE20 (m) | **0,464** | 0,471 | Bản mới kém 0,007 m. |
| minFDE20 (m) | **0,607** | 0,615 | Bản mới kém 0,008 m. |
| Ravg (%) | **97,47** | 96,60 | Bản mới thấp hơn 0,87 điểm %. |
| Rmin (%) | **94,06** | 90,72 | Bản mới thấp hơn 3,35 điểm %. |
| S68 (m²/s) | **1,358** | 1,629 | Vùng dự báo của bản mới rộng hơn. |
| S95 (m²/s) | **6,061** | 7,394 | Vùng dự báo của bản mới rộng hơn. |
| ASAEE (m/s) | **0,235** | 0,242 | Bản mới kém 0,007 m/s. |

NLL biên mỗi vị trí của baseline là -1,092, còn bản mới là -0,989 (thấp hơn tốt hơn). NLL quỹ đạo chung của bản mới là -1,433 mỗi bước; không dùng con số đó để so trực tiếp với NLL biên của baseline.

**Kết luận:** Phương pháp mới tạo được mode nhất quán trên toàn quỹ đạo, giảm 15,4% số tham số và có minADE tốt hơn trong phép lấy mẫu quỹ đạo bổ sung. Tuy nhiên, ở evaluator chính thức, mọi chỉ số trên đều kém baseline M3; hiện chưa có bằng chứng rằng đây là một cải tiến về độ chính xác hoặc chất lượng bất định. Kết quả âm này nên được trình bày trung thực cùng động cơ phương pháp.

## Hình và dữ liệu

- [Kết quả đánh giá bổ sung](../results/trained_models/mode_consistent_mdn/imptc/mode_consistent_peds_imptc/runs/imptc_mode_consistent_seed2024/testing/evaluation.json)
- [Ba trường hợp test: tốt, bất định, thất bại](../results/trained_models/mode_consistent_mdn/imptc/mode_consistent_peds_imptc/runs/imptc_mode_consistent_seed2024/figures/07_test_case_studies.png)
- [Checkpoint và quá trình huấn luyện](../results/trained_models/mode_consistent_mdn/imptc/mode_consistent_peds_imptc/runs/imptc_mode_consistent_seed2024/run_manifest.json)

Các số của baseline lấy từ `results/comparisons/imptc_m1_vs_m3/comparison.json`; các số bản mới lấy từ `testing/evaluation.json`. Cả hai dùng checkpoint tốt nhất theo validation. Bảng bổ sung và bảng chính thức sử dụng thủ tục lấy mẫu khác nhau, vì vậy không trộn các số minADE/minFDE giữa hai bảng.
