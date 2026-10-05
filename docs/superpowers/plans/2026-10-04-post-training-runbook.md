# Việc cần làm khi các run train xong (runbook, 2026-10-04)

Đọc file này để biết làm gì tiếp, không cần nhớ lại cuộc trò chuyện. Spec của từng hướng nằm ở `docs/superpowers/specs/`; plan triển khai của growing_mdn ở [2026-10-04-growing-mdn.md](2026-10-04-growing-mdn.md). **sparsemax_mdn chưa có file plan riêng**: nó có spec ([sparsemax spec](../specs/2026-10-04-sparsemax-mdn-design.md)), được cài đặt theo một brief tạm không lưu trong repo, và phần "làm gì sau khi train" nằm ở đây.

## 0. Quy tắc phải giữ (người dùng đã yêu cầu)
- Không sửa `base_mdn/`. Mọi biến thể nằm ở thư mục riêng (`growing_mdn/`, `sparsemax_mdn/`).
- Không chạy đánh giá trên tập **test** khi chưa hỏi người dùng. Test chỉ dùng **một lần**, cho cấu hình đã công bố trước.
- Chưa commit gì vì người dùng chưa cho phép; mọi thay đổi đang nằm trong working tree.
- Thầy của người dùng không chấp nhận "chạy nhiều biến thể rồi chọn" là cải tiến. Vì vậy sparsemax (một lần train, cấu hình công bố trước) là cải tiến chính; growing_mdn chỉ là phân tích bổ trợ về K.
- Không đưa ra kết luận từ một seed; nói rõ khi chênh lệch nhỏ.

## 1. Trạng thái các run (cập nhật sau khi cả hai run xong)
| Run | Cấu hình | Trạng thái |
|---|---|---|
| `sparsemax_k16_seed2024` | sparsemax thuần, K_max 16, 2500 epoch, seed 2024 | **đã xong**; đã so metric validation với baseline và phân tích K trên toàn bộ validation (đã ghi vào `docs/sparsemax_mdn/SPARSEMAX_MDN.md` và `MIDTERM_NOTES.md`) |
| `growing_k12_v2_seed2024` | growing_mdn sau Revision 1, K 1→12 | **đã xong**; `select_k` đã chạy (K chọn = 11, `selection.json` đã ghi), kết quả ở `docs/growing_mdn/GROWING_MDN.md` mục 7.6 |
| `growing_k12_seed2024` (v1) | bản có lỗi tách | đã dừng, giữ làm bằng chứng |
| `sparsemax_k8_seed2024` | K_max 8 | đã dừng sau 48 epoch, giữ artifact |

**Đã làm:** test của sparsemax đã chạy đúng một lần (kết quả ở `docs/sparsemax_mdn/SPARSEMAX_MDN.md` mục 6.7; file `<run>/evaluation/test_best_clampedpi.json` và `_persample.npz`; lock file `test_evaluation_started.lock` giữ nguyên, không xoá). **Việc còn lại (cần quyết định của người dùng):** (1) test cho growing_mdn K = 11 (`growing_mdn.evaluate --official`) nếu muốn; tính số liệu sharpness từng mẫu cho baseline K = 3, K = 8 trên test để đối chiếu đuôi; (2) nhiều seed cho sparsemax và baseline nếu muốn khẳng định hơn; (3) thí nghiệm bổ sung nếu muốn xử lý độ sắc nét hoặc quá khớp (chặn dưới σ, hoặc K theo từng mẫu bằng gate); (4) commit khi được phép. Các mục 2 và 3 dưới đây là bản gốc kế hoạch, giữ để tham chiếu cách làm lại.

## 2. Khi `sparsemax_k16_seed2024` xong
Kiểm tra: `tmux ls | grep sparsemax`, manifest `status == completed`, `history.csv` có 2500 hàng.

1. **Chỉ số theo epoch** (đã có trong `history.csv`): validation NLL, `mean_support_size`, `fraction_full_support`, `dead_components`. Kiểm tra trần: nếu `fraction_full_support` ở cuối không còn ≈ 0 thì K_max = 16 đang chặn, báo là cận dưới của K.
2. **Metric chính thức theo epoch** (đã có, mỗi 250 epoch, tập validation): `metrics/epoch_0250.json` … `epoch_2500.json` của sparsemax so với cùng tên file của baseline K = 8 (`results/trained_models/base_mdn/imptc/m8_peds_imptc/runs/imptc_m8_seed2024/metrics/`) và baseline K = 3 (`.../default_peds_imptc/runs/imptc_baseline_seed2024/metrics/`). Cùng giao thức và cùng seed, nên so sánh trực tiếp được; báo các khóa `ravg_percent`, `rmin_percent`, `s68_m2_per_s`, `s95_m2_per_s`, `asaee_m_per_s`, `minade20_m`, `minfde20_m` và NLL.
3. **Phân tích K trên validation** (không dùng test):
   `.venv/bin/python -m sparsemax_mdn.analyze_k --run-id sparsemax_k16_seed2024 --config sparsemax_k16_peds_imptc --gpu -1`
   (bỏ `--limit` để dùng toàn bộ validation). Kết quả ở `<run>/analysis/k_usage.json`. Báo: phân bố K(x,t), K theo bước dự báo, thành phần chết, quan hệ với độ khó (đang yếu ở epoch 115 đến 140).
4. **Đối chiếu với tiêu chí đã công bố** (spec mục 5): (1) NLL, Ravg, Rmin không kém baseline và S68/S95 không xấu đi; (2) K trung bình < K_max và thay đổi theo mẫu/bước; (3) K gắn với độ khó. Báo từng điểm đạt hay không đạt, kể cả khi không đạt.
5. **Đánh giá và việc còn thiếu:**
   - `sparsemax_mdn/evaluate.py` **đã có**: NLL chính xác + metric chính thức (`--split validation|test`), tùy chọn `--exact-pi` (loại hẳn thành phần π = 0, spec mục 8), `--limit N` (chỉ validation). Test chỉ chạy được một lần cho mỗi run và cần `--confirm-test-once`; chỉ chạy khi người dùng yêu cầu rõ ràng. Cách dùng ở `sparsemax_mdn/README.md`.
   - Một so sánh có phương sai: nếu muốn kết luận về "tốt hơn baseline" thì cần nhiều seed cho cả sparsemax và baseline (chưa chạy).
6. **Cập nhật tài liệu:** chạy lại `.venv/bin/python docs/sparsemax_mdn/generate_figures.py`, sửa phần 6 và phần trạng thái ở đầu [SPARSEMAX_MDN.md](../../sparsemax_mdn/SPARSEMAX_MDN.md) bằng kết quả cuối (hiện là ảnh chụp giữa chừng), thêm một mục về `sparsemax_mdn` vào `MIDTERM_NOTES.md` (chưa có).

## 3. Khi cần dùng kết quả growing_mdn v2 (đã xong)
1. **Chọn K trên validation:** `.venv/bin/python -m growing_mdn.select_k --run-id growing_k12_v2_seed2024`. Ngưỡng đã đóng băng trong cấu hình của run (ε = 0.02, 0.5 điểm phần trăm, 10%); không đổi. Kết quả ở `<run>/selection.json`. Lệnh từ chối chạy lại nếu `selection.json` đã tồn tại.
2. Đọc `phase_summary.json` (NLL và metric validation của từng K), `growth_history.jsonl` (kiểm tra mọi lần tách có `aborted: false` và |ΔNLL| nhỏ), manifest. Lưu ý: metric từng phase chỉ tính ở cuối phase.
3. Cập nhật [GROWING_MDN.md](../../growing_mdn/GROWING_MDN.md) (mục 7.5 hiện chỉ có lần tách đầu; hình 21 sẽ tự cập nhật khi chạy lại `docs/growing_mdn/generate_figures.py`) và nêu rõ đây là **phân tích về K**, không phải cải tiến chính.
4. **Đánh giá test:** `growing_mdn.evaluate --official` chỉ chạy cho K trong `selection.json`, một lần, và **phải hỏi người dùng trước**. Không dùng test để so sánh các K khác.
5. Cân nhắc trung thực khi trình bày: K=1 chạy 400 epoch (baseline 2500) nên các K nhỏ chưa hội tụ như baseline; mọi phase là khởi động ấm nối tiếp, so với baseline K = 3 và K = 8 phải so theo tổng compute.

## 4. Việc chung sau cùng
- Viết/cập nhật `MIDTERM_NOTES.md` (sparsemax chưa có mục) và `README_LOCAL.md` nếu cần hướng dẫn chạy.
- Nếu người dùng cho phép: commit (các thư mục `growing_mdn/`, `sparsemax_mdn/`, `docs/growing_mdn/`, `docs/sparsemax_mdn/`, `docs/superpowers/`); trước khi commit chạy `git status` để xem đúng các file được thêm, tránh đưa vào `results/` và `data/` (đã gitignore).
- Chạy lại các kiểm tra: `.venv/bin/python -m unittest discover -s growing_mdn/tests -t .`, `.venv/bin/python -m unittest discover -s sparsemax_mdn/tests -t .`, `.venv/bin/python -m growing_mdn.base_hashes --check`, `.venv/bin/python -m sparsemax_mdn.base_hashes --check` (cả hai phải báo `base_mdn unchanged`; ba file đã bị xoá trong `base_mdn/` có từ trước và đã nằm trong bản chụp hash).
- Hướng chưa làm, chỉ làm nếu người dùng yêu cầu: chặn dưới cho σ hoặc chính quy hóa σ để xử lý mối lo quá khớp; K theo từng mẫu bằng gate phụ thuộc đầu vào; nhiều seed.
