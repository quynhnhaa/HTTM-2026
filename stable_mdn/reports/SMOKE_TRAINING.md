# Kiểm chứng smoke training Stable MDN

Kết quả: **PASS**. Báo cáo máy đọc: [SMOKE_TRAINING.json](SMOKE_TRAINING.json).

## Phạm vi

- IMPTC thật, kiến trúc LSTM hidden 8, K=3, input 32, output 48.
- Parameterization stable: sigma floor 0.01 m, exp guard 1000 m, rho limit 0.999.
- Ba epoch; 189 mẫu train mỗi epoch, batch 128.
- Toàn bộ 19.148 mẫu validation để tính NLL và chọn best.
- Metric ADE/FDE và reliability trên toàn validation mỗi epoch, MC 64.
- Sharpness/ASAEE kiểm tra riêng trên tám mẫu validation cố định, mesh 1 m.
- Chưa chạy full training hoặc đánh giá test để báo kết quả đồ án.

## Các kiểm chứng đã qua

1. Run liên tục hoàn tất ba epoch và ghi history/LR/NLL.
2. Resume từ checkpoint epoch 1 chạy tới epoch 3 cho model, optimizer và
   scheduler giống hệt run liên tục; train/validation NLL và LR cũng trùng.
3. Best, last, final và checkpoint định kỳ được ghi. Run liên tục có epoch 1–3;
   run resume có epoch 2–3 và lịch sử checkpoint bao gồm epoch 1–3.
4. Tám sample IDs cố định khớp các prediction snapshots; X và ground truth
   nằm trong inputs.npz dùng chung cho mọi checkpoint.
5. Raw output, pi, mu, sigma, rho và covariance hữu hạn; pi tổng bằng 1,
   sigma đạt sàn và trị tuyệt đối rho nhỏ hơn 1.
6. Tính lại validation NLL cuối bằng checkpoint và toàn bộ dữ liệu:
   6.281075902938444, khớp logging trong sai số CPU/GPU cho phép 1e-4.
7. Metric snapshots ghi version stable_v2_ecdf_rng, seed và full validation.
8. Mọi nhánh metric có kết quả hữu hạn. Số smoke không dùng để so chất lượng
   với baseline full training vì khác ngân sách, MC và mesh.
9. Hash mọi file base_mdn không thay đổi.

## Giới hạn và bước tiếp theo

- Ba epoch chưa chứng minh chống divergence ở epoch muộn hoặc cải thiện dự báo.
- Smoke dùng train subset, batch và MC khác full config; full run cần theo
  stable_peds_imptc.json đã review.
- Resume được kiểm tra bằng fork checkpoint epoch 1 vào run riêng để đối chiếu;
  chưa mô phỏng crash giữa epoch.
- Chưa tạo variant attention stable hoặc ablation không attention.
- Trước full training, chốt sigma/rho và ngân sách thí nghiệm; không tự chạy
  full training nếu chưa được yêu cầu.

Script tái kiểm chứng: `stable_mdn/tests/smoke_training.py`.
Config smoke: `stable_mdn/configs/imptc/smoke_stable_peds_imptc.json`.
Các đường dẫn run thực tế được ghi trong báo cáo JSON.
