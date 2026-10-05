> Báo cáo lịch sử của smoke v1. Bản triển khai hiện tại và smoke mới: [PREDICTIVE_SMOKE.md](PREDICTIVE_SMOKE.md).

# Kiểm tra triển khai Bayesian MDN

Đã pass 3 unit tests và smoke 3 epoch IMPTC. Mỗi epoch lấy 189 mẫu train, validation đủ 19.148 mẫu. Validation NLL: 6,39506 → 6,37082 → 6,35838. Đây là kiểm tra pipeline, không đánh giá chất lượng model đã hội tụ.

Đã kiểm tra best/last/final và checkpoint epoch1/2/3; optimizer, scheduler, RNG và loader state. Có 8 fixed sample IDs, input/ground truth, 5 prediction archives gồm raw/decoded MDN parameters và posterior weights. Các archive số học hữu hạn, pi chuẩn hóa. Cả 7 official metric branches đã chạy ở 3 epoch trên 8 fixed validation samples với MC64/mesh1m. Limited test16 hoàn tất. Hash các file baseline khớp snapshot trước đó.

K diagnostic theo 99% global mass = 7, **đã bằng 7 tại prior ban đầu**. Mean global weights gần như chưa đổi sau 3 epoch. Không có bằng chứng học thành công số Gaussian, không loại thành phần và chưa chạy full training.

Điểm cần review trước full: prior concentration1, conditional logits bounded±1, KL/(N_ref*48), threshold99%, plug-in inference thay cho posterior predictive. Những lựa chọn này ảnh hưởng K và chất lượng. Đọc ../README.md để xem mô hình và giới hạn.

Báo cáo máy đọc: SMOKE.json. Chạy lại review: `.venv/bin/python -m bayesian_mdn.review_smoke`.
