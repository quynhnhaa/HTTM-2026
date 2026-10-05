# Residual MDN: protocol và review trước full training

## Phạm vi đã chốt

- Model dựa trên baseline stable, giữ 8224 tham số và khởi tạo giống baseline
  với cùng seed; chỉ cộng prior CV vào hai nhóm mean đầu ra.
- Prior dùng mean vx/vy của 5 bước quan sát cuối, timestep h=1..48, delta_t=0.1.
- Covariance stable, marginal NLL, Adam, full validation, train reduction0.5,
  batch4096 và evaluator giống baseline stable.
- Train từ đầu 2500 epoch; sau đó fork continuation riêng 500 epoch, LR1e-5
  đến1e-7, giữ Adam và best parent. Đây là lịch giống đối chứng stable hiện có.
- Không chọn window velocity hoặc số epoch bằng test. Test chưa được chạy.
- Không sửa base_mdn, stable_mdn hoặc các thư mục attention.

## Review đã qua

1. Unit test: residual zero cho đúng CV, timestep đầu 0.1 s; không dùng velocity
   trước window đã chốt.
2. Chỉ mean đổi, covariance/pi giữ nguyên với cùng raw head; số tham số và
   khởi tạo trùng baseline.
3. NLL vị trí và gradient hữu hạn.
4. Smoke ba epoch trên IMPTC, 189 mẫu train/epoch, full19148 validation.
5. Có best/last/final và checkpoint1/2/3, history và 8 fixed sample IDs.
6. Capture raw output, GMM, CV và residual hữu hạn; CV+residual khớp mean cuối.
7. Các nhánh metric đều chạy được; sharpness/ASAEE smoke trên fixed8, mesh1m,
   MC64; các số đó không dùng so chất lượng full model.
8. Continuation smoke thêm2 epoch: Adam step count tiếp tục, scheduler reset đúng.
9. Hình smoke đã được xem; component means còn gần CV ở phần lớn mẫu vì chỉ
   mới3 epoch. Không suy ra chất lượng của full model từ hình smoke.

Chi tiết máy đọc: SMOKE.json. Hình: SMOKE_CV_COMPONENT_MEANS.png.

## Artifact bổ sung

Fixed prediction NPZ giữ các field của tracker stable và bổ sung:

- cv_trajectory: [8,48,2];
- residual_mu: [8,48,3,2];
- raw_residual_output: [8,48,18].

mu và raw_output gốc trong NPZ là mean/output vị trí cuối đã cộng CV, đúng với
loss và evaluator. Các đường nối cùng component index không chứng minh joint
mode xuyên thời gian; mô hình vẫn tạo GMM biên tại từng timestep.

## Giới hạn

Smoke không chứng minh model sẽ hội tụ tốt hoặc chống divergence ở epoch muộn.
Window5 là lựa chọn thử nghiệm; vận tốc không đổi có thể sai khi người rẽ/dừng,
nên mạng cần học residual đủ lớn trong các trường hợp đó. Giảm residual không
tự động bảo đảm sharpness hoặc reliability tốt hơn. Cần evaluate cùng giao thức
với baseline stable sau training; hiện chỉ có một seed.
