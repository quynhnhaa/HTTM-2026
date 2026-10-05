# Phân tích S68 của baseline và attention legacy

`analyze_s68.py` đọc checkpoint best của hai run legacy, không sửa model/loss và không train. Chạy từ gốc repo:

```bash
.venv/bin/python analysis/analyze_s68.py
```

Mặc định phân tích toàn bộ 56.694 mẫu test, sáu horizon chính thức, K=3, 1.000 mẫu Monte Carlo, lưới [-18,18]² với bước 0,1 m. Kết quả: `results/analysis/legacy_s68/`.

- `per_sample.csv`: A68/A95 và GT thuộc vùng 68% cho mỗi mẫu/horizon; đây là diện tích m², không phải điểm S68 m²/s.
- `per_sample.npz`: raw outputs, thresholds, diện tích và ID ổn định.
- `case_parameters.csv`: pi, mu, sigma, rho của các trường hợp được chọn.
- `summary.json`: nguồn chọn mẫu, diện tích, tham số, S68/S95 tính lại và phân rã đóng góp percentile.
- `overview.png`: diện tích theo horizon, phân vị diện tích và đóng góp vào điểm S68.
- PNG đối chiếu các mẫu lớn nhất, tăng mạnh nhất và giảm mạnh nhất; hai nhóm đầu có thể chọn cùng mẫu.

Chỉ vẽ lại bằng cache:

```bash
.venv/bin/python analysis/analyze_s68.py --resume
```

Không thay checkpoint, seed, tập dữ liệu hoặc protocol khi dùng cache; hãy chọn thư mục output mới cho thí nghiệm khác. Chạy tập con nếu cần:

```bash
.venv/bin/python analysis/analyze_s68.py --limit 32 --output results/analysis/legacy_s68_subset
```

## Định nghĩa và giới hạn

Evaluator định nghĩa confidence tại điểm z là tỷ lệ mẫu GMM có mật độ lớn hơn mật độ tại z. Vùng kappa là các điểm có confidence <= kappa. Script dùng threshold thứ tự của chính các sampled log densities để đếm vùng đó; tương đương phép so sánh confidence, với cùng n và cùng grid.

Để tăng tốc, bỏ qua những điểm chắc chắn ngoài vùng 95% bằng bất đẳng thức mật độ GMM <= mật độ component lớn nhất. Rectangle bao các ellipse superlevel của từng component, kể cả khi rho khác 0. Điểm số được chia cho **toàn bộ** số điểm grid và nhân 1296 m²; không chia cho số điểm trong rectangle. Đây là tối ưu tính toán, không đổi grid hoặc định nghĩa vùng.

Các ngưỡng dùng những lượt Monte Carlo mới, không dùng chính RNG stream của lượt test trước. Vì vậy kết quả tính lại có thể lệch so với metrics chính thức đã lưu. Không thay hoặc ghi đè evaluation.json.

S68 chính thức tổng hợp trung bình của 101 percentile (0,1,...,100) theo từng horizon rồi chuẩn hóa thời gian. **Percentile 100 là maximum**, có trọng số 1/101 tại mỗi horizon; cực trị có thể ảnh hưởng rất mạnh. Cột/biểu đồ phân rã chỉ giải thích cách tổng hợp, không đề xuất thay định nghĩa metric.

Diện tích được giới hạn trong [-18,18]². Với Gaussian cực rộng, vùng thực tế có thể vượt lưới; số đo trên lưới không phải diện tích đầy đủ ngoài miền. Ellipse nét đứt là ellipse 1 std của component, không phải vùng GMM 68%. Đường expectation nối trung bình từng bước, không phải một quỹ đạo mode nhất quán.

Một ground truth đơn lẻ không chứng minh một phân phối quá rộng so với độ khó thực sự. Coverage thực nghiệm, NLL, lỗi vị trí và nhiều trường hợp cần được xem cùng nhau. Không suy ra nguyên nhân kiến trúc từ các hình này.
