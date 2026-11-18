# Kế hoạch triển khai theo cổng nghiệm thu

Mỗi cổng có đầu ra kiểm tra được. Không vượt qua cổng correctness bằng cách gắn nhãn TODO cho phần lõi. Khi có trở ngại môi trường, vẫn bàn giao phần đã làm nhưng ghi BLOCKED ở phần chưa chạy.

## G0 — kiểm tra repository và khả năng build

Đọc hướng dẫn có sẵn, kiểm tra git status và không xóa thay đổi của người dùng. Xác định OS, compiler, CMake, Python và OMPL; dùng dependency đã có nếu phù hợp. Ghim phiên bản/commit thực tế; không ghi “latest” như một phiên bản tái lập. Kiểm tra các header/API BIT*, objective, sampler, benchmark/callback.

Tạo bộ khung build và executable in cấu hình. Tạo `docs/ENVIRONMENT.md`, `docs/OMPL_INTEGRATION.md` và `docs/STATUS.md`. Mục tiêu build phải chạy thực, không chỉ tạo CMakeLists.txt. Không cần ROS/Gazebo/GPU.

**Nghiệm thu:** có phiên bản compiler/OMPL, dòng lệnh build, exit code, log; có một test hoặc executable chạy thành công. Nếu dependency bị chặn, ghi dependency và lỗi cụ thể, không cài vượt quyền.

## G1 — địa hình, tính hợp lệ và chi phí

Hiện thực loader/generator tối thiểu; quy ước row/column/diagonal; chi phí bằng giao biên tam giác; footprint và capsule kiểm toàn cạnh. Viết test T01–T10 và fixture NoData/invalid input. Hiện thực verifier độc lập cho các bài toán nhỏ.

Giữ geometry/cost độc lập với OMPL để kiểm thử dễ. Có thể bắt đầu bằng brute-force rồi tăng tốc sau, nhưng không dùng một phương pháp “nhanh” mà bỏ sót cell/triangle. Thay đổi thuật toán geometry phải giữ fixture regression.

**Nghiệm thu:** test giải tích và đối chiếu độc lập pass trong tolerance; vật cản mỏng bị phát hiện; offset độ cao và split invariants đúng; không NaN/Infinity ngoài contract.

## G2 — tích hợp BIT* A0/A3 và smoke

Dùng `RealVectorStateSpace(2)`, validator riêng, objective c_H, hai policy A0/A3. Kiểm tra source OMPL và ghi hook map. Sampler phải hỗ trợ một/hai ngưỡng; bounded attempts; informed measure được khai báo đúng; k-nearest 2D/JIT off.

Cài trial worker, ghi raw log và snapshots. Thử fixture phẳng/nghiêng có đường thẳng biết trước. Chạy 24 trial smoke theo YAML hiện tại. Có thêm unit/integration test cho goal, degeneracy và timeout.

**Nghiệm thu:** chứng minh BIT* thật được gọi, sampler/heuristic đúng policy; mọi đường báo success vượt verifier; mỗi trial có metadata/counter/status; không yêu cầu 100% success trên mọi bài toán ngẫu nhiên trong ngân sách nhỏ. Không được gắn A3 nếu requested pruning/sampling không hoạt động như đã mô tả.

Sau cổng này dừng theo prompt 01. Không tự chạy pilot/full.

## G3 — baseline và ablation

Thêm RRT*/Informed RRT* cùng c_H/validator. Test rằng tất cả đọc cùng map/model hash và fixed goal. Chọn tham số trên pilot, ghi effective values.

Rà soát A1/A2. Nếu cần, tách queue-policy và state-pruning/sampler-policy bằng patch nhỏ, có source commit, license và regression test. A1/A2 chưa tách đúng thì để UNSUPPORTED. Không thay đổi label để che thiếu tính năng.

**Nghiệm thu:** bốn phương pháp bắt buộc chạy trên cùng bài toán; kết quả có metric terrain thay vì metric XY mặc định; ablation nào công bố thì phải có bằng chứng policy độc lập.

## G4 — pipeline, pilot và phân tích

Hiện thực generator đầy đủ, worker subprocess, run manifest, trial order, config hash, resume, dry-run, watchdog, CSV/JSONL export và bộ phân tích từ log thật. Chạy pilot theo YAML; quy mô hiện tại 400 trial, không phải benchmark hoàn chỉnh để khẳng định thắng/thua cuối cùng.

Báo cáo success theo executed trials và completion coverage theo planned trials, xử lý timeout đúng, chi phí anytime theo event, thống kê theo family và quá trình kiểm chứng độc lập. Phân tích từ log phải có test parser bằng fixture được gắn nhãn và tách khỏi results.

**Nghiệm thu:** report truy được về từng log; có các trường hợp thất bại; không biến dữ liệu thiếu thành 0; nguồn c_ref và loại reference rõ; mọi câu “nhanh hơn” gắn với nhóm thử và số liệu thực tế.

Sau cổng này dừng theo prompt 02. Đề xuất thông số full nhưng không tự chạy.

## G5 — thử nghiệm khóa cấu hình

Chỉ thực hiện theo prompt 03. Kiểm tra lại G0–G4 và config/test split. `--dry-run` phải tạo manifest và số trial trước khi worker chạy. `--allow-full` mới cho phép profile full. Không thuê tài nguyên hoặc đẩy dữ liệu ngoài.

Khóa map seeds và query IDs; lưu cấu hình đã tune, machine/dependency/code versions. Có resume từ tiến độ thật khi bị ngắt. Khi đã chạy full, không đổi cấu hình trên cùng experiment_id; phát sinh version mới.

**Nghiệm thu:** complete/partial được ghi đúng; mọi output đã kiểm chứng; số trial và tỷ lệ failures minh bạch; có lệnh tái lập/resume; kết luận không vượt bằng chứng.

## Giao diện dòng lệnh cần tạo

Các lệnh này là mục tiêu triển khai, chưa tồn tại trong gói handoff:

```bash
# Build và test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure

# Smoke
python tools/generate_scenarios.py --config configs/experiments.yaml --profile smoke
python tools/run_experiments.py --config configs/experiments.yaml --profile smoke --output results/smoke --jobs 1
python tools/validate_results.py --input results/smoke
python tools/analyze_results.py --input results/smoke --output reports/smoke

# Chuẩn bị full, chưa chạy planner
python tools/run_experiments.py --config configs/experiments.yaml --profile full --output results/full --dry-run

# Chỉ chạy khi được giao prompt 03
python tools/run_experiments.py --config configs/experiments.yaml --profile full --output results/full --jobs 1 --allow-full

# Tiếp tục đúng cấu hình/hash, không chạy lại chọn lọc các seed thất bại
python tools/run_experiments.py --config configs/experiments.yaml --profile full --output results/full --jobs 1 --allow-full --resume
```

Nếu repository dùng entry point khác, cập nhật toàn bộ tài liệu và kiểm tra các lệnh mới nhất quán. Không giữ lệnh mẫu sai để trông như đã đủ tính năng.

## Mẫu STATUS.md cần tạo khi bắt đầu triển khai

```text
# Trạng thái thực thi
Cập nhật UTC:
Code commit/dirty hash:
Environment summary:

G0: NOT_RUN | bằng chứng:
G1: NOT_RUN | bằng chứng:
G2: NOT_RUN | bằng chứng:
G3: NOT_RUN | bằng chứng:
G4: NOT_RUN | bằng chứng:
G5: NOT_RUN | bằng chứng:

Lệnh đã chạy + exit code + log:
Test pass/fail/block:
Trial dự kiến / đã chạy / success / timeout / invalid / crash / chưa chạy:
Các metric thật sự đã đo:
Ablation effective / unsupported:
Hạn chế còn lại:
Lệnh tiếp tục đã kiểm tra:
```
