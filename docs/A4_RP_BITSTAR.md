# A4 / RP-BIT* (Risk-Potential BIT*) — `a4_rpbitstar_v1`

Thuật toán dựa trên BIT* cho bài toán tìm đường 2.5D có xét rủi ro địa hình, được thiết kế để so với
SEM-RRT* trên các metric của paper (PSR, APL, PRI, STI, Time). Tài liệu này mô tả thuật toán, lập luận
đúng đắn, cách chạy và kết quả. Kế hoạch được duyệt ngày 16/11/2026; mã nguồn cũ (A3, A3-T và các
biến thể) không bị sửa.

## 1. Mô hình và những gì giữ nguyên

- Trạng thái q = (x, y), độ cao z = H(x, y) lấy từ `Terrain`; không thêm chiều trạng thái.
- Planner là `ompl::geometric::BITstar` của OMPL 1.7.0, **không patch, không thay thế**: k-nearest,
  JIT tắt, pruning bật, lân cận gần nhất theo khoảng cách Euclid XY, batch 100 mẫu rồi 200 mẫu sau
  nghiệm đầu tiên, rewire 1,1.
- Miền an toàn là `SafetyPolicy` đã có: mọi ô SEM mà đường tâm chạm phải có support và risk ≤ 40,
  kiểm tra va chạm toàn bộ cạnh với footprint, độ dốc quét ≤ 35°. Đích là đúng điểm goal.
- Trường rủi ro là `SAFE_SEM_FIELD_V1` dựng từ point cloud, cùng định nghĩa với bộ chấm điểm độc lập.
  Kết quả của SEM-RRT* không được dùng trong planner (`sem_reference_used = false`).

## 2. Thuật toán

### 2.1 Hàm mục tiêu

BIT* tối thiểu trực tiếp

    J_λ(đường) = L_H + λ · E / 40,     E = ∫ F_R ds_H

trong đó L_H là chiều dài trên mặt địa hình, F_R là risk của ô SEM (hằng trên từng ô 0,1 m), ds_H là
độ dài trên mặt. Đây là objective `surface_length_sem_exposure` đã có của dự án. Trọng số của
SEM-RRT* (0,48·chiều dài + 0,36·risk) tương đương λ = 30, nên **J₃₀ chính là hàm mục tiêu của
SEM-RRT*** và được dùng làm thước đo chung, tính độc lập cho mọi planner.

### 2.2 Tích phân cạnh nhanh (`RpEdgeIntegrator`)

Bản tích phân bằng số hữu tỉ `semExposureCanonical` tốn khoảng 0,6 ms mỗi cạnh và chiếm 70–96% thời
gian solve của biến thể SEM-exposure cũ. `RpEdgeIntegrator` tính cùng đại lượng trong một lượt bằng số
thực double: trộn ba dãy điểm cắt đã đơn điệu (tam giác địa hình, đường lưới SEM dọc, đường lưới SEM
ngang). Ô được xác định bằng so sánh chính xác với đường lưới rồi bước chỉ số, nên tách một cạnh
thành hai vẫn cho cùng tổng; trên đoạn nằm đúng đường lưới lấy max của các ô kề như bản canonical.
Đo trên test: **3,0 µs/cạnh so với 629 µs/cạnh (nhanh hơn ≈ 209 lần), sai lệch tương đối lớn nhất
2·10⁻¹⁵**. Khi ghi kết quả, worker tính lại mọi đường công bố bằng bản canonical và báo lỗi nếu lệch
quá 10⁻⁸; mọi số trong báo cáo đến từ bản canonical và từ oracle Python độc lập.

### 2.3 Risk potential — cận dưới "biết rủi ro" cho J (`RiskPotential`)

Heuristic cũ d_Φ chỉ biết chiều dài, nên rất lỏng khi λ lớn. Risk potential cho cận dưới của J giữa
một nguồn cố định (start hoặc goal) và mọi điểm:

1. **Lưới góc ô.** Nút là các góc ô SEM, 8-liên thông. Ô *mở* là ô có support và risk ≤ 40, với
   trọng số w_c = 1 + (λ/40)·F_c. Cạnh bên có chi phí res·min(w của các ô mở kề cạnh đó); hai đường
   chéo của một ô mở có chi phí √2·res·w_c; ô chặn không có đường chéo.
2. **Dijkstra đa nguồn** từ bốn góc của (các) ô chứa nguồn, cho U tại mọi góc ô.
3. **Co theo κ = cos(π/8)** (làm tròn xuống: 0,92387953251). Đường trên lưới 8-liên thông dài hơn
   đường Euclid nhiều nhất 1/cos(π/8) lần, nên κ·U là dữ liệu biên hợp lệ.
4. **Mở rộng vào trong ô từ bốn cạnh.** Trong mỗi ô mở dùng hàm w_c-Lipschitz **lớn nhất** khớp với
   dữ liệu trên bốn cạnh: P(x) = min_b [κU(b) + w_c·|x − b|] (dạng đóng cho từng cạnh). Riêng (các)
   ô chứa nguồn, nơi dữ liệu biên bằng 0, dùng hàm nhỏ nhất P(x) = −w_c·dist(x, biên ô).

**Vì sao hợp lệ.** (a) Trên biên của mỗi ô mở, dữ liệu κU tuyến tính từng cạnh là w_c-Lipschitz theo
khoảng cách Euclid: các ràng buộc cạnh bên (≤ w_c·res) và đường chéo (≤ √2·w_c·res) làm U là
w_c-Lipschitz theo chuẩn octile, và chuẩn octile ≤ Euclid / cos(π/8). (b) Vì vậy hai phép mở rộng ở
bước 4 đều trùng dữ liệu trên biên, nên P **liên tục qua các ô** và w_c-Lipschitz trong từng ô.
(c) Với một hàm như vậy, dọc theo mọi đường hợp lệ γ từ x tới y:
|P(x) − P(y)| ≤ ∫_γ w ds_xy ≤ J(γ), vì ds_H ≥ ds_xy và mật độ risk dọc γ không nhỏ hơn F_c của ô
đang đi qua (trên đường lưới dùng max các ô kề). Một bản chỉ nội suy bốn góc không liên tục qua biên
ô nên **không** dùng được cho cận cạnh; test có kiểm tra tính liên tục này.

Từ đó, với P_s (nguồn = start) và P_g (nguồn = goal):

- ĝ(x) = max(d_Φ(s, x), P_s(x) − P_s(s)), ĥ(x) = max(d_Φ(x, g), P_g(x) − P_g(g));
- ĉ(x, y) = max(d_Φ(x, y)·(1 + (λ/40)·F_min), |P_g(x) − P_g(y)|, |P_s(x) − P_s(y)|), trong đó F_min là
  risk nhỏ nhất của ô mở trong một hình vuông bao hộp giới hạn của cạnh (tra O(1) bằng `RiskBoxMin`).

**An toàn số học** (BIT* tự cắt chuỗi tổ tiên của nghiệm đang giữ nếu heuristic ước lượng quá):
ĥ(goal) = ĝ(start) = 0 chính xác; mọi giá trị potential bị trừ 10⁻¹¹ tương đối và 10⁻⁹ tuyệt đối rồi
làm tròn xuống một ULP; không bao giờ trả NaN (ô chặn hoặc không tới được trên lưới nới lỏng cho +∞).
Lúc chạy, tại **mỗi** incumbent thật, worker kiểm lại ĝ ≤ chi phí tiền tố, ĥ ≤ chi phí hậu tố và
ĉ ≤ chi phí cạnh; số vi phạm được ghi (`rp_counters.heuristic_violations`) và validator coi khác 0 là
lỗi correctness.

Vật cản hình học của địa hình chưa được đưa vào lưới potential (cận dưới vẫn hợp lệ, chỉ lỏng hơn
quanh vật cản).

### 2.4 Lấy mẫu informed theo rủi ro (`RpInformedSampler`)

Mỗi ô có cận dưới của ĝ + ĥ (góc nhỏ nhất của hai potential). Khi BIT* yêu cầu mẫu với J < J_best,
sampler giữ tiền tố các ô có cận dưới < J_best (tập bao của tập informed), chọn đều một ô, một điểm
đều trong ô, rồi **loại bỏ chính xác** theo ĝ + ĥ ∈ [min, max). Kết quả là mẫu đều trên
{ĝ + ĥ < J_best}. Khi tiền tố rỗng sampler trả `false` ngay. Trước nghiệm đầu, mẫu đều trên các ô mở
tới được.

### 2.5 Dừng khi hội tụ

Điều kiện dừng được BIT* hỏi mỗi vòng lặp. Tại mỗi ranh giới batch: nếu J đã công bố không giảm ít
nhất ε_rel = 10⁻³ (tương đối) qua M = 5 batch liên tiếp thì dừng (`CONVERGED_NO_IMPROVEMENT`); luôn
có trần thời gian (`TIME_BUDGET`). Kích hoạt theo batch chứ không theo đồng hồ nên lượt chạy
fixed-work tái lập được. Đồng hồ solve bắt đầu **trước** khi dựng safety policy, potential và planner.

### 2.6 Tinh chỉnh đường theo J (khai báo công khai, có ablation)

Tại ranh giới batch, incumbent mới của BIT* được tinh chỉnh bằng `rpRefine` (tất định, không dùng số
ngẫu nhiên, không tác động ngược vào BIT*):

1. **Cắt ngắn**: thay một chuỗi đỉnh bằng một cạnh thẳng khi J không tăng và cạnh mới an toàn.
2. **Dịch đỉnh**: chia cạnh dài hơn 1,5 m, rồi pattern search 8 hướng (bước 0,4 → 0,05 m) trên từng
   đỉnh trong; chỉ nhận khi J giảm và hai cạnh mới an toàn.
3. **Bo góc**: góc đổi hướng > 30° được thay bằng một cung gồm các dây bằng nhau (hoặc bỏ đỉnh). Cung
   có thể dịch ra phía ngoài để vòng qua thứ mà góc đang ôm. Mọi đoạn mới dài ít nhất
   `min_chord_m` = 0,3 m nên **không thể giấu góc gấp sau các đoạn siêu nhỏ**. Một phép thay chỉ được
   nhận khi số góc gấp cục bộ giảm và mọi cạnh mới an toàn; cả bước bo góc chỉ được làm J tăng tối
   đa `round_cost_ratio` (mặc định 0,5%).

Đường tinh chỉnh chỉ được công bố nếu mọi cạnh qua `SafetyPolicy` và J thấp hơn đường đã công bố
trước đó (J công bố không bao giờ tăng). Không có bước nào kéo dài đường để hạ PRI. Ablation:
`refine: false` (chỉ công bố incumbent thô của BIT*), `round_turns: false`, `potential: false`
(cận d_Φ và sampler atlas cũ, vẫn dùng tích phân nhanh).

## 3. Tham số

| Tham số | Mặc định trong mã | Cấu hình chính đã khóa (`a4_rpbit_l33_m20`) | Ghi chú |
|---|---|---|---|
| λ | 30 | **33** | 30 = trọng số của SEM-RRT*; chọn trên dev (mục 6.3) |
| batch | 100 → 200 sau nghiệm đầu | như mặc định | như A3-T |
| ε_rel, M | 10⁻³, 5 batch | 10⁻³, **20 batch** | luật dừng hội tụ; chọn trên dev |
| cạnh tối đa khi chia / bước dịch đỉnh | 1,5 m / 0,4 → 0,05 m | như mặc định | tinh chỉnh |
| ngưỡng góc gấp, dây tối thiểu | 30°, 0,3 m | như mặc định | bo góc |
| trần tăng J khi bo góc | 0,5% | như mặc định | 0,2% và 1% cho kết quả giống hệt trên dev |
| trần đánh giá cạnh mỗi lần tinh chỉnh | 20 000 | như mặc định | |

## 4. Mã nguồn và kiểm thử

- Mới: `include/terrain25d/RpBit.hpp`, `src/RpBit.cpp`, `apps/RpBitWorker.inc`, `cmake/RpBitTargets.cmake`,
  `tests/integration/rp_bit_tests.cpp`, `tests/test_rp_bit_worker.py`, `tests/test_rp_bit_tooling.py`,
  `tools/rp_bit_protocol.py`, `tools/run_rp_bit.py`, `tools/rp_bit_validation.py`, `tools/analyze_rp_bit.py`,
  `configs/a4_rpbitstar_v1.json`.
- Sửa tối thiểu: `apps/terrain_benchmark.cpp` (2 include + 1 dòng dispatch theo khóa `rp_bit`),
  `CMakeLists.txt` (1 dòng include). Worker phát đúng schema SEM-exposure nên validator độc lập
  `sem_exposure_validation.validate_trial` áp dụng nguyên trạng.
- `rp_bit_core` (C++): tích phân nhanh so với canonical trên 4 000 cạnh kể cả các ca biên trên đường
  lưới/góc ô, tính cộng và đảo chiều; potential bằng 0 tại nguồn, Lipschitz trong ô, liên tục qua
  cạnh ô, +∞ sau tường chặn, từ chối raster bị cắt; ĉ ≤ chi phí thật trên 60 000 cạnh ngẫu nhiên hợp lệ;
  sampler đúng khoảng và phát hiện tập rỗng; BIT* thật với kiểm tra cận tại mọi callback, có và không
  có potential; bo góc sát vật cản và trong vùng trống; chạy fixed-work tái lập, nhãn dừng, truy vấn
  không hợp lệ và truy vấn trùng điểm.
- `rp_bit_worker` (Python + worker thật): 5 lượt thật qua validator nghiêm ngặt trên mọi snapshot,
  12 request sai phải bị từ chối với mã 2. `rp_bit_tooling`: giao thức, chốt held-out, phân tích.

## 5. Cách chạy

```powershell
cmake --build build-a4-rpbit --parallel 3
ctest --test-dir build-a4-rpbit -L rp_bit --output-on-failure -j 1
ctest --test-dir build-a4-rpbit -L "sem_exposure|pri_segment|pri_pathsearch|safe_length" --output-on-failure -j 1
python -B tools/run_rp_bit.py run --phase smoke --output results/<thư mục> --worker build-a4-rpbit/terrain_benchmark.exe --bridge build-a4-rpbit/libterrain_compare_c.dll --full-validation
python -B tools/run_rp_bit.py run --phase dev_a4_core --output results/<thư mục> --worker ... --bridge ...
python -B tools/analyze_rp_bit.py --input results/<thư mục>/<pha> [...] --output reports/<thư mục>
```

Runner chạy mỗi trial trong một tiến trình mới, ghim một lõi, không ghi đè và không chạy lại trial đã
bắt đầu; pha có tên bắt đầu bằng `heldout` chỉ chạy với `--allow-heldout`.

## 6. Kết quả đến 17/11/2026 (smoke, dev và held-out trên Kaggle ở mục 6.6)

Mọi số dưới đây do oracle Python độc lập tính lại từ bản đồ và point cloud gốc. SEM-RRT* là bản tái
dựng Python đã khóa của dự án (không phải mã tác giả), ngân sách 20 s, trung bình 7 425 vòng lặp mỗi
lượt (paper dùng khoảng 5 000). A4 là C++; so sánh Time vì vậy là khác ngôn ngữ và phải đọc kèm ghi
chú này. Dev dùng 15 bản đồ **đã xem** (5 họ × seed 41001–41003) × 5 planner seed = 75 lượt cho mỗi
phương pháp; đây là dữ liệu để chỉnh tham số, **không phải bằng chứng tổng quát hóa**.

### 6.1 Smoke (7 fixture, 2 s, validator nghiêm ngặt trên mọi snapshot)

Ba lượt smoke (`results/a4_rpbit_smoke_20261116`, `..._smoke_r2_...` trên binary cuối,
`..._smoke_final_...` với đúng bộ phương pháp của held-out): 0 lỗi correctness. A4 thành công 7/7
fixture ở mọi lượt; lượt không thành công duy nhất là SEM-RRT* trên `nodata_thin_obstacle` (đường cuối
chạm ô SEM không có support), giống baseline trước khi sửa mã.

### 6.2 Dev: A4 so với SEM-RRT* 20 s

| Phương pháp | PSR | APL (m) | PRI | STI (1/m) | Góc gấp/đường | Time (s) | J₃₀ | ΔAPL gộp [cận trên 95%] | ΔPRI gộp [cận trên 95%] | ΔJ₃₀ [cận trên 95%] | Cặp có J₃₀ thấp hơn SEM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEM-RRT* (`sem_native`) | 75/75 | 36,445 | 3,9608 | 0,03146 | 1,147 | 20,00 | 152,18 | — | — | — | — |
| **A4 chính: λ=33, M=20** | 75/75 | 36,315 | 3,9426 | 0,00037 | 0,013 | 1,47 | 151,36 | −0,36% [+0,01%] | −0,46% [+0,14%] | −0,818 [−0,575] | 75/75 |
| A4 λ=30, M=20 | 75/75 | 36,244 | 3,9575 | 0,00110 | 0,040 | 1,51 | 151,36 | −0,55% [−0,23%] | −0,09% [+0,53%] | −0,816 [−0,573] | 75/75 |
| A4 λ=36, M=20 | 75/75 | 36,317 | 3,9430 | 0,00184 | 0,067 | 1,55 | 151,36 | −0,35% [+0,01%] | −0,45% [+0,19%] | −0,816 [−0,574] | 75/75 |
| A4 λ=30, M=5 (nhanh) | 75/75 | 36,274 | 3,9532 | 0,00147 | 0,053 | 0,50 | 151,46 | −0,47% [−0,01%] | −0,19% [+0,43%] | −0,713 [−0,458] | 74/75 |
| A4 λ=30, không dừng sớm | 75/75 | 36,225 | 3,9613 | 0,00074 | 0,027 | 20,00 | 151,34 | −0,60% [−0,28%] | +0,01% [+0,57%] | −0,831 [−0,586] | 75/75 |
| A4 λ=24 / 27 / 36 / 45 (M=5) | 75/75 | 36,136 / 36,172 / 36,434 / 36,660 | 3,9852 / 3,9685 / 3,9283 / 3,8944 | ≤ 0,0036 | ≤ 0,133 | ≈ 0,5 | 151,64 / 151,46 / 151,62 / 151,77 | −0,85 / −0,75 / −0,03 / +0,59% | +0,62 / +0,19 / −0,82 / −1,68% | −0,54 / −0,72 / −0,56 / −0,41 | 72 / 73 / 71 / 69 |
| Ablation: tắt tinh chỉnh (λ=30, M=5) | 75/75 | 36,507 | 3,9440 | 0,01388 | 0,507 | 0,58 | 152,28 | +0,17% [+0,74%] | −0,42% [+0,44%] | +0,101 [+0,550] | 41/75 |
| Ablation: tắt potential (λ=30, M=5) | 75/75 | 36,388 | 3,9380 | 0,00147 | 0,053 | 1,20 | 151,67 | −0,16% [+0,59%] | −0,58% [+0,32%] | −0,504 [−0,114] | 72/75 |
| Ablation: tắt bo góc (λ=30, M=5) | 75/75 | 36,274 | 3,9535 | 0,00662 | 0,240 | 0,51 | 151,46 | −0,47% [−0,01%] | −0,19% [+0,44%] | −0,711 [−0,456] | 74/75 |
| A3-T SEM-exposure cũ, λ=30 (15 lượt) | 15/15 | 37,185 | 3,8948 | 0,02331 | 0,867 | 20,00 | 154,19 | +2,05% [+3,56%] | −1,80% [+0,25%] | +2,099 [+3,633] | 3/15 |

Δ gộp = Σ(A4 − SEM)/Σ SEM trên các cặp cùng map/seed; cận trên là bootstrap phân cấp map → seed
(20 000 lần, một phía 95%). Tiền xử lý trước đồng hồ solve: A4 ≈ 0,53 s, SEM ≈ 0,38 s.

Đọc bảng:

- **Cấu hình chính đạt mọi tiêu chí đã đăng ký trên dev**: PSR bằng nhau (100%), STI thấp hơn khoảng
  86 lần, Time 1,47 s so với 20 s, J₃₀ thấp hơn ở 75/75 cặp, APL và PRI gộp đều thấp hơn. Riêng APL và
  PRI là **khác biệt nhỏ**: cận trên 95% của hai hiệu vẫn chạm hoặc vượt 0 một chút (+0,01% và +0,14%),
  nên trên dev chúng là "không kém, thiên về tốt hơn", chưa phải thắng có ý nghĩa thống kê.
- Theo từng họ (cấu hình chính): J₃₀ và STI thấp hơn ở cả 5 họ; APL thấp hơn ở 4/5 họ
  (`mixed_obstacles` +0,51%); PRI thấp hơn ở 4/5 họ (`narrow_passage` +0,83%).
- **Anytime** (cấu hình chính so với SEM tại cùng thời điểm, J₃₀): 0,5 s: 151,46 so với 182,84;
  1 s: 151,36 so với 156,68; 2 s: 151,36 so với 154,05; 5 s: 152,73; 10 s: 152,37; 20 s: 152,18.
  Cấu hình chính công bố nghiệm đầu trong 0,15 s ở mọi lượt dev (trung bình 0,11 s) và tự dừng sau
  1,47 s trung bình (lớn nhất 2,61 s); 75/75 lượt dừng do hội tụ. Trên vết J₃₀ do oracle tính lại
  cho mọi snapshot (validate nghiêm ngặt 75/75 lượt, 0 lỗi), nó đạt J₃₀ cuối 20 s của SEM cùng
  map/seed ở 75/75 cặp: trung vị 0,13 s, trung bình 0,16 s, lớn nhất 0,57 s.
- **Đóng góp từng thành phần**: tắt tinh chỉnh thì BIT* thô trong 0,58 s ngang SEM 20 s về J₃₀
  (+0,07%) và còn 0,51 góc gấp/đường — phần vượt SEM về J₃₀ và STI đến từ bước tinh chỉnh; tắt
  potential thì chậm hơn 2,4 lần và J₃₀ kém hơn một chút; biến thể A3-T cũ (tích phân hữu tỉ chậm,
  heuristic d_Φ, không tinh chỉnh) sau 20 s vẫn kém SEM. Chạy đủ 20 s chỉ thêm −0,02 J so với M=20:
  A4 đã hội tụ ở khoảng 1,5 s.
- 0 vi phạm cận heuristic trên mọi incumbent của 1 050 lượt A4 dev; 0 lỗi correctness.

### 6.3 Cách chọn cấu hình chính

Chỉ dùng dev. (1) λ: quét 24/27/30/36/45 ở M=5 cho thấy λ dịch chuyển đánh đổi APL ↔ PRI, còn J₃₀
gần như phẳng quanh 30. (2) M: 20 batch lấy gần hết phần chất lượng còn lại với Time 1,5 s. (3) Quy tắc
đăng ký trước khi chạy pha `dev_a4_balance`: trong λ ∈ {30, 33, 36} ở M=20, lấy cấu hình đạt mọi tiêu
chí và có max(cận trên ΔAPL%, cận trên ΔPRI%) nhỏ nhất → λ=33 (0,14% so với 0,53% và 0,19%).

### 6.4 Held-out đã đăng ký (chạy trên Kaggle với các thay đổi ở mục 6.5; kết quả ở mục 6.6)

Pha `heldout` trong `configs/a4_rpbitstar_v1.json`: 5 họ × seed bản đồ 61001–61010 (chưa từng xuất
hiện trong `configs/` hay `tools/`) × planner seed 616001–616005 = 250 slot; phương pháp:
`sem_native` (20 s), `a4_rpbit_l33_m20` (**chính**), ablation `a4_rpbit_l33_m20_norefine`,
`a4_rpbit_l33_m20_nopotential`, preset nhanh `a4_rpbit_l33` (M=5) và `a3t_sem_exposure_l33` (A3-T cũ
cùng λ, 20 s); tổng 1 500 lượt. Tiêu chí như mục 5.3 của kế hoạch, không đổi sau khi xem kết quả.
Đường sinh bản đồ đã được diễn tập với seed 51001 (không thuộc held-out): 10/10 lượt đạt.

### 6.5 Held-out trên Kaggle: các thay đổi đã ghi trước khi chạy

Ngày 16/11/2026, người dùng chọn chạy held-out trên Kaggle 4 vCPU. Biên bản thay đổi nằm ở
`logs/a4_rpbit_kaggle_20261116/deviation_plan.json`. Controller ghi thêm `heldout_deviation_record.json`,
kèm hash của binary Linux, trước lượt held-out đầu tiên.

- Cấu hình `configs/a4_rpbitstar_v1_kaggle.json`, pha `heldout_kaggle`: cùng 1 500 mã trial, chạy 4 làn × 1 thread.
- **SEM-RRT* chạy cố định 7 425 vòng lặp** (trung bình 75 lượt dev, trần 180 s) thay vì 20 s.
  - Lý do: vCPU của Kaggle chậm hơn và có thể dùng chung lõi vật lý. Với 20 s đồng hồ, SEM sẽ được ít vòng lặp
    hơn trên dev. A4 dừng theo số batch nên gần như không bị ảnh hưởng, vì vậy so sánh APL/PRI sẽ lệch về phía A4.
  - Time của SEM là thời gian thật để chạy đủ số vòng lặp đó. A3-T và các biến thể A4 giữ ngân sách 20 s.
- Trước held-out phải đạt các cổng: 30 nhóm CTest trên Linux, `smoke_final`, `rehearsal` và `kaggle_gate`
  (seed 51001; mọi lượt SEM phải chạy đủ 7 425 vòng lặp).
- Kết quả held-out là số đo trên Linux/Kaggle. Không so trực tiếp số Time với dev trên Windows.

### 6.6 Kết quả held-out (Kaggle, 16/11/2026)

**Lượt chạy và kiểm chứng.**
- Kernel `notebook66864104b1`, Xeon 2,2 GHz, 4 vCPU = 2 lõi vật lý; 1 500/1 500 lượt, 0 lỗi correctness.
- Validate nghiêm ngặt mọi snapshot của cấu hình chính: 250/250 đạt.
- Biên bản thay đổi được ghi trước lượt held-out đầu tiên. Báo cáo dựng lại trên máy cục bộ trùng từng khóa với
  báo cáo của Kaggle (`logs/a4_rpbit_kaggle_20261116/heldout_receipt.json`).
- Báo cáo đầy đủ ở `reports/a4_rpbit_heldout_kaggle_20261116/heldout_kaggle/REPORT.md`; verdict ở
  `results/a4_rpbit_heldout_kaggle_20261116/`.

| Phương pháp | PSR | APL (m) | PRI | STI (1/m) | Góc gấp/đường | Time (s) | J₃₀ | ΔAPL gộp [cận trên 95%] | ΔPRI gộp [cận trên 95%] | ΔJ₃₀ [cận trên 95%] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEM-RRT* (7 425 vòng lặp) | 249/250 | 39,694 | 3,750 | 0,0245 | 0,97 | 90,2 | 154,63 | — | — | — |
| **A4 chính: λ=33, M=20** | 250/250 | 39,515 | 3,745 | 0,0015 | 0,06 | 5,37 | 153,61 | −0,41% [−0,24%] | −0,13% [+0,17%] | −0,954 [−0,707] |
| A4 tắt tinh chỉnh | 250/250 | 39,633 | 3,746 | 0,0081 | 0,32 | 10,3 | 154,04 | −0,11% [+0,05%] | −0,10% [+0,15%] | −0,520 [−0,326] |
| A4 tắt potential | 250/250 | 39,518 | 3,744 | 0,0014 | 0,056 | 15,6 | 153,62 | −0,40% [−0,23%] | −0,18% [+0,13%] | −0,943 [−0,697] |
| A4 λ=33, M=5 | 250/250 | 39,511 | 3,747 | 0,0007 | 0,028 | 1,77 | 153,64 | −0,42% [−0,24%] | −0,09% [+0,21%] | −0,921 [−0,678] |
| A3-T SEM-exposure cũ, λ=33 (20 s) | 250/250 | 40,394 | 3,786 | 0,0351 | 1,42 | 20,0 | 159,00 | +1,81% [+2,87%] | +0,95% [+2,24%] | +4,45 [+5,80] |

**Đọc bảng.**
- Cấu hình chính **đạt mọi tiêu chí đã đăng ký** (mục 5.3):
  - PSR không thấp hơn SEM, cả gộp lẫn từng họ;
  - STI và Time thấp hơn;
  - J₃₀ thấp hơn ở 249/249 cặp, cận trên âm;
  - APL và PRI không kém quá 1%, trung bình gộp đều thấp hơn.
- So với dev:
  - APL nay thấp hơn có ý nghĩa thống kê: thấp hơn ở 192/249 cặp, cận trên −0,24%.
  - PRI vẫn chỉ "không kém, thiên về tốt hơn": thấp hơn ở 139/249 cặp, khoảng tin cậy 95% của hiệu là [−0,48%; +0,22%].
- Theo họ bản đồ, J₃₀ và STI thấp hơn ở 5/5 họ, APL và PRI thấp hơn ở 4/5 họ. Hai ngoại lệ giống hệt dev:
  APL của `mixed_obstacles` cao hơn 0,09%, PRI của `narrow_passage` cao hơn 0,10%.
- Cấu hình chính tự dừng do hội tụ ở cả 250/250 lượt (lâu nhất 17,5 s). Nó đạt J₃₀ cuối của SEM ở
  249/249 cặp, trung vị 0,44 s, lâu nhất 2,32 s.
- Về anytime (J₃₀), A4 so với SEM tại cùng thời điểm: 1 s: 153,89 so với 199,27; 5 s: 153,61 so với 158,81;
  20 s: 153,61 so với 155,29. Tại 0,5 s mới có 236/250 lượt A4 có nghiệm, do CPU Kaggle chậm hơn.
- Ablation:
  - Tắt tinh chỉnh vẫn đạt mọi tiêu chí, nhưng lợi thế J₃₀ giảm còn khoảng một nửa.
  - Tắt potential cho chất lượng ngang cấu hình chính nhưng chậm gấp 2,9 lần. Potential chủ yếu đem lại tốc độ.
  - M = 5 cho chất lượng ngang M = 20 với 1,77 s.
  - Hai ablation tắt tinh chỉnh và tắt potential bị cắt ở trần 20 s ở 24 và 66 lượt.

### 6.7 Thí nghiệm cùng thời gian với SEM-RRT* bản C++ (đã đăng ký 17/11/2026; đã chạy trên Kaggle 17/11/2026, kết quả chưa đưa vào repo)

**Mục đích.** Trả lời câu hỏi: khi cả hai planner đều viết bằng C++, chạy trên cùng phần cứng và cùng ngân sách
20 s đồng hồ thật, cấu hình chính đã khóa của A4 có còn đạt mọi tiêu chí của mục 5.3 khi so với SEM-RRT* không?
Held-out ở mục 6.6 đã dùng xong và không được chạy lại hay gộp số liệu với thí nghiệm này.

**Bản C++ của SEM-RRT*.**
- Mã: `include/terrain25d/SemRrtStar.hpp`, `src/SemRrtStar.cpp`, thư viện `sem_rrtstar_c`.
- Bản này trùng từng bit với bản tái dựng Python: 30/30 lượt dev ở 7 425 vòng lặp trên Windows, nhanh hơn
  khoảng 48 lần (xem `docs/STATUS.md`).
- Raster native vẫn dựng bằng Python và tính vào tiền xử lý; chỉ phần tìm kiếm chạy bằng C++.

**Đăng ký.**
- Biên bản: `logs/a4_rpbit_fairtime_20261117/registration.json`.
- Cấu hình: `configs/a4_rpbitstar_v1_fairtime_kaggle.json`. Controller trên Kaggle từ chối chạy nếu hash
  của file cấu hình hoặc của bất kỳ pha nào khác với biên bản.

| Mục | Giá trị |
|---|---|
| Phương pháp | `sem_native` với `"engine": "cpp"`, 20 s, không giới hạn số vòng lặp; `a4_rpbit_l33_m20` y như bản khóa |
| Held-out mới `heldout_fairtime` | 5 họ × seed bản đồ 71001–71010 × planner seed 716001–716005 = 250 cặp, 500 lượt |
| Lõi | `cores [0]`: 1 làn, mỗi lúc một lượt, hai phương pháp không bao giờ chạy cùng lúc |
| Đồng hồ | như mục 5.3: A4 tính từ trước khi dựng potential; SEM tính từ lúc dựng planner |
| Phân tích | giữ nguyên khối `analysis` (bootstrap 20 000 lần, seed 61016, biên không kém 1%) |

- Seed 71001–71010 và 716001–716005 chưa từng xuất hiện trong cấu hình, công cụ, test hay tài liệu nào.
- Hai phương pháp của cùng một cặp (map, seed) chạy liền nhau, thứ tự đảo giữa các block.
- Bản đăng ký r1 ghi `cores [0, 1]`. Bộ chạy dùng chung chỉ nhận 1 hoặc 4 làn, nên r1 không chạy được.
  Lỗi này lộ ra khi diễn tập các cổng trên Windows, trước khi bất kỳ lượt nào của thí nghiệm bắt đầu. Bản r2
  thay r1 và chỉ đổi số lõi; r1 được giữ ở `registration_r1_superseded.json`.

**Cổng phải đạt trước held-out.**
1. 31 nhóm CTest trên Linux: 30 nhóm cũ cộng `sem_native_cpp`.
2. `smoke_fairtime`: 7 fixture, 2 s.
3. `gate_fairtime`: seed bản đồ 51001, planner seed 516001, 20 s. Ở cả hai cổng phải có 0 lỗi correctness và A4
   thành công mọi lượt. Validator ghi lỗi `sem_engine_mismatch` nếu lượt SEM không chạy đúng engine C++ và đúng
   thư viện đã ghi hash.
4. `sem_cpp_bit_identity`: trên 5 map của `gate_fairtime`, planner seed 516001 và 516002, SEM Python và SEM C++
   chạy 7 425 vòng lặp. Cả 10/10 lượt phải trùng hoàn toàn về trạng thái, bộ đếm và mọi incumbent, ngay trên
   runtime Linux của campaign (`tools/sem_cpp_equivalence.py`).

**Diễn tập trên Windows (chỉ dùng đầu vào của cổng, không phải held-out).** Có ba lượt diễn tập:
- **r1** để lộ giới hạn 1 hoặc 4 làn của bộ chạy dùng chung. Đây là lý do bản đăng ký r2 chuyển sang 1 làn.
- **r2** để lộ một lỗi ở wrapper C++. Wrapper báo thời gian incumbent theo đồng hồ C++, mà đồng hồ này chậm hơn
  đồng hồ của worker một khoảng bằng chi phí gọi ctypes. Khi hai incumbent đến cách nhau dưới khoảng đó, worker
  coi lượt SEM là lỗi thứ tự thời gian và tính là planner thất bại. Lỗi như vậy sẽ làm SEM thua oan về PSR.
  Bản sửa đọc thời gian từ `perf_counter` của bên gọi, giống bản Python. Test
  `test_incumbent_times_are_on_the_caller_clock` dùng đồng hồ giả nhảy 1 ms mỗi lần đọc, nên tái hiện lỗi một
  cách tất định.
- **r3** (`results/a4_rpbit_fairtime_rehearsal_r3_20261117`):
  - cả hai cổng đạt theo đúng luật chấm của controller;
  - `smoke_fairtime` 13/14, lượt hỏng duy nhất là lỗi SEM đã biết trên `nodata_thin_obstacle`;
  - `gate_fairtime` 10/10, SEM C++ chạy 45 277–62 062 vòng lặp trong 20 s;
  - trùng từng bit 10/10.

  Năm cặp của cổng chỉ dùng để kiểm tra pipeline, không phải bằng chứng.

**Tiêu chí chính:** mục 5.3, không đổi.

**Phân tích phụ, khai báo trước:**
- bảng anytime tại T ∈ {0,5; 1; 2; 5; 10; 20} s, kèm cận trên bootstrap của hiệu ghép cặp tại cùng T;
- thời gian A4 đạt J₃₀ cuối của SEM;
- số vòng lặp SEM mỗi lượt.

**Kỳ vọng ghi trước khi chạy.**
- Với 20 s, SEM C++ chạy nhiều vòng lặp hơn 7 425 vài lần. Trên dev (Windows), mức này cho chi phí native thấp
  hơn 0,42–3,0%.
- Lợi thế J₃₀ của A4 ở mục 6.6 (khoảng 0,6%) vì vậy có thể thu hẹp, biến mất hoặc đảo chiều.
- STI và Time dự kiến vẫn nghiêng về A4.
- Kết quả nào cũng được báo cáo.

### 6.8 Thí nghiệm cả hai cùng tự dừng khi hội tụ (đã đăng ký 17/11/2026, chưa chạy)

**Mục đích.**
- Ở mục 6.7, SEM-RRT* không có luật dừng nên luôn dùng hết 20 s. Tiêu chí Time vì vậy đem thời gian tự dừng của A4
  so với ngân sách của SEM.
- Ở đây SEM-RRT* cũng tự dừng, theo đúng luật của A4.
- Người dùng chọn cách làm này ngày 17/11/2026:
  - luật dừng giống hệt A4, không chỉnh tham số nào theo dữ liệu;
  - xem trước trên dev, rồi mới chạy Kaggle với seed mới.

**Luật dừng của SEM-RRT\*.**
- Tại mỗi ranh giới vòng lặp sau khi có nghiệm đầu:
  - nếu chi phí native đã công bố thấp hơn mốc × (1 − 10⁻³), chi phí đó trở thành mốc mới;
  - nếu không, và đã qua 4 000 vòng lặp kể từ lần đặt mốc gần nhất, SEM dừng với lý do `CONVERGED_NO_IMPROVEMENT`.
- Trần 20 s vẫn giữ.
- Vì sao chọn 4 000: sau nghiệm đầu, mỗi batch của A4 có 200 mẫu, nên 20 batch là 4 000 mẫu. Mỗi vòng lặp SEM rút
  một mẫu. Cửa sổ được khớp theo số mẫu, không theo thời gian hay khối lượng tính toán.
- Mã nguồn:
  - `SemRrtStar::setConvergenceStop`;
  - C ABI `sem_set_convergence_stop` và `sem_convergence`;
  - `tools/sem_native_cpp.py`;
  - trường request `sem_convergence_stop`;
  - khóa `convergence_stop` của method `sem_native`, chỉ hợp lệ khi `engine: "cpp"`.
- Luật mặc định tắt, nên mọi lượt cũ không đổi.
- Luật chỉ đọc chi phí đã công bố: không rút số ngẫu nhiên, không sửa cây. Vì vậy một lượt có luật dừng trùng từng
  bit với lượt không dừng bị cắt ở cùng số vòng lặp. Test `test_convergence_stop_is_an_exact_prefix` kiểm tra điều
  này bằng `==` với cả bản C++ lẫn bản Python.
- Validator coi hai trường hợp sau là lỗi correctness:
  - `sem_convergence_stop_mismatch`: luật ghi trong worker khác config.
  - `sem_convergence_inconsistent`: lượt dừng do hội tụ nhưng số vòng lặp khác mốc + 4 000, hoặc chi phí cuối nằm
    ngoài [mốc × (1 − 10⁻³), mốc].
- Báo cáo có thêm lý do dừng và thời gian dừng của từng phương pháp.

**Xem trước trên dev.**
- Thiết lập: map dev đã thấy, chạy trên Windows, 1 làn ở lõi 2. Cấu hình `configs/a4_rpbitstar_v1_selfstop_dev.json`,
  báo cáo `reports/a4_rpbit_selfstop_dev_20261117/REPORT.md`.
- 150/150 verdict, 0 lỗi correctness.
- SEM dừng do hội tụ ở 75/75 lượt: trung vị 2,08 s, trung bình 2,83 s, lâu nhất 10,1 s; mỗi lượt 6 915–37 106
  vòng lặp.
- A4 cũng dừng do hội tụ ở 75/75 lượt, trung bình 1,52 s.
- Đạt cả 10 tiêu chí:
  - J₃₀ thấp hơn ở 75/75 cặp: −1,02, cận trên −0,69;
  - APL −0,68%, cận trên −0,13%;
  - PRI −0,33%, cận trên +0,51%;
  - STI 0,0004 so với 0,084.
- **Time là biên hẹp nhất:** A4 nhanh hơn ở 51/75 cặp, trung bình nhanh hơn 1,31 s.
- Không tham số nào bị đổi sau khi xem.

**Đăng ký.**
- Biên bản: `logs/a4_rpbit_selfstop_20261117/registration.json`.
- Cấu hình: `configs/a4_rpbitstar_v1_selfstop_kaggle.json`.
- Controller dùng chung với 6.7: `kaggle_a4_rpbit/fairtime.py`, mục `EXPERIMENTS['selfstop']`. Điểm vào là
  `kaggle_a4_rpbit/selfstop.py`.

| Mục | Giá trị |
|---|---|
| Phương pháp | `sem_native` engine C++ với `convergence_stop` {ε = 10⁻³, 4 000 vòng lặp}, trần 20 s; `a4_rpbit_l33_m20` y như bản khóa |
| Held-out mới `heldout_selfstop` | 5 họ × seed bản đồ 81001–81010 × planner seed 816001–816005 = 250 cặp, 500 lượt |
| Lõi, đồng hồ, phân tích | như mục 6.7 |

- Seed 81001–81010 và 816001–816005 chưa từng xuất hiện trong cấu hình, công cụ, test, tài liệu hay thư mục kết quả nào.
- Tiêu chí chính: mục 5.3, không đổi. Time là hiệu ghép cặp giữa thời gian tự dừng của hai phương pháp.
- Cổng giống mục 6.7, đổi tên thành `smoke_selfstop` và `gate_selfstop`. CTest `sem_native_cpp` nay kiểm thêm tính
  chất trùng tiền tố. Cổng trùng từng bit Python/C++ chạy với luật dừng tắt.
- **Kỳ vọng ghi trước khi chạy:**
  - Trên Kaggle, A4 chậm đi khoảng 2,4 lần so với Windows trong các lần chạy trước, còn SEM chỉ chậm khoảng 1,4 lần.
    Biên Time vì vậy có thể thu hẹp hoặc đảo chiều.
  - J₃₀ và STI dự kiến vẫn nghiêng về A4.
  - Kết quả nào cũng được báo cáo.

**Diễn tập trên Windows** (chỉ dùng đầu vào cổng; `results/a4_rpbit_selfstop_rehearsal_20261117`, log ở
`logs/a4_rpbit_selfstop_20261117/`):
- `smoke_selfstop` đạt 13/14; lượt hỏng là lỗi SEM đã biết trên `nodata_thin_obstacle`.
- `gate_selfstop` đạt 10/10, 0 lỗi correctness. SEM dừng sau 1,0–19,7 s, A4 sau 1,1–2,4 s.
- SEM Python và C++ trùng từng bit 10/10.
- 31/31 nhóm CTest bắt buộc đạt.

### 6.9 Thí nghiệm 3×2 tách đóng góp của rpRefine (kế hoạch duyệt 17/11/2026; đang xem trước trên dev)

**Câu hỏi.** Ablation ở 6.2 và 6.6 cho thấy một phần lớn lợi thế J₃₀/STI của A4 đến từ `rpRefine`. Bước này áp
được cho đường của mọi planner. Vậy nếu SEM-RRT* cũng được hậu xử lý bằng đúng bộ refine đó, A4 còn hơn không?
Kế hoạch: `C:\Users\Long\.claude\plans\parallel-questing-lemon.md`. Người dùng chọn đủ 6 nhánh, refine trên SEM dùng
λ = 33 như A4.

| Planner \ refine | tắt | bật |
|---|---|---|
| SEM-RRT* C++, tự dừng như 6.8 | `sem_native` | `sem_refine` (mới, trial dẫn xuất) |
| BIT* không potential | `a4_bit_plain_l33_m20` (mới, chỉ là cấu hình) | `a4_rpbit_l33_m20_nopotential` |
| A4 | `a4_rpbit_l33_m20_norefine` | `a4_rpbit_l33_m20` (cấu hình chính đã khóa) |

- **BIT* không potential.** Vẫn dùng cận d_Φ, sampler atlas, tích phân cạnh nhanh và luật dừng 20 batch. Đây là
  "BIT* bỏ hai đóng góp của A4", không phải OMPL mặc định.
- **`sem_refine`.** Đọc trial `sem_native` cùng slot rồi replay trong C++ (`apps/RpRefineReplayWorker.inc`, khóa
  request `rp_refine_replay`):
  - **Đường thô:** mỗi incumbent của SEM được kiểm bằng `rpPathCost` và `SafetyPolicy` của A4, tại đúng thời điểm
    SEM công bố nó.
  - **Lịch refine:** tại mỗi r ∈ {0,5; 1; 2; 5; 10; 20} s trước khi SEM dừng, và một lần tại lúc SEM dừng. Mỗi lần
    refine incumbent đang đứng bằng `rpRefine` (λ = 33, tùy chọn mặc định).
    - Mỗi incumbent chỉ refine một lần.
    - Incumbent không an toàn thì không công bố và không refine.
  - **Thời gian:** mọi bước đo được cộng vào đồng hồ solve của SEM, như thể chạy nội tuyến.
  - **Công bố:** theo luật của A4 (J giảm hơn 10⁻⁶ tương đối, dấu thời gian ≤ 20 s).
  - **Tọa độ:** đi qua request dưới dạng mẫu bit IEEE-754, vì Boost.JSON 1.82 không parse số thập phân chính xác.
  - **Kiểm chứng:** `tools/rp_bit_validation.py::_replay_checks` buộc trial với nguồn (sha của `worker.json`), dựng lại
    lịch, dấu thời gian và luật công bố, và so đường thô với nguồn.
- **Luật quyết định** (đăng ký trước; tương phản gộp `a4_rpbit_l33_m20 − sem_refine`, bootstrap map → seed một phía 95%):
  - **A:** ΔJ₃₀ < 0 và cận trên < 0. Viết bài về A4 trọn vẹn.
  - **B:** không đạt A, ΔTime < 0 với cận trên < 0, và J₃₀ không kém quá 1%. Viết bài về tốc độ.
  - **C:** còn lại. Viết bài về rpRefine như bộ hậu xử lý.

  A được xét trước B, nên xác suất kết luận sai tối đa 2α; đây là quyết định hướng bài, không phải claim.
- **Mã.**
  - `tools/rp_bit_protocol.py`: kind `sem_refine`; nhánh dẫn xuất chạy ngay sau nguồn trong cùng block; khối tùy chọn
    `analysis.contrasts`/`decision`.
  - `tools/analyze_rp_bit.py`: tham số `baseline`; mục 7–9 của báo cáo chỉ xuất hiện khi config có tương phản.
  - `tools/sem_exposure_validation.py`: chấp nhận `planner_id = rp_refine_replay`.
  - `apps/terrain_benchmark.cpp` (file khóa): thêm đúng 2 dòng.
  - Không đổi `src/RpBit.cpp`, `RpBit.hpp`, `RpBitWorker.inc`.
- **Kiểm tra trước khi chạy dev** (build `build-a4-factorial`):
  - **Test mới** `rp_refine_replay_worker` (16 case):
    - đường refine của replay trùng từng byte với đường refine của A4 từ cùng incumbent thô;
    - lịch, thời gian, cắt ngân sách, nguồn không an toàn, chạy lại tất định;
    - 5 kiểu dữ liệu bị sửa đều bị bắt;
    - 9 request sai đều bị từ chối.
  - **Test A4 cũ:** `test_rp_bit_worker.py` cho cùng số liệu với binary cũ.
  - **Báo cáo cũ:** chạy lại phân tích `dev_selfstop` cho `report.json` và `REPORT.md` trùng từng byte
    (sha `9c1792b1…`); 6 verdict chấm lại trùng.
  - **CTest:** 32/32 nhóm đạt.
- **Smoke cục bộ** (`results/a4_rpbit_factorial_dev_20261117/smoke_factorial`, 42 lượt, 2 s):
  - Lần chấm đầu báo 1 lỗi correctness `requested_replay_mismatch`. Nguyên nhân: Boost.JSON parse
    `source_preprocessing_s` lệch 1 ULP, trong khi phần echo bị so bằng `==`.
  - Đã sửa validator: số thực trong echo so với dung sai 10⁻¹²; planner và metric không đổi. Verdict cũ giữ ở
    `validation_full_r1_before_echo_fix/`.
  - Chấm lại: 41/42 đạt, 0 lỗi. Lượt hỏng là lỗi SEM đã biết trên `nodata_thin_obstacle` (`missing_sem_support`).
  - Trên chính map đó `sem_refine` đạt: replay loại 30 incumbent không an toàn của SEM và công bố một incumbent an toàn
    có trước. Đây là hiệu ứng chọn lọc đã khai báo: PSR của `sem_refine` có thể cao hơn `sem_native`.
- **Cấu hình dev:** `configs/a4_rpbitstar_v1_refine_factorial_dev.json`.
  - Phase `smoke_factorial`, `gate_factorial`, `dev_factorial`; map dev đã thấy 41001–41003 × 416001–416005,
    75 slot × 6 = 450 lượt.
  - Held-out Kaggle (seed 91001–91010 × 916001–916005) chỉ đăng ký sau khi người dùng xem kết quả dev.
- **Kết quả dev** (`results/a4_rpbit_factorial_dev_20261117/dev_factorial`; báo cáo
  `reports/a4_rpbit_factorial_dev_20261117/`, `report.json` sha `dcd1b095…`, `REPORT.md` sha `82e2dcc9…`; map đã thấy):
  - **Thu thập:** 450/450 verdict đạt, 0 lỗi correctness.
  - **Hồi quy:** đường cuối của `sem_native` và `a4_rpbit_l33_m20` trùng từng byte với `dev_selfstop` ở cả 150 trial.
  - **Bảng gộp (75 lượt mỗi nhánh, PSR 100% cả 6):**

    | Nhánh | APL (m) | PRI | STI (1/m) | Time (s) | J₃₀ |
    |---|---:|---:|---:|---:|---:|
    | `sem_native` | 36.563 | 3.956 | 0.0835 | 2.73 | 152.38 |
    | `sem_refine` | 36.317 | 3.946 | 0.00147 | 2.91 | 151.37 |
    | `a4_bit_plain_l33_m20` | 36.438 | 3.954 | 0.0128 | 9.41 | 152.00 |
    | `a4_rpbit_l33_m20_nopotential` | 36.325 | 3.942 | 0.00147 | 4.41 | 151.37 |
    | `a4_rpbit_l33_m20_norefine` | 36.392 | 3.949 | 0.00806 | 3.01 | 151.82 |
    | `a4_rpbit_l33_m20` | 36.315 | 3.943 | 0.000367 | 1.43 | 151.36 |

  - **Luật quyết định:** ra **A**.
    - ΔJ₃₀ = −0.0167, cận trên −8.1·10⁻⁵.
    - Độ lớn chỉ bằng −0.013% J₃₀; trung vị −0.003.
    - 50/75 slot A4 tốt hơn. Chỉ 2 slot lệch quá 0.1%, và 5 slot chiếm 91% tổng hiệu.
    - 12/15 map âm; bỏ bất kỳ map nào thì hiệu vẫn âm.

    Điều kiện B cũng đạt: ΔTime = −1.48 s, cận trên −0.83 s.
  - **Đọc kết quả:**
    - Khi bật refine, cả ba planner hội tụ về gần cùng một J₃₀ (151.36–151.37).
    - Planner chỉ còn khác ở tốc độ:
      - Time: A4 1.43 s, SEM + refine 2.91 s, BIT* không potential + refine 4.41 s;
      - anytime tại 0.5 s, A4 − `sem_refine`: ΔJ₃₀ −0.56 (cận trên −0.35). Từ 1 s trở đi hiệu < 0.03.
    - `nopotential − sem_refine`: J₃₀ ngang nhau (cận trên +0.010), chậm hơn 1.5 s.
  - **Luật thiếu ngưỡng hiệu tối thiểu.** Luật A đã đăng ký không có ngưỡng hiệu tối thiểu, nên một hiệu 0.01% vẫn ra A.
    Có thêm ngưỡng hay không là việc người dùng quyết trước khi đăng ký held-out. Nếu thêm thì phải ghi là
    "đổi sau khi đã thấy dev".
  - **Replay:**
    - 248 lần refine, 7 lần bỏ qua vì đã refine. Lần refine nào cũng tốt hơn đường thô của chính nó (248/248).
    - 103 lần không hạ được J đã công bố, vì thua một đường đã refine trước đó.
    - 2 đường thô SEM bị loại vì không an toàn.
    - 0 ứng viên quá ngân sách; 75/75 lượt dừng do hội tụ.
    - Đường cuối của cả 75 lượt là đường đã refine.
    - Overhead trung bình 0.176 s, lớn nhất 0.398 s.

## 7. Hạn chế đã biết

- Trên held-out, PRI chỉ thấp hơn SEM 0,13% và chưa tách khỏi 0 về thống kê. Lợi thế rõ và ổn định là Time,
  STI, J₃₀ và (trên held-out) APL.
- Held-out chạy trên Kaggle với các thay đổi đã ghi ở mục 6.5. Đáng chú ý nhất là SEM chạy cố định 7 425 vòng
  lặp thay vì 20 s; trên CPU Kaggle, SEM cần trung bình 90 s để chạy đủ số vòng lặp đó.
- Time ở mục 6.2 và 6.6 so A4 viết bằng C++ với SEM-RRT* tái dựng bằng Python. Bản SEM-RRT* C++ đã có. Thí nghiệm
  so sánh cùng ngôn ngữ ở mục 6.7 đã chạy, nhưng ở đó SEM luôn dùng đủ 20 s. Thí nghiệm cả hai cùng tự dừng ở mục 6.8
  đã đăng ký nhưng chưa chạy.
- Phần vượt SEM về J₃₀ phụ thuộc bước tinh chỉnh (xem ablation); đây là hậu xử lý tất định, được khai
  báo, không phải thuộc tính của riêng BIT*.
- Potential chưa mô hình hóa vật cản hình học; cận dưới quanh vật cản lỏng hơn (khoảng cách chứng chỉ
  J/ĝ(goal) − 1 của cấu hình chính trên dev: 1,6–13,5%, trung vị 6,2%), nên A4 chưa chứng minh được
  tối ưu toàn cục.
- Góc gấp tại ranh giới rủi ro có độ tương phản rất cao có thể không bo được trong trần 0,5% J.
- Chưa làm so với kế hoạch: kiểm định chi-square cho tính đều của sampler; đối chứng âm "potential chỉ
  nội suy bốn góc phải trượt test liên tục"; test "cùng tập mẫu thì nghiệm trùng với d_Φ" (thay bằng
  so sánh hai biến thể trên fixture có nghiệm giải tích).
