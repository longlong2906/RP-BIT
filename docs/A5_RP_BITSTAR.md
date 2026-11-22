# A5 / RP-BIT* nhanh — `a5_rpbitstar_v1`

A5 là A4 (`docs/A4_RP_BITSTAR.md`) được làm nhanh hơn mà không đổi hàm mục tiêu, các cận dưới, sampler, miền an
toàn hay bước tinh chỉnh. Tài liệu này mô tả bốn cơ chế thêm vào, lập luận đúng đắn, cách chạy và kết quả trên các
bản đồ dev **đã thấy**. Kế hoạch được duyệt ngày 21/11/2026. Thiết kế của C3 và C4 đã đổi sau probe 1 (mục 6.2);
người dùng chấp nhận thay đổi đó cùng ngày. Held-out trên Kaggle được đăng ký ngày 22/11/2026 (mục 6.10) và đã chạy:
nhánh chính đạt cả bốn tiêu chí, SEM / A5 = 6,20 (mục 6.11).

## 1. Mục tiêu và những gì giữ nguyên

- **Lý do.** A4 chỉ nhanh hơn SEM-RRT* bản C++ tự dừng 1,5–1,9 lần. A4 công bố nghiệm đầu sau khoảng 0,11 s nhưng
  tự dừng sau khoảng 1,4 s: phần lớn thời gian là chờ luật dừng 20 batch.
- **Giữ nguyên, dùng lại qua API công khai của A4:** J với λ = 33, `RpEdgeIntegrator`, hai `RiskPotential` và các cận
  ĝ/ĥ/ĉ, `RpInformedSampler`, `SafetyPolicy`, `rpRefine` với tham số mặc định, ε = 10⁻³, gốc đồng hồ (trước khi dựng
  safety policy, potential và planner), kiểm tra cận tại mọi incumbent thật, kiểm toán canonical khi ghi.
- **Lõi vẫn là `ompl::geometric::BITstar` thật**, không patch, cùng cấu hình: k-nearest, JIT tắt, pruning bật,
  rewire 1,1, batch đầu 100 mẫu.
- Không file nào của A4 bị sửa. Với mọi cờ tắt, `batch_min` = 200 và `window_samples` = 200·M, một lượt A5 trùng
  từng bit với lượt A4 (có test).

## 2. Bốn cơ chế, mỗi cơ chế một cờ

### 2.1 C1 — cửa sổ hội tụ đếm theo mẫu, batch lớn dần (`batch_growth`)

A4 dừng khi J đã công bố không giảm ít nhất ε qua M batch 200 mẫu; M = 20 là 4 000 mẫu. Mỗi batch BIT* khởi động
lại hàng đợi và duyệt lại cây, nên 20 batch nhỏ tốn hơn nhiều so với vài batch lớn.

A5 giữ tiêu chí đó nhưng đếm theo mẫu. Tại mỗi ranh giới batch, sau khi đã có đường công bố:

- nếu J đã công bố thấp hơn mốc × (1 − ε): đặt mốc mới, số mẫu cũ về 0;
- nếu không: cộng số mẫu của batch vừa xử lý xong vào số mẫu cũ; khi số mẫu cũ ≥ W (`window_samples`) thì dừng với
  lý do `CONVERGED_NO_IMPROVEMENT`.

Cỡ batch kế tiếp = max(`batch_min`, số mẫu cũ + số mẫu của batch đang mở), không vượt phần còn lại của cửa sổ. Với
`batch_min` = 500 và W = 4 000, cửa sổ kiểm chứng gồm 4 batch 500 → 500 → 1 000 → 2 000 thay vì 20 batch. Tắt
`batch_growth` thì mọi batch có `batch_min` mẫu.

- W = 4 000 đúng bằng cửa sổ của A4 với M = 20 và của SEM-RRT* tự dừng (4 000 vòng lặp, mỗi vòng một mẫu).
- Trước nghiệm đầu, batch vẫn là 100 mẫu như A4. BIT* chuyển sang `batch_min` tại incumbent đầu tiên, hoặc ngay từ
  đầu nếu nghiệm mồi (2.4) được công bố.
- Khi A5 nhìn thấy một ranh giới, BIT* đã rút xong batch mới. Cỡ đặt tại ranh giới vì vậy áp dụng cho batch sau đó,
  và lúc dừng BIT* đã rút thêm một batch `batch_min` mẫu chưa xử lý.

### 2.2 C2 — bỏ qua phép tính độ dốc quét khi đã có chứng chỉ (`fast_validity`)

`SafetyPolicy::rejectionReason` kiểm tra theo thứ tự: ô SEM (support, risk ≤ 40) → hình học footprint → độ dốc
quét lớn nhất ≤ 35° (`Terrain::maximumSweptSlope`, duyệt mọi tam giác trong capsule). Phép cuối đắt nhất.

`RpA5SlopeEnvelope` được dựng một lần, khoảng 2 ms trên bản đồ dev:

1. Với mỗi ô địa hình, lấy gradient lớn hơn trong hai tam giác SW-SE-NE và SW-NE-NW (cùng cách chia với
   `maximumSweptSlope`). Ô có nút thiếu độ cao nhận +∞.
2. Lấy max trong cửa sổ ±reach ô theo mỗi trục, reach = ⌈bán kính footprint / cỡ ô⌉ + 2.

Một cạnh được **chứng nhận** khi tại mọi điểm lấy cách nhau nửa ô trên cạnh, giá trị envelope của ô chứa điểm
đó ≤ tan(35°)·(1 − 10⁻⁹). Khi đó mọi tam giác trong capsule đều thoải hơn trần, phép tính chính xác chắc chắn đạt,
nên được bỏ qua. Cạnh không được chứng nhận thì gọi phép tính chính xác như cũ.

- **Vì sao reach đủ.** Tam giác cách một điểm p của cạnh không quá bán kính thì nằm trong ô lệch tối đa
  ⌈bán kính / cỡ ô⌉ chỉ số so với ô chứa p. Điểm p cách điểm lấy mẫu gần nhất không quá 1/4 ô, nên lệch thêm tối đa
  1 chỉ số.
- **Vì sao thiếu dữ liệu không lọt.** `Terrain` coi một ô là có dữ liệu khi cả bốn nút hữu hạn. Một nút thiếu độ cao
  làm cả bốn ô quanh nó nhận +∞ trong envelope, nên không cạnh nào chạm ô đó được chứng nhận.
- Thứ tự kiểm tra và lý do từ chối giữ nguyên. **Quyết định hợp lệ trùng từng cạnh với `SafetyPolicy`.**
- Chỉ validator của BIT* được thay. `rpRefine` và bước kiểm tra trước khi công bố vẫn dùng `SafetyPolicy` chính xác.

### 2.3 C3 — tinh chỉnh lười (`lazy_refine`) — thiết kế đổi sau probe 1

A4 gọi `rpRefine` tại mọi ranh giới batch có incumbent mới. Ở A5, incumbent mới của BIT* chỉ được tinh chỉnh và
công bố khi một trong ba điều sau đúng:

- chưa có đường nào được công bố;
- chi phí thô của nó thấp hơn ít nhất ε so với chi phí thô thấp nhất trong các incumbent đã tinh chỉnh;
- cửa sổ sẽ đầy ngay tại ranh giới này.

Trường hợp khác incumbent chờ (bộ đếm `refine_deferred`). Nhờ điều thứ ba, lượt chạy không bao giờ dừng trên một
incumbent chưa tinh chỉnh; nếu lần tinh chỉnh đó hạ J ít nhất ε thì cửa sổ chạy lại. Với ε = 0, tinh chỉnh lười
trùng từng bit với tinh chỉnh ở mọi ranh giới (có test).

### 2.4 C4 — nghiệm mồi từ lưới và cận lấy mẫu theo J đã công bố (`seed`) — thiết kế đổi sau probe 1

Trước khi BIT* chạy:

1. **A\* trên tâm các ô SEM mở**, 8-liên thông; bước chéo chỉ được đi khi hai ô kề cạnh cũng mở. Chi phí một bước =
   độ dài 3D × (1 + λ/40 × trung bình risk của hai ô). Dẫn hướng bằng potential-tới-đích của A4 chia κ; đây chỉ là
   dẫn hướng, không tuyên bố tối ưu. Tối đa 60 000 lần mở rộng.
2. **Gộp và tinh chỉnh:** `rpRefine` chỉ-cắt-ngắn để gộp các bước lưới, rồi `rpRefine` đầy đủ.
3. **Kiểm tra và công bố:** mọi cạnh qua `SafetyPolicy`; đạt thì công bố với nguồn `a5_lattice_seed_refined`, batch 0.
4. **BIT\* chạy như cũ** trên cùng đối tượng planner và cùng cấu hình. Sampler informed nhận thêm một cận trên là J
   đã công bố: không rút mẫu nào có ĝ + ĥ ≥ J đã công bố.

BIT* **không** nhận trạng thái nào của nghiệm mồi. Nó vẫn tự tìm nghiệm của mình; incumbent của nó chỉ được công bố
khi sau tinh chỉnh thấp hơn J đã công bố.

Mọi thất bại (`no_lattice_path`, `lattice_path_undefined`, `seed_unsafe`) đều dẫn tới không có nghiệm mồi và khởi
động như A4.

### 2.5 Vì sao vẫn đúng

- Các cận ĝ, ĥ, ĉ không đổi. Worker vẫn kiểm chúng tại mọi incumbent thật; khác 0 vi phạm là lỗi correctness.
- C1 chỉ đổi cách chia batch và cách đếm cửa sổ. C3 chỉ đổi thời điểm gọi `rpRefine`.
- C2 không đổi quyết định hợp lệ nào (lập luận ở 2.2, kiểm bằng `==` trên 80 000 cạnh ngẫu nhiên).
- C4: mọi đường tốt hơn J đã công bố chỉ đi qua các điểm có ĝ + ĥ nhỏ hơn J đó, nên cận lấy mẫu chặt hơn không loại
  nghiệm tốt hơn nào. Nghiệm mồi phải qua `SafetyPolicy` như mọi đường khác.
- Mọi đường công bố vẫn qua `SafetyPolicy` chính xác, kiểm toán canonical lúc ghi và validator Python độc lập trên
  mọi snapshot.
- Validator có thêm ba kiểm tra cho A5: tùy chọn worker báo phải trùng request; luật dừng và lịch batch được dựng
  lại từ `batch_log`; sự kiện nghiệm mồi chỉ được đứng đầu và chỉ khi bật `seed`.

## 3. Tham số và các nhánh

| Tham số | Nhánh chính `a5_rpbit_l33_w4000` | Ghi chú |
|---|---|---|
| λ, ε | 33, 10⁻³ | như A4 đã khóa |
| `window_samples` (W) | 4 000 | bằng cửa sổ của A4 M = 20 và của SEM-RRT* tự dừng |
| `batch_min` | 500 | cỡ batch sau nghiệm đầu; A4 dùng 200 |
| `batch_growth`, `fast_validity`, `lazy_refine`, `seed` | bật cả bốn | mỗi request phải ghi rõ cả bốn cờ |
| trần mở rộng của A* mồi | 60 000 | hằng số `RP_A5_SEED_EXPANSIONS` |
| tham số `rpRefine` | mặc định của A4 | không đổi |

Các nhánh của `configs/a4_rpbitstar_v1_a5_dev.json`:

| Nhánh | Nội dung |
|---|---|
| `sem_native` | SEM-RRT* C++, tự dừng 0,1% / 4 000 vòng lặp, trần 20 s |
| `a4_rpbit_l33_m20` | A4 chính đã khóa (M = 20) |
| `a4_rpbit_l33` | A4 với M = 5, tức cửa sổ 1 000 mẫu |
| `a5_rpbit_l33_w4000` | **A5 chính:** C1 + C2 + C3 + C4, W = 4 000 |
| `a5_rpbit_l33_w1000` | A5 đầy đủ, W = 1 000 (thăm dò; SEM không có nhánh đối xứng) |
| `a5_rpbit_l33_w4000_noseed` | C1 + C2 + C3 |
| `a5_rpbit_l33_w4000_eager` | C1 + C2 |
| `a5_rpbit_l33_w4000_batchonly` | C1 |

## 4. Mã nguồn và kiểm thử

- **Mới:** `include/terrain25d/RpBitA5.hpp`, `src/RpBitA5.cpp`, `apps/RpBitA5Worker.inc`,
  `cmake/RpBitA5Targets.cmake`, `tests/integration/rp_bit_a5_tests.cpp`, `tests/test_rp_bit_a5_worker.py`,
  `configs/a4_rpbitstar_v1_a5_dev.json`, `tools/rp_bit_a5_field.py` (container nhị phân của trường rủi ro, mục 6.8).
- **Sửa tối thiểu:**
  - `apps/terrain_benchmark.cpp`: 1 dòng include và 1 dòng dispatch theo khóa `rp_bit_a5`;
  - `CMakeLists.txt`: 1 dòng include;
  - `tools/rp_bit_protocol.py`: kind `rp_bit_a5`, kiểm tra khóa, nhánh `request_for`;
  - `tools/rp_bit_validation.py`: `_a5_checks`, thêm kind vào `events_j_sem`;
  - `tools/run_sem_exposure.py`: tham số tùy chọn `field_exchange` của `_collect_trial`; mặc định không đổi;
  - `tools/run_rp_bit.py`: chọn container nhị phân cho kind `rp_bit_a5`, mọi kind khác giữ file văn bản;
  - `tests/test_rp_bit_tooling.py`: `A5ConfigTests`.
- **Sửa mã dùng chung, theo yêu cầu của người dùng (mục 6.9), có hiệu lực với mọi pipeline:**
  - `src/Terrain.cpp`: `Terrain::load` đọc file nhanh hơn; tập file được chấp nhận và giá trị đọc ra không đổi;
  - `tools/native_paper_worker.py` và `tools/run_sem_exposure.py`: khởi động tiến trình nằm ngoài đồng hồ trước solve;
  - `tests/unit/terrain_tests.cpp`: nhóm `G_LOAD` cho `Terrain::load`.
- **Không đổi** (hash trùng `logs/a4_rpbit_20261116/lock_manifest.json`, kiểm ngày 21/11/2026): `src/RpBit.cpp`,
  `include/terrain25d/RpBit.hpp`, `apps/RpBitWorker.inc`, `cmake/RpBitTargets.cmake`,
  `tests/integration/rp_bit_tests.cpp`, `tests/test_rp_bit_worker.py`, `configs/a4_rpbitstar_v1.json` và hai binary
  trong `build-a4-rpbit`.
- **`rp_bit_a5_core` (C++):**
  - lịch batch: 500/500/1 000/2 000 với W = 4 000; tổng đúng bằng W với nhiều cặp (W, `batch_min`); tắt tăng batch
    cho đúng 20 batch 200 mẫu;
  - mọi cờ tắt: công bố, số vòng lặp, số batch và lý do dừng trùng `runRpBit` của A4, kể cả khi tắt tinh chỉnh và
    khi dừng bằng cửa sổ 600 mẫu so với 3 batch của A4;
  - C2: 80 000 cạnh và điểm ngẫu nhiên trên fixture có dốc khoảng 40°, lỗ NoData và vật cản, lý do từ chối trùng
    `SafetyPolicy::rejectionReason`; bật hay tắt C2 cho cùng công bố, số vòng lặp và bộ đếm;
  - C3: tái lập được; không dừng trên incumbent chưa tinh chỉnh; ε = 0 trùng từng bit với tinh chỉnh ở mọi ranh giới;
  - C4: nghiệm mồi là công bố đầu tiên và tất định; tường chặn mà trường SEM không biết cho `seed_unsafe` rồi khởi
    động như A4; trường bị chia cắt cho `no_lattice_path`; trần mở rộng có hiệu lực;
  - lượt đầy đủ trên fixture dốc; tổ hợp tùy chọn sai và truy vấn sai bị từ chối;
  - container nhị phân của trường rủi ro cho đúng từng bit các ô của file văn bản; file cụt, thừa byte, support khác
    0/1, lưới rỗng và risk trên ô không có support đều bị từ chối.
- **`rp_bit_a5_worker` (Python + worker thật):** 13 lượt thật qua validator nghiêm ngặt trên mọi snapshot, trong đó
  một lượt đọc trường rủi ro từ container nhị phân và phải cho đúng các byte đường đi của lượt đọc file văn bản; 9 bản
  ghi bị sửa tay phải bị scorer báo đúng lỗi; 21 request sai phải bị từ chối với mã 2.

## 5. Cách chạy

```powershell
cmake --build build-a5 --parallel 3
ctest --test-dir build-a5 -L rp_bit --output-on-failure -j 1
ctest --test-dir build-a5 -L "sem_cpp|sem_exposure|pri_segment|pri_pathsearch|safe_length" --output-on-failure -j 1
python -B tools/run_rp_bit.py run --phase dev_a5 --config configs/a4_rpbitstar_v1_a5_dev.json `
  --output results/<thư mục> --worker build-a5/terrain_benchmark.exe `
  --bridge build-a5/libterrain_compare_c.dll --full-validation
python -B tools/analyze_rp_bit.py --input results/<thư mục>/dev_a5 --output reports/<thư mục>
```

Chạy `ctest` từ PowerShell. Trong Git Bash, `safe_length_unit` và `safe_length_oracle` nạp nhầm DLL runtime của Git
và báo lỗi giả (đã gặp ngày 21/11/2026; hai test đó đạt khi chạy lại từ PowerShell).

## 6. Kết quả: dev ngày 21/11/2026 trên bản đồ đã thấy (6.1–6.9), held-out Kaggle ngày 22/11/2026 (6.11)

Mọi số do oracle Python độc lập tính lại từ bản đồ và point cloud gốc. SEM-RRT* là bản C++ của dự án, tự dừng theo
đúng luật của A4. Time là thời gian solve tới lúc tự dừng; với A4 và A5 nó gồm dựng potential, BIT* và tinh chỉnh.
Phần trước đồng hồ solve (nạp dữ liệu, dựng trường SEM) báo riêng: ở lượt dev đầu A4 và A5 khoảng 0,51 s, SEM khoảng
0,41 s; sau mục 6.8 và 6.9, A5 còn 0,10 s và SEM còn 0,12 s.
Dev dùng 15 bản đồ **đã thấy** (5 họ × seed 41001–41003) × 5 planner seed = 75 slot. Đây là dữ liệu phát triển,
**không phải bằng chứng tổng quát hóa**.

### 6.1 Smoke (7 fixture, 2 s, validator nghiêm ngặt trên mọi snapshot)

Hai lượt: `results/a4_rpbit_a5_smoke_20261121` (thiết kế trước probe 1) và
`results/a4_rpbit_a5_smoke_final_20261121` (binary cuối `688eac33…`). Mỗi lượt 56 trial, 55 thành công, 0 lỗi
correctness. Mọi nhánh A4 và A5 đạt 7/7. Lượt không thành công duy nhất là `sem_native` trên
`nodata_thin_obstacle`, lỗi đã biết của baseline.

### 6.2 Probe 1 và thay đổi thiết kế của C3, C4

Probe = 15 bản đồ dev × planner seed 416001 × 8 nhánh = 120 lượt. Probe 1
(`results/a4_rpbit_a5_probe1_20261121`, binary `19cdc2c0…`, 120/120 thành công, 0 lỗi correctness) chạy thiết kế
ghi trong kế hoạch:

- C3 cũ: bơm các điểm của đường đã tinh chỉnh ngược vào BIT* qua sampler, để BIT* nối lại và hạ cận của nó;
- C4 cũ: nghiệm mồi được bơm vào batch đầu của BIT* theo cùng đường đó.

| Nhánh (probe 1) | Time (s) | SEM / nhánh | J₃₀ so với A4 | Slot kém A4 quá 0,1% (tệ nhất) |
|---|---:|---:|---:|---:|
| SEM-RRT* | 2,260 | 1 | +0,67% | — |
| A4 chính | 1,463 | 1,54 | — | — |
| C1 | 0,694 | 3,26 | +0,002% | 0/15 |
| C1 + C2 | 0,444 | 5,09 | +0,002% | 0/15 |
| C1 + C2 + C3 cũ | 0,450 | 5,02 | +0,326% | 3/15 (+2,17%) |
| Đầy đủ (C3, C4 cũ), W = 4 000 | 0,435 | 5,20 | +0,055% | 2/15 (+0,38%) |
| Đầy đủ (C3, C4 cũ), W = 1 000 | 0,187 | 12,10 | +0,103% | 4/15 (+0,45%) |

Phản hồi vào BIT* không rút ngắn thời gian (0,444 → 0,450 s) mà làm J kém đi. Nó bị gỡ khỏi mã và thay bằng thiết
kế ở mục 2.3 và 2.4. Cần đọc kết quả bên dưới kèm ba điều sau:

- Thay đổi này được làm **sau khi xem số liệu probe 1**, trên chính các bản đồ dev.
- Phương án lùi ghi trong kế hoạch là tắt C3 và C4 ở nhánh chính, không phải thiết kế lại.
- Người dùng chấp nhận thiết kế mới và giữ nhánh đầy đủ làm nhánh chính ngày 21/11/2026, trước khi chạy dev.

Mã của thiết kế cũ không còn trong cây nguồn; chỉ còn thư mục kết quả probe 1.

### 6.3 Probe 2 (thiết kế hiện tại, binary cuối)

`results/a4_rpbit_a5_probe2_20261121`: 120/120 thành công, 0 lỗi correctness.

| Nhánh (probe 2) | Time (s) | SEM / nhánh | J₃₀ so với A4 | Slot kém A4 quá 0,1% (tệ nhất) |
|---|---:|---:|---:|---:|
| SEM-RRT* | 2,258 | 1 | +0,67% | — |
| A4 chính | 1,449 | 1,56 | — | — |
| C1 | 0,684 | 3,30 | +0,002% | 0/15 |
| C1 + C2 | 0,436 | 5,18 | +0,002% | 0/15 |
| C1 + C2 + C3 | 0,428 | 5,27 | +0,002% | 0/15 |
| **A5 chính**, W = 4 000 | 0,427 | 5,29 | −0,001% | 0/15 |
| A5 đầy đủ, W = 1 000 | 0,201 | 11,24 | +0,092% | 4/15 (+0,28%) |

Kế hoạch ước tính 7–10 lần cho nhánh chính; probe đo được 5,3 lần. Người dùng được báo điều này và vẫn duyệt chạy
dev đầy đủ.

### 6.4 Dev đầy đủ (75 slot × 8 nhánh = 600 lượt)

`results/a4_rpbit_a5_dev_20261121/dev_a5`, một phiên: thu thập 1 283 s trên lõi 2, validate 793 s trên 4 lõi.
600/600 verdict nghiêm ngặt, 600 thành công, 0 lỗi correctness; mọi lượt của 8 nhánh dừng do hội tụ. Báo cáo:
`reports/a4_rpbit_a5_dev_20261121/REPORT.md`, `report.json` và `a5_diagnostics.json`.

| Nhánh | PSR | APL (m) | PRI | Góc gấp trên 75 đường | Time (s) | SEM / nhánh | Cặp nhanh hơn SEM | J₃₀ | J₃₀ so với A4 [cận trên 95%] | Slot kém A4 quá 0,1% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEM-RRT* (`sem_native`) | 75/75 | 36,563 | 3,9558 | 229 | 2,626 | 1 | — | 152,376 | +0,67% | — |
| A4 chính (`a4_rpbit_l33_m20`) | 75/75 | 36,315 | 3,9426 | 1 | 1,379 | 1,90 | 54/75 | 151,357 | — | — |
| A4, M = 5 (`a4_rpbit_l33`) | 75/75 | 36,293 | 3,9487 | 5 | 0,448 | 5,87 | 74/75 | 151,373 | +0,010% | 0 |
| C1 (`…_batchonly`) | 75/75 | 36,315 | 3,9433 | 6 | 0,760 | 3,46 | 68/75 | 151,363 | +0,004% [+0,008%] | 0 |
| C1 + C2 (`…_eager`) | 75/75 | 36,315 | 3,9433 | 6 | 0,496 | 5,29 | 74/75 | 151,363 | +0,004% | 0 |
| C1 + C2 + C3 (`…_noseed`) | 75/75 | 36,313 | 3,9433 | 7 | 0,485 | 5,42 | 74/75 | 151,364 | +0,004% | 0 |
| **A5 chính (`a5_rpbit_l33_w4000`)** | 75/75 | 36,325 | 3,9370 | 5 | **0,419** | **6,28** | 74/75 | 151,386 | +0,019% [+0,049%] | 2 |
| A5 đầy đủ, W = 1 000 | 75/75 | 36,425 | 3,9092 | 5 | 0,211 | 12,45 | 75/75 | 151,447 | +0,059% [+0,106%] | 9 |

"SEM / nhánh" là Time trung bình của SEM chia Time trung bình của nhánh trên 75 cặp. Cận trên là bootstrap phân cấp
map → seed (20 000 lần, một phía 95%) và chỉ có cho các tương phản đã đăng ký.

Nhánh chính theo từng họ (Time của SEM / Time của A5): `traversable_hill` 3,91; `narrow_passage` 4,58;
`ridge_pass` 5,47; `mixed_obstacles` 7,29; `rolling_terrain` 9,70. A5 nhanh hơn SEM ở 74/75 cặp; cặp còn lại là
`rolling_terrain_41001` seed 416003 (0,434 s so với 0,425 s).

### 6.5 Ba tiêu chí đã đăng ký trong kế hoạch, chấm trên nhánh chính

| Tiêu chí | Kết quả | Kết luận |
|---|---|---|
| 1. Đúng đắn | 0 lỗi correctness; validate nghiêm ngặt mọi snapshot; 0 vi phạm cận; PSR 75/75 | đạt |
| 2a. Mười tiêu chí so với SEM | 10/10: ΔJ₃₀ −0,990 [−0,672], APL −0,65% [−0,03%], PRI −0,48% [+0,55%], 5 góc gấp so với 229 | đạt |
| 2b. Không kém A4 cùng pha | ΔJ₃₀ gộp +0,019%, cận trên 95% +0,049% (ngưỡng +0,1%); 2 slot kém quá 0,1% | đạt |
| 3. Tốc độ | 2,626 / 0,419 = 6,28 (ngưỡng 5); nhanh hơn ở 74/75 cặp = 98,7% (ngưỡng 90%) | đạt |
| Mục tiêu kỳ vọng 7–10 lần | 6,28 | **không đạt** |

Không tiêu chí hay tham số nào bị đổi sau khi xem số liệu dev.

### 6.6 Đóng góp của từng cơ chế

Tương phản đã đăng ký, mỗi bậc so với bậc trước (cận trên 95% trong ngoặc):

| Cơ chế | ΔTime (s) | ΔJ₃₀ gộp | Ghi chú |
|---|---:|---:|---|
| C1 (so với A4 chính) | −0,619 [−0,533] | +0,004% [+0,008%] | 6,8 batch mỗi lượt |
| C2 | −0,263 [−0,223] | 0 | đường cuối trùng nhau ở 75/75 slot |
| C3 | −0,012 [−0,006] | +0,001% [+0,001%] | 5,83 → 5,49 lần tinh chỉnh mỗi lượt |
| C4 | −0,066 [−0,030] | +0,015% [+0,044%] | 5,49 → 4,09 lần tinh chỉnh mỗi lượt |

- **C2** chứng nhận 100% phép kiểm của BIT* trên dev: không lần nào phải tính độ dốc chính xác. Phần tìm kiếm giảm
  từ 0,51 s xuống 0,24 s mỗi lượt.
- **C3** gần như không đóng góp.
- **C4** có ba tác dụng đo được:
  - Nghiệm mồi được công bố ở 75/75 lượt, thuộc đủ 5 họ, tại 0,102 s.
  - Nghiệm công bố đầu tiên gần nghiệm cuối: J cao hơn J cuối trung bình 0,13%, nhiều nhất 0,57%. Không có nghiệm
    mồi, 25/75 lượt có nghiệm đầu cao hơn nghiệm cuối trên 1%, 10/75 lượt trên 10%, tệ nhất 4,5 lần. Tại 0,25 s,
    J₃₀ là 151,45 so với 151,87 của A4 (Δ −0,418 [−0,023]).
  - Time giảm 0,066 s.
- **Giá phải trả của C4:**
  - BIT* cải thiện được nghiệm mồi ở 69/75 lượt (trung vị 0,03%, nhiều nhất 0,57%). Ở 6/75 lượt, đường cuối chính là
    nghiệm mồi.
  - Hai slot `ridge_pass_41003` (seed 416002 và 416003) kết thúc trên nghiệm mồi, J = 314,133, kém A4 0,29% và 0,27%.
    Ở slot đầu J₃₀ còn cao hơn SEM 0,07%, nên nhánh chính thấp hơn SEM ở 74/75 cặp, còn A4 là 75/75.
- **Phân rã Time của nhánh chính:** 0,419 s = dựng policy và potential 0,056 + nghiệm mồi 0,047 + pha BIT* 0,316.
  `rpRefine` chiếm tổng cộng 0,132 s (31%).
- **W = 1 000** (thăm dò): 0,211 s, nhanh hơn SEM ở 75/75 cặp. J₃₀ cao hơn A4 0,059%, cận trên +0,106%, vượt nhẹ
  ngưỡng 0,1%; 9 slot kém A4 quá 0,1%; đường cuối là nghiệm mồi ở 18/75 lượt. So với A4 M = 5, cùng cửa sổ 1 000
  mẫu: nhanh hơn 0,237 s [0,201], J₃₀ +0,049% [+0,094%].

### 6.7 Hồi quy

- Đường cuối của `a4_rpbit_l33_m20` và `sem_native` trùng từng byte với cùng slot trong
  `results/a4_rpbit_selfstop_dev_20261117/dev_selfstop`: 75/75 và 75/75.
- Sau lượt dev, 7 file lõi của A4 và hai binary `build-a4-rpbit` vẫn trùng hash với lock
  (`logs/a4_rpbit_a5_20261121/lock_rehash_after_dev.json`).
- `ctest -L rp_bit` 6/6 và 28/28 nhóm của các nhãn còn lại đạt trên `build-a5`.

### 6.8 Tiền xử lý: trao đổi trường rủi ro dạng nhị phân (dev lần 2)

Ở lượt dev đầu, phần trước đồng hồ solve của A5 là 508 ms, dài hơn chính thời gian solve (419 ms). Người dùng yêu
cầu giảm phần này ngày 21/11/2026. Đo từng bước trên `rolling_terrain_41002` trước khi sửa:

| Bước | Thời gian |
|---|---:|
| Python định dạng 160 000 dòng văn bản của trường rủi ro | 114 ms |
| C++ đọc lại file văn bản đó (`SafetyField::load`) | 139 ms |
| Import `scipy.ndimage` trong tiến trình mới | 63 ms |
| Dựng trường rủi ro (`build_field`) | 62 ms |
| Băm point cloud 16 MB và file bản đồ, mỗi thứ ba lần | 44 ms |
| C++ nạp bản đồ (`Terrain::load`) | 33–47 ms |

Một nửa thời gian là để chuyển trường rủi ro từ Python sang C++ qua file văn bản; phần tính toán thật chỉ khoảng 62 ms.

**Thay đổi, chỉ áp dụng cho kind `rp_bit_a5`:**

- `tools/rp_bit_a5_field.py` ghi trường rủi ro vào container nhị phân `SAFE_SEM_FIELD_B1` (file
  `planner_sem_field.bin`): header 56 byte, rồi 1 byte support và 8 byte risk cho mỗi ô. Trường vẫn do
  `safe_length_field.build_field` dựng; bộ dựng không đổi.
- Point cloud được đọc một lần và băm trên đúng các byte đã dùng để dựng trường, thay cho ba lần băm file.
- `rpA5LoadField` đọc container; file văn bản vẫn đọc được qua `SafetyField::load`. `SafetyField` kiểm mọi ô như cũ.
- Ở bước này mốc bấm giờ chưa đổi: import `scipy` vẫn nằm trong thời gian dựng trường.
- Ở bước này A4 và SEM-RRT* giữ nguyên đường cũ; `src/Safety.cpp`, `src/Terrain.cpp` và `tools/safe_length_field.py`
  không đổi. Mục 6.9 mới đổi mốc bấm giờ và `src/Terrain.cpp`.

**Kiểm chứng:**

- Trên 15 bản đồ dev (2 400 000 ô), container nhị phân cho đúng từng bit giá trị mà `SafetyField::load` đọc từ file
  văn bản.
- Dev lần 2 (`results/a4_rpbit_a5_dev_r2_20261121`, binary `03da8ea6…`): 600/600 thành công, 0 lỗi correctness.
  Đường cuối trùng từng byte với lượt dev đầu ở 75/75 slot của cả 8 nhánh; số vòng lặp và J cuối cũng trùng.
- Smoke qua runner thật (`results/a4_rpbit_a5_smoke_r2_20261121`): 55/56 như trước, 56/56 đường cuối trùng lần smoke
  trước.
- `ctest -L rp_bit` 6/6 và 35/35 nhóm của các nhãn còn lại (thêm `pri_portfolio`) đạt trên binary mới. Bảy file lõi
  của A4 và hai binary đã khóa vẫn trùng hash (`logs/a4_rpbit_a5_20261121/lock_rehash_after_dev_r2.json`).

**Kết quả, trung bình 75 slot:**

| | Lượt dev đầu | Dev lần 2 |
|---|---:|---:|
| A5: dựng và ghi trường rủi ro (Python) | 316 ms | 160 ms |
| A5: worker nạp bản đồ và trường rủi ro | 192 ms | 40 ms |
| **A5: tổng trước đồng hồ solve** | 508 ms | **200 ms** |
| SEM-RRT*: tổng trước đồng hồ solve | 412 ms | 412 ms |
| A4: tổng trước đồng hồ solve | 515 ms | 513 ms |
| A5 chính: solve cộng phần trước solve | 926 ms | 620 ms |
| SEM-RRT*: solve cộng phần trước solve | 3 038 ms | 3 044 ms |
| SEM / A5, gồm phần trước solve | 3,28 | **4,91** |
| Cặp A5 nhanh hơn, gồm phần trước solve | 74/75 | 75/75 |
| SEM / A5, chỉ solve | 6,28 | 6,26 |

Gồm phần trước solve, tỉ số theo họ ở dev lần 2 là: `traversable_hill` 3,19; `narrow_passage` 3,85;
`ridge_pass` 4,41; `mixed_obstacles` 5,55; `rolling_terrain` 7,29.

200 ms còn lại của A5 gồm: import `scipy` khoảng 63 ms, dựng trường khoảng 60 ms, nạp bản đồ khoảng 40 ms, đọc và băm
point cloud khoảng 20 ms, ghi file dưới 10 ms.

**Khi so phần trước solve giữa hai pipeline, cần biết ba điều** (đo riêng trên cùng bản đồ; đều có từ trước, không do
thay đổi ở mục này):

- Phần trước solve của SEM-RRT* gồm import cả `numpy` lẫn `scipy` (khoảng 127 ms). Của A5 chỉ gồm `scipy` (63 ms), vì
  `numpy` đã được nạp trước khi đồng hồ của A5 bắt đầu.
- SEM-RRT* nạp file bản đồ hai lần (khoảng 75 ms), A5 một lần.
- Nếu bỏ import ở cả hai phía và tính SEM nạp bản đồ một lần, tỉ số gồm phần trước solve là khoảng 5,2. Nếu chỉ bỏ
  hai khoản dư của SEM, tỉ số là khoảng 4,75. Kết luận ít phụ thuộc cách tính, vì Time của SEM chủ yếu là 2,6 s solve.

Mục 6.9 xử lý cả ba điều này.

### 6.9 `Terrain::load` nhanh hơn và đồng hồ không tính khởi động tiến trình (dev lần 3)

Người dùng duyệt hai thay đổi vào mã dùng chung ngày 21/11/2026. Cả hai có hiệu lực với mọi pipeline, không riêng A5.

**`Terrain::load` (`src/Terrain.cpp`):**

- File được đọc một lần vào bộ nhớ, tách token theo khoảng trắng, đổi số bằng `std::from_chars`. Trước đó mỗi số đi
  qua `operator>>` và `std::stod`.
- Dạng số mà `std::from_chars` không nhận (dấu cộng, số hex), số dưới chuẩn và số ngoài miền vẫn chuyển cho
  `std::stod`. Tập file được chấp nhận và giá trị đọc ra vì vậy không đổi.
- `std::from_chars` và `std::stod` cho cùng từng bit trên 6 669 878 số của 167 file `.t25d` trong dự án và trên
  5 994 239 chuỗi ngẫu nhiên.
- Với cả 167 file đó, mọi thứ quan sát được qua API (lưới, biên, bán kính, độ cao tại mọi nút, tính hợp lệ tại tâm
  mọi ô) trùng nhau giữa bản cũ và bản mới.
- Trên bản đồ dev (201 × 201 nút): 33 ms → 3 ms.

**Đồng hồ trước solve không còn tính khởi động tiến trình:**

- `tools/run_sem_exposure.py`: `scipy` được nạp trước khi bấm giờ dựng trường rủi ro. Áp dụng cho mọi kind chạy
  worker C++, gồm cả A4.
- `tools/native_paper_worker.py` (worker của SEM-RRT* và PUTN): `numpy`, `scipy` và các module của worker được nạp
  trước khi đồng hồ chạy.
- Cũng trong worker đó, `platform.platform()` được gọi trước khi đồng hồ chạy. Lần gọi đầu mất khoảng 70 ms trên
  Windows. Nó không phải import, nhưng cùng là chi phí khởi động và chỉ SEM-RRT* bị tính; phát hiện khi đo smoke.
- Phần solve không đổi.

**Kiểm chứng:**

- Toàn bộ 127 nhóm CTest: 125 đạt ở lần chạy đầu. `a3_pilot_campaign` lỗi một lần và kéo theo `planner_pipeline`
  không chạy; chạy lại thì cả hai đạt, và chạy riêng 6 lần (3 với binary mới, 3 với binary cũ) đều đạt. Test đó chập
  chờn từ trước: nó so `(28800 + t) − t` với 28800, và phép tính có thể ra nhỏ hơn 28800 do làm tròn khi đồng hồ chưa
  kịp nhảy. Test này không bị sửa.
- Sau chỉnh sửa cuối, 46/46 nhóm của các nhãn dùng worker (`sem_cpp`, `rp_bit`, `native_paper_v3`, `native_paper_v4`,
  `sem_exposure`) đạt.
- Smoke với mã cuối (`results/a4_rpbit_a5_smoke_r4_20261121`): 55/56, 0 lỗi correctness. Các lượt hội tụ trùng đường
  với các lần smoke trước; một lượt SEM bị cắt bởi trần 2 s kết thúc ở đường khác vì số vòng lặp phụ thuộc thời gian.
- Dev lần 3 (`results/a4_rpbit_a5_dev_r3_20261121`, binary `e58db889…`): 600/600 thành công, 0 lỗi correctness. Đường
  cuối trùng từng byte với dev lần 2 ở 75/75 slot của cả 8 nhánh; số vòng lặp và J cuối cũng trùng.
- Bảy file lõi của A4 và hai binary đã khóa vẫn trùng hash (`logs/a4_rpbit_a5_20261121/lock_rehash_after_dev_r3.json`).

**Kết quả, trung bình 75 slot:**

| | Lượt dev đầu | Dev lần 2 | Dev lần 3 |
|---|---:|---:|---:|
| SEM-RRT*: phần trước solve | 412 ms | 412 ms | 121 ms |
| A4 chính: phần trước solve | 515 ms | 513 ms | 418 ms |
| **A5 chính: phần trước solve** | 508 ms | 200 ms | **102 ms** |
| SEM-RRT*: solve cộng phần trước solve | 3 038 ms | 3 044 ms | 2 760 ms |
| A5 chính: solve cộng phần trước solve | 926 ms | 620 ms | 524 ms |
| SEM / A5, gồm phần trước solve | 3,28 | 4,91 | **5,26** |
| Cặp A5 nhanh hơn, gồm phần trước solve | 74/75 | 75/75 | 74/75 |
| SEM / A5, chỉ solve | 6,28 | 6,26 | 6,25 |

- Gồm phần trước solve, tỉ số theo họ ở dev lần 3 là: `traversable_hill` 3,25; `narrow_passage` 3,99;
  `ridge_pass` 4,63; `mixed_obstacles` 6,14; `rolling_terrain` 7,99.
- A4 còn 418 ms vì vẫn trao đổi trường rủi ro qua file văn bản; mục 6.8 chỉ đổi cho A5.
- 102 ms của A5 gồm: dựng trường rủi ro khoảng 60 ms, đọc và băm point cloud khoảng 20 ms, ghi file và việc khác
  khoảng 18 ms, worker nạp bản đồ và trường rủi ro 4 ms.
- Số "tiền xử lý" của SEM-RRT* và A4 trong mọi báo cáo trước ngày này còn gồm các khoản khởi động nói trên, nên không
  so trực tiếp được với các lượt sau.

### 6.10 Đăng ký held-out trên Kaggle (22/11/2026; đã chạy, kết quả ở 6.11)

Biên bản: `logs/a4_rpbit_a5_heldout_20261122/registration.json`, cấu hình `configs/a4_rpbitstar_v1_a5_kaggle.json`.
Quyết định của người dùng: giữ A5 đầy đủ làm nhánh chính, trần 20 s, chạy trên Kaggle, Time là thời gian solve còn
phần trước solve chỉ báo kèm; giữ nhánh C1 + C2; tiêu chí 1 không đòi PSR 250/250; giữ ngưỡng tốc độ 5.

| Mục | Nội dung |
|---|---|
| Nhánh | `sem_native`, `a4_rpbit_l33_m20`, `a5_rpbit_l33_w4000` (chính), `a5_rpbit_l33_w4000_eager` (C1 + C2) |
| Đầu vào | 5 họ × seed bản đồ 111001–111010 × planner seed 1116001–1116005 = 250 slot; chưa dùng ở đâu trước đó |
| Pha | `smoke_a5_kaggle` 28, `gate_a5_kaggle` 20 (seed 51001), `heldout_a5_kaggle` 1 000 lượt |
| Thực thi | Kaggle, 1 làn trên CPU 0, mỗi lúc một lượt; validate trên 4 lõi |

Bốn tiêu chí, chấm trên nhánh chính bằng `tools/analyze_rp_bit.py` và `tools/analyze_rp_bit_a5.py`:

1. Đúng đắn: đủ verdict, 0 lỗi correctness, nhánh chính được validate nghiêm ngặt. PSR chỉ báo, được chấm ở 2a.
2. a) Đủ 10 tiêu chí mục 5.3 so với SEM-RRT*; b) so với A4: cận trên 95% của ΔJ₃₀ gộp tương đối ≤ +0,1%.
3. Tốc độ: Time trung bình SEM / A5 ≥ 5 và A5 nhanh hơn ở ≥ 90% số cặp (chỉ thời gian solve).

`tools/analyze_rp_bit_a5.py` chạy lại trên dev lần 3 cho đúng mọi số của script chẩn đoán dev (0 sai khác) và kết luận
"đạt cả bốn". Biên bản ghi hash của cấu hình, từng pha và 26 file nguồn; bộ điều khiển Kaggle (`kaggle_a4_rpbit/a5.py`)
từ chối chạy nếu một hash lệch. Dự đoán ghi trước: tiêu chí 3 có thể sát ngưỡng, vì trên các held-out Kaggle trước tỉ
số SEM / A4 thấp hơn dev (1,57 và 1,49 so với 1,90); nếu A5 hụt cùng tỉ lệ, tỉ số khoảng 5,1–5,2.

### 6.11 Kết quả held-out trên Kaggle (22/11/2026)

**Lượt chạy và kiểm chứng.**
- Kernel `notebook48a9ab731c`, Xeon 2,2 GHz. Một phiên từ 18:45 đến 21:51 UTC ngày 21/11 (22/11 giờ Việt Nam), khoảng
  3,1 giờ. Thu thập chia 2 đoạn trong cùng phiên; đoạn sau chỉ chạy 313 lượt chưa bắt đầu.
- Cổng trên Linux: 34/34 nhóm CTest. `smoke_a5_kaggle` thành công 27/28 (lượt hỏng là SEM-RRT* trên
  `nodata_thin_obstacle`, giống diễn tập Windows) và `gate_a5_kaggle` 20/20, mọi lượt A4 và A5 đều thành công, 0 lỗi
  correctness. SEM Python và C++ trùng từng bit 10/10.
- Held-out: 1 000/1 000 lượt, mỗi lượt bắt đầu đúng một lần, tất cả trên CPU 0 và không lúc nào hai lượt chạy cùng
  nhau; 0 lỗi correctness. Validate nghiêm ngặt mọi snapshot của nhánh chính: 250/250 đạt.
- Biên bản thay đổi ghi lúc 19:17:08, trước lượt held-out đầu tiên (19:20:30). Hash của biên bản đăng ký
  (`649343fd…`), của config, từng pha và binary Linux trùng nhau ở mọi chỗ ghi; 26 file nguồn đã đăng ký trùng với cây
  làm việc.
- Báo cáo dựng lại trên máy cục bộ bằng `tools/analyze_rp_bit.py` và `tools/analyze_rp_bit_a5.py` trùng báo cáo Kaggle
  ở mọi khóa, chỉ khác đường dẫn thư mục (`logs/a4_rpbit_a5_heldout_20261122/heldout_receipt.json`).
- Báo cáo đầy đủ ở `reports/a4_rpbit_a5_kaggle_20261122/heldout_a5_kaggle/` (`REPORT.md`, `A5_VERDICT.md`,
  `TABLE2_STYLE.md`); verdict ở `results/a4_rpbit_a5_kaggle_20261122/`. Kho trial thô 2,65 GB chỉ nằm trong
  `C:\Users\Long\Downloads\notebook48a9ab731c_output`.

**Bốn tiêu chí đã đăng ký** (nhánh chính `a5_rpbit_l33_w4000`; trong ngoặc là cận trên một phía 95%):

| Tiêu chí | Số liệu | Kết luận |
|---|---|---|
| 1. Đúng đắn | đủ 1 000 verdict, 0 lỗi correctness, validate nghiêm ngặt 250/250, PSR 250/250 | đạt |
| 2a. Mười tiêu chí mục 5.3 so với SEM-RRT* | 10/10 | đạt |
| 2b. J₃₀ không kém A4 quá 0,1% | gộp −0,003% [+0,008%] | đạt |
| 3. Tốc độ | SEM / A5 = 4,895 / 0,790 = **6,20** (ngưỡng 5); A5 nhanh hơn ở 249/250 cặp (ngưỡng 90%) | đạt |

**Thời gian** (giây, trung bình 250 lượt; Time là thời gian solve, phần trước solve chỉ báo kèm):

| Nhánh | Solve | Trước solve | Tổng | SEM / nhánh (solve) | SEM / nhánh (tổng) | Nhanh hơn SEM (solve) |
|---|---:|---:|---:|---:|---:|---:|
| `sem_native` | 4,895 | 0,238 | 5,133 | — | — | — |
| `a4_rpbit_l33_m20` | 3,003 | 0,668 | 3,671 | 1,63 | 1,40 | 165/250 |
| **`a5_rpbit_l33_w4000`** | **0,790** | 0,246 | **1,036** | **6,20** | **4,95** | **249/250** |
| `a5_rpbit_l33_w4000_eager` (C1 + C2) | 1,025 | 0,244 | 1,269 | 4,78 | 4,04 | 245/250 |

Theo họ, SEM / A5 (solve): `mixed_obstacles` 8,26; `rolling_terrain` 7,73; `narrow_passage` 5,16;
`traversable_hill` 4,78; `ridge_pass` 4,14. A5 nhanh hơn SEM ở mọi cặp trừ một (`ridge_pass_111010`, seed 1116004:
1,76 s so với 1,12 s).

**Bảng theo kiểu Table II của bài SEM-RRT*** (IROS 2025). PSR = số lượt thành công / số lượt; APL, PRI, STI và Time là
trung bình ± độ lệch chuẩn (quần thể) trên các đường hợp lệ. STI ở bảng này tính cho từng đường (số góc gấp > 30° /
L_H) rồi lấy trung bình, nên khác STI gộp (Σ góc gấp / Σ L_H) của `REPORT.md`. Time là thời gian solve (ms). In đậm là
giá trị tốt nhất trong từng nhóm, so trên số chưa làm tròn. Mỗi họ có 50 lượt (10 bản đồ × 5 planner seed), khác bài
gốc (100 lượt cho một bài toán).

| Scenario | Method | PSR ↑ | APL[m] ↓ | PRI ↓ | STI ↓ | Time[ms] ↓ |
|---|---|---:|---:|---:|---:|---:|
| traversable_hill | SEM-RRT* | **100%** (50/50) | 34,29 ± 1,26 | 1,04 ± 0,44 | 0,010 ± 0,025 | 2984 ± 1422 |
|  | A4 / RP-BIT* | **100%** (50/50) | **34,24 ± 1,25** | **1,04 ± 0,44** | **0,000 ± 0,000** | 2011 ± 288 |
|  | A5 chỉ C1 + C2 | **100%** (50/50) | 34,24 ± 1,25 | 1,04 ± 0,44 | **0,000 ± 0,000** | 771 ± 79 |
|  | A5 / RP-BIT* (chính) | **100%** (50/50) | 34,24 ± 1,25 | 1,04 ± 0,44 | **0,000 ± 0,000** | **624 ± 118** |
| ridge_pass | SEM-RRT* | **100%** (50/50) | 38,93 ± 1,82 | **8,62 ± 1,98** | 0,079 ± 0,047 | 2889 ± 1581 |
|  | A4 / RP-BIT* | **100%** (50/50) | 38,68 ± 1,87 | 8,64 ± 1,99 | 0,009 ± 0,015 | 2761 ± 613 |
|  | A5 chỉ C1 + C2 | **100%** (50/50) | 38,80 ± 2,21 | 8,62 ± 1,98 | **0,009 ± 0,014** | 1037 ± 287 |
|  | A5 / RP-BIT* (chính) | **100%** (50/50) | **38,67 ± 1,91** | 8,64 ± 1,97 | 0,012 ± 0,018 | **698 ± 228** |
| rolling_terrain | SEM-RRT* | **100%** (50/50) | 36,01 ± 1,05 | 5,12 ± 1,72 | 0,109 ± 0,087 | 6135 ± 4397 |
|  | A4 / RP-BIT* | **100%** (50/50) | 35,77 ± 1,02 | **5,11 ± 1,71** | 0,004 ± 0,012 | 3297 ± 702 |
|  | A5 chỉ C1 + C2 | **100%** (50/50) | **35,76 ± 1,01** | 5,12 ± 1,71 | 0,004 ± 0,013 | 1037 ± 182 |
|  | A5 / RP-BIT* (chính) | **100%** (50/50) | 35,78 ± 1,00 | 5,12 ± 1,71 | **0,003 ± 0,009** | **794 ± 75** |
| narrow_passage | SEM-RRT* | **100%** (50/50) | 36,55 ± 2,13 | 2,77 ± 0,88 | 0,074 ± 0,063 | 4474 ± 2757 |
|  | A4 / RP-BIT* | **100%** (50/50) | 36,37 ± 2,13 | 2,77 ± 0,88 | **0,000 ± 0,000** | 3326 ± 741 |
|  | A5 chỉ C1 + C2 | **100%** (50/50) | 36,45 ± 2,04 | **2,76 ± 0,89** | **0,000 ± 0,000** | 1092 ± 377 |
|  | A5 / RP-BIT* (chính) | **100%** (50/50) | **36,26 ± 2,26** | 2,77 ± 0,88 | **0,000 ± 0,000** | **867 ± 201** |
| mixed_obstacles | SEM-RRT* | **100%** (50/50) | 41,04 ± 6,79 | 4,19 ± 0,40 | 0,109 ± 0,077 | 7992 ± 5038 |
|  | A4 / RP-BIT* | **100%** (50/50) | **40,82 ± 6,76** | 4,18 ± 0,39 | **0,003 ± 0,008** | 3620 ± 816 |
|  | A5 chỉ C1 + C2 | **100%** (50/50) | 40,87 ± 6,80 | 4,17 ± 0,39 | 0,005 ± 0,010 | 1188 ± 333 |
|  | A5 / RP-BIT* (chính) | **100%** (50/50) | 40,89 ± 6,75 | **4,17 ± 0,39** | 0,006 ± 0,010 | **968 ± 362** |
| Tất cả (250 cặp) | SEM-RRT* | **100%** (250/250) | 37,36 ± 4,11 | 4,35 ± 2,84 | 0,076 ± 0,073 | 4895 ± 3894 |
|  | A4 / RP-BIT* | **100%** (250/250) | 37,18 ± 4,08 | 4,35 ± 2,85 | **0,003 ± 0,010** | 3003 ± 869 |
|  | A5 chỉ C1 + C2 | **100%** (250/250) | 37,23 ± 4,13 | **4,34 ± 2,84** | 0,004 ± 0,010 | 1025 ± 307 |
|  | A5 / RP-BIT* (chính) | **100%** (250/250) | **37,17 ± 4,11** | 4,35 ± 2,84 | 0,004 ± 0,011 | **790 ± 252** |

**So với kỳ vọng ghi trước.**
- Tiêu chí 1, 2a và 2b đạt như kỳ vọng. Tiêu chí 3 tốt hơn kỳ vọng: 6,20 so với dự đoán 5,1–5,2.
- Theo họ: `traversable_hill` dưới 5 như dự đoán; `narrow_passage` trên 5 (dự đoán dưới 5); `ridge_pass` dưới 5 (không
  được dự đoán).
- Tính cả phần trước solve, tỉ số là 4,95 (chỉ báo kèm, không chấm; dev lần 3: 5,26).
- Nhánh eager chậm hơn nhánh chính (1,025 so với 0,790 s) với J₃₀ gần bằng A4 (+0,035%), như dự đoán.

**Đọc kết quả.**
- **Tỉ số 6,20 là tỉ số của hai trung bình** (đúng tiêu chí đã đăng ký) và chịu ảnh hưởng của các lượt SEM dài. Tỉ số
  hai trung vị là 4,92, trung bình nhân của tỉ số từng cặp là 5,00, bỏ 25 lượt SEM dài nhất còn 4,94. Hai lượt SEM chạm
  trần 20 s (`mixed_obstacles`) làm SEM có lợi; bỏ hai cặp đó tỉ số là 6,07.
- **Chất lượng so với SEM:** APL −0,52% [−0,38%], PRI −0,03% [+0,21%], ΔJ₃₀ −0,92 [−0,78], STI gộp 0,0047 so với 0,078.
  **So với A4:** J₃₀ −0,003%; 1 slot kém hơn 0,1% (+0,102%, `ridge_pass_111009` seed 1116001), 4 slot tốt hơn 0,1%.
- **Nghiệm mồi (C4):** được công bố ở 245/250 lượt; 5 lượt nghiệm mồi không an toàn, đều ở `ridge_pass`. Đường cuối
  chính là nghiệm mồi ở 47/250 lượt (dev: 6/75). BIT* cải thiện nghiệm mồi trung bình 0,048%, nhiều nhất 0,59%. Nghiệm
  đầu tiên của A5 công bố lúc khoảng 0,227 s và chỉ cách J₃₀ cuối trung bình 0,06%; với A4 và nhánh eager, 92 và 90
  lượt có nghiệm đầu kém J₃₀ cuối hơn 1%.
- **C3 + C4 đóng góp nhiều hơn trên dev:** từ nhánh eager sang nhánh chính, Time giảm 0,235 s (23%), so với khoảng
  0,08 s trên dev. Không có C3 + C4, tỉ số chỉ là 4,78.
- **Chia thời gian solve của nhánh chính** (trung bình): setup 0,128 s, `rpRefine` 0,253 s (gồm tinh chỉnh nghiệm mồi),
  phần còn lại 0,409 s (BIT* và A* trên lưới của nghiệm mồi); riêng nghiệm mồi 0,096 s.
- **Anytime:** tại 0,1 s chưa nhánh RP-BIT* nào có đường, còn SEM có 250/250. Tại 0,25 s A5 có đường an toàn ở 220/250
  lượt; từ đó trở đi, trên các cặp đều có đường, J₃₀ của A5 thấp hơn SEM ở mọi checkpoint (cận trên đều âm).
- A4 trên lượt này: SEM / A4 = 1,63 (held-out selfstop 1,57, PUTN 1,49).
- Held-out này đã dùng: không chạy lại, không thêm seed, không gộp với các held-out trước; Time trên Kaggle không so
  với số đo trên Windows.

### 6.12 Đăng ký held-out thứ hai trên Kaggle: chỉ A5 và SEM-RRT*, trần 10 s (22/11/2026; chưa chạy)

Biên bản: `logs/a4_rpbit_a5_10s_20261122/registration.json`, cấu hình `configs/a4_rpbitstar_v1_a5_10s_kaggle.json`.
Người dùng yêu cầu sau khi xem kết quả 6.11. Họ chọn trần 10 s, chỉ chạy hai phương pháp, dùng seed mới, và giữ
tiêu chí 1, 2a, 3 với ngưỡng cũ. Không chạy A4 nên không chấm 2b.

| Mục | Nội dung |
|---|---|
| Nhánh | `sem_native`, `a5_rpbit_l33_w4000` (chính); tham số y như 6.10 |
| Đầu vào | 5 họ × seed bản đồ 121001–121010 × planner seed 1216001–1216005 = 250 slot; chưa dùng ở đâu trước đó |
| Pha | `smoke_a5_10s` 14, `gate_a5_10s` 10 (seed 51001, 10 s), `heldout_a5_10s` 500 lượt |
| Thực thi | Kaggle, 1 làn trên CPU 0, mỗi lúc một lượt; validate trên 4 lõi; 34 nhóm CTest như 6.10 |
| Chấm | `tools/analyze_rp_bit.py` và `tools/analyze_rp_bit_a5_sem.py`: tiêu chí 1, 2a, 3 với hàm và ngưỡng của `tools/analyze_rp_bit_a5.py` |

Trên dữ liệu 20 s của mục 6.11, công cụ chấm mới cho đúng các tiêu chí 1, 2a, 3 và mục thời gian của SEM và A5 như
công cụ cũ. Biên bản ghi hash của cấu hình, từng pha và 27 file nguồn: 26 file của 6.10, không đổi, cộng công cụ
chấm mới. Bộ điều khiển là `kaggle_a4_rpbit/a5_10s.py`.

Dự đoán ghi trước, tính từ 6.11:
- A5 không chạy quá 2,7 s, nên trần 10 s không ảnh hưởng A5.
- Có 29/250 lượt SEM chạy quá 10 s. Nếu cắt các lượt này ở 10 s, tỉ số SEM / A5 giảm từ 6,20 xuống 5,62.
- Chất lượng của SEM ở 10 s bằng ở 20 s.
- Kiểm tra dễ trượt nhất trong 2a là `pri_target_lower`. Ở 20 s, chênh lệch PRI gộp là −0,03%, khoảng tin cậy 95% là
  [−0,31%; +0,26%].

## 7. Hạn chế đã biết

- **C3, C4 được thiết kế lại sau khi xem probe trên bản đồ dev.** Held-out (mục 6.11) là lần thử đầu tiên của chúng
  trên bản đồ chưa thấy, và nhánh chính đạt cả bốn tiêu chí. Các số ở mục 6.1–6.9 vẫn chỉ từ bản đồ đã thấy.
- **Mục tiêu 7–10 lần không đạt.** Tỉ số gộp là 6,28 trên dev và 6,20 trên held-out. Có hai họ dưới 5: trên dev là
  `traversable_hill` 3,91 và `narrow_passage` 4,58; trên held-out là `traversable_hill` 4,78 và `ridge_pass` 4,14.
- **Tỉ số 6,28 chỉ tính thời gian solve.** Tính cả phần trước đồng hồ solve, tỉ số là 5,26 sau mục 6.8 và 6.9; ở
  lượt dev đầu là 3,28. Phần trước solve vẫn gồm băm file để kiểm toàn vẹn ở cả hai pipeline. Trên held-out: 6,20 chỉ
  tính solve, 4,95 tính cả phần trước solve.
- **Tỉ số giữa hai trung bình chịu ảnh hưởng của các lượt SEM dài.** Trên held-out, tỉ số hai trung vị là 4,92 và
  trung bình nhân của tỉ số từng cặp là 5,00 (mục 6.11).
- **A4 với M = 5 đã đạt 0,448 s** nhờ cửa sổ ngắn 1 000 mẫu; A5 chính chỉ nhanh hơn nó ở 48/75 cặp. Lợi ích của A5
  chính là đạt thời gian đó mà vẫn giữ cửa sổ kiểm chứng 4 000 mẫu, đối xứng với luật dừng của SEM.
- **C2 phụ thuộc địa hình.** Trên dev mọi cạnh đều được chứng nhận vì địa hình thoải so với trần 35°. Địa hình dốc
  hơn được lợi ít hơn; quyết định hợp lệ thì không đổi.
- **Nghiệm mồi không phải là BIT\*.** Nó là A* trên lưới cộng `rpRefine`, được ghi nguồn riêng trong mọi bản ghi.
  Nhánh `…_noseed` cho thấy BIT* tự đạt cùng J cuối khi không có nó. Trên held-out, đường cuối chính là nghiệm mồi ở
  47/250 lượt, và BIT* cải thiện nghiệm mồi trung bình 0,048%.
- **Góc gấp:** 5 trên 75 đường, so với 1 của A4 chính và 229 của SEM.
- **Phần còn lại của Time:** `rpRefine` 31%, dựng potential 13%. Muốn nhanh hơn nữa với W = 4 000 cần cơ chế mới,
  nằm ngoài kế hoạch đã duyệt.
