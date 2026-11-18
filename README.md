# BIT* 2.5D — A4 / RP-BIT*

A4 / RP-BIT* (`a4_rpbitstar_v1`) là planner tìm đường cho robot trên bản đồ địa hình 2.5D (z = H(x, y)), có xét
rủi ro địa hình. Lõi là `ompl::geometric::BITstar` thật của OMPL 1.7.0, không patch. A4 thêm hàm mục tiêu
chiều dài + rủi ro, cận dưới "biết rủi ro" (risk potential), lấy mẫu informed theo rủi ro, luật dừng khi hội tụ và
một bước tinh chỉnh đường có ablation. Baseline so sánh là SEM-RRT* (bản tái dựng Python, kèm bản C++ trùng từng bit).

Mô tả thuật toán, tham số, tiêu chí đã đăng ký và toàn bộ kết quả: [docs/A4_RP_BITSTAR.md](docs/A4_RP_BITSTAR.md).

## Kết quả chính

Held-out đã đăng ký trước, chạy trên Kaggle 4 vCPU (16/11/2026): 50 bản đồ mới × 5 seed, 1 500 lượt, 0 lỗi
correctness. Cấu hình chính đã khóa (λ = 33, M = 20) đạt mọi tiêu chí đã đăng ký so với SEM-RRT* 7 425 vòng lặp:

| Phương pháp | PSR | APL (m) | PRI | STI (1/m) | Time (s) | J₃₀ |
|---|---:|---:|---:|---:|---:|---:|
| SEM-RRT* | 249/250 | 39,694 | 3,750 | 0,0245 | 90,2 | 154,63 |
| A4 chính | 250/250 | 39,515 | 3,745 | 0,0015 | 5,37 | 153,61 |

Cách đọc bảng, ablation, các thí nghiệm cùng thời gian với SEM-RRT* bản C++ và các hạn chế: mục 6–7 của
[docs/A4_RP_BITSTAR.md](docs/A4_RP_BITSTAR.md). Dữ liệu thô và báo cáo đầy đủ không nằm trong repo này.

## Repo này có gì

- Mã C++17 (`include/`, `src/`, `apps/`), CMake (`CMakeLists.txt`, `cmake/`) và hai bản vá OMPL (`patches/`).
- Runner, validator độc lập và công cụ phân tích bằng Python (`tools/`), test (`tests/`), cấu hình (`configs/`).
- Bộ điều khiển Kaggle của A4 (`kaggle_a4_rpbit/`) và vài helper dùng chung.
- Lock của cấu hình chính và biên bản đăng ký các thí nghiệm (`logs/`). Code A4 đọc các tệp này, đừng sửa chúng.

Không có trong repo: dependency (tải bằng script bên dưới), bản đồ dev, kết quả thô. Vì vậy các pha `dev_*`
không chạy được từ bản clone. Pha `smoke` và các pha held-out tự sinh bản đồ từ seed.

## Build trên Linux x86_64

Đây là cách build đã chạy trên Kaggle (Linux x86_64). Cần `g++`, `gcc`, `cmake`, `ninja`, `bash` và Python 3.12
(Kaggle dùng 3.12.7).

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements-native-paper-v4.txt   # numpy, scipy, matplotlib, PyYAML, psutil

python tools/fetch_dependencies.py              # tải OMPL 1.7.0, Eigen 3.4.0, Boost 1.82.0 vào .deps/, kiểm SHA-256
python tools/build_dependencies.py --jobs 4     # build Boost, Eigen, OMPL (+ 2 bản vá) vào .deps/install

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_PREFIX_PATH="$PWD/.deps/install" \
  -DBOOST_ROOT="$PWD/.deps/boost_1_82_0" -DBOOST_LIBRARYDIR="$PWD/.deps/boost_1_82_0/stage/lib" \
  -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON -DBoost_USE_STATIC_LIBS=ON \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
cmake --build build --parallel 4
```

`fetch_dependencies.py` dừng lại nếu archive tải về không khớp SHA-256 ghim trong `configs/dependencies.lock.json`.
Build Boost và OMPL mất khá lâu (cỡ vài chục phút với 4 lõi).

## Build trên Windows

Đã dùng MinGW-w64 g++ và Ninja đặt tại `C:/mingw64/bin`. Một số test trong `cmake/` ghi cứng đường dẫn này.
Hãy clone vào một đường dẫn ngắn, ví dụ `C:\src\bit25d`. Một số test tạo thư mục tạm khá sâu, và nếu clone nằm
sâu (ví dụ dưới `AppData\Local\Temp\...`) thì đường dẫn vượt giới hạn 260 ký tự của Windows làm test thất bại.
Cần Python 3.12 với `numpy==1.26.4 scipy==1.11.4 matplotlib==3.10.7 PyYAML==6.0.2 psutil==5.9.8`. Tệp lock
`requirements-*.txt` chỉ có hash cho wheel Linux.

```powershell
python tools/fetch_dependencies.py
python tools/build_dependencies.py --jobs 4
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CXX_COMPILER=C:/mingw64/bin/g++.exe' `
  '-DCMAKE_MAKE_PROGRAM=C:/mingw64/bin/ninja.exe' "-DCMAKE_PREFIX_PATH=$PWD/.deps/install" `
  "-DBOOST_ROOT=$PWD/.deps/boost_1_82_0" "-DBOOST_LIBRARYDIR=$PWD/.deps/boost_1_82_0/stage/lib" `
  -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON -DBoost_USE_STATIC_LIBS=ON `
  "-DPython3_EXECUTABLE=$((Get-Command python).Source)"
cmake --build build --parallel 4
```

## Kiểm thử

```bash
ctest --test-dir build -L rp_bit --output-on-failure -j 1
ctest --test-dir build -L "sem_exposure|pri_segment|pri_pathsearch|safe_length" --output-on-failure -j 1
```

## Chạy thử (smoke)

Pha smoke gồm 7 fixture, mỗi lượt 2 s. Lệnh dưới đây validate nghiêm ngặt mọi snapshot, rồi tạo báo cáo:

```bash
python -B tools/run_rp_bit.py run --phase smoke --output results/a4_smoke \
  --worker build/terrain_benchmark --bridge build/libterrain_compare_c.so --full-validation
python -B tools/analyze_rp_bit.py --input results/a4_smoke/smoke --output reports/a4_smoke
```

Trên Windows dùng `build/terrain_benchmark.exe` và `build/libterrain_compare_c.dll`.

Runner chạy mỗi lượt trong một tiến trình riêng và không ghi đè thư mục output đã có. Lượt đã bắt đầu thì không
bao giờ chạy lại. Pha có tên bắt đầu bằng `heldout` chỉ chạy khi có `--allow-heldout`. Các held-out đã đăng ký
trong `configs/` đều đã chạy xong, nên nếu chạy lại thì không còn là held-out.

## License

MIT, xem [LICENSE](LICENSE). Các tệp trong `patches/` sửa OMPL nên vẫn theo license BSD của OMPL
([patches/OMPL-LICENSE.txt](patches/OMPL-LICENSE.txt)).
