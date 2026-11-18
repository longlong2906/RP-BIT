# Đặc tả thực nghiệm BIT* 2.5D bằng lấy mẫu thấp chiều

Phiên bản đặc tả: 1.0 — ngày 19/09/2026.

Tài liệu này giao việc triển khai và đo lường. Các cấu hình và ngưỡng là lựa chọn đề xuất, không phải số liệu thực nghiệm đã có. Nguồn API được đánh dấu [Sxx] và liệt kê trong `docs/SOURCES.md`. Phải đối chiếu source/header với phiên bản OMPL thực sự cài đặt.

## 1. Mục tiêu, giả thuyết và phạm vi

Xây dựng pipeline tái lập để so sánh BIT* dùng heuristic khoảng cách mặt phẳng với BIT* dùng cận dưới khoảng cách giữa các điểm đã ánh xạ lên địa hình. Cả hai đều lấy mẫu q=(x,y) và tối ưu chiều dài đường bám cùng một bề mặt.

Giả thuyết H1: mô hình giảm chiều vẫn biểu diễn đúng đường tham chiếu trên heightfield đơn trị. H2: cận dưới có xét độ cao và miền informed tương ứng có thể giảm công việc tìm kiếm hoặc cải thiện chi phí ở cùng ngân sách. H3: lợi ích phụ thuộc cấu trúc địa hình; có trường hợp overhead vượt lợi ích. H1 có phần lập luận toán học và phần kiểm chứng triển khai; H2/H3 cần số liệu thật.

Không coi “mọi đường đều tìm thấy” là tiêu chí mặc định của một planner lấy mẫu trong ngân sách hữu hạn. Không dùng thực nghiệm để khẳng định lần đầu tiên trên thế giới, đầy đủ theo xác suất hay tối ưu tiệm cận.

Mô hình cố định: bản đồ tĩnh, có dữ liệu độ cao; robot có footprint tròn; không yaw, nonholonomic constraint, động lực học, slip, học máy hoặc vật cản động. Trạng thái là điểm tham chiếu trên mặt đất, không phải đầy đủ pose thân robot. Không thêm z độc lập. Không tìm đường trong thể tích 3D cho phép đi phía trên mặt đất.

Ưu tiên phiên bản nền chạy được, kiểm tra được, sau đó mới CGAL, bản đồ thực, ROS/Gazebo, AIT*/ABIT* hoặc đối chứng constrained-3D.

## 2. Mô hình toán học phải giữ nguyên

Cho D là miền hình chữ nhật trên mặt phẳng, H là heightfield đơn trị. Đặt Phi(q)=(x,y,H(x,y)). H liên tục, affine trên mỗi tam giác của lưới trong miền có dữ liệu. Không nội suy bắc cầu qua NoData hoặc vách đã đánh dấu cấm.

Với hai trạng thái u,v, local connector là q(t)=(1-t)u+t v, 0<=t<=1. Đường thực được kiểm tra là Gamma(t)=Phi(q(t)). Không dùng chord 3D Phi(u)Phi(v) làm đường thực thi.

Chi phí cạnh hợp lệ:

```text
c_H(u,v) = integral_0^1 sqrt(||q_dot||^2 + (grad(H(q)) dot q_dot)^2) dt
```

Tách q(t) tại tất cả giao điểm với biên tam giác, thành q_0=u,...,q_M=v. Khi đó:

```text
c_H(u,v) = sum_{i=0}^{M-1} ||Phi(q_{i+1}) - Phi(q_i)||_2
```

Đây là chi phí trên bề mặt rời rạc tuyến tính từng phần, không phải trên một địa hình thực chưa biết. Tổng chi phí đường là tổng c_H của các cạnh. Chi phí không âm, cộng được, đối xứng trong mô hình hiện tại. Chi phí một local connector không được giả định là metric giữa mọi cặp trạng thái.

Định nghĩa hai cận dưới:

```text
d_xy(u,v)  = ||u-v||_2
d_phi(u,v) = ||Phi(u)-Phi(v)||_2
```

Đối với cạnh hợp lệ, d_xy<=d_phi<=c_H. Cận dưới đường từ start qua q đến goal là f_xy(q)=d_xy(start,q)+d_xy(q,goal), hoặc f_phi(q)=d_phi(start,q)+d_phi(q,goal).

Với c_best là chi phí bề mặt của incumbent hợp lệ:

```text
E_xy(c_best) = {q: f_xy(q) < c_best}
I_phi(c_best) = {q: f_phi(q) < c_best}
I_phi subset E_xy
```

Mọi điểm trên đường cải thiện đều thỏa f_phi(q)<=chi phí đường<c_best. Đây là lý do phép loại theo I_phi không bỏ một đường cải thiện, trong mô hình toán học chính xác. Sai số số học cần chính sách bảo thủ và test; không trừ một epsilon lớn khỏi c_best làm cắt nhầm đường.

Điểm đích là một trạng thái cố định được sample tường minh. Không dùng vùng đích bán kính lớn nhưng vẫn lấy khoảng cách tới tâm làm heuristic cho vùng đó. Nếu thư viện dùng tolerance, phân biệt tolerance roundoff với một vùng đích vật lý; xuất đường tới q_goal và tính đầy đủ kết nối cuối. Lập trình goal và heuristic nhất quán, có test start=goal và near-goal. Không sửa c_best sau khi solve một cách làm sampler đã cắt theo chi phí thiếu đoạn cuối.

## 3. Dữ liệu bản đồ, tọa độ và khả năng đi qua

### 3.1. Dữ liệu và quy ước

Chọn định dạng nhẹ để cả C++ và Python đọc được, ví dụ JSON metadata + CSV/array nhị phân chiều cao/masks. Ưu tiên thư viện có sẵn trong repository; không thêm format phức tạp nếu chưa cần. Ghi schema và kiểm tra dữ liệu đầu vào.

Metadata tối thiểu: map_id, generator_version, generator_seed, extent_m, origin_xy, spacing_xy, node_count_xy, height_units, row_axis, column_axis, diagonal convention, NoData encoding và SHA-256 của dữ liệu chuẩn hóa.

Đề xuất height[j,i] tại (x0+i*dx,y0+j*dy), với trục y tăng theo j. Đảo hàng chỉ ở bước hiển thị nếu cần, không âm thầm trong loader. Lưới 40x40 m với bước 0.1 m có 400x400 ô và 401x401 nút.

Chia mỗi ô theo cùng đường chéo SW–NE. H affine trên hai tam giác và dùng chung giá trị nút. Truy vấn ngoài bản đồ trả về trạng thái không hợp lệ; NoData không chuyển thành 0. Bắt lỗi kích thước, dx<=0, đơn vị không nhất quán và dữ liệu vô hạn.

### 3.2. Footprint và miền hợp lệ

Bán kính hiệu dụng r_eff=r_robot+r_margin. Chạm vùng cấm hoặc biên ngoài không được tính là hợp lệ trong cấu hình nền.

Miền cấm F bao gồm vật cản, ô NoData, ô vượt ngưỡng địa hình và phần bên ngoài D. Phiên bản nền có thể đánh dấu cả ô cấm nếu một trong hai tam giác vượt ngưỡng dốc: đây là xấp xỉ bảo thủ phải ghi trong metadata, và áp dụng giống nhau cho mọi planner.

Độ dốc tam giác H=ax+by+c: alpha=atan(sqrt(a*a+b*b)). Nếu có roughness, dùng RMS phần dư mặt phẳng trên cửa sổ có kích thước vật lý xác định; không tính từ đúng ba đỉnh của một tam giác. roughness_limit_m=null nghĩa là chưa bật ràng buộc này và phải ghi rõ trong báo cáo.

Một trạng thái q hợp lệ khi đĩa footprint không giao F. Một cạnh hợp lệ khi capsule quét bởi đĩa dọc đoạn uv không giao F. Có thể hiện thực bằng khoảng cách liên tục từ điểm/đoạn tới các hình chữ nhật ô cấm và polygon vật cản, kết hợp kiểm tra biên. Dùng spatial index để lọc ứng viên sau khi thuật toán brute-force đã được kiểm thử.

Không được chỉ so khoảng cách tới tâm ô vật cản. Không được kiểm tra dải mẫu cách xa hơn vật cản mỏng rồi gọi là kiểm tra liên tục. Nếu thay bằng occupancy erosion, phải mô tả phép bảo thủ và không giãn footprint hai lần.

Phiên bản mask/polygon, footprint và những ngưỡng trên xác định cùng một D_free cho mọi baseline. Các chỗ gấp trên bề mặt không tự được xử lý như động lực học tiếp xúc; chỉ những quy tắc mô hình đã ghi mới có hiệu lực.

## 4. Các module cần hiện thực

Cấu trúc đề xuất, được phép đổi tên nếu repository hiện có có quy ước khác:

```text
CMakeLists.txt
include/terrain25d/{HeightField,Traversability,SurfaceGeometry}.hpp
include/terrain25d/{TerrainMotionValidator,TerrainObjective}.hpp
include/terrain25d/{TerrainInformedSampler,PlannerFactory,RunRecorder}.hpp
src/...
apps/terrain_benchmark.cpp
tests/unit/...
tests/integration/...
tests/fixtures/...
tools/generate_scenarios.py
tools/run_experiments.py
tools/validate_results.py
tools/analyze_results.py
configs/experiments.yaml
docs/ENVIRONMENT.md
docs/OMPL_INTEGRATION.md
docs/STATUS.md
results/<profile>/<experiment_id>/<trial_id>/...
reports/<experiment_id>/...
```

HeightField đọc dữ liệu và nội suy, trả về trạng thái có/không dữ liệu. SurfaceGeometry duyệt giao biên tam giác và tính c_H. Traversability xử lý footprint và miền cấm. TerrainMotionValidator triển khai toàn bộ overload cần thiết của checkMotion theo API thực tế, bao gồm last-valid khi được yêu cầu [S03].

TerrainObjective kế thừa OptimizationObjective trực tiếp, hoặc một cách khác đã kiểm chứng không kế thừa nhầm chi phí/heuristic Euclid [S02]. motionCost trả chi phí bề mặt; motionCostHeuristic chỉ trả cận dưới. Đường đi xuyên vật cản có thể có chiều dài hình học hữu hạn nhưng vẫn phải bị validator bác bỏ trước khi được chấp nhận. Đi qua NoData không được nội suy để có chi phí giả. Bộ EvaluateTerrainEdge dùng cho test phải trả cả valid flag, cost, số mặt duyệt và nguyên nhân lỗi.

Tách counter validity và counter geometry-cost; không tăng giảm số đếm giả để làm cải tiến trông tốt hơn. Không dùng metric 2D của nearest-neighbor thay objective. Nếu cost được tính trước collision trong planner, đó không phải lỗi, miễn là đường được chấp nhận đã qua cùng validator.

Edge cache mặc định tắt. Nếu bổ sung, dùng key biểu diễn chính xác trạng thái/ID cạnh và map/model hash; không gộp hai cạnh khác nhau bằng làm tròn tọa độ không được kiểm chứng. Reset cache giữa các trial và dùng giống nhau cho các phương pháp so sánh.

## 5. BIT* thật, baseline và ablation

### 5.1. Hai cấu hình bắt buộc đầu tiên

| ID | Planner | Cận dưới ưu tiên/cắt tỉa | Lấy mẫu sau khi có nghiệm | Chi phí thật |
|---|---|---|---|---|
| bitstar_a0 | OMPL BIT* | d_xy, f_xy | E_xy, sau đó kiểm tra validity | c_H |
| bitstar_a3 | OMPL BIT* | d_phi, f_phi | Đề xuất E_xy, lọc I_phi, kiểm tra validity | c_H |

Trước nghiệm đầu tiên, lấy mẫu đều trong D rồi kiểm tra validity. Tập mẫu cuối cùng được phân bố theo diện tích tham số (x,y), không theo diện tích mặt 3D. Thứ tự kiểm tra bounds, NoData, heuristic và validity phải được khai báo; tránh thực hiện cùng phép kiểm tra đắt tiền hai lần mà không ghi nhận.

Dùng k-nearest 2D, cùng samples-per-batch và rewire factor cho A0/A3. Tắt just-in-time sampling ở cấu hình nền. OMPL có API cấu hình các thành phần này, nhưng phải kiểm tra giá trị thực sau setup [S04]. Không tự suy ra ngưỡng bán kính/k mới rồi gắn bảo đảm của BIT* chuẩn.

Mỗi run manifest phải ghi requested_policy và effective_policy. Nếu requested A3 nhưng sampler hoặc pruning vẫn theo 2D, fail rõ ràng hoặc đổi nhãn thành cấu hình chẩn đoán đúng nghĩa; không báo A3 hoàn chỉnh.

### 5.2. Kiểm tra API và các điểm có thể phải patch

Trong docs/OMPL_INTEGRATION.md, chỉ rõ phiên bản, lớp/hàm/source line thực hiện: cost-to-come lower bound, edge lower bound, cost-to-go, vertex pruning, sample pruning, sampler allocation, informed measure, RNG, callbacks và connectivity.

Không cho rằng chỉ gọi setCostToGoHeuristic là đã đổi mọi thành phần bên trong BIT*. Xem source của phiên bản đã chọn và xác nhận bằng trace/test [S02,S04].

A1/A2 là thí nghiệm mở rộng:

| ID | Ưu tiên/cận dưới xét cạnh | Miền mẫu và cắt tỉa trạng thái |
|---|---|---|
| bitstar_a1 | d_phi | f_xy/E_xy |
| bitstar_a2 | d_xy | f_phi/I_phi |

Các chính sách này có thể đang dùng chung hook trong OMPL. Chỉ bật A1/A2 sau khi có thay đổi chính sách thật, test trên trạng thái f_xy<c_best<=f_phi và trace cho thấy queue/pruning/sampler đúng cấu hình. Nếu API chưa cho tách, patch tối thiểu và lưu patch; nếu chưa làm được, ghi UNSUPPORTED, không tạo log giả. A0/A3 vẫn phải chạy được độc lập với phần mở rộng.

### 5.3. Baseline bổ sung

G3 thêm OMPL RRT* và Informed RRT*. Cả hai dùng cùng H, D_free, goal, c_H và validator. Informed RRT* được dùng d_phi/I_phi khi tích hợp đúng; ghi rõ cấu hình. Ghi range/goal bias/rewiring và toàn bộ effective parameters của mỗi planner. Không ép các tham số khác nghĩa phải cùng giá trị chỉ để gọi là công bằng.

A*, CGAL, AIT*, ABIT* là mở rộng, không bắt buộc trước khi có hệ thống chính chạy được. A* trên đồ thị hữu hạn chỉ là nghiệm tối ưu trên đồ thị đó. Một baseline 3D phải bị ràng buộc đúng trên cùng mặt đất, không được bay qua vật cản. A0/A3 tự chúng không đo riêng lợi ích của giảm chiều.

## 6. Hợp đồng informed sampler

Cài đúng overload sampleUniform(maxCost) và sampleUniform(minCost,maxCost), theo khoảng cận dưới [minCost,maxCost), và bounded attempts [S05]. Không trả một mẫu ngoài khoảng chỉ để tránh thất bại.

Với maxCost vô hạn: dùng miền D và kiểm tra điều kiện còn lại. Với maxCost hữu hạn: sinh đều trong đĩa đơn vị (bán kính sqrt(U), góc 2*pi*V), biến đổi thành E_xy có tâm (s+g)/2, a=maxCost/2, b=sqrt(maxCost^2-||s-g||^2)/2. Xoay theo hướng s->g. Có thể chọn proposal là D thay vì elip khi nó tiết kiệm hơn, nhưng mọi mẫu vẫn phải qua cùng điều kiện để giữ đúng phân bố chấp nhận. Không clamp điểm ngoài bounds vào biên vì tạo sai lệch phân bố.

A0 dùng f_xy cho sublevel set. A3 dùng f_phi; trong A3, E_xy chỉ là proposal, không phải diện tích chính xác của tập mẫu. Thực hiện điều kiện minCost cho overload hai ngưỡng. Lưu số đề xuất, số loại bởi bounds/NoData/heuristic/validity và số không tìm được mẫu trong giới hạn.

Xử lý start=goal, empty domain, zero/negative maxCost, maxCost thấp hơn lower bound, elip gần suy biến, NaN và giới hạn vòng lặp. Không tuyên bố tối ưu hoặc vô nghiệm chỉ vì một lần gọi sampler trả false. Khi incumbent đạt cận dưới toàn bài toán trong tolerance công bố, có thể ghi nhận dừng theo chứng chỉ đó; nếu không có chứng chỉ, tuân theo ngân sách.

Không biết diện tích chính xác I_phi thì không báo hasInformedMeasure=true với pi*a*b. Dùng nhánh không có measure hoặc upper bound bảo thủ theo đúng hợp đồng source/API thực tế [S05]. Ghi rõ lựa chọn ảnh hưởng đến pruning, sample density và connectivity. Dùng k-nearest không miễn việc kiểm tra các chỗ khác cần measure. Nếu OMPL vô hiệu hóa pruning khi measure chưa biết, phải khắc phục hợp lệ hoặc báo effective configuration không có pruning; không giấu cảnh báo.

Không trả diện tích 0 chỉ vì lấy max-bound minus min-bound của hai fallback diện tích bằng nhau mà caller cần một interval measure thực sự. Không bật JIT ở bản nền. Có test xác nhận sampler I_phi không suy giảm âm thầm thành E_xy hoặc uniform-D.

## 7. Bộ test bắt buộc

Đặt tolerance mặc định |a-b| <= 1e-8 m + 1e-8*max(|a|,|b|), rồi kiểm tra phù hợp thang bản đồ. Không nới tolerance để che lỗi. Các bất đẳng thức được kiểm với dung sai số học tương ứng.

| Test | Dữ liệu | Yêu cầu |
|---|---|---|
| T01 | H=h0 | c_H=d_xy trên cạnh; trên bài toán không vật cản đường thẳng là tối ưu. |
| T02 | H=ax+by+c | Chi phí cạnh bằng sqrt(dx^2+dy^2+(a*dx+b*dy)^2). |
| T03 | Đồi gấp khúc, hai đầu cùng cao | c_H lớn hơn chord 3D; kiểm đúng từng lần cắt tam giác. |
| T04 | Bất kỳ cạnh hợp lệ | d_xy<=d_phi<=c_H, symmetry và split additivity. |
| T05 | Đường dọc/cắt cạnh mesh, qua đỉnh, đoạn độ dài 0 | Không double-count, không mất đoạn, không NaN. |
| T06 | Vật cản mỏng giữa hai đầu hợp lệ | checkMotion bác bỏ dù hai trạng thái đầu–cuối hợp lệ. |
| T07 | Hành lang quanh 2*r_eff | Bị chặn/đi được theo khoảng hở, không theo tâm ô. |
| T08 | NoData, ngoài bounds, invalid start/goal | Kết quả lỗi/hợp lệ đúng; không dùng height=0 thay dữ liệu thiếu. |
| T09 | Heightfield+constant offset | Chi phí và heuristic d_phi không đổi. |
| T10 | Uniform scale tọa độ, độ cao, footprint và cửa sổ vật lý | Chi phí nhân theo scale; dốc và tính khả thi giữ như mô hình. |
| T11 | Mẫu với c_min,c_max | Mọi mẫu đúng khoảng; A3 đáp ứng f_phi; bounded attempts và elip suy biến an toàn. |
| T12 | Known improving path/đồ thị nhỏ | Cận dưới không vượt cost-to-go Dijkstra trên đồ thị; không prune nhầm điểm trên đường cải thiện. |
| T13 | Rewiring và incumbent | Cây không chu trình, chi phí cha-con khớp; incumbent không tăng. |
| T14 | Bộ pipeline và seed | Cùng map/config/seed tạo cùng dữ liệu đầu vào; deterministic work-budget test tái lập được. |
| T15 | Timeout/empty free/no path | Thoát theo ngân sách/watchdog; không gọi timeout là chứng minh vô nghiệm. |
| T16 | Approximate goal result | Không tính như thành công chính; chi phí/đích được kiểm nhất quán. |
| T17 | Logger và JSON | Null/status cho unavailable; không NaN/Infinity trong JSON chuẩn; failure vẫn có record. |

T12 dùng Dijkstra trên một đồ thị nhỏ có trọng số c_H để phát hiện heuristic bị đánh giá cao; nó không chứng minh tối ưu của bài toán liên tục. Có test đặc biệt g_T(q)+h(q)>=c_best nhưng g_hat(q)+h(q)<c_best: không loại vĩnh viễn vị trí chỉ vì cây hiện tại đi vòng.

Với T13, nếu không truy cập được nội bộ OMPL để kiểm tra mọi lần rewiring, xuất PlannerData ở các checkpoint kiểm thử/debug và nêu phạm vi test; không nói đã kiểm tra invariant bên trong khi chưa có instrumentation.

Bộ verifier độc lập ở Python hoặc module riêng đọc raw heightfield, mask và path. Dùng một cách tính khác, ví dụ brute-force intersect với biên mesh để đối chiếu traversal nhanh, cùng phép tính capsule-distance độc lập. Chỉ lấy mẫu dày đường đi là diagnostic, không phải chứng minh không va chạm liên tục.

Test path cả ở điểm gãy tam giác, chứ không chỉ dùng các đỉnh cây đã lift. Lưu các fixture nhân tạo trong tests/fixtures, tách biệt results/.

## 8. Generator và bộ scenario

Các family phải có công thức/tham số, seed và expected properties rõ ràng:

- flat: H không đổi; tilted_plane: H=ax+by+c. Dùng kiểm tra giải tích.
- traversable_hill: H=h0+A*exp(-0.5*((x-cx)^2/sx^2+(y-cy)^2/sy^2)); chọn A/s đủ thấp để đồi vẫn đi được ở cấu hình chính. Thêm fixture khi chi phí đi vòng và vượt đồi cạnh tranh nhau.
- ridge_pass: một sống núi hoặc dải cấm có đèo; biến đổi rộng/cao sống núi, vị trí/rộng đèo. Lưu rõ phần nào bị cấm và phần nào chỉ làm đường dài hơn.
- rolling_terrain: nhiều dao động/đồi liên tiếp; có trường hợp chênh cao đầu–cuối nhỏ nhưng đường lên xuống dài. Ghi biên độ, bước sóng và pha.
- narrow_passage: vật cản tạo hành lang trên địa hình; tham số hóa clearance của tâm w_free=w_corridor-2*r_eff. Có cả trường hợp feasible và blocked riêng.
- mixed_obstacles: kết hợp vùng phẳng, đồi, vật cản và NoData; sinh deterministically.

Tách deterministic correctness fixtures khỏi stochastic performance suite. Các bài toán được ghi solvable phải có witness path hợp lệ hoặc chứng cứ kiểm tra thích hợp; kiểm tra liên thông lưới thô một mình không chứng minh vô nghiệm liên tục. Không loại map/query vì BIT* mới thất bại.

Query generation dùng seed riêng, chỉ chấp nhận start/goal hợp lệ; phân tầng theo chiều dài và cấu trúc đường. Chia tập tune/test theo map seed, không chỉ planner seed. Công thức, tham số đã tune trên pilot được khóa trước full.

Profile smoke có 4*1*1*3*2=24 trial. Pilot có 5*2*2*5*4=400 trial. Full có 5*5*4*30*4=12000 trial với YAML hiện tại, chưa gồm ablation mở rộng. Code phải tính lại từ manifest, không hardcode các số này. Không chạy full theo mặc định. Các profile có thể dùng độ phân giải khác nhau; không lấy chênh lệch thời gian giữa profile làm bằng chứng tăng tốc cùng bài toán.

## 9. Trial runner, thời gian và tái lập

Dùng một worker process cho mỗi (map,query,planner,seed,config). Khởi tạo seed trước khi dựng đối tượng dùng RNG. Bộ điều phối chạy tuần tự jobs=1 trong phép thử chính; đảo thứ tự bằng shuffle seed đã lưu. Không tái sử dụng cây/cache của trial trước.

Worker dùng OMPL Benchmark với một cấu hình và runCount=1, hoặc thiết kế tương đương giữ raw OMPL log và mọi metadata. Nếu dùng runCount trong OMPL cho nhiều lượt, phải chứng minh reset RNG/cache đúng; ưu tiên subprocess để dễ kiểm soát. Thiết lập simplify=false tường minh [S06,S07].

Tách các đồng hồ: load, preprocess, planner setup, solve, serialize và independent validation. Planning clock bắt đầu ngay trước solve, bao gồm sampling, evaluation và overhead logging trong quá trình solve. Không trừ overhead chỉ ở một planner. Dùng steady/monotonic clock.

Ghi first-solution và improvements qua callback/API được xác nhận. Nếu chỉ có polling, ghi sai số thời điểm hoặc khoảng [lần poll trước,lần poll đầu phát hiện], không gọi thời gian polling là timestamp chính xác. Không dùng solve_duration làm first_solution_time khi planner tiếp tục tối ưu. Không ghi một callback không bao giờ được planner gọi mà vẫn coi như đã có tiến trình.

Dùng timeout trong planner và watchdog ngoài process để xử lý loop/call bị treo. Lưu time_budget_s, actual_solve_wall_s, termination_reason và overshoot. Watchdog quá hạn là trạng thái riêng, không một trial thành công bình thường.

Callback lưu event và path snapshot nhẹ trong RAM rồi ghi ra sau nếu có thể; không recompute toàn bộ cost và serialize mesh trong hot loop. Mọi policy logging phải giống nhau giữa baseline. Nếu đọc progress từ thread khác, counter và dữ liệu phải an toàn; không dereference cây đang bị mutate.

Wall-clock budget có thể làm số iteration khác nhau dù cùng seed. Tái lập nghiêm ngặt nên có thêm test dùng số iteration/edge evaluation cố định; benchmark thời gian báo biến động thay vì hứa bitwise reproducibility.

## 10. Output schema và dữ liệu thô

Mỗi experiment có manifest, environment, resolved_config, map hashes và trial order. Mỗi trial có:

```text
trial.json
stdout.log
stderr.log
ompl.log
progress.jsonl
path_xy.csv
path_surface.csv
validation.json
```

Không tạo path file giả nếu chưa có path; khai báo path_available=false. Snapshot improvements có ID và checksum riêng. path_surface phải có các breakpoint giao biên tam giác; chỉ lift đỉnh cây rồi nối chord 3D sẽ sai.

trial.json tối thiểu chứa schema_version, experiment_id, trial_id, map_id/hash, query_id, planner_id, requested/effective policies, map_seed, planner_seed, shuffle_order, config_hash, code_commit, dirty_worktree_hash nếu có, dependency version, CPU/RAM/OS/compiler/build flags, thời điểm UTC và các đồng hồ.

Trạng thái tách thành planner_status, execution_status và independent_valid. Ví dụ execution_status=COMPLETED/TIMEOUT/WATCHDOG/CRASH/BLOCKED; planner_status=EXACT_SOLUTION/APPROXIMATE_SOLUTION/NO_SOLUTION/INVALID_START/INVALID_GOAL. Một exact solution là đến đúng goal theo contract; không có nghĩa tối ưu toàn cục. Chỉ exact + independent_valid=true, được tìm thấy trong ngân sách đo, mới là success chính. Một planner anytime dừng bình thường do đã dùng hết ngân sách nhưng có incumbent hợp lệ vẫn là success; không đánh đồng termination_reason=TIME_BUDGET với thất bại tìm đường. WATCHDOG/CRASH phải được báo riêng ngay cả khi đã lưu được snapshot trước đó.

Các trường kết quả: first_solution_time_s hoặc interval, best_cost_terrain_m, length_xy_m, independently_recomputed_cost_m, cost_abs_error_m, minimum_footprint_clearance_m nếu đã tính đúng toàn cạnh, maximum_slope_deg, sample_proposals, samples_accepted, rejected_by_reason, state_validity_calls, motion_validity_calls, full_surface_cost_calls, triangle_visits, height_queries, peak_rss_mb và time breakdown khi instrumentation có.

Không dùng tên num_collision_checks để trộn lẫn 5 loại counter. Nếu metric không khả dụng, ghi null và lý do; không suy đoán từ num_samples. `PathGeometric::length()` là độ dài theo state-space distance, không thay cho terrain objective [S08]. Built-in clearance nếu chỉ lấy trung bình các đỉnh không được gắn nhãn minimum clearance toàn đường.

JSON không chứa NaN/Infinity. Chưa có nghiệm thì best_cost_terrain_m=null và feasible=false. Event có elapsed_s, feasible, best_cost_terrain_m, sample/edge counters và snapshot_id. Không tạo event “đã có nghiệm” chỉ vì một candidate chưa được chấp nhận.

Validator độc lập kiểm tra đường cuối và snapshots được dùng cho đồ thị anytime. Đường không hợp lệ phải bị gắn nhãn, không lặng lẽ sửa trước báo cáo. Không chuyển đường invalid thành success bằng hậu xử lý âm thầm.

Resume dựa trên trial identity + config/map/code hash. Không dùng lại kết quả cũ sau khi objective/validity thay đổi. Trial chưa chạy không phải thất bại, nhưng phải ghi rõ số lượng chưa chạy và không gọi report là complete. Trial đã chạy mà timeout/invalid vẫn nằm trong mẫu số thành công.

## 11. Metrics và phân tích

Metrics chính: success rate hợp lệ, invalid-output rate, thời gian first solution, c_best(t) theo chiều dài bề mặt, số phép đánh giá cạnh, lượng truy vấn địa hình, chi phí sampler và peak memory. Median/IQR và confidence interval theo map/query/seed; không chỉ mean trên các trial thành công.

Tại mỗi mốc t, tái dựng incumbent bằng step function từ event đã xảy ra (right-continuous); không nội suy tuyến tính giữa hai improvements hoặc backfill nghiệm tương lai. Chưa có nghiệm được biểu diễn missing/infinite về mặt toán, không 0. Mọi biểu đồ cost có điều kiện theo success cần hiển thị coverage đi kèm.

Nếu có c_ref cho từng query, T_eps=inf{t:c_best(t)<=(1+eps)c_ref}. F_eps(t)=số trial đã thực sự có đường hợp lệ đạt ngưỡng trước t / tổng trial đã được thực hiện của nhóm. Giữ timeout trong mẫu số. Không gán time-to-solution=T_budget cho timeout như một quan sát hoàn chỉnh.

Nguồn reference phải có reference_type=analytic/cgal_matched_model/best_known và metadata. Với analytic optimum, relative gap=(c-c_star)/c_star. Với best_known chỉ gọi reference-relative difference; có thể âm nếu thuật toán đang đánh giá tìm được đường tốt hơn, không clamp về 0 hoặc đổi reference âm thầm. c_star=0 có quy tắc riêng. Dijkstra/A* chỉ cho optimum trên đồ thị của chúng; không tự gọi là tối ưu liên tục.

Lựa chọn CGAL hoặc DEM thực là extension tắt mặc định; không tự tải dữ liệu ngoài cho smoke. Nếu chưa có reference đáng tin, vẫn báo success/cost/time/effort nhưng bỏ mục gap hoặc target không xác định, không dựng đường tối ưu giả.

Thêm bốn nhóm hình: attainment theo thời gian; surface-cost anytime kèm coverage; quality theo edge evaluations; hiệu quả theo family (kể cả family overhead lớn). Hình 2D và 3D path là minh họa, không thay metrics. Đơn vị và số trial phải có trên hình/caption. Python plotting dùng matplotlib, không cần seaborn; dùng bảng màu mặc định trừ khi chủ dự án yêu cầu khác.

Báo cáo paired comparison ở cùng map/query; bootstrap phân cấp theo map rồi query rồi seed nếu đủ mẫu. Pilot nhỏ không được gọi là kiểm định kết luận cuối. Tham số chỉ tune trên pilot; khi thay tham số, tạo config version mới. Không “best of many seeds” cho phương pháp mới nhưng mean của baseline.

## 12. Acceptance và bàn giao

G0: có environment report, dependency version và lệnh build thực. G1: test geometry/validity/heuristic pass và có oracle độc lập. G2: A0/A3 thật chạy smoke, không có invalid path bị tính là success, có bằng chứng effective policies. G3: các baseline chính dùng cùng problem/objective; ablation được chứng minh hoặc ghi unsupported. G4: pipeline thống kê đọc raw logs và xử lý failure/timeout đúng. G5: full chỉ theo prompt/cờ cho phép, có manifest complete hoặc partial được khai báo trung thực.

Không yêu cầu thuật toán mới thắng để được nghiệm thu mã nguồn. Yêu cầu mã đúng mô hình, số liệu tái lập được trong phạm vi môi trường đã ghi và không có kết luận vượt quá bằng chứng.
