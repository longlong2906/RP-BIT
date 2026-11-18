# BIT* 2.5D — quy tắc cho coding agent

Đọc `docs/EXPERIMENT_SPEC.md`, `docs/IMPLEMENTATION_PLAN.md` và `configs/experiments.yaml` trước khi triển khai. Đặc tả là đề xuất nghiên cứu; không được biến giả thuyết thành kết quả đã có.

## Phạm vi cố định

- Lấy mẫu trạng thái q=(x,y), không thêm z độc lập. z=H(x,y).
- Bản đồ tĩnh, đã biết, liên tục tuyến tính từng tam giác trong miền có dữ liệu.
- Footprint tròn; xét bán kính và biên an toàn. Không có yaw, động học, động lực học, robot thật, ROS/Gazebo hoặc học máy trong phiên bản nền.
- Mọi planner tối thiểu hóa cùng chiều dài đường bám bề mặt. Vật cản, độ dốc và vùng thiếu dữ liệu là ràng buộc, không tự thêm trọng số phạt.
- Giữ khoảng cách láng giềng Euclid 2D. Không đổi metric của state space thành chi phí cạnh địa hình.

## Quy tắc triển khai

- Kiểm tra repository và hướng dẫn sẵn có; không xóa, ghi đè hoặc hoàn tác thay đổi không thuộc nhiệm vụ.
- Ưu tiên C++/CMake/OMPL cho lõi; Python cho sinh bản đồ, điều phối và phân tích. Ghi lại phiên bản thực sự dùng.
- Dùng BIT* thật của OMPL. Không thay bằng A*, RRT* tự viết hoặc mô phỏng giả rồi giữ nhãn BIT*.
- Xác minh API bằng source/header của phiên bản đã ghim; không đoán tên hàm hoặc tùy chọn.
- Tách motionCost, motionCostHeuristic, heuristic đến đích và informed sampler. Kiểm tra chúng thực sự được planner sử dụng.
- Kiểm tra toàn cạnh và footprint; không chỉ hai đầu. Không xuyên qua NoData hoặc vách cấm.
- Dùng bộ kiểm chứng độc lập; không tái sử dụng nguyên routine cần kiểm thử làm oracle.
- Chỉ chạy cấu hình ablation thực sự được hỗ trợ và có test xác nhận. Không gắn nhãn ablation cho các cờ không có hiệu lực.
- Không cần sửa OMPL nếu extension point đã đủ. Nếu bắt buộc sửa, giữ patch tối thiểu trong repository, ghi base commit và giấy phép; không sửa âm thầm thư viện hệ thống.
- Không thay đổi mặc định bảo mật, quyền truy cập, cấu hình agent toàn cục hoặc dùng dịch vụ tính phí. Không tải dữ liệu/tạo tài nguyên bên ngoài khi chưa được cho phép.

## Kiểm thử, số liệu và báo cáo

- Build thật, chạy test thật và lưu câu lệnh/exit code/log. Không có dependency thì ghi BLOCKED; không dùng kết quả giả để lấp chỗ trống.
- Chỉ `exact goal reached + independent path validation` được tính là thành công. “EXACT_SOLUTION” của OMPL không đồng nghĩa “tối ưu toàn cục”.
- Không dùng PathGeometric::length() như chiều dài địa hình; lưu riêng cost_terrain_m và length_xy_m.
- Không gọi timeout là vô nghiệm. Không gọi nghiệm tốt nhất đã biết là nghiệm tối ưu đã chứng minh.
- Lưu lần thất bại, timeout, crash, dữ liệu thô và cấu hình. Không chọn lại seed để loại kết quả bất lợi.
- Không tạo biểu đồ từ dữ liệu giả trong thư mục kết quả thực nghiệm. Dữ liệu giả cho test parser phải ở tests/fixtures và được gắn nhãn.
- Chỉ chạy smoke mặc định. Pilot và full phải theo prompt được giao; full yêu cầu bật cờ rõ ràng, không tự chạy hàng nghìn lượt.
- Sau mỗi giai đoạn cập nhật docs/STATUS.md: DONE / FAILED / BLOCKED / NOT_RUN, bằng chứng test, hạn chế và bước tiếp theo.
- Báo cáo cho người dùng bằng tiếng Việt; code identifier và lệnh CLI dùng tiếng Anh.
