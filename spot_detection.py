import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------- 配置参数 ----------------------
# YOLO 相关
YOLO_UPDATE_INTERVAL = 60  # 正常跟踪时，降低检测频率，减少卡顿
YOLO_RETRY_INTERVAL = 30  # 未检测到目标时，YOLO 重试间隔帧数
MIN_FEATURES = 5           
CONFIDENCE_THRESHOLD = 0.5 

# 光斑检测相关（动态阈值）
HSV_LOWER = np.array([0, 0, 225])  # 初始阈值（H_min, S_min, V_min）
HSV_UPPER = np.array([179, 67, 255])  # 初始阈值（H_max, S_max, V_max）
KERNEL_SIZE = (5, 5)  # 形态学操作核大小
MIN_SPOT_AREA = 10  # 最小光斑面积（像素²），可调整
MAX_SPOT_AREA = 200  # 最大光斑面积（像素²），可调整
WHITE_RGB_THRESHOLD = 180  # R、G、B均 > 此值视为白色背景
WHITE_AREA_RATIO = 0.3  # 白色背景最小占比
CONSECUTIVE_FRAMES = 5  # 连续检测到光斑的帧数阈值

# 加载模型和摄像头
model = YOLO("D:/guangdian/code/code/runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建窗口（调试用）
cv2.namedWindow("Real-time Tracking", cv2.WINDOW_NORMAL)
cv2.namedWindow("HSV Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Binary Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Threshold Adjust", cv2.WINDOW_NORMAL)  # 新增阈值调整窗口
cv2.resizeWindow("Real-time Tracking", 800, 600)

# 初始化阈值调整滑动条
def on_trackbar_change(val):
    global HSV_LOWER, HSV_UPPER
    HSV_LOWER[0] = cv2.getTrackbarPos('H_min', 'Threshold Adjust')
    HSV_LOWER[1] = cv2.getTrackbarPos('S_min', 'Threshold Adjust')
    HSV_LOWER[2] = cv2.getTrackbarPos('V_min', 'Threshold Adjust')
    HSV_UPPER[0] = cv2.getTrackbarPos('H_max', 'Threshold Adjust')
    HSV_UPPER[1] = cv2.getTrackbarPos('S_max', 'Threshold Adjust')
    HSV_UPPER[2] = cv2.getTrackbarPos('V_max', 'Threshold Adjust')


cv2.createTrackbar('H_min', 'Threshold Adjust', 0, 179, on_trackbar_change)
cv2.createTrackbar('S_min', 'Threshold Adjust', 0, 255, on_trackbar_change)
cv2.createTrackbar('V_min', 'Threshold Adjust', 225, 255, on_trackbar_change)
cv2.createTrackbar('H_max', 'Threshold Adjust', 179, 179, on_trackbar_change)
cv2.createTrackbar('S_max', 'Threshold Adjust', 67, 255, on_trackbar_change)
cv2.createTrackbar('V_max', 'Threshold Adjust', 255, 255, on_trackbar_change)

# 初始化变量
frame_count = 0
track_bbox = None
prev_gray = None
prev_points = None
initial_y = None
initial_h = None
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 用于记录连续检测到光斑的帧数
consecutive_spot_frames = 0
# 用于未检测到目标时，计数到一定帧数才重试 YOLO 推理
yolo_retry_counter = 0  

# ---------------------- 核心逻辑（主循环） ----------------------
while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧，退出...")
        break  # 退出主循环

    # 显示FPS
    if frame_count == 0:
        start_time = cv2.getTickCount()
    else:
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time) * frame_count
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # ---------------------- 1. YOLO检测（初始化/更新） ----------------------
    # 判断是否需要进行 YOLO 推理：跟踪框为空 或 达到更新间隔 或 特征点不足 或 达到重试间隔
    need_yolo = (track_bbox is None 
                 or frame_count % YOLO_UPDATE_INTERVAL == 0 
                 or (prev_points is not None and len(prev_points) < MIN_FEATURES) 
                 or (track_bbox is None and yolo_retry_counter % YOLO_RETRY_INTERVAL == 0))
    
    if need_yolo:
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            max_idx = np.argmax(confs)
            x1, y1, x2, y2 = boxes[max_idx]
            track_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            print(f"YOLO更新框：{track_bbox} (置信度: {confs[max_idx]:.2f})")

            initial_y = track_bbox[1]
            initial_h = track_bbox[3]

            mask = np.zeros_like(gray)
            mask[track_bbox[1]:track_bbox[1] + track_bbox[3],
            track_bbox[0]:track_bbox[0] + track_bbox[2]] = 255
            prev_points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100,
                                                   qualityLevel=0.3, minDistance=7, blockSize=7)
            prev_gray = gray.copy()
            # 重置重试计数器
            yolo_retry_counter = 0  
        else:
            print("YOLO未检测到目标，继续跟踪...")
            yolo_retry_counter += 1
            # 这里可以考虑一些降级处理，比如保持上一帧的跟踪框等，而不是直接使用可能未定义的prev_points
            if prev_points is not None:
                next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
                if next_points is not None:
                    good_points = next_points[status == 1]
                    if len(good_points) > 0:
                        prev_points = good_points.reshape(-1, 1, 2)
                else:
                    # 光流跟踪也失败，重置跟踪相关变量
                    track_bbox = None
                    prev_points = None
                    initial_y = None
                    initial_h = None
                    prev_gray = None
            else:
                # 没有之前的跟踪点，暂时无法跟踪，等待YOLO重新检测
                pass
    else:
        yolo_retry_counter += 1

    # ---------------------- 2. 光流跟踪（水平约束） ----------------------
    if track_bbox is not None and prev_points is not None:
        if prev_gray is not None:
            next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

            if next_points is not None:
                good_points = next_points[status == 1]
                prev_good_points = prev_points[status == 1]

                if len(good_points) > MIN_FEATURES:
                    x, y, w, h = cv2.boundingRect(good_points.astype(np.int32))
                    y = initial_y
                    h = initial_h
                    track_bbox = (x, y, w, h)
                    prev_points = good_points.reshape(-1, 1, 2)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    x, y, w, h = track_bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Low Features", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    print(f"特征点不足（{len(good_points)}），等待YOLO重新检测")
            else:
                x, y, w, h = track_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Tracking Lost", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            prev_gray = gray.copy()

    # ---------------------- 3. 光斑检测（动态HSV阈值 + 帧连续性、面积判断 + 白色背景验证） ----------------------
    if track_bbox is not None and prev_points is not None and len(prev_points) > MIN_FEATURES:
        x, y, w, h = track_bbox
        roi = frame[y:y + h, x:x + w]
        if roi.size != 0:
            # ---------------------- 3.1 动态HSV颜色检测 ----------------------
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(hsv_roi, HSV_LOWER, HSV_UPPER)

            # 形态学操作：开运算去噪点，闭运算填充内部
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)
            hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
            hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)

            cv2.imshow("HSV Mask", hsv_mask)

            # ---------------------- 3.2 显示二值化图 ----------------------
            _, binary_mask = cv2.threshold(hsv_mask, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow("Binary Mask", binary_mask)

            # ---------------------- 3.3 面积与帧连续性判断 ----------------------
            spot_area = cv2.countNonZero(binary_mask)
            if MIN_SPOT_AREA <= spot_area <= MAX_SPOT_AREA:
                consecutive_spot_frames += 1
                if consecutive_spot_frames >= CONSECUTIVE_FRAMES:
                    # ---------------------- 3.4 白色背景验证 ----------------------
                    # 白色背景判断（RGB均值法）
                    avg_b, avg_g, avg_r = np.mean(roi, axis=(0, 1))
                    is_white_bg = (avg_r > WHITE_RGB_THRESHOLD and
                                   avg_g > WHITE_RGB_THRESHOLD and
                                   avg_b > WHITE_RGB_THRESHOLD)

                    if is_white_bg:
                        # 在视频帧中显示结果，先显示spot，再根据是否在白色区显示success
                        cv2.putText(frame, "Spot (Success)", (x, y - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Spot", (x, y - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                consecutive_spot_frames = 0
                # 清除之前的显示（可选）
                cv2.putText(frame, "                ", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # ---------------------- 4. 显示与退出 ----------------------
    cv2.imshow("Real-time Tracking", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC键退出
        break  # 退出主循环
    elif key == ord('r'):  # R键重置跟踪
        track_bbox = None
        prev_points = None
        initial_y = None
        initial_h = None
        consecutive_spot_frames = 0
        print("已重置跟踪")

# 释放资源
cap.release()
cv2.destroyAllWindows()