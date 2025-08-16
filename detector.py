import cv2
import numpy as np
from ultralytics import YOLO
import time # 引入time模块

class SpotDetector:
    def __init__(self, yolo_model_path, camera_index=0):
        # ---------------------- 1. 配置参数 ----------------------
        # YOLO & 光流
        self.YOLO_UPDATE_INTERVAL = 30  # 从50降低到30，更频繁地更新YOLO检测
        self.YOLO_RETRY_INTERVAL = 60   # 从40增加到60，第一次检测到目标框之前的重试间隔更长
        self.MIN_FEATURES = 5
        self.CONFIDENCE_THRESHOLD = 0.5
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # 光斑检测
        self.HSV_LOWER = np.array([159, 0, 197])  # 降低亮度下限，更容易检测到小光斑
        self.HSV_UPPER = np.array([179, 170, 255])  # 放宽饱和度上限
        self.KERNEL_SIZE = (3, 3)  # 减小核大小，保留更多细节
        self.MIN_SPOT_AREA = 80  # 由于光影更换，光斑变大，适当提高最小面积
        self.MAX_SPOT_AREA = 2000  # 由于光影更换，光斑变大，进一步增加最大面积
        
        # 形状检测参数
        self.MIN_CIRCULARITY = 0.25  # 由于光影更换，光斑形状可能略有变化，进一步降低圆度要求
        self.MAX_ASPECT_RATIO = 3.0  # 由于光影更换，进一步放宽长宽比限制

        # "胜利"条件
        self.VICTORY_DURATION = 2.5  # 胜利需要持续的时间（秒）
        self.SPOT_MEMORY_DURATION = 0.5 # 允许光斑消失的最长时间（秒），用于处理偶尔的丢帧，增加到0.5秒提高容错性
        
        # ---------------------- 2. 初始化模型和设备 ----------------------
        self.model = YOLO(yolo_model_path)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError("无法打开摄像头")

        # ---------------------- 3. 初始化状态变量 ----------------------
        self.frame_count = 0
        self.track_bbox = None
        self.prev_gray = None
        self.prev_points = None
        self.initial_y = None
        self.initial_h = None
        self.yolo_retry_counter = 0

        # "胜利"逻辑相关变量
        self.spot_detected = False
        self.first_spot_time = None  # 首次检测到光斑的时间
        self.last_spot_time = None   # 最后一次检测到光斑的时间
        self.victory_achieved = False

    def process_frame(self):
        # --- 新增：在这里从摄像头读取一帧 ---
        ret, frame = self.cap.read()
        if not ret:
            print("无法从摄像头读取帧或视频已结束。")
            return None, None, False

        return self.process_frame_with_image(frame)

    def process_frame_with_image(self, frame):
        """处理外部传入的图像帧"""
        if frame is None:
            return None, None, False

        vis_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        # --- YOLO检测（初始化/更新）---
        need_yolo = (self.track_bbox is None or
                     self.frame_count % self.YOLO_UPDATE_INTERVAL == 0 or
                     (self.prev_points is not None and len(self.prev_points) < self.MIN_FEATURES) or
                     (self.track_bbox is None and self.yolo_retry_counter % self.YOLO_RETRY_INTERVAL == 0))

        if need_yolo:
            results = self.model(frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False) # verbose=False关闭YOLO的打印输出
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                max_idx = np.argmax(confs)
                x1, y1, x2, y2 = boxes[max_idx]
                self.track_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                
                self.initial_y = self.track_bbox[1]
                self.initial_h = self.track_bbox[3]
                
                mask = np.zeros_like(gray)
                mask[self.track_bbox[1]:self.track_bbox[1] + self.track_bbox[3], self.track_bbox[0]:self.track_bbox[0] + self.track_bbox[2]] = 255
                self.prev_points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                self.prev_gray = gray.copy()
                self.yolo_retry_counter = 0
            else:
                self.yolo_retry_counter += 1
        else:
            self.yolo_retry_counter += 1

        # --- 光流跟踪 (已用平均运动向量法改进) ---
        if self.track_bbox is not None and self.prev_points is not None:
            if self.prev_gray is not None:
                next_points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **self.lk_params)

                if next_points is not None:
                    good_new_points = next_points[status == 1]
                    
                    if len(good_new_points) > self.MIN_FEATURES:
                        good_old_points = self.prev_points[status == 1]

                        # 计算平均运动向量
                        delta_x = np.mean(good_new_points[:, 0] - good_old_points[:, 0])

                        # 应用位移到旧的框上
                        prev_x, _, prev_w, _ = self.track_bbox
                        new_x = int(prev_x + delta_x)
                        
                        # 使用新的位置和旧的/初始的尺寸更新框
                        self.track_bbox = (new_x, self.initial_y, prev_w, self.initial_h)
                        
                        # 更新特征点以供下一帧使用
                        self.prev_points = good_new_points.reshape(-1, 1, 2)
                    else:
                        # 特征点太少，跟踪不可靠，清空跟踪状态，等待YOLO重新检测
                        self.track_bbox = None
                        self.prev_points = None

            # 始终更新prev_gray以供下一帧使用
            self.prev_gray = gray.copy()

        # --- 光斑检测与胜利逻辑 ---
        current_spot_found = False
        if self.track_bbox is not None:
            x, y, w, h = self.track_bbox
            
            # 向内缩进到原来的85%，避免边框干扰
            shrink_factor = 0.85
            shrink_w = int(w * shrink_factor)
            shrink_h = int(h * shrink_factor)
            shrink_x = x + int(w * (1 - shrink_factor) / 2)
            shrink_y = y + int(h * (1 - shrink_factor) / 2)
            
            roi = frame[shrink_y:shrink_y + shrink_h, shrink_x:shrink_x + shrink_w]
            if roi.size > 0:
                 # 对ROI进行适度的对比度增强，增强红色光斑与白色背景的区分
                 # 使用CLAHE（对比度限制自适应直方图均衡）进行局部对比度增强
                 lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                 lab_roi[:,:,0] = clahe.apply(lab_roi[:,:,0])  # 只对亮度通道进行增强
                 enhanced_roi = cv2.cvtColor(lab_roi, cv2.COLOR_LAB2BGR)
                 
                 hsv_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2HSV)
                 hsv_mask = cv2.inRange(hsv_roi, self.HSV_LOWER, self.HSV_UPPER)
                 
                 # 检查白色区域占比，避免环境光源干扰
                 white_area = cv2.countNonZero(hsv_mask)
                 total_area = roi.shape[0] * roi.shape[1]
                 white_ratio = white_area / total_area
                 
                 # 如果白色区域超过1/3，跳过光斑检测
                 if white_ratio > 1/3:
                     print(f"环境光源干扰，白色区域占比: {white_ratio:.2%}，跳过光斑检测")
                     current_spot_found = False
                 else:
                     # 正常进行光斑检测
                     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.KERNEL_SIZE)
                     binary_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
                     binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                     spot_area = cv2.countNonZero(binary_mask)
                     
                     # 形状检测
                     shape_valid = self.check_spot_shape(binary_mask)
                     
                     if self.MIN_SPOT_AREA <= spot_area <= self.MAX_SPOT_AREA and shape_valid:
                         current_spot_found = True

        # --- 更新胜利计时器 ---
        if not self.victory_achieved:
            now = time.time()
            if current_spot_found:
                self.last_spot_time = now # 只要找到，就更新最后看到的时间
                if self.first_spot_time is None:
                    self.first_spot_time = now # 如果是第一次看到，记录开始时间
            
            # 检查光斑是否在允许的时间内消失了
            if self.first_spot_time is not None:
                if (now - self.last_spot_time) > self.SPOT_MEMORY_DURATION:
                    # 如果光斑消失太久，重置计时器
                    self.first_spot_time = None
                    self.last_spot_time = None
                    print("光斑丢失，重置计时。")
                elif (now - self.first_spot_time) >= self.VICTORY_DURATION:
                    # 持续时间达到，宣布胜利
                    self.victory_achieved = True
                    print("胜利！持续检测到光斑超过3秒。")
        
        # --- (可选) 可视化，用于调试 ---
        # 如果需要在边缘端也显示，可以取消下面的注释
        # self.visualize(frame)
        
        # --- HSV可视化 ---
        if self.track_bbox is not None:
            vis_frame = self.add_hsv_visualization(vis_frame, frame)
            
        # --- 在检测到的光斑上标注HSV和RGB值 ---
        if self.track_bbox is not None:
            vis_frame = self.annotate_spot_colors(vis_frame, frame)

        # --- 返回结果 ---
        return vis_frame, self.track_bbox, self.victory_achieved

    def add_hsv_visualization(self, vis_frame, original_frame):
        """添加HSV可视化到帧上"""
        if self.track_bbox is None:
            return vis_frame
        
        x, y, w, h = self.track_bbox
        
        # 向内缩进到原来的85%，避免边框干扰
        shrink_factor = 0.85
        shrink_w = int(w * shrink_factor)
        shrink_h = int(h * shrink_factor)
        shrink_x = x + int(w * (1 - shrink_factor) / 2)
        shrink_y = y + int(h * (1 - shrink_factor) / 2)
        
        # 提取ROI
        roi = original_frame[shrink_y:shrink_y + shrink_h, shrink_x:shrink_x + shrink_w]
        if roi.size == 0:
            return vis_frame
        
        # 对ROI进行适度的对比度增强，与检测逻辑保持一致
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_roi[:,:,0] = clahe.apply(lab_roi[:,:,0])  # 只对亮度通道进行增强
        enhanced_roi = cv2.cvtColor(lab_roi, cv2.COLOR_LAB2BGR)
        
        # 转换到HSV
        hsv_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2HSV)
        
        # 创建HSV掩码
        hsv_mask = cv2.inRange(hsv_roi, self.HSV_LOWER, self.HSV_UPPER)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.KERNEL_SIZE)
        binary_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 计算光斑面积和形状
        spot_area = cv2.countNonZero(binary_mask)
        shape_valid = self.check_spot_shape(binary_mask)
        
        # 获取形状信息
        shape_info = self.get_shape_info(binary_mask)
        
        # 创建可视化图像
        # 1. 原始ROI
        roi_resized = cv2.resize(roi, (160, 120))
        vis_frame[10:130, 10:170] = roi_resized
        cv2.putText(vis_frame, "Original ROI", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 2. HSV掩码
        mask_colored = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
        mask_resized = cv2.resize(mask_colored, (160, 120))
        vis_frame[10:130, 180:340] = mask_resized
        cv2.putText(vis_frame, "HSV Mask", (180, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 3. 形态学处理后的掩码
        binary_colored = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        binary_resized = cv2.resize(binary_colored, (160, 120))
        vis_frame[10:130, 350:510] = binary_resized
        cv2.putText(vis_frame, "Binary Mask", (350, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 4. 显示检测信息
        info_y = 160
        cv2.putText(vis_frame, f"Spot Area: {spot_area}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Min Area: {self.MIN_SPOT_AREA}", (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Max Area: {self.MAX_SPOT_AREA}", (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 5. 显示形状信息
        shape_y = info_y + 60
        cv2.putText(vis_frame, f"Aspect Ratio: {shape_info['aspect_ratio']:.2f}", (10, shape_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Circularity: {shape_info['circularity']:.2f}", (10, shape_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 显示形状约束信息
        cv2.putText(vis_frame, f"Max Aspect: {self.MAX_ASPECT_RATIO}", (10, shape_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Min Circularity: {self.MIN_CIRCULARITY}", (10, shape_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 6. 显示检测状态
        if self.MIN_SPOT_AREA <= spot_area <= self.MAX_SPOT_AREA and shape_valid:
            status_color = (0, 255, 0)  # 绿色
            status_text = "SPOT DETECTED"
        else:
            status_color = (0, 0, 255)  # 红色
            status_text = "NO SPOT"
        
        cv2.putText(vis_frame, status_text, (10, shape_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # 7. 显示HSV阈值信息
        hsv_info_y = shape_y + 80
        cv2.putText(vis_frame, f"HSV Lower: {self.HSV_LOWER}", (10, hsv_info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"HSV Upper: {self.HSV_UPPER}", (10, hsv_info_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 8. 在HSV掩码上标注成功检测的光斑
        if self.MIN_SPOT_AREA <= spot_area <= self.MAX_SPOT_AREA and shape_valid:
            # 查找轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                # 计算光斑中心点
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 在HSV掩码上绘制光斑中心点和轮廓
                    # 将坐标转换到HSV掩码的显示区域
                    mask_x = 180 + int(cx * 160 / roi.shape[1])
                    mask_y = 10 + int(cy * 120 / roi.shape[0])
                    
                    # 在HSV掩码上绘制小圆圈
                    cv2.circle(vis_frame, (mask_x, mask_y), 3, (0, 255, 0), -1)
                    
                    # 在HSV掩码上绘制轮廓（需要缩放）
                    contour_resized = largest_contour.copy()
                    # 正确转换坐标：先转换为float，然后缩放，最后转换为int
                    contour_resized = contour_resized.astype(np.float32)
                    contour_resized[:, :, 0] = 180 + (contour_resized[:, :, 0] * 160 / roi.shape[1])
                    contour_resized[:, :, 1] = 10 + (contour_resized[:, :, 1] * 120 / roi.shape[0])
                    contour_resized = contour_resized.astype(np.int32)
                    cv2.drawContours(vis_frame, [contour_resized], -1, (0, 255, 0), 2)
                    
                    # 在HSV掩码上显示检测状态
                    cv2.putText(vis_frame, "DETECTED", (mask_x + 5, mask_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    
                    # 在形态学处理后的掩码上也标注成功检测的光斑
                    # 将坐标转换到形态学掩码的显示区域
                    binary_x = 350 + int(cx * 160 / roi.shape[1])
                    binary_y = 10 + int(cy * 120 / roi.shape[0])
                    
                    # 在形态学掩码上绘制小圆圈
                    cv2.circle(vis_frame, (binary_x, binary_y), 3, (0, 255, 0), -1)
                    
                    # 在形态学掩码上绘制轮廓（需要缩放）
                    contour_binary = largest_contour.copy()
                    # 正确转换坐标：先转换为float，然后缩放，最后转换为int
                    contour_binary = contour_binary.astype(np.float32)
                    contour_binary[:, :, 0] = 350 + (contour_binary[:, :, 0] * 160 / roi.shape[1])
                    contour_binary[:, :, 1] = 10 + (contour_binary[:, :, 1] * 120 / roi.shape[0])
                    contour_binary = contour_binary.astype(np.int32)
                    cv2.drawContours(vis_frame, [contour_binary], -1, (0, 255, 0), 2)
                    
                    # 在形态学掩码上显示检测状态
                    cv2.putText(vis_frame, "DETECTED", (binary_x + 5, binary_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return vis_frame

    def annotate_spot_colors(self, vis_frame, original_frame):
        """在检测到的光斑上标注HSV和RGB值"""
        if self.track_bbox is None:
            return vis_frame
        
        x, y, w, h = self.track_bbox
        
        # 向内缩进到原来的85%，避免边框干扰
        shrink_factor = 0.68
        shrink_w = int(w * shrink_factor)
        shrink_h = int(h * shrink_factor)
        shrink_x = x + int(w * (1 - shrink_factor) / 2)
        shrink_y = y + int(h * (1 - shrink_factor) / 2)
        
        # 提取ROI
        roi = original_frame[shrink_y:shrink_y + shrink_h, shrink_x:shrink_x + shrink_w]
        if roi.size == 0:
            return vis_frame
        
        # 对ROI进行适度的对比度增强，与检测逻辑保持一致
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_roi[:,:,0] = clahe.apply(lab_roi[:,:,0])  # 只对亮度通道进行增强
        enhanced_roi = cv2.cvtColor(lab_roi, cv2.COLOR_LAB2BGR)
        
        # 转换到HSV
        hsv_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2HSV)
        
        # 创建HSV掩码
        hsv_mask = cv2.inRange(hsv_roi, self.HSV_LOWER, self.HSV_UPPER)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.KERNEL_SIZE)
        binary_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return vis_frame
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.MIN_SPOT_AREA:
            return vis_frame
        
        # 计算轮廓的边界矩形
        x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(largest_contour)
        
        # 检查长宽比
        if w_contour == 0 or h_contour == 0:
            return vis_frame
        
        aspect_ratio = max(w_contour, h_contour) / min(w_contour, h_contour)
        
        # 对所有光斑都应用圆形约束
        if aspect_ratio > self.MAX_ASPECT_RATIO:
            return vis_frame  # 太细长，不是圆形
        
        # 计算圆度
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return vis_frame
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 检查圆度
        if circularity < self.MIN_CIRCULARITY:
            return vis_frame  # 不够圆
        
        # 如果通过了所有检测，在光斑中心标注颜色信息
        # 计算光斑中心点（在ROI坐标系中）
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx_roi = int(M["m10"] / M["m00"])
            cy_roi = int(M["m01"] / M["m00"])
            
            # 转换到全局坐标系（考虑缩进）
            cx_global = shrink_x + cx_roi
            cy_global = shrink_y + cy_roi
            
            # 获取中心点的颜色值
            if 0 <= cx_roi < roi.shape[1] and 0 <= cy_roi < roi.shape[0]:
                # 获取BGR值
                bgr_color = roi[cy_roi, cx_roi]
                b, g, r = bgr_color
                
                # 获取HSV值
                hsv_color = hsv_roi[cy_roi, cx_roi]
                h, s, v = hsv_color
                
                # 在光斑中心绘制小圆圈
                cv2.circle(vis_frame, (cx_global, cy_global), 3, (0, 255, 0), -1)
                
                # 标注HSV值
                hsv_text = f"H:{h} S:{s} V:{v}"
                cv2.putText(vis_frame, hsv_text, (cx_global + 5, cy_global - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # 标注RGB值
                rgb_text = f"R:{r} G:{g} B:{b}"
                cv2.putText(vis_frame, rgb_text, (cx_global + 5, cy_global + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # 在光斑轮廓上绘制边界
                # 将轮廓点转换到全局坐标系（考虑缩进）
                contour_global = largest_contour.copy()
                contour_global[:, :, 0] += shrink_x
                contour_global[:, :, 1] += shrink_y
                cv2.drawContours(vis_frame, [contour_global], -1, (0, 255, 0), 2)
        
        return vis_frame

    def check_spot_shape(self, binary_mask):
        """检查光斑形状是否符合要求（圆形斑点）"""
        if binary_mask is None or cv2.countNonZero(binary_mask) == 0:
            return False
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.MIN_SPOT_AREA:
            return False
        
        # 计算轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 检查长宽比
        if w == 0 or h == 0:
            return False
        
        aspect_ratio = max(w, h) / min(w, h)
        
        # 对所有光斑都应用圆形约束
        if aspect_ratio > self.MAX_ASPECT_RATIO:
            return False  # 太细长，不是圆形
        
        # 计算圆度 (4π*面积/周长²)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return False
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 检查圆度
        if circularity < self.MIN_CIRCULARITY:
            return False  # 不够圆
        
        return True

    def get_shape_info(self, binary_mask):
        """获取形状信息用于显示"""
        shape_info = {
            'aspect_ratio': 0.0,
            'circularity': 0.0
        }
        
        if binary_mask is None or cv2.countNonZero(binary_mask) == 0:
            return shape_info
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return shape_info
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.MIN_SPOT_AREA:
            return shape_info
        
        # 计算轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 检查长宽比
        if w == 0 or h == 0:
            return shape_info
        
        aspect_ratio = max(w, h) / min(w, h)
        shape_info['aspect_ratio'] = aspect_ratio
        
        # 计算圆度
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            shape_info['circularity'] = circularity
        
        return shape_info

    def visualize(self, frame):
        """用于调试的可视化函数"""
        # 显示跟踪框
        if self.track_bbox:
            x, y, w, h = self.track_bbox
            # 检查prev_points是否为None，避免TypeError
            if self.prev_points is not None and len(self.prev_points) > self.MIN_FEATURES:
                color = (0, 255, 0)  # 绿色表示跟踪正常
            else:
                color = (0, 0, 255)  # 红色表示跟踪异常
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 显示胜利状态
        if self.victory_achieved:
            cv2.putText(frame, "VICTORY!", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 215, 255), 3)
        elif self.first_spot_time is not None:
             # 显示计时进度
            progress = (time.time() - self.first_spot_time) / self.VICTORY_DURATION
            cv2.putText(frame, f"Spot Detected: {progress:.1%}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Real-time Tracking", frame)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


# ---------------------- 如何使用这个类 ----------------------
if __name__ == '__main__':
    # 这是一个本地测试的例子，不是ROS节点
    # 在ROS中，你会在节点初始化时创建detector对象，在回调函数或主循环中调用process_frame()
    
    YOLO_MODEL_PATH = "D:/guangdian/code/code/runs/detect/train/weights/best.pt"
    
    try:
        detector = SpotDetector(yolo_model_path=YOLO_MODEL_PATH)
        
        while True:
            # 1. 核心调用：处理一帧并获取结果
            frame, bbox, victory = detector.process_frame()

            if frame is None:
                break

            # 2. 在这里，你可以使用返回的数据
            if bbox:
                print(f"当前目标框位置: {bbox}")
                # 在ROS中，这里就是你发布消息的地方
                # bbox_msg = RegionOfInterest()
                # bbox_msg.x_offset, bbox_msg.y_offset, bbox_msg.width, bbox_msg.height = bbox
                # self.bbox_pub.publish(bbox_msg)
                pass

            if victory:
                print("任务完成，可以执行停止跟踪等操作。")
                # 可以在这里停止舵机等
                
            # 3. 可视化（本地测试时使用）
            detector.visualize(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    finally:
        detector.cleanup()