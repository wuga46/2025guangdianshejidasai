import cv2
import numpy as np

def update_thresholds(val):
    """滑动条回调函数，用于更新阈值"""
    pass

def detect_spot(frame):
    """检测光斑的函数"""
    # 获取当前滑动条的值
    h_min1 = cv2.getTrackbarPos('H_min1', 'Threshold Tuner')
    s_min1 = cv2.getTrackbarPos('S_min1', 'Threshold Tuner')
    v_min1 = cv2.getTrackbarPos('V_min1', 'Threshold Tuner')
    h_max1 = cv2.getTrackbarPos('H_max1', 'Threshold Tuner')
    s_max1 = cv2.getTrackbarPos('S_max1', 'Threshold Tuner')
    v_max1 = cv2.getTrackbarPos('V_max1', 'Threshold Tuner')
    
    h_min2 = cv2.getTrackbarPos('H_min2', 'Threshold Tuner')
    s_min2 = cv2.getTrackbarPos('S_min2', 'Threshold Tuner')
    v_min2 = cv2.getTrackbarPos('V_min2', 'Threshold Tuner')
    h_max2 = cv2.getTrackbarPos('H_max2', 'Threshold Tuner')
    s_max2 = cv2.getTrackbarPos('S_max2', 'Threshold Tuner')
    v_max2 = cv2.getTrackbarPos('V_max2', 'Threshold Tuner')
    
    white_thresh = cv2.getTrackbarPos('White_Thresh', 'Threshold Tuner')
    
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 创建红色范围1的掩码
    lower1 = np.array([h_min1, s_min1, v_min1])
    upper1 = np.array([h_max1, s_max1, v_max1])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    
    # 创建红色范围2的掩码
    lower2 = np.array([h_min2, s_min2, v_min2])
    upper2 = np.array([h_max2, s_max2, v_max2])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    
    # 合并两个红色掩码
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 创建白色掩码
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)[1]
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # 找到轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在原图上绘制检测结果
    result = frame.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # 过滤小面积
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f'Area: {area:.0f}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 创建显示图像
    display = np.hstack([
        frame,
        cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    ])
    
    # 返回参数列表
    params = [h_min1, s_min1, v_min1, h_max1, s_max1, v_max1,
              h_min2, s_min2, v_min2, h_max2, s_max2, v_max2, white_thresh]
    
    return display, params

def main():
    # 创建窗口和滑动条
    cv2.namedWindow('Threshold Tuner')
    
    # 创建红色范围1的滑动条
    cv2.createTrackbar('H_min1', 'Threshold Tuner', 0, 179, update_thresholds)
    cv2.createTrackbar('S_min1', 'Threshold Tuner', 120, 255, update_thresholds)
    cv2.createTrackbar('V_min1', 'Threshold Tuner', 120, 255, update_thresholds)
    cv2.createTrackbar('H_max1', 'Threshold Tuner', 10, 179, update_thresholds)
    cv2.createTrackbar('S_max1', 'Threshold Tuner', 255, 255, update_thresholds)
    cv2.createTrackbar('V_max1', 'Threshold Tuner', 255, 255, update_thresholds)
    
    # 创建红色范围2的滑动条
    cv2.createTrackbar('H_min2', 'Threshold Tuner', 160, 179, update_thresholds)
    cv2.createTrackbar('S_min2', 'Threshold Tuner', 120, 255, update_thresholds)
    cv2.createTrackbar('V_min2', 'Threshold Tuner', 120, 255, update_thresholds)
    cv2.createTrackbar('H_max2', 'Threshold Tuner', 180, 179, update_thresholds)  # HSV中H最大179，这里初始设180可能有问题，实际会被限制在179
    cv2.createTrackbar('S_max2', 'Threshold Tuner', 255, 255, update_thresholds)
    cv2.createTrackbar('V_max2', 'Threshold Tuner', 255, 255, update_thresholds)
    
    # 创建白色阈值滑动条
    cv2.createTrackbar('White_Thresh', 'Threshold Tuner', 200, 255, update_thresholds)
    
    # 读取截图或从摄像头获取图像
    print("请选择输入方式:")
    print("1. 从摄像头捕获图像")
    print("2. 读取本地截图")
    choice = input("输入选项 (1/2): ")
    
    if choice == '1':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        ret, frame = cap.read()
        if not ret:
            print("无法捕获图像")
            return
        cap.release()
    else:
        img_path = input("请输入截图路径: ")
        frame = cv2.imread(img_path)
        if frame is None:
            print("无法读取图像，请检查路径")
            return
    
    # 循环调整阈值
    while True:
        display, params = detect_spot(frame)
        cv2.imshow('Threshold Tuner', display)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC退出
            break
        elif key == ord('s'):  # S保存参数
            with open('spot_thresholds.txt', 'w') as f:
                f.write(f"# 红色光斑检测阈值参数\n")
                f.write(f"RED_LOWER1 = np.array([{params[0]}, {params[1]}, {params[2]}])\n")
                f.write(f"RED_UPPER1 = np.array([{params[3]}, {params[4]}, {params[5]}])\n")
                f.write(f"RED_LOWER2 = np.array([{params[6]}, {params[7]}, {params[8]}])\n")
                f.write(f"RED_UPPER2 = np.array([{params[9]}, {params[10]}, {params[11]}])\n")
                f.write(f"WHITE_THRESHOLD = {params[12]}\n")
            print("阈值参数已保存到 spot_thresholds.txt")
        elif key == ord('r'):  # R重置参数
            cv2.setTrackbarPos('H_min1', 'Threshold Tuner', 0)
            cv2.setTrackbarPos('S_min1', 'Threshold Tuner', 120)
            cv2.setTrackbarPos('V_min1', 'Threshold Tuner', 120)
            cv2.setTrackbarPos('H_max1', 'Threshold Tuner', 10)
            cv2.setTrackbarPos('S_max1', 'Threshold Tuner', 255)
            cv2.setTrackbarPos('V_max1', 'Threshold Tuner', 255)
            
            cv2.setTrackbarPos('H_min2', 'Threshold Tuner', 160)
            cv2.setTrackbarPos('S_min2', 'Threshold Tuner', 120)
            cv2.setTrackbarPos('V_min2', 'Threshold Tuner', 120)
            cv2.setTrackbarPos('H_max2', 'Threshold Tuner', 179)
            cv2.setTrackbarPos('S_max2', 'Threshold Tuner', 255)
            cv2.setTrackbarPos('V_max2', 'Threshold Tuner', 255)
            
            cv2.setTrackbarPos('White_Thresh', 'Threshold Tuner', 200)
            print("参数已重置")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()