#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import time
import requests
from detector import SpotDetector
from uart.process import checksum_crc8
import gpiod
import time

class DetectorTracker:
    def __init__(self, yolo_model_path, camera_index=0, uart_port='/dev/ttyACM0', uart_baud=1000000):
        # ---------------------- 1. 初始化检测器 ----------------------
        self.detector = SpotDetector(yolo_model_path, camera_index)
        
        # ---------------------- 2. 初始化UART通信 ----------------------
        self.uart_port = uart_port
        self.uart_baud = uart_baud
        self.ser = None
        
        # ---------------------- 3. 舵机控制参数 ----------------------
        # 舵机当前位置（初始化为120°）
        self.servo1_current_angle = 120  # Y方向舵机
        self.servo2_current_angle = 90  # X方向舵机
        
        # 舵机角度限制
        self.servo1_min_angle = 60   # Y轴最低60°
        self.servo1_max_angle = 160  # Y轴最高160°
        self.servo2_min_angle = 0
        self.servo2_max_angle = 180
        
        # 跟踪控制参数
        self.track_interval = 15  # 每15帧执行一次跟踪（进一步降低频率）
        self.frame_counter = 0
        
        # 比例控制参数
        self.x_scale_factor = 0.4  # X方向比例系数（降低）
        self.y_scale_factor = 0.80  # Y方向比例系数（增大到0.75）
        self.min_move_threshold = 1  # 最小移动阈值（度）
        
        # 摄像头参数（自动获取）
        self.camera_center_x = None  # 摄像头中心X坐标
        self.camera_center_y = None  # 摄像头中心Y坐标
        
        # ---------------------- 4. 获取摄像头分辨率 ----------------------
        self.get_camera_center()
        
        # ---------------------- 5. 初始化舵机 ----------------------
        self.init_servos()
    
    def get_camera_center(self):
        """获取摄像头中心点坐标"""
        # 已知摄像头分辨率为640x480
        self.camera_center_x = 640 // 2  # 320
        self.camera_center_y = 480 // 2 +50  # 232 (中心下方8像素，比之前增加7像素)
        print(f'摄像头分辨率: 640 x 480')
        print(f'目标跟踪中心点: ({self.camera_center_x}, {self.camera_center_y})')
    
    def init_servos(self):
        """初始化舵机到指定位置"""
        try:
            self.ser = serial.Serial(self.uart_port, self.uart_baud, timeout=1)
            print(f'UART OPEN: {self.ser.name}')
            
            # 舵机1初始化到120°，舵机2初始化到90°
            cmd = self.create_dual_servo_command(1, 120, 2, 90, 1000)
            print(f'初始化舵机1到120°，舵机2到90°: {cmd.hex(" ").upper()}')
            self.ser.write(cmd)
            time.sleep(1.5)  # 等待舵机运动完成
            
            print('舵机初始化完成')
            
        except Exception as e:
            print(f'舵机初始化失败: {e}')
            self.ser = None
    
    def angle_to_pulse_width(self, angle):
        """将角度转换为脉宽"""
        return int(500 + (angle / 180) * 2000)
    
    def create_single_servo_command(self, servo_id, angle, duration_ms=1000):
        """创建单舵机控制命令（子命令0x03）"""
        pulse_width = self.angle_to_pulse_width(angle)
        
        data = bytes([
            0x04,  # 功能码 PACKET_FUNC_PWM_SERVO
            0x06,  # 长度
            0x03,  # 子命令（单舵机）
            duration_ms & 0xFF, (duration_ms >> 8) & 0xFF,  # 运动时间(ms)
            0x01,  # 舵机数量
            servo_id,  # 舵机ID
            pulse_width & 0xFF, (pulse_width >> 8) & 0xFF  # 脉宽
        ])
        
        crc = checksum_crc8(data)
        return bytes([0xAA, 0x55]) + data + bytes([crc])
    
    def create_dual_servo_command(self, servo1_id, angle1, servo2_id, angle2, duration_ms=1000):
        """创建双舵机控制命令（子命令0x01）"""
        pulse_width1 = self.angle_to_pulse_width(angle1)
        pulse_width2 = self.angle_to_pulse_width(angle2)
        
        data = bytes([
            0x04,  # 功能码 PACKET_FUNC_PWM_SERVO
            0x0A,  # 长度 (3*2+4=10)
            0x01,  # 子命令（多舵机）
            duration_ms & 0xFF, (duration_ms >> 8) & 0xFF,  # 运动时间(ms)
            0x02,  # 舵机数量
            servo1_id,  # 舵机1 ID
            pulse_width1 & 0xFF, (pulse_width1 >> 8) & 0xFF,  # 脉宽1
            servo2_id,  # 舵机2 ID
            pulse_width2 & 0xFF, (pulse_width2 >> 8) & 0xFF   # 脉宽2
        ])
        
        crc = checksum_crc8(data)
        return bytes([0xAA, 0x55]) + data + bytes([crc])
    
    def calculate_tracking_error(self, bbox):
        """计算跟踪误差"""
        if bbox is None:
            return 0, 0
        
        # 计算目标框中心点
        x, y, w, h = bbox
        bbox_center_x = x + w // 2
        bbox_center_y = y + h // 2
        
        # 计算与摄像头中心的差值
        error_x = bbox_center_x - self.camera_center_x
        error_y = bbox_center_y - self.camera_center_y
        
        return error_x, error_y
    
    def update_servo_angles(self, error_x, error_y):
        """根据误差更新舵机角度"""
        if self.ser is None:
            return
        
        # 计算角度变化量（比例控制）
        angle_change_x = -error_x * self.x_scale_factor  # 负号是因为舵机2的0°在最右边
        angle_change_y = error_y * self.y_scale_factor
        
        # 限制角度变化量
        angle_change_x = max(-5, min(5, angle_change_x))  # 限制每次最大变化5度
        angle_change_y = max(-1, min(1, angle_change_y))  # Y轴限制每次最大变化1度
        
        # 检查是否超过最小移动阈值
        if abs(angle_change_x) < self.min_move_threshold and abs(angle_change_y) < self.min_move_threshold:
            return
        
        # 计算新的舵机角度
        new_servo1_angle = self.servo1_current_angle + angle_change_y
        new_servo2_angle = self.servo2_current_angle + angle_change_x
        
        # 限制舵机角度范围
        new_servo1_angle = max(self.servo1_min_angle, min(self.servo1_max_angle, new_servo1_angle))
        new_servo2_angle = max(self.servo2_min_angle, min(self.servo2_max_angle, new_servo2_angle))
        
        # 发送舵机控制命令
        cmd = self.create_dual_servo_command(1, new_servo1_angle, 2, new_servo2_angle, 1500)  # 增加运动时间到1.5秒
        self.ser.write(cmd)
        
        # 更新记录的舵机角度
        self.servo1_current_angle = new_servo1_angle
        self.servo2_current_angle = new_servo2_angle
        
        print(f'舵机角度更新: 舵机1={new_servo1_angle:.1f}°, 舵机2={new_servo2_angle:.1f}°')
    
    def process_frame(self):
        """处理一帧图像并执行跟踪"""
        # 使用检测器处理帧
        frame, bbox, victory = self.detector.process_frame()
        
        if frame is None:
            return None, None, False
        
        # 在帧上绘制检测框和跟踪信息
        vis_frame = self.draw_tracking_info(frame, bbox)
        
        # 更新帧计数器
        self.frame_counter += 1
        
        # 每10帧执行一次跟踪控制
        if self.frame_counter % self.track_interval == 0 and bbox is not None:
            # 计算跟踪误差
            error_x, error_y = self.calculate_tracking_error(bbox)
            
            # 更新舵机角度
            self.update_servo_angles(error_x, error_y)
        
        return vis_frame, bbox, victory
    
    def draw_tracking_info(self, frame, bbox):
        """在帧上绘制跟踪信息"""
        vis_frame = frame.copy()
        
        # 绘制目标跟踪中心点
        cv2.circle(vis_frame, (self.camera_center_x, self.camera_center_y), 5, (0, 255, 255), -1)
        cv2.putText(vis_frame, "Target Center", (self.camera_center_x + 10, self.camera_center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 绘制检测框
        if bbox is not None:
            x, y, w, h = bbox
            # 绘制检测框
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 计算并绘制目标框中心点
            bbox_center_x = x + w // 2
            bbox_center_y = y + h // 2
            cv2.circle(vis_frame, (bbox_center_x, bbox_center_y), 3, (255, 0, 0), -1)
            
            # 绘制误差线
            cv2.line(vis_frame, (self.camera_center_x, self.camera_center_y), 
                     (bbox_center_x, bbox_center_y), (0, 0, 255), 2)
            
            # 显示误差信息
            error_x, error_y = self.calculate_tracking_error(bbox)
            cv2.putText(vis_frame, f"Error: X={error_x}, Y={error_y}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示舵机角度信息
        cv2.putText(vis_frame, f"Servo1: {self.servo1_current_angle:.1f}°", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Servo2: {self.servo2_current_angle:.1f}°", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示帧计数器
        cv2.putText(vis_frame, f"Frame: {self.frame_counter}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def cleanup(self):
        """清理资源"""
        if self.ser:
            self.ser.close()
        self.detector.cleanup()


# ---------------------- 测试代码 ----------------------
if __name__ == '__main__':
    YOLO_MODEL_PATH = "/home/pi/yolo/optic/best.pt"
    tracker = None

    chip = gpiod.Chip('gpiochip4')
    line = chip.get_line(23)
    line.request(
        consumer='gpio_output',      
        type=gpiod.LINE_REQ_DIR_OUT, 
        default_vals=[0]             
    )


    
    try:
        print("正在初始化跟踪器...")
        tracker = DetectorTracker(yolo_model_path=YOLO_MODEL_PATH)
        print("跟踪器初始化完成，开始运行...")
        
        while True:
            frame, bbox, victory = tracker.process_frame()
            
            if frame is None:
                print("无法读取摄像头帧")
                break
            
            # 显示跟踪信息
            if bbox:
                x, y, w, h = bbox
                error_x, error_y = tracker.calculate_tracking_error(bbox)
                print(f'目标框: {bbox}, 误差: X={error_x}, Y={error_y}')
            
            if victory:
                print("任务完成！")
                
                # 发送UART完成指令
                if tracker.ser:
                    completion_cmd = bytes.fromhex("AA 55 02 08 E8 03 F4 01 2C 01 0A 00 8B")
                    tracker.ser.write(completion_cmd)
                    print("已发送任务完成UART指令")
                
                line.set_value(1)
                time.sleep(10)

                line.release()
                chip.close()

                break
            
            # 显示图像
            cv2.imshow("Tracking", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    except FileNotFoundError as e:
        print(f"模型文件未找到: {e}")
        print("请确保 'best.pt' 文件存在于当前目录")
    except requests.exceptions.ConnectionError as e:
        print(f"网络连接错误: {e}")
        print("请检查网络连接或确保模型文件已下载")
    except Exception as e:
        print(f"发生未知错误: {e}")
    finally:
        if tracker is not None:
            tracker.cleanup()
        cv2.destroyAllWindows() 