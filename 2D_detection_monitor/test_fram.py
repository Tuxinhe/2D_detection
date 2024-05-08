import numpy as np
import pandas as pd
import os

class frame_datas():
    def __init__(self):
        self.frame_data = None
        self.image = None
        self.results = None
        self.mp_pose =  None

        self.prev_left_heel = [] # 脚跟
        self.prev_right_heel = [] # 脚跟
        self.prev_left_foot_index = [] # 脚趾
        self.prev_right_foot_index = [] # 脚趾
        self.detection_times = []
        
    def point_xyz(self):
            attributes = [
                'left_heel_x', 'left_heel_y', 'left_heel_z', 'left_heel_visibility', 
               'right_heel_x', 'right_heel_y', 'right_heel_z', 'right_heel_visibility', 
                'left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z', 'left_foot_index_visibility', 
               'right_foot_index_x', 'right_foot_index_y', 'right_foot_index_z', 'right_foot_index_visibility'
            ]
            for attr in attributes:
                setattr(self, attr, None)
            self.time = None
            self.detection_times.append(self.time)
            
            
    def point_prev(self, results, mp_pose, image_height,image_width,detection_time):
        
        self.results = results
        self.mp_pose = mp_pose
         
        self.left_heel_x = int(self.results.pose_landmarks.landmark[29].x * image_width)
        self.left_heel_y = int(self.results.pose_landmarks.landmark[29].y * image_height)
        self.left_heel_z = int(self.results.pose_landmarks.landmark[29].z)
        self.left_heel_visibility = float(self.results.pose_landmarks.landmark[29].visibility)
            #visibility
        self.right_heel_x = int(self.results.pose_landmarks.landmark[30].x * image_width)
        self.right_heel_y = int(self.results.pose_landmarks.landmark[30].y * image_height)
        self.right_heel_z = int(self.results.pose_landmarks.landmark[30].z)
        self.right_heel_visibility = float(self.results.pose_landmarks.landmark[30].visibility)
            #
        self.left_foot_index_x = int(self.results.pose_landmarks.landmark[31].x * image_width)
        self.left_foot_index_y = int(self.results.pose_landmarks.landmark[31].y * image_height)
        self.left_foot_index_z = int(self.results.pose_landmarks.landmark[31].z)
        self.left_foot_index_visibility = float(self.results.pose_landmarks.landmark[31].visibility)
            #
        self.right_foot_index_x = int(self.results.pose_landmarks.landmark[32].x * image_width)
        self.right_foot_index_y = int(self.results.pose_landmarks.landmark[32].y * image_height)
        self.right_foot_index_z = int(self.results.pose_landmarks.landmark[32].z * image_height)
        self.right_foot_index_visibility = float(self.results.pose_landmarks.landmark[32].visibility)
        
        self.detection_times.append(detection_time)
            
    def append_pose_landmarks(self):
            # 左脚跟坐标
            self.prev_left_heel.append([self.left_heel_x, self.left_heel_y, self.left_heel_z, self.left_heel_visibility])
            # 右脚跟坐标
            self.prev_right_heel.append([self.right_heel_x, self.right_heel_y, self.right_heel_z, self.right_heel_visibility])
            # 左脚趾坐标
            self.prev_left_foot_index.append([self.left_foot_index_x, self.left_foot_index_y, self.left_foot_index_z, self.left_foot_index_visibility])
            # 右脚趾坐标
            self.prev_right_foot_index.append([self.right_foot_index_x, self.right_foot_index_y, self.right_foot_index_z, self.right_foot_index_visibility])
        
            
    def combine_pose_data(self, file_path,output_file):
        # 将所有输入转换为NumPy数组
        left_heel = np.array(self.prev_left_heel)
        right_heel = np.array(self.prev_right_heel)
        left_foot_index = np.array(self.prev_left_foot_index)
        right_foot_index = np.array(self.prev_right_foot_index)
        time = np.array(self.detection_times)
        # 水平堆叠所有数组
        pose_data = np.hstack((left_heel, right_heel, left_foot_index, right_foot_index))
        
        df = pd.DataFrame(pose_data, columns=[
                'left_heel_x', 'left_heel_y', 'left_heel_z', 'left_heel_visibility',
                'right_heel_x', 'right_heel_y', 'right_heel_z', 'right_heel_visibility',
                'left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z', 'left_foot_index_visibility',
                'right_foot_index_x', 'right_foot_index_y', 'right_foot_index_z', 'right_foot_index_visibility'
                ])
        df['time'] = time 
        for path in file_path:
            # 提取文件名（不含扩展名）
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            json_file_path = output_file +"\\"+ file_name + ".json"
            # 将DataFrame转换为JSON并导出到文件
            #df.to_csv(json_file_path, index=False)
            df.to_json(json_file_path, orient="records")
            print("jason 以保存在",json_file_path)
            return json_file_path
        
        