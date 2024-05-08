import cv2
import mediapipe as mp
from mp_setting import mp_set
from test_fram import frame_datas
from data_a import data_analysis
#from test_iris import DataAnalyzer

class VideoPlayer:
    def __init__(self, video_filename):
        self.video_filename = video_filename
        self.mp_set  = mp_set(self.video_filename)
        self.frame_data = frame_datas()
        self.data_analysis = data_analysis()
        #self.analyzer = DataAnalyzer()
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        self.image = None
        self.results = None
        self.mp_pose =  None
        self.detection_time = []
        self.output_file_path = None
        

    def play(self, file_path):
        with self.mp_set.pose:
            while self.mp_set.cap.isOpened():
                self.image, self.results, self.mp_pose = self.mp_set.mp_setting()
                if self.image is not None:    
                    if self.mp_set.results.pose_landmarks:
                        image_height, image_width, _ = self.image.shape
                        self.detection_time = (self.mp_set.cap.get(cv2.CAP_PROP_POS_MSEC) /1000)  # 獲取當前影格的偵測時間
                        self.frame_data.point_prev(self.results, self.mp_pose, image_height,image_width, self.detection_time)
                        self.frame_data.append_pose_landmarks()
                    else:
                        self.frame_data.point_xyz()
                        self.frame_data.append_pose_landmarks()
                    cv2.imshow('video',self.image)
                    if cv2.waitKey(1) & 0xFF == 27 :
                        break
                else :
                        break
            self.output_file_path = self.frame_data.combine_pose_data(self.video_filename, file_path)
            self.mp_set.cap.release()
            cv2.destroyAllWindows()
            self.data_analysis.data_analysis(self.output_file_path,file_path)
            

        

        
    