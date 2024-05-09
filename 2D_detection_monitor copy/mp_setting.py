import cv2
import mediapipe as mp

class mp_set():
    def __init__(self,file_list):
        self.cap = cv2.VideoCapture(file_list)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode = True,
            enable_segmentation = True,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.image =None
        self.results = None
    
   
    def mp_setting(self):
        self.success ,self.image =self.cap.read() 
        if not self.success:
            print("Failed to read frame from video.")
            return None, None, None          
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.image)

        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            self.image,
            self.results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        return self.image, self.results, self.mp_pose


    

