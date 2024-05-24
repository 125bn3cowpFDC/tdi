import cv2
import mediapipe as mp
import numpy as np
import json
from collections import OrderedDict
a = []
pose_list = []
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
c=1
cap = cv2.VideoCapture("/home/smaipcjjh/Documents/project/ai_project/STGCN/연습_발사위2_kor.mp4")
file_data = OrderedDict()
file_data["data"] = []
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
        # Recolor image to RGB
         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         image.flags.writeable = False
      
        # Make detection
         results = pose.process(image)
    
        # Recolor back to BGR
         image.flags.writeable = True
         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                # Extract landmarks 11~28
         try:
            landmarks = results.pose_landmarks.landmark
            for i in range(11,29):
              print('i for save skeleton = ', i)
              a.append(format(landmarks[i].x, ".3f"))
              a.append(format(landmarks[i].y, ".3f"))
             
            frame_c = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print('pose count',len(a))
            file_data["data"].append({'frame_index': round(frame_c) ,"skeleton":[{'pose':a,'score':["잇힝몰랑"]}]})    
            a=[]
         except:
            pass
        
        # Render detections
         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               

         #print(type(landmarks))
         #print("apaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",c)
         c+=1
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q') or ret == False:
            break
    
    
    file_data["label"] = "잇힝이름"
    file_data["label_index"] = 1
    #저ㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓㅓ장ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ)
    with open('ithing.json', 'w', encoding="utf-8") as make_file:
      json.dump(file_data, make_file, ensure_ascii=False)    
    
    #print(a)
    cap.release()
    cv2.destroyAllWindows()
    print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
    #print(json.dumps(file_data, ensure_ascii=False, indent="\t"))
    