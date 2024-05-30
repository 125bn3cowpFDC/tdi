# Korean Traditional Dance Lesson
+ Language : Python3.7
+ Framework & Model : Mediapipe-Pose, ST-GCN, Pytorch

---
## ML Model
### ST-GCN

![gra](https://github.com/125bn3cowpFDC/tdi/assets/170291905/4f6cfe8f-c21e-4f5d-a9bd-990594d23a20)



- 스켈레톤의 시퀀스로부터 신체 포인트의 시공간적 연결을 통한 그래프를 통한 동작추정


 ### Dataset
data custumizing
 ```
# Extract landmarks 0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28
# Joint index:
    # 0 = {0,  "Nose"} -> 0
    # (11+12)/2 = {1,  "Neck"} -> 17,
    # 11 = {2,  "RShoulder"} -> 5,
    # 13 = {3,  "RElbow"} -> 7,
    # 15 = {4,  "RWrist"} -> 9,
    # 12 = {5,  "LShoulder"} -> 6,
    # 14 = {6,  "LElbow"} -> 8,
    # 16 = {7,  "LWrist"} -> 10,
    # 23 = {8,  "RHip"} -> 11,
    # 25 = {9,  "RKnee"} -> 13,
    # 27 = {10, "RAnkle"} -> 15,
    # 24 = {11, "LHip"} -> 12,
    # 26 = {12, "LKnee"} -> 14,
    # 28 = {13, "LAnkle"} -> 16,
    # 2 = {14, "REye"} -> 1,
    # 5 = {15, "LEye"} -> 2,
    # 7 = {16, "REar"} -> 3,
    # 8 = {17, "LEar"} -> 4,

for i in range(0, 29):
    #print('CHECK i for save skeleton = ', i)

    if ((i==0) or (i==2) or (i==5) or (i==7) or (i==8) or (i==11) or (i==12) or (i==13) or (i==14) 
            or (i==15) or (i==16) or (i==23) or (i==24) or (i==25) or (i==26) or (i==27) or (i==28)):
        a.append(round(landmarks[i].x, 3))
        a.append(round(landmarks[i].y, 3))

    a.append(round((landmarks[11].x + landmarks[12].x)/2, 3))
    a.append(round((landmarks[11].y + landmarks[12].y)/2, 3))

 ```
+ 기존 st-gcn에 사용된 skeleton모델은 openPose에서 사용 -> mediapipe-pose로 추출방법 변경에 따른 point 인지
```
if count <= 100:
    file_data["data"].append({'frame_index': round(count) ,"skeleton":[{'pose':a,'score':[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]}]})
    a=[]

    elif count > 100:
    print("LENGTH: ", len(file_data["data"]))
    file_data["data"].pop(0)
    file_data["data"].append({'frame_index': round(count) ,"skeleton":[{'pose':a,'score':[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]}]})
    for i in range(100):
        file_data["data"][i]['frame_index'] = i+1
    a=[]
```
```
if ((count>=100) and ((count % 10)==0)):
    print('###### SAVE THE DATA ######')
    file_data["label"] = label
    file_data["label_index"] = label_index

```
+ 데이터 하나 당 100프레임 간 정보 저장
+ 이후 10프레임 씩 영상이 진행되며 데이터 생성
---
###Piprline
![stm](https://github.com/125bn3cowpFDC/tdi/assets/170291905/881590f5-ed4f-411c-8dd7-bb361c01d375)

