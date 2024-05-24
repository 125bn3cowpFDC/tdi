import json
from collections import OrderedDict

output_json_name = 'hanchoom_val_label.json'

file_data = OrderedDict()

for i in range (140, 172):
    video_name = 'breathing_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "breathing", "label_index": 0}   

for i in range(53, 66):
    video_name = 'walking_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "walking_sawi", "label_index": 1}

for i in range(118, 131):
    video_name = 'walking_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "walking_sawi", "label_index": 1}

for i in range(167, 175):
    video_name = 'walking_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "walking_sawi", "label_index": 1}

for i in range(209, 217):
    video_name = 'walking_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "walking_sawi", "label_index": 1}

for i in range(40, 49):
    video_name = 'jump_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "jump_sawi", "label_index": 2}

for i in range(63, 67):
    video_name = 'jump_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "jump_sawi", "label_index": 2}

for i in range(40, 49):
    video_name = 'turn_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "turn_sawi", "label_index": 3}

for i in range(63, 68):
    video_name = 'turn_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "turn_sawi", "label_index": 3}

for i in range(43, 53):
    video_name = 'normp_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi", "label_index": 4}

for i in range(90, 99):
    video_name = 'normp_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi", "label_index": 4}

for i in range(134, 143):
    video_name = 'normp_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi", "label_index": 4}

for i in range(37, 46):
    video_name = 'normp_sawi2_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi2", "label_index": 5}

for i in range(96, 108):
    video_name = 'normp_sawi2_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi2", "label_index": 5}

for i in range(35, 43):
    video_name = 'everyone_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "everyone_sawi", "label_index": 6}

for i in range(79, 88):
    video_name = 'everyone_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "everyone_sawi", "label_index": 6}

for i in range(22, 27):
    video_name = 'wind_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "wind_sawi", "label_index": 7}

for i in range(50, 55):
    video_name = 'wind_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "wind_sawi", "label_index": 7}

with open(output_json_name, 'w', encoding="utf-8") as make_file:
              json.dump(file_data, make_file, ensure_ascii=False)