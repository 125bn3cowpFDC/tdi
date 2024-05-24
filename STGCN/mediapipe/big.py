import json
from collections import OrderedDict

output_json_name = 'hanchoom_train_label.json'

file_data = OrderedDict()

for i in range (1, 140):
    video_name = 'breathing_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "breathing", "label_index": 0}   

for i in range(1, 53):
    video_name = 'walking_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "walking_sawi", "label_index": 1}

for i in range(66, 118):
    video_name = 'walking_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "walking_sawi", "label_index": 1}

for i in range(131, 167):
    video_name = 'walking_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "walking_sawi", "label_index": 1}

for i in range(175, 209):
    video_name = 'walking_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "walking_sawi", "label_index": 1}

for i in range(1, 40):
    video_name = 'jump_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "jump_sawi", "label_index": 2}

for i in range(49, 63):
    video_name = 'jump_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "jump_sawi", "label_index": 2}

for i in range(1, 40):
    video_name = 'turn_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "turn_sawi", "label_index": 3}

for i in range(49, 68):
    video_name = 'turn_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "turn_sawi", "label_index": 3}

for i in range(1, 43):
    video_name = 'normp_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi", "label_index": 4}

for i in range(53, 90):
    video_name = 'normp_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi", "label_index": 4}

for i in range(99, 134):
    video_name = 'normp_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi", "label_index": 4}

for i in range(1, 37):
    video_name = 'normp_sawi2_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi2", "label_index": 5}

for i in range(46, 96):
    video_name = 'normp_sawi2_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "normp_sawi2", "label_index": 5}

for i in range(1, 35):
    video_name = 'everyone_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "everyone_sawi", "label_index": 6}

for i in range(43, 79):
    video_name = 'everyone_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "everyone_sawi", "label_index": 6}

for i in range(1, 22):
    video_name = 'wind_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "wind_sawi", "label_index": 7}

for i in range(27, 50):
    video_name = 'wind_sawi_' + str(i)
    file_data[video_name] = {"has_skeleton": bool('true'), "label": "wind_sawi", "label_index": 7}

with open(output_json_name, 'w', encoding="utf-8") as make_file:
              json.dump(file_data, make_file, ensure_ascii=False)