import mediapipe as mp
import cv2
import numpy as np

import torch
import torch.nn as nn

import argparse
import yaml
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():

    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default='./config/NTU-RGB-D/xview/ST_GCN.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=128,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')

    return parser


def point_return(landmark):
    '''
    for i in range(0, 29):
        if ((i==0) or (i==2) or (i==5) or (i==7) or (i==8) or (i==11) or (i==12) or (i==13) or (i==14) 
            or (i==15) or (i==16) or (i==23) or (i==24) or (i==25) or (i==26) or (i==27) or (i==28)):

            a.append(round(landmark[i].x, 3))
            a.append(round(landmark[i].y, 3))
            
        a.append(round((landmark[11].x + landmark[12].x)/2, 3))
        a.append(round((landmark[11].y + landmark[12].y)/2, 3))
    '''

    a = []

    a.append(round(landmark[0].x, 3))
    a.append(round(landmark[0].y, 3))

    a.append(round(landmark[2].x, 3))
    a.append(round(landmark[2].y, 3))

    a.append(round(landmark[5].x, 3))
    a.append(round(landmark[5].y, 3))

    a.append(round(landmark[7].x, 3))
    a.append(round(landmark[7].y, 3))

    a.append(round(landmark[8].x, 3))
    a.append(round(landmark[8].y, 3))

    a.append(round(landmark[11].x, 3))
    a.append(round(landmark[11].y, 3))

    a.append(round(landmark[12].x, 3))
    a.append(round(landmark[12].y, 3))

    a.append(round(landmark[13].x, 3))
    a.append(round(landmark[13].y, 3))

    a.append(round(landmark[14].x, 3))
    a.append(round(landmark[14].y, 3))

    a.append(round(landmark[15].x, 3))
    a.append(round(landmark[15].y, 3))

    a.append(round(landmark[16].x, 3))
    a.append(round(landmark[16].y, 3))

    a.append(round(landmark[23].x, 3))
    a.append(round(landmark[23].y, 3))

    a.append(round(landmark[24].x, 3))
    a.append(round(landmark[24].y, 3))

    a.append(round(landmark[25].x, 3))
    a.append(round(landmark[25].y, 3))

    a.append(round(landmark[26].x, 3))
    a.append(round(landmark[26].y, 3))

    a.append(round(landmark[27].x, 3))
    a.append(round(landmark[27].y, 3))

    a.append(round(landmark[28].x, 3))
    a.append(round(landmark[28].y, 3))

    a.append(round((landmark[11].x + landmark[12].x)/2, 3))
    a.append(round((landmark[11].y + landmark[12].y)/2, 3))

    return a



if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        print("THIS IS: ", p.config)
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    #output_device = arg.device[0] if type(arg.device) is list else arg.device
    #output_device = arg.device[0]


    #from st_gcn.net import st
    Model = import_class(arg.model)
    #model = Model(**arg.model_args).cuda(output_device)
    #model = Model(**arg.model_args).cuda(0)
    model = Model(**arg.model_args)
    model.load_state_dict(torch.load('/home/smaipcjjh/Documents/project/ai_project/STGCN/work_dir/Hanchoom/ST_GCN/Final4/epoch30_model.pt'))
    model = model.cuda(0)
    model.eval()

    '''
    from st_gcn.net import st_gcn
    model = st_gcn.Model(**arg.model_args)
    model.load_state_dict(torch.load('/home/smaipcjjh/Documents/project/ai_project/STGCN/work_dir/Hanchoom/ST_GCN/20221212/epoch60_model.pt'))
    model = model.cuda(0)
    model.eval()
    '''
    '''
    from st_gcn.net import st_gcn
    from st_gcn.graph import hanchoom
    #from st_gcn.graph import Hanchoom
    model = st_gcn.Model(channel=3, num_class=4, window_size=100, num_point=18, dropout=0 ,graph=hanchoom.Graph(), mask_learning = True, use_data_bn=True)
    model.load_state_dict(torch.load('/home/smaipcjjh/Documents/project/ai_project/STGCN/work_dir/Hanchoom/ST_GCN/20221212/epoch50_model.pt'))
    model = model.cuda(0)
    model2 = model.eval()
    '''
    # -----------------------------------------------------------------------------------------------

    label_path = '/home/smaipcjjh/Documents/project/ai_project/STGCN/8class_labels.txt'
    labeling = {}
    f = open(label_path, 'r')

    while True:
        line = f.readline()
        if not line: break
    
        strings = line.split(' ')
        labeling[int(strings[0])] = strings[1][:-1]
    f.close()


    # -----------------------------------------------------------------------------------------------

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    datapath = "/home/smaipcjjh/Documents/project/ai_project/STGCN/mediapipe/AIproject_DB/연습_발사위5_kor.mp4"
    #datapath = '/home/smaipcjjh/Documents/project/ai_project/STGCN/518_0022.MXF'
    cap = cv2.VideoCapture(datapath)

    start_frame = 1896

    # CAP SETTING
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_TEST_turn2.mp4', fourcc, fps, (frame_width, frame_height))

    cnt = 0
    landmark_list = []
    data_numpy = np.zeros((3, 100, 18, 1))
    data_numpy_in = np.zeros((3, 100, 18, 1))
    score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    temp_cnt = 0
    show_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print('CANT OPEN VIDEO')
                break

            else:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detection
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                # Get landmarks
                print('*** NOW_FRAME: ', cnt)
                landmarks = results.pose_landmarks.landmark

                landmark_list = point_return(landmarks)
                #print('LANDMARK_LIST: ', landmark_list)
                
                if cnt < 100:
                    data_numpy[0, cnt, :, 0] = landmark_list[0::2]
                    data_numpy[1, cnt, :, 0] = landmark_list[1::2]
                    data_numpy[2, cnt, :, 0] = score

                elif cnt >= 100:
                    #print('LL: ', landmark_list)
                    #print('BEFORE DATA: ', data_numpy)
                    #print('run change')
                    #print('AAA_AAA1: ', data_numpy[0, 1, 0, 0])
                    #print('AAA_AAA2: ', data_numpy[0, 2, 0, 0])
                    for i in range(99):
                        data_numpy[:, i, :, 0] = data_numpy[:, i+1, :, 0]
                    data_numpy[0, 99, :, 0] = landmark_list[0::2]
                    data_numpy[1, 99, :, 0] = landmark_list[1::2]
                    data_numpy[2, 99, :, 0] = score
                    #print('AAA_BBB1: ', data_numpy[0, 0, 0, 0])
                    #print('AAA_BBB2: ', data_numpy[0, 1, 0, 0])
                    #print('AAA_CCC: ', data_numpy[0, 99, 0, 0])
                    #print('AFTER DATA: ', data_numpy)
                    
                    if cnt % 10 == 0:
                        data_numpy_p = data_numpy.tolist()
                        data_numpy_p = np.array(data_numpy_p)
                        data_numpy_p[0:2] = data_numpy_p[0:2] - 0.5
                        data_numpy_p[0][data_numpy_p[2] == 0] = 0
                        data_numpy_p[1][data_numpy_p[2] == 0] = 0

                        #sort_index = (-data_numpy_in[2, :, :, :].sum(axis=1)).argsort(axis=1)
                        #for t, s in enumerate(sort_index):
                            #data_numpy_in[:, t, :, :] = data_numpy_in[:, t, :, s].transpose((1, 2, 0))
                        #data_numpy_in = data_numpy_in[:, :, :, 0: 1]


                        data_numpy_p = np.reshape(data_numpy_p, (1, 3, 100, 18, 1))
                        data_torch = torch.Tensor(data_numpy_p)

                    
                        data = Variable(data_torch.float().cuda(0), requires_grad=False, volatile=True)

                        outputs = model(data)
                        _, predicted = torch.max(outputs, 1)

                        softmax_out = torch.nn.functional.softmax(outputs, dim=1)

                        #softmax_out = torch.nn.functional.softmax(outputs, dim=1)
                        print('OUTPUT: ', outputs)
                        print('SOFTMAX out: ', softmax_out)
                        print('predict: ', predicted)
                        show_count += 1
                    
                    label_name = labeling[int(predicted[0])]
                    this_out = float(softmax_out[0, int(predicted[0])]) * 100
                    cv2.putText(image, label_name +':  ' + str(this_out), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


                    # -----------------------------------------------------------------------------------------------

                    breathing = float(softmax_out[0, 0]) * 100
                    walking_sawi = float(softmax_out[0, 1]) * 100
                    jump_sawi = float(softmax_out[0, 2]) * 100
                    turn_sawi = float(softmax_out[0, 3]) * 100
                    normp_sawi = float(softmax_out[0, 4]) * 100
                    normp_sawi2 = float(softmax_out[0, 5]) * 100
                    everyone_sawi = float(softmax_out[0, 6]) * 100
                    wind_sawi = float(softmax_out[0, 7]) * 100

                    breathing = round(breathing,2)
                    walking_sawi = round(walking_sawi,2)
                    jump_sawi = round(jump_sawi,2)
                    turn_sawi = round(turn_sawi,2)
                    normp_sawi = round(normp_sawi,2)
                    normp_sawi2 = round(normp_sawi2,2)
                    everyone_sawi = round(everyone_sawi,2)
                    wind_sawi = round(wind_sawi,2)

                    cv2.putText(image, 'breathing: ' + str(breathing), (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'walking_sawi: ' + str(walking_sawi), (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'jump_sawi: ' + str(jump_sawi), (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'turn_sawi: ' + str(turn_sawi), (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'normp_sawi: ' + str(normp_sawi), (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'normp_sawi2: ' + str(normp_sawi2), (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'everyone_sawi: ' + str(everyone_sawi), (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'wind_sawi: ' + str(wind_sawi), (10, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(image, 'predict conunt: ' + str(show_count), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, 'pos frame: ' + str(cnt), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)



                # Processing

                    
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               

                cnt += 1
                temp_cnt += 1
            out.write(image)
            cv2.imshow('Mediapipe Feeder', image)

            if cv2.waitKey(10) & 0xFF == ord('q') or ret == False:
                break 
            
        #print(a)
        cap.release()
        cv2.destroyAllWindows()
        out.release()