import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

import time

import os
import sys

####sound
#from pydub import AudioSegment
#from pydub.playback import play

#####line notification
from line_notify import *
####end line notification

'''import imagezmq
import threading
import socket

# Helper class implementing an IO deamon thread
class VideoStreamSubscriber:
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Timeout while reading from subscriber tcp://{}:{}".format(self.hostname, self.port))
        self._data_ready.clear()
        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        while not self._stop:
            self._data = receiver.recv_jpg()
            self._data_ready.set()
        # Close socket here, not implemented in ImageHub :(
        # zmq_socket.close()

    def close(self):
        self._stop = True

hostname = "192.168.1.128"
port = 5555
receiver = VideoStreamSubscriber(hostname, port)'''

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def cal_slope(pt1,pt2):
    return (pt1[1]-pt2[1])/(pt1[0]-pt2[0])

def run_demo(net, height_size, track, smooth, record_vid, camera_type):
    net = net.eval()
    net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []

    ##Tarit defined
    slope_threshold = 0.4
    ear_slope_threshold = 0.5
    eye_ear_slope_threshold = 0.5
    not_detected = (-1,-1)
    sleep_confirmation_time = 45 #in seconds

    #flags to detect whether the person is sleeping or not
    sleeping = False

    timer_started = False

    time_notified = 0
    time_not_sleep = 0

    while True:
        #msg, frame = receiver.receive(timeout = 60.0)
        #img = cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)
        img = cap.read()
        if camera_type == "jetson":
            img = img[1300:1780,1320:1960]

        #start_time = time.time()
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        '''for pose in current_poses:
            pose.draw(img)'''

        ##find longest_nect_to_nose_dst and select that pose
        longest_nect_to_nose_dst = 0
        for pose in current_poses:
            nose = tuple(pose.keypoints[0])
            neck = tuple(pose.keypoints[1])
            ##pythagoras
            nect_to_nose_dst = pow((pow(abs(nose[0] - neck[0]) ,2)) + (pow(abs(nose[1] - neck[1]) ,2)),1/2)
            if nect_to_nose_dst > longest_nect_to_nose_dst:
                longest_nect_to_nose_dst = nect_to_nose_dst
                selected_pose = pose

        selected_pose.draw(img)

        nose = tuple(selected_pose.keypoints[0])
        neck = tuple(selected_pose.keypoints[1])
        l_ear = tuple(selected_pose.keypoints[16])
        r_ear = tuple(selected_pose.keypoints[17])
        l_eye = tuple(selected_pose.keypoints[15])
        r_eye = tuple(selected_pose.keypoints[14])

        print(cal_slope(l_eye,l_ear),cal_slope(r_eye,r_ear))
        
        ##detect if the person back if facing to the camera
        if nose == (-1,-1):
            if l_ear != not_detected and r_ear != not_detected:
                ear_slope = abs(l_ear[1] - r_ear[1])/abs(l_ear[0]-r_ear[0])
                cv2.circle(img,l_ear,5,(255,0,0),3)
                cv2.circle(img,r_ear,5,(0,255,0),3)
                if ear_slope > ear_slope_threshold:
                    sleeping = True
                    print("sleeping")
                else:
                    sleeping = False
            else:
                ##out of condition, can't detect
                sleeping = False
        else:
            cv2.circle(img,nose,5,(255,0,0),3)
            cv2.circle(img,neck,5,(0,255,0),3)

            slope_inverse = (nose[0] - neck[0])/ (nose[1] - neck[1])
            l_ear_eye_slope = cal_slope(l_eye,l_ear)
            r_ear_eye_slope = cal_slope(r_eye,r_ear)
            
            #increase the slope_threshold if the person is turning their head
            #print(pose.keypoints[16],pose.keypoints[17]) #print ear location
            if l_ear == (-1,-1) or r_ear == (-1,-1):
                slope_threshold = 1
                print("one ear missing , Increasing slope_threshold")
            else:
                slope_threshold = 0.4

            if abs(slope_inverse) > slope_threshold:
                #cv2.putText(img,"".join([str(pose.id),"sleeping"]),(20,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)
                print("Sleeping (neck bend more than threshold)") 
                #cv2.putText(img,"sleeping",(20,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)
                sleeping = True
            
            elif l_eye == not_detected or r_eye == not_detected:
                sleeping = True
                print("Sleeping (not seeing both eyes)")     

            elif l_ear_eye_slope < -0.6 or r_ear_eye_slope > 0.6 or l_ear_eye_slope > eye_ear_slope_threshold or r_ear_eye_slope < -eye_ear_slope_threshold:
                sleeping = True
                print("Sleeping (ears higher/lower than eyes)")  

            else:
                time_not_sleep += 1
                print("not sleeping",time_not_sleep)
                if time_not_sleep > 10:
                    sleeping = False
                    time_not_sleep = 0

        if sleeping:
            if not timer_started:
                t_start_sleep = time.time()
                timer_started = True
            else:
                if time.time() - t_start_sleep > sleep_confirmation_time:
                    print("sending line message")
                    pic_name = "".join(["log_data/",str(time_notified),".jpg"])
                    cv2.imwrite(pic_name,img)
                    #lineNotify("Elderly sleeping %d"%time_notified)
                    notifyFile("Elderly sleeping %d"%time_notified,pic_name)
                    time_notified += 1
                    timer_started = False
                    sleeping = False
        else:
            timer_started = False
            
            #song = AudioSegment.from_mp3("Alarm_Clock_Sound.mp3")
            #play(song)
            #print("poped",pose.id)

        img = cv2.addWeighted(orig_img, 0.6, img, 0.6, 0)

        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        cv2.imshow('Sleep detector', img)
        if record_vid:
            out_raw.write(orig_img)
            out_pose.write(img)

        #print((1/(time.time()-start_time)))
        if cv2.waitKey(1) == 27:  # esc
            #receiver.close()
            cap.stop()
            if record_vid:
                out_raw.release()
                out_pose.release()
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, default="checkpoint_iter_370000.pth", help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--camera', type=str,required=True, help='select jetson or webcam')
    parser.add_argument('--record', action="store_true", help='record to video file')
    args = parser.parse_args()

    if not os.path.isdir("log_data"):
        os.mkdir("log_data")
    if args.record:
        if os.path.isfile("log_data/out_no_vis.avi") or os.path.isfile("log_data/out_with_vis.avi"):
            print("video exist, quitting")
            sys.exit()

        fourcc = cv2.VideoWriter_fourcc(*'X264')
    
        out_raw = cv2.VideoWriter("log_data/out_no_vis.avi",fourcc,5.0,(640,480))
        out_pose = cv2.VideoWriter("log_data/out_with_vis.avi",fourcc,5.0,(640,480))

    if args.camera == "webcam":
        from threaded_cam import ThreadedCamera
        cap = ThreadedCamera()

    elif args.camera == "jetson":
        from threaded_cam import jetson_csi_camera
        camSet = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, framerate=21/1, format=NV12 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
        cap = jetson_csi_camera(camSet)

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    run_demo(net, args.height_size, args.track, args.smooth,args.record, args.camera)
