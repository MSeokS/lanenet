#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import serial

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger


angleK = 18 # 회전 각도 배율 

lineR = 470 # 기본 라인 위치 설정
lineL = 170

RMax = 520 # 우측 임계값 (좌측 차선) -> 좌측 차선에 민감하게 반응하면 올리고 둔감하면 내리기
RMin = 420 # 좌측 임계값 (우측 차선) -> 우측 차선에 민감하게 반응하면 내리고 둔감하면 올리기

laneMove = 0.25 # 라인에서 벗어났을 때 움직이는 정도
laneC = 20 # 라인 보정치 -> 웬만하면 건들지 말고 라인 보정 이후에 정신 못차리면 수정(lane Move 크게 바꾸면 비례해서 수정 추천)

Svalue = 50 # 색 구분값 초록색 인식되면 내리고 흰색 안보이면 올리기

sess, input_tensor, binary_seg_ret, instance_seg_ret, postprocessor = None, None, None, None, None

class ComputerVision(object):
    def canny_edge(self, img, lth, hth):
        return cv2.Canny(img.copy(), lth, hth)

    def hough_transform(self, img, rho=None, theta=None, threshold=None, mll=None, mlg=None, mode="lineP"):
        if mode == "line":
            return cv2.HoughLines(img.copy(), rho, theta, threshold)
        elif mode == "lineP":
            return cv2.HoughLinesP(img.copy(), rho, theta, threshold, lines=np.array([]),
                                   minLineLength=mll, maxLineGap=mlg)
        elif mode == "circle":
            return cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                                    param1=200, param2=10, minRadius=40, maxRadius=100)

    def calculation(self, img,  lines):
        total = 0.0
        cnt = 0
            
        if lines is None:
            return None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                cnt += 1
            else:
                m = (y2 - y1) / (x2 - x1)
                if abs(m) > 1:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    theta = np.arctan(abs(m)) - np.pi / 2
                    if m > 0:
                        theta = theta * -1

                    total += theta
                    cnt += 1 
       
        
        self.plothistogram(img)
        
        global lineL, lineR, Rmin, Rmax, laneC

        if cnt == 0:
            return -999

        result = total / cnt

        if lineR > RMax:
            lineL -= laneC
            lineR -= laneC
            return min(-1 * laneMove, result)
        if lineR < RMin:
            lineL += laneC
            lineR += laneC
            return max(laneMove, result)
        
        return result

    def plothistogram(self, image):
        global lineL, lineR
        histogram = np.sum(image[7 * image.shape[0]//8:, :], axis=0)
        indices = np.where(histogram >= 7000)[0]
        if indices.size > 0:
            tempR = indices[np.argmin(np.abs(indices - lineR))]
            tempL = indices[np.argmin(np.abs(indices - lineL))]
            if np.abs(tempR - lineR) < 30:
                a = tempR - lineR
                lineR += a
                lineL += a
                return
            if np.abs(tempL - lineL) < 30:
                a = tempL - lineL
                lineR += a
                lineL += a
                return

    def wrapping(self, image):
        points = [[ 87, 357],
    [559, 357],
    [433, 119],
    [226, 119]]

        height, width = image.shape[0], image.shape[1]
        scaled_points = [(int(p[0]), int(p[1])) for p in points]
        
        src_points = np.float32([scaled_points[0], scaled_points[1], scaled_points[3], scaled_points[2]])
        dst_points = np.float32([[160, height], [width - 160, height], [160, 0], [width - 160, 0]])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        bird_eye_view = cv2.warpPerspective(image, matrix, (width, height))
        return bird_eye_view

    def detect(self, cap):

        if not cap.isOpened():
            return None
        
        ret, img = cap.read()
        if not ret:
            return None

        img = test_lanenet(img)
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)

        bird_eye_view = self.wrapping(img) 

        # 그레이스케일 변환

        # 가우시안 블러 적용
        cv2.imshow("BEW", bird_eye_view)
       
        canny = self.canny_edge(bird_eye_view, 150, 200)
        lines = self.hough_transform(canny, 1, np.pi/180, 50, 50, 20, mode="lineP")

        reward = self.calculation(bird_eye_view, lines)

        return reward

class Arduino:
    def __init__(self):
        # 초기 상태 설정
        self.cap = cv2.VideoCapture("curv.mp4")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        #self.arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)
        time.sleep(2.0)
        self.computerVision = ComputerVision()

    def __delete__(self):
        self.cap.release()
        self.arduino.release()
        global sess
        sess.close()

    def move(self):
        # 모터 회전 후 이동
        try:
            check = 0
            while True:
                start = time.time()
                self.reward = self.take_picture()
                
                if self.reward is None:
                    print("Camera Disconnected")
                    angle = 999
                elif self.reward == -999:
                    print("No Line")
                    continue
                else:
                    angle = self.reward * angleK + 17
                                
                angle = int(angle)
                print(lineL, lineR, angle)
                
                end = time.time()
                if end - start < 0.1:
                    time.sleep(0.1 - (end - start))
                #self.arduino.flush()
                #self.arduino.write(f"{angle}\n".encode('utf-8'))
                cv2.waitKey(1)
        except serial.SerialException as e:
            print(f"Serial Error : {e}")
        except KeyboardInterrupt:
            print("stop")
        finally:
            time.sleep(1.0)
            #self.arduino.write("999\n".encode('utf-8'))
    
    def take_picture(self):
        reward = self.computerVision.detect(self.cap)
        return reward

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--with_lane_fit', type=args_str2bool, help='If need to do lane fit', default=True)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def init_lanenet(weights_path):
    global sess, input_tensor, binary_seg_ret, instance_seg_ret, postprocessor
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # 세션 구성 설정
    """
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)
    """
    sess_config = tf.ConfigProto(device_count={'GPU': 0})  # GPU를 사용하지 않음
    sess_config.gpu_options.allow_growth = True  # GPU 메모리 사용 설정을 CPU에 맞게 변경

    sess = tf.Session(config=sess_config)

    # 이동 평균 변수 설정
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # saver 정의
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

    return sess, input_tensor, binary_seg_ret, instance_seg_ret, postprocessor

def test_lanenet(image):
    """

    :param image_path:
    :param weights_path:
    :param with_lane_fit:
    :return:
    """
    
    t_start = time.time()
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0

    binary_seg_image, instance_seg_image = sess.run(
        [binary_seg_ret, instance_seg_ret],
        feed_dict={input_tensor: [image]}
    )

    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    return (binary_seg_image[0] * 255).astype(np.uint8)


if __name__ == '__main__':
    init_lanenet("./weights/tusimple_lanenet/tusimple_lanenet.ckpt")
    run = Arduino()
    run.move()
