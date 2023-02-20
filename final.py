from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import numpy as np
import cv2
import time
import torch
import math
from model2 import MyResNest50


def test_images(model_path,images_dir):
    Angle=[]
    threshold=10
    image_size = 112  # 112
    model = MyResNest50(nums_class=136)
    model = torch.load(model_path, map_location= torch.device('cpu'))
    model.eval()
    image_files = os.listdir(images_dir)
    for index, image_file in enumerate(image_files):
        coor = []
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        if image is None:
            print(image_path)
        image = cv2.resize(image, (width, height))
        cv2.waitKey(0) 
        x1 = 0
        y1 = 0
        x2 = int(width)
        y2 = int(height)
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size = int(max([w, h])*1.1)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        cropped = image[y1:y2, x1:x2]
        box_w = x2-x1
        box_h = y2-y1
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)  # top,bottom,left,right

        input = cv2.resize(cropped, (image_size, image_size))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = input.astype(np.float32) / 255.0
        input = np.expand_dims(input, 0)
        input = torch.Tensor(input.transpose((0, 3, 1, 2)))

        cv2.waitKey(0)

        pre_landmarks, _ = model(input)
        pre_landmark = pre_landmarks[0].cpu().detach().numpy()
        pre_landmark = pre_landmark.reshape(-1, 2)
        pre_landmark = pre_landmark*[box_w+dx+edx, box_h+dy+edy]-[dx, dy]
        
        for (x, y) in pre_landmark.astype(np.int32):
            coor.append((x,y))
            cv2.circle(image, (x1 + x, y1 + y),2 , (0, 0, 255), -1)
        print(f"Number of landmark of image {image_file}: ",len(coor))
        if len(coor) != 0:
            vector1 = (coor[38][0] - coor[36][0],coor[38][1] - coor[36][1])
            vector2 = (coor[39][0] - coor[36][0],coor[39][1] - coor[36][1])
            mul = vector1[0]*vector2[0] + vector1[1]*vector2[1]
            length_vec1 = math.sqrt(vector1[0]*vector1[0] + vector1[1]*vector1[1])
            length_vec2 = math.sqrt(vector2[0]*vector2[0] + vector2[1]*vector2[1])
            div = mul/(length_vec1*length_vec2)
            angle = math.acos(div)*180 / math.pi
            close_eye = float(coor[39][1]) - float(coor[38][1])
            if close_eye < 0:
                angle = -1 * angle
            if angle > threshold:
                print("open")
            else :
                print("closed")
        image_name = "result" + image_file
        cv2.imwrite(image_name,image)
        if cv2.waitKey(0) == 13:
            continue
        elif cv2.waitKey(0) == 27:
            break
   
if __name__ == '__main__':
    
    model_path = './resnest50.pth'
    images_dir =   './test'
    test_images(model_path,images_dir)
