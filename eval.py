from glob import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
import cv2
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import time

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield(x, y, image[y:y+windowSize[1], x:x+windowSize[0]])
 
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")    
       
model = torch.load('C:/Users/Arthur/Desktop/test/model.pth')
img = cv2.imread('7.jpg')
print(img.shape)
img = np.array(img)
imgc = cv2.imread('7c.jpg')
rect1 = []
rect2 = []
rect3 = []
rect4 = []
scales = [(40,42), (50,51), (56, 60),(63,65)]
#scales = [(53,55)]
for (wh,ww) in scales:
    for (x, y, window) in sliding_window(img, stepSize=5, windowSize=(wh,ww)):
        if window.shape[0] == ww and window.shape[1] == wh:
            cv2.imwrite('test.png', window)
            imgs = Image.open('test.png')
            #window = Image.fromarray(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
            #window = torch.tensor(window)
        #c = window.numpy()
        #window = window.permute(2,0,1)
        #print(window.shape)
        #print(c.shape)
            simple_transform = transforms.Compose([transforms.Resize((40,40)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            window = simple_transform(imgs)
            inputs = Variable(torch.unsqueeze(window, dim=0).float(), requires_grad=False).cuda()
            outputs = model(inputs)
            probability = torch.nn.functional.softmax(outputs,dim=1)
            max_value,index = torch.max(probability,1)
            print(index)
            if max_value > 0.98:
                if index == 0:
                    rect1.append([x, y, x+wh,y+ww,index])
                elif index == 1:
                    rect2.append([x, y, x+wh,y+ww,index])
                elif index == 2:
                    rect3.append([x, y, x+wh,y+ww,index])
                elif index == 3:
                    rect4.append([x, y, x+wh,y+ww,index])
                    
r1 = np.array(rect1+rect3)
r4 = np.array(rect4+rect2)
p1 = non_max_suppression(r1,overlapThresh=0.05)
p2 = non_max_suppression(r4,overlapThresh=0.05)
for (xA, yA, xB, yB, idx) in p1:
        if idx == 0:
            cv2.rectangle(imgc, (xA, yA), (xB, yB), (0, 255, 0), 2)
        else:
            cv2.rectangle(imgc, (xA, yA), (xB, yB), (255, 0, 0), 2)
for (xA, yA, xB, yB, inde) in p2:
        if idx == 1:
            cv2.rectangle(imgc, (xA, yA), (xB, yB), (0, 0, 255), 2)
        else:
            cv2.rectangle(imgc, (xA, yA), (xB, yB), (0, 255, 255), 2)
cv2.imwrite('result.png', imgc)
