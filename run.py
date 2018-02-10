import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

import coco
import utils
import model as modellib
import visualize

# get_ipython().magic('matplotlib inline')

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
SAVE_DIR = os.path.join(ROOT_DIR, "saves")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
			   'bus', 'train', 'truck', 'boat', 'traffic light',
			   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
			   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
			   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
			   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
			   'kite', 'baseball bat', 'baseball glove', 'skateboard',
			   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
			   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
			   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
			   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
			   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
			   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
			   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
			   'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection

# Load images from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
BSIZE = 1000
STARTPOINT = 0
img_names = []
images = []
with open('labels.csv') as labels:
	for label in labels:
		label = label.strip()
		label = label.split(',')
		if label[0] == 'id':
			continue
		img_names.append(label)

for b in range(BSIZE):
	if b < STARTPOINT:
		continue
	img_path = os.path.join(IMAGE_DIR, img_names[b][0]+'.jpg')
	print(img_path)
	image = skimage.io.imread(img_path)

	results = model.detect([image], verbose=1)
	result = results[0]

	# COPY FROM CROP.PY--------------------------------------

	rois = result['rois']
	masks = result['masks']
	class_ids = result['class_ids'] 
	scores = result['scores']
#	print(scores)
#	print(class_ids)
	num = class_ids.shape[0]
	print(num)

	# class_id of the object to be cropped
	id = 17		# class id of dog

	# To get the index of dog region with the highest score
	for i in range(num):
		if class_ids[i] == id:  	
			cp = image.copy()
			# Box
			y1, x1, y2, x2 = result['rois'][i]
			cropped_img = cp[y1:y2, x1:x2]
			sav_path = os.path.join(SAVE_DIR, img_names[b][0])
			skimage.io.imsave(sav_path+'_b.png', cropped_img)

			# Mask
			# If pixel selected in mask - mask value 1
			# Otherwise - mask value 0
			mask = masks[:, :, i]
			row, col = image.shape[0], image.shape[1]
			# cp = image.copy()
			for r in range(row):
			    for c in range(col):
			        if mask[r][c] == 0:
			            cp[r][c]=[0,0,0] 	# set unwanted pixels to be black in colour

			detected = cp[y1:y2, x1:x2]		# crop the image according to the box
			#viewer = ImageViewer(detected)
			#viewer.show()
			skimage.io.imsave(sav_path+'_m.png', detected)

	# --------------------------------------------------------------------