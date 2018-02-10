import cv2
import sys
import os

class data_reader():

	def __init__(self, fname, colnames=1):
		
		self.fname = fname
		self.ROOT_DIR = os.getcwd()

		# default: column name is present - skip first row when reading the data
		self.colnames = colnames

		# To be changed accordingly
		self.BBOX_IMAGE_PATH = os.path.join(self.ROOT_DIR, 'bbox')	
		self.MASKED_IMAGE_PATH = os.path.join(self.ROOT_DIR, 'mask')

	def get_img_data(self, path, format=".png"):
		data = []
		print('Reading image data from ' + path)

		with open(self.fname, 'r') as f:
			if self.colnames == 1:
				next(f)			
			for line in f:
				line_data = line.split(',')
				img_name = line_data[1].strip('"')
				img_path = os.path.join(path, img_name) + format
#				print(img_path)
				breed_id = int(line_data[2].strip('\n'))
				img = cv2.imread(img_path)
				img = cv2.resize(img, (256, 256))		# resize to (256, 256) - tbc
				data_row = [img, breed_id]
				data.append(data_row)

		return data

	def get_bboxed_img(self):
		bboxed_img = self.get_img_data(self.BBOX_IMAGE_PATH)
		return bboxed_img

	def get_masked_img(self):
		masked_img = self.get_img_data(self.MASKED_IMAGE_PATH)
		return masked_img

def testing_reader():
	fname = 'label_with_id.csv'
	reader = data_reader(fname)
	TESTING_READER_PATH = os.path.join(reader.ROOT_DIR, 'train')
	data = reader.get_img_data(TESTING_READER_PATH, format=".jpg")
	
	for i in range(10):
		img = data[i][0]
		breed_id = data[i][1]
		print("breed_id: ", breed_id)
		cv2.imshow('image', img)
		cv2.waitKey(0)

#testing_reader()


