def crop(r, image):
	'''
	- crop the image based on box
	- only show the colour if the mask value is 1
	- pad pixel with mask value 0 with black colour [0, 0, 0]
	- input: detection results of an image and the image
		in demo.ipynb:
		results = model.detect([image], verbose=1)
		r = results[0]
	- return: cropped image with black background
	- return 
	'''

	rois = r['rois']
	masks = r['masks']
	class_ids = r['class_ids'] 
	scores = r['scores']
#	print(scores)
#	print(class_ids)
	num = class_ids.shape[0]
	print(num)

	# class_id of the object to be cropped
	id = 17		# class id of dog

	# To get the index of dog region with the highest score
	for i in range(num):
		if class_ids[i] == id:  
			break
		if i == num-1:
			return None	# no dog detected from the image, return None
	
	# Box
	y1, x1, y2, x2 = r['rois'][i]

	# Mask
	# If pixel selected in mask - mask value 1
	# Otherwise - mask value 0
	mask = masks[:, :, i]
	row, col = image.shape[0], image.shape[1]
	cp = image.copy()
	for r in range(row):
	    for c in range(col):
	        if mask[r][c] == 0:
	            cp[r][c]=[0,0,0] 	# set unwanted pixels to be black in colour

	detected = cp[y1:y2, x1:x2]		# crop the image according to the box
#	cv2.imshow('image', detected)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()
#	cv2.waitKey(1)
#	print("done")
	return detected
