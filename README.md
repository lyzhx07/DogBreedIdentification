# DogBreedIdentification

## Pathways
To accomplish the classification of dog breeds, we came up with three different pathways.
1. Crop body & classification
2. Facial key point detection & calculate and crop facial region & classification 
3. Crop face & classification **~~FIRST, WE WILL TRY THIS ONE~~**
4. Use pre-trained model to mark the region of the dog, crop down the region, put it on a black canvas, then do the claasification **NOW LETS TRY THIS ONE**

## WORKING PROCEDURE								
|FROM						|USE 						|GET 						|
|---------------------------|---------------------------|---------------------------|
|~~Original pics~~			|~~Manually label~~			|~~600 labeled data~~		|
|~~Pics & labels~~			|~~Resize function~~		|~~Shaped pictures & labels~~|
|~~600 Shaped labeled data~~|~~Train network1~~			|~~Trained label_AI~~		|
|~~Shaped pics~~ 			|~~Trained label_AI~~		|~~10k shaped labels~~		|
|~~10k Shaped labels~~		|~~Reversed resize function~~|~~Original labels~~		|
|Original pics				|Pre-trained model			|10k labels					|
|Original labels & pics 	|Crop(opencv2)				|10k ~~faces~~bodies(different sizes)|
|10k ~~faces~~bodies 		|Resize 					|10k resized pics 			|
|10k resized pics 			|Train network2				|Trained main_model			|
|**COMBINE** 				|**EVERYTHING**				|**TOGETHER**				|
|10k test pics 				|Combined Trained network 	|Test result 				|

## Current Works
1. Create a image name list.(copy function from label.py) and generate bbox and mask from Mask-RCNN.
2. Post-process Mask_RCNN output.
3. Original pictures cropping function.

## Options
1. Change classifier.
2. Use multiple clsasifiers and combine the results.
3. Use different models and combine the results.
4. Pretreat the pics with more openCV functions including change brightness, contrast, saturation, rotation, etc.
5. **Give up.**

## Something about pre-trained model
Mask_RCNN not accurate enough.

P3 cannot distinguish two dogs next to each other.

Temporary best solution: get masks and boxes from Mask_RCNN, put mask region of the picture into the box with pure black background.
