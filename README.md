# Object detect based on Caffe and Opencv 
=========================================
Describe your images by calling an API (in your server)

# Requirement :
	- pip install -r requirement.txt (python lib needed)
	- Don't forget to download caffe model and cascade model
	[
		put caffemodel and deploy file into (you can also put mean file on it or put your own model):
			cv_api/face_detector/bvlc_model/bvlc_alexnet/
		        cv_api/face_detector/bvlc_model/bvlc_googlenet/ <-- NOT USE
		        cv_api/face_detector/bvlc_model/bvlc_reference_caffenet/ <-- NOT USE
		        cv_api/face_detector/bvlc_model/bvlc_reference_rcnn_ilsvrc13/ <-- NOT USE
		        cv_api/face_detector/bvlc_model/finetune_flickr_style/ <-- NOT USE
	]		
# Run the server :
	cd cv_api
	python manger.py runserver
	
# Usages :
	- local data : curl -X POST -F image=@<path/to/imagefile> 'http://localhost:8000/face_detection/detect/' ; echo ""
	- image from internet : curl -X POST 'http://localhost:8000/face_detection/detect/' -d 'url=http://www.pyimagesearch.com/wp-content/uploads/2015/05/obama.jpg' ; echo ""


# Caffe model ZOO:
	- https://github.com/BVLC/caffe/wiki/Model-Zoo

#Refrences:
	- Face detection api : http://www.pyimagesearch.com/2015/05/11/creating-a-face-detection-api-with-python-and-opencv-in-just-5-minutes/
	- Caffe : http://caffe.berkeleyvision.org/
#TO-DO:
	- Data base for recgnition 
	- Increase the accuracy of prediction by trained more the model
