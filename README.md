# Object detect based on Caffe and Opencv 
=========================================
	
Call as API in your server

# Requirement :
	pip install -r requirement.txt
	
# Run the server :
	cd cv_api
	python manger.py runserver
	
# Usages :
	- local data : curl -X POST -F image=@<path/to/imagefile> 'http://localhost:8000/face_detection/detect/' ; echo ""
	- image from internet : curl -X POST 'http://localhost:8000/face_detection/detect/' -d 'url=http://www.pyimagesearch.com/wp-content/uploads/2015/05/obama.jpg' ; echo ""


Caffe model ZOO:
	- https://github.com/BVLC/caffe/wiki/Model-Zoo

#Refrences:
	- Face detection api : http://www.pyimagesearch.com/2015/05/11/creating-a-face-detection-api-with-python-and-opencv-in-just-5-minutes/
	- Caffe : http://caffe.berkeleyvision.org/
