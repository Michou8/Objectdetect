import cv2
import os

# For face recognition we will the the LBPH Face Recognizer
####Input radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold=DBL_MAX
recognizer_LBH = cv2.createLBPHFaceRecognizer(neighbors=100)
#### Input num_components=0, double threshold=DBL_MAX
recognizer_Eigen = cv2.createEigenFaceRecognizer(num_components=100)
recognizer_Fisher = cv2.createFisherFaceRecognizer(num_components=100)




FACE_MODEL_FOLDER = os.path.abspath(os.path.dirname(__file__))+'/cascade/'
detector = []
listFolder = os.listdir(FACE_MODEL_FOLDER)




for file in listFolder:
	detect = cv2.CascadeClassifier(FACE_MODEL_FOLDER+file)
	detector.append(detect)
def face_detector(image):
	""" face detection with multiple cascade model
	
	"""
        listFolder = os.listdir(FACE_MODEL_FOLDER)
        rectangles = []
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for detect in detector:
                rects = detect.detectMultiScale(image,1.1,2, cv2.cv.CV_HAAR_SCALE_IMAGE,(64,64))
		#TO-DO : crop image with more accuracy
                for r in rects:
                        r = list(r)
                        if r not in rectangles:
                                rectangles.append(r)
        res = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rectangles]
        return res
# TO-DO: implement a fucntion to get labelized face
# Call the get_images_and_labels function and get the face images and the corresponding labels
def recognition(image,x_=64,y_=64):
        images, labels = get_images_and_labels()
        labels,reverse =  encoderlabel(labels)
        labels = np.array(labels)
        if len(list(set(labels)))<2:
                return []
        recognizer_Eigen.train(images, labels)
        recognizer_Fisher.train(images, labels)
        #recognizer_LBH.train(images, labels)
        # Append the images with the extension .sad into image_paths
        #{for image_path in image_paths:
        #predict_image_pil = Image.open(image_path).convert('L')
        predict_image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #predict_image_pil = cv2.resize()
        #predict_image = np.array(predict_image_pil, 'uint8')
        faces = reco.face_detector(image_path)
        #faces = faceCascade.detectMultiScale(predict_image_pil)
        label = []
        for (x, y, w, h) in faces:
                newimage = cv2.resize(predict_image_pil[y: y + h, x: x + w],(x_,y_))
                nbr_predicted, conf = recognizer_Eigen.predict(newimage)
                result = 1-conf/(x_*y_)
                label.append({'crop':(x, y, w, h),'label':reverse[nbr_predicted],'confiance':result,'algo':'EIGEN'})
                ###################################################
                nbr_predicted_f, conf_f = recognizer_Fisher.predict(newimage)
                result = 1-conf_f/(x_*y_)
                label.append({'crop':(x, y, w, h),'label':reverse[nbr_predicted_f],'confiance':result,'algo':'FISHER'})
                ###################################################
                #nbr_predicted_l, conf_l = recognizer_LBH.predict(newimage)
                #result = 1-conf_l/(x_*y_)
        return label
