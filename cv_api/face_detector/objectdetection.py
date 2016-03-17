import json
import numpy as np
import operator
import cv2
import os
# Define your caffe directory
caffe_root = '/home/devuser/caffe/'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
CAFFE_MODEL_FOLDER = os.path.abspath(os.path.dirname(__file__))+'/bvlc_model/'

def prediction(deploy,caffe_model):
        net = caffe.Net(deploy,
                        caffe_model,
                        caffe.TEST)# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        # set net to batch size of 50
        net.blobs['data'].reshape(50,3,227,227)
	#img = path
	#transformer = {}
	return net,transformer
	#net.blobs['data'].data[...] = transformer.preprocess('data', img)
def predictionClassi(model_file,pretrained):
        net = caffe.Classifier(model_file, pretrained,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
	return net
	
fRoot = CAFFE_MODEL_FOLDER

######## ALEX archi #########
deploy = 'bvlc_alexnet/deploy.prototxt'
caffe_model = 'bvlc_alexnet/bvlc_alexnet.caffemodel'
deploy = fRoot + deploy
caffe_model = fRoot + caffe_model
#########################################
net_alex,transformer_alex = prediction(deploy=deploy,caffe_model=caffe_model)
def alex(img):
	net_alex.blobs['data'].data[...] = transformer_alex.preprocess('data', img)
	#print net_alex.predict([img])
	out = net_alex.forward()
	present = {}
	cat =  out.keys()[0]
	target = ''
	if cat != 'prob':
	        target = fRoot+'target_region.json'
	else:
	        target = fRoot+'target_desc.json'
	with open(target) as f:
	        label = json.load(f)
	out = out[cat]
	for i in xrange(len(out[0])):
	        p = out[0][i]
	        present[label[str(i)]] = p
	return present


########## GOOGLENET DESCRIP ############
deploy = 'bvlc_googlenet/deploy.prototxt'
caffe_model= 'bvlc_googlenet/bvlc_googlenet.caffemodel'
deploy = fRoot + deploy
caffe_model = fRoot + caffe_model
#net_google =  predictionClassi(model_file=deploy,pretrained=caffe_model)
#########################################
file = '/home/devuser/Objectdetect/cv_api/images/343811.jpg'
img = cv2.imread(file)
#print net_google.predict([img])
def googlenet(img):
	prediction = net_google.predict([img])
	target = fRoot+'target_desc.json'
        present = {}
        with open(target) as f:
                label = json.load(f)
        out = prediction[0]
        for i in xrange(len(out)):
                p = out[i]
                present[label[str(i)]] = p
        return present

########## RCNN ILSVRC13   ##############
caffe_model = 'bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'
deploy = 'bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'
deploy = fRoot + deploy
caffe_model = fRoot + caffe_model
#########################################
#net_RCNN,transformer_RCNN = prediction(deploy=deploy,caffe_model=caffe_model)
def RCNN(img):
	net_RCNN.blobs['data'].data[...] = transformer_RCNN.preprocess('data', img)
        out = net_RCNN.forward()
        present = {}
        cat =  out.keys()[0]
        target = ''
        if cat != 'prob':
                target = fRoot+'target_region.json'
        else:
                target = fRoot+'target_desc.json'
        with open(target) as f:
                label = json.load(f)
        out = out[cat]
        for i in xrange(len(out[0])):
                p = out[0][i]
                present[label[str(i)]] = p
        return present
########## FLICKR STYLE    ##############
caffe_model = 'finetune_flickr_style/finetune_flickr_style.caffemodel'
deploy = 'finetune_flickr_style/deploy.prototxt'
deploy = fRoot + deploy
caffe_model = fRoot + caffe_model

#########################################
#net_flickr,transformer_flickr = prediction(deploy=deploy,caffe_model=caffe_model)
def FLICKR(img):
	net_flickr.blobs['data'].data[...] = transformer_flickr.preprocess('data', img)
        out = net_flickr.forward()
        present = {}
        cat =  out.keys()[0]
        target = ''
        if cat != 'prob':
                target = fRoot+'target_region.json'
        else:
                target = fRoot+'target_desc.json'
        with open(target) as f:
                label = json.load(f)
        out = out[cat]
        for i in xrange(len(out[0])):
                p = out[0][i]
                present[label[str(i)]] = p
        return present

