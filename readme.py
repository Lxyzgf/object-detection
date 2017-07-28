#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:05:44 2017

@author: lxy
date: 2017/04/11
Project: FPGA-Accelerate-CNN
revertion:v.0.1
tool: tensorflow
net:  MsCNN
dataset: 
"""
############################################
the project files

    mscnn.py
        ----Creating MsCnn module file
        
    boxes.py
        ----generating the Boxes layer,output used for roipooling layer
        
    roiPooling.py
        ----generating the RoiPooling layer,output the fixed features
        
    pd_caffemodel
        ----This file includes the parameter file trained by caffe.
        ----"mscnn-model-1s-480-cut.h5"
    
    data_h5
        ----This file includes the caffe output files
        
    tensorflow_h5
        ----This file includes the Mscnn-tensorflow model output files
        
###############################################
In mscnn.py, there are a few functions.

    Mscnn(image,param_fil,sess,t_d)
        ----'image': input image for the model
        ----'param_fil': parameters including w and b, saved path
        ----'sess': Tensorflow Session()
        ----'t_d': define the data type used in Mscnn 
        ----the class Mscnn is defined to create Mscnn model, including
        ----2 functions.
        ----'convlayers(self,t_dtype=tf.float32)' creates the Convolutional
        ----layers in Mscnn.
        ----'load_weights(self,weight_fil,sess)' loads the parameters from 
        ----the h5 file.
        
    h5_show(fil):
        ----'fil': saving h5 file path
        ---- this function loads h5 file and save the datas into a txt file.
        
    psnr(fil1,fil2):
        ----'fil1': saving h5 file path
        ----'fil2': saving h5 file path
        ----this function compare the two files if they are the same.And
        ----using the difference of squres of errors function.
        
In boxes.py, there are 2 functions.

    Gen_boxs(propal):
        ----'propal': input is proposal layer of Mscnn. propal=[height,width,chal]
        ----this function implements boxes layer just like in caffe
        ----output is boxes=[x1,y1,x2,y2,score]; chal=[N,P,x,y,bw,bh]
        
    non_max_suppression(boxes,overlapthresh):
        ----'boxes': input boxes=[x1,y1,x2,y2,score]
        ----'overlapthresh': select boxes according to threshold
        ----this function select boxes 
