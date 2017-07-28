#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
"""
Created on Wed Mar 22 09:59:07 2017

@author: lxy
"""
import numpy as np
import tensorflow as tf
import glob
from scipy.misc import imread,imresize,imshow
import h5py
import cv2
import boxes
import roiPooling
import pdb
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

class Mscnn:
    def __init__(self,imgs,roi_fc,weights=None,sess=None,t_dtype=tf.float32):
        self.imgs=imgs
        self.roi_fc=roi_fc
        self.t_dtype=t_dtype
        self.convlayers()        
        self.fc_layers()        
        self.data_out=[self.propal,self.conv4_3,self.pool6]
        self.final_out=[self.cls_pred,self.box_pred,self.roi_c1,self.fc1]
        if weights is not None and sess is not None:
            self.load_weights(weights,sess)
    def convlayers(self):
        self.parameters=[]
        #self.t_dtype=t_dtype
        with tf.name_scope('conv1_1') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,3,64],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.imgs,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,shape=[64],dtype=self.t_dtype),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv1_1=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        with tf.name_scope('conv1_2') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,64,64],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.conv1_1,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,shape=[64],dtype=self.t_dtype),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.con1_2=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
            
        self.pool1=tf.nn.max_pool(self.con1_2,
                                  ksize=[1,2,2,1],
                                  strides=[1,2,2,1],
                                  padding='VALID',
                                  name='pool1')
        
        with tf.name_scope('conv2_1') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,64,128],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.pool1,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,shape=[128],dtype=self.t_dtype),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv2_1=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        with tf.name_scope('conv2_2') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,128,128],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.conv2_1,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,shape=[128],dtype=self.t_dtype),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv2_2=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        
        self.pool2=tf.nn.max_pool(self.conv2_2,
                                  ksize=[1,2,2,1],
                                  strides=[1,2,2,1],
                                  padding='VALID',
                                  name='pool2')
       
        with tf.name_scope('conv3_1') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,128,256],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.pool2,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[256]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv3_1=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        with tf.name_scope('conv3_2') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.conv3_1,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[256]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv3_2=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        with tf.name_scope('conv3_3') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weight')
            conv=tf.nn.conv2d(self.conv3_2,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[256]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv3_3=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
            
        self.pool3=tf.nn.max_pool(self.conv3_3,
                                  ksize=[1,2,2,1],
                                  strides=[1,2,2,1],
                                  padding='VALID',
                                  name='pool3')
        
        with tf.name_scope('conv4_1') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,256,512],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.pool3,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[512]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv4_1=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        with tf.name_scope('conv4_2') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,512,512],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.conv4_1,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[512]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv4_2=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        with tf.name_scope('conv4_3') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,512,512],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.conv4_2,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[512]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv4_3=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
            
        self.pool4=tf.nn.max_pool(self.conv4_3,
                                  ksize=[1,2,2,1],
                                  strides=[1,2,2,1],
                                  padding='VALID',
                                  name='pool4')
        
        with tf.name_scope('conv5_1') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,512,512],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.pool4,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[512]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv5_1=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        with tf.name_scope('conv5_2') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,512,512],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.conv5_1,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[512]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv5_2=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        with tf.name_scope('conv5_3') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,512,512],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.conv5_2,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[512]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv5_3=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
        
        self.pool5=tf.nn.max_pool(self.conv5_3,
                                  ksize=[1,2,2,1],
                                  strides=[1,2,2,1],
                                  padding='VALID',
                                  name='pool5')
        
        with tf.name_scope('conv6_1') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,512,512],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.pool5,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[512]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.conv6_1=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
            
        self.pool6=tf.nn.max_pool(self.conv6_1,
                                  ksize=[1,2,2,1],
                                  strides=[1,2,2,1],
                                  padding='SAME',
                                  name='pool6')
        
        with tf.name_scope('LFCN_4_3') as scope:
            kernal=tf.Variable(tf.truncated_normal([5,3,512,6],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.pool6,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[6]),
                               name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.propal=out
            self.parameters+=[kernal,biases]
            #self.parameters=np.asarray(self.parameters)
        
#        with tf.name_scope('propal') as scope:
#            box_p=np.asarray(self.propal)
#            print np.shape(box_p)
#            self.box=boxes.Gen_boxs(box_p[0,:,:,:])
#            
#        with tf.name_scope('roiPooling') as scope:
#            (r,c)=np.shape(self.box)
#            self.roi_ctx=np.zeros([r,8,4,512])
#            self.roi_org=np.zeros([r,8,4,512])
#            pad_ratio=0.25
#            for i in range(r):
#                self.roi_ctx[i,:,:,:]=roiPooling.Roi_pooling(self.conv4_3[0,:,:,:],self.box[i,:],pad_ratio)
#                self.roi_org[i,:,:,:]=roiPooling.Roi_pooling(self.conv4_3[0,:,:,:],self.box[i,:],0)
                 
#        with tf.name_scope('concat') as scope:
#            roi_shape=np.shape(self.roi_ctx)
#            con_shape=int(np.prod(roi_shape[1:]))
#            box_num=roi_shape[0]
#            roi_pool_r=np.zeros([box_num,con_shape])
#            for i in range(box_num):
#                roi_ctx_r=np.reshape(self.roi_ctx[i,:,:,:],con_shape)
#                roi_org_r=np.reshape(self.roi_org[i,:,:,:],con_shape)
#                roi_pool_r[i,:]=np.concatenate((roi_org_r,roi_ctx_r),axis=0)
#            self.roi_pool=roi_pool_r
#            
        with tf.name_scope('roi_c1') as scope:
            kernal=tf.Variable(tf.truncated_normal([3,3,1024,512],dtype=self.t_dtype,
                                                   stddev=1e-1),name='weights')
            conv=tf.nn.conv2d(self.roi_fc,kernal,strides=[1,1,1,1],padding='SAME')
            biases=tf.Variable(tf.constant(0,dtype=self.t_dtype,shape=[512]),
                             name='biases',trainable=True)
            out=tf.nn.bias_add(conv,biases)
            self.roi_c1=tf.nn.relu(out,name=scope)
            self.parameters+=[kernal,biases]
            
    def fc_layers(self):        
        with tf.name_scope('fc1') as scope:
            fc_shape=int(np.prod(self.roi_c1.get_shape()[1:]))
            fc1_w=tf.Variable(tf.truncated_normal([fc_shape,2048],dtype=self.t_dtype,
                                                  stddev=1e-1),name='weights')
            fc1_b=tf.Variable(tf.constant(1,dtype=self.t_dtype,shape=[2048]),
                              name='biases',trainable=True)
            c1=tf.transpose(self.roi_c1[0,:,:,:],(2,0,1))
            roi_flat=tf.reshape(c1,[-1,fc_shape])
            #roi_flat=tf.reshape(self.roi_c1,[-1,fc_shape])
            fc1l=tf.nn.bias_add(tf.matmul(roi_flat,fc1_w),fc1_b)
            self.fc1=tf.nn.relu(fc1l,name=scope)
            #self.fc1=tf.nn.dropout(self.fc1,0.5)
            self.parameters+=[fc1_w,fc1_b]            
        with tf.name_scope('cls_pred') as scope:
            cls_w=tf.Variable(tf.truncated_normal([2048,2],dtype=self.t_dtype,
                                                  stddev=1e-1),name='weights')
            cls_b=tf.Variable(tf.constant(1,dtype=self.t_dtype,shape=[2]),
                              name='biases',trainable=True)
            self.cls_pred=tf.nn.bias_add(tf.matmul(self.fc1,cls_w),cls_b)
            self.parameters+=[cls_w,cls_b]
        with tf.name_scope('box_pred') as scope:
            box_w=tf.Variable(tf.truncated_normal([2048,8],dtype=self.t_dtype,
                                                  stddev=1e-1),name='weights')
            box_b=tf.Variable(tf.constant(1,dtype=self.t_dtype,shape=[8]),
                              name='biases',trainable=True)
            self.box_pred=tf.nn.bias_add(tf.matmul(self.fc1,box_w),box_b)
            self.parameters+=[box_w,box_b]
           
    def load_weights(self,weight_fil,sess):
        f=h5py.File(weight_fil,'r')
        i=0
        j=0
        for name in f:
            i+=1
            print name
            if i==1:
                w=np.asarray(f[name+'/weight'])
                w=w.transpose((2,3,1,0))
                b=np.asarray(f[name+'/bias'])
                sess.run(self.parameters[28].assign(w))
                sess.run(self.parameters[29].assign(b))
            if i==2:
                w=np.asarray(f[name+'/weight'])
                w=w.transpose((1,0))
                b=np.asarray(f[name+'/bias'])
                sess.run(self.parameters[36].assign(w))
                sess.run(self.parameters[37].assign(b))
            if i==3:
                w=np.asarray(f[name+'/weight'])
                w=w.transpose((1,0))
                b=np.asarray(f[name+'/bias'])
                sess.run(self.parameters[34].assign(w))
                sess.run(self.parameters[35].assign(b))
            if i>=4 and i<18:
                w=np.asarray(f[name+'/weight'])
                w=w.transpose((2,3,1,0))
                b=np.asarray(f[name+'/bias'])
#                print np.shape(w)
#                print 'the origin paramter is'
#                print self.parameters[j].get_shape()
#                print 'the j is %d' %j
                sess.run(self.parameters[j].assign(w))
#                print self.parameters[j].get_shape()
                sess.run(self.parameters[j+1].assign(b))
                j+=2
            if i==18:
                w=np.asarray(f[name+'/weight'])
                w=w.transpose((1,0))
                print 'the fc6 shape is'
                print np.shape(w)
                b=np.asarray(f[name+'/bias'])
                sess.run(self.parameters[32].assign(w))
                sess.run(self.parameters[33].assign(b))
            if i==19:
                w=np.asarray(f[name+'/weight'])
                w=w.transpose((2,3,1,0))
                print 'the roi_c1 shape is'
                print np.shape(w)
                b=np.asarray(f[name+'/bias'])
                sess.run(self.parameters[30].assign(w))
                sess.run(self.parameters[31].assign(b))
        

def psnr(fil1,fil2):
    f1=h5py.File(fil1,'r')
    a=np.asarray(f1['data'])
    f1.close
    f2=h5py.File(fil2,'r')
    for name in f2:
        b=np.asarray(f2[name])
    f2.close
    #b=np.transpose(b,(0,2,3,1))
    #b=np.transpose(b,(1,2,0))
    assert np.shape(a)==np.shape(b)
    print np.shape(b)
    c=b-a
    print 'the max is %d' %(np.max(c))
    print 'the min is %d' %(np.min(c))
    #c=np.ceil(c)
    #print 'the max is %d' %(np.max(c))
    #print 'the min is %d' %(np.min(c))
    mse=np.sum(c**2)/np.prod(np.shape(a))
#    (row,col,n)=np.shape(a)
#    k=0
#    for i in range(row):
#        for j in range(col):
#            for chal in range(n):
#                if np.abs(c[i,j,chal] )> 1 :
#                    k+=1
#                    #print c[i,j,chal]
#    print k
    return np.sqrt(mse)
    #return b
    
def h5_read(fil):
    f=h5py.File(fil,'r')
    for name in f:
        print name
   # a=np.asarray(f[name])
    #print a.shape
    f.close
   # return a

def h5_write(fil,a):
    f=h5py.File(fil,'w')
    f['data']=a
    f.close

#if __name__=='__main__':
def test():
    with tf.device('/cpu:0'):
        sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
        t_d=tf.float32
        image=tf.placeholder(dtype=tf.float32,shape=[None,480,640,3])
        #p_6=tf.placeholder(dtype=tf.float32,shape=[1,8,10,512])
        roi_fc=tf.placeholder(dtype=tf.float32,shape=[None,8,4,1024])
        img_fil='./001_Eth-set04_V000_I00000.jpg'
        img1 = imread(img_fil, mode='RGB')
        img1 = imresize(img1, (480, 640))
        img1 = np.float32(img1)
        img2=np.zeros([480,640,3],dtype=np.float32)
    #    B=img1[:,:,2]
    #    G=img1[:,:,1]
    #    R=img1[:,:,0]
    #    img2[:,:,0]=B-104
    #    img2[:,:,1]=G-117
    #    img2[:,:,2]=R-123
        img2[:,:,0]=img1[:,:,2]-104.0
        img2[:,:,1]=img1[:,:,1]-117.0
        img2[:,:,2]=img1[:,:,0]-123.0
        param_fil='./mscnn-model-1s-480-cut.h5'
        #param_fil='./pd_caffemodel/mscnn-model.h5'
        #t_d=tf.float64
        mscnn=Mscnn(image,roi_fc,param_fil,sess,t_d)
#        pool=h5_read('./data_h5/pool_6.h5')
#        pool=np.transpose(pool,(1,2,0))
#        pool=np.expand_dims(pool,axis=0)
        img2=np.expand_dims(img2,axis=0)
        roi_test=np.zeros([1,8,4,1024])
#    pdb.set_trace()
    a=sess.run(mscnn.data_out,feed_dict={mscnn.imgs:img2,mscnn.roi_fc:roi_test})
    box_i=a[0]
###########################################################################
# generate boxes
################
#    fil_caf='./data_h5/proposal.h5'
#    box_2=h5_read(fil_caf)
#    box_i=np.transpose(box_2,(1,2,0))
#    print np.shape(box_i)
#    f_test='./tensorflow_h5/proposal.h5'
#    a_cur=a[0]
#    b=a_cur[0,:,:,:]
#    h5_write(f_test,b)     
    print 'begin to generante boxes'
    box=boxes.Gen_boxs(box_i[0,:,:,:])
    #box=boxes.Gen_boxs(box_i)
#    print box
    print 'boxes is successful'
#    f_4_3='./data_h5/conv4_3.h5'
#    conv_data=h5_read(f_4_3)
#    conv_data=np.transpose(conv_data,(1,2,0))
##################################################
# generate roipooling
#########################
    print 'begin to roipooling'
#    print np.shape(conv_data)
    (r,c)=np.shape(box)
    roi_ctx=np.zeros([r,8,4,512])
    roi_org=np.zeros([r,8,4,512])
    roi_concat=np.zeros([r,8,4,1024])
    cls_pred=np.zeros([r,2])
    box_pred=np.zeros([r,8])
    roi_c1=np.zeros([3,8,4,512])
    fc6=np.zeros([3,2048])
    pad_ratio=0.25
    conv_data=a[1]
    for i in range(r):
#        roi_ctx[i,:,:,:]=roiPooling.Roi_pooling(conv_data,box[i,:],pad_ratio)
#        roi_org[i,:,:,:]=roiPooling.Roi_pooling(conv_data,box[i,:],0)
        roi_ctx[i,:,:,:]=roiPooling.Roi_pooling(conv_data[0,:,:,:],box[i,:],pad_ratio)
        roi_org[i,:,:,:]=roiPooling.Roi_pooling(conv_data[0,:,:,:],box[i,:],0)
#    print np.shape(roi_ctx)
#    f1='./tensorflow_h5/roipool_ctx.h5'
#    f2='./tensorflow_h5/roipool_org.h5'
#    h5_write(f1,roi_ctx)
#    h5_write(f2,roi_org)
##########################
#concat
###################
        org=roi_org[i,:,:,:]
        ctx=roi_ctx[i,:,:,:]
        roi_concat[i,:,:,:]=np.concatenate((org,ctx),axis=2)
################################################
##detect layers
################################
        concat_expd=np.expand_dims(roi_concat[i,:,:,:],axis=0)
        final_out=sess.run(mscnn.final_out,feed_dict={mscnn.imgs:img2,mscnn.roi_fc:concat_expd})
        cls_pred[i,]=final_out[0]
        box_pred[i,]=final_out[1]
        roi_c1[i,:,:,:]=final_out[2]
        fc6[i,:]=final_out[3]
    print cls_pred
    print box_pred
    box_pred=np.int32(box_pred[:,:])
    print 'box is:'
    print box_pred
    img3=cv2.rectangle(img1,(box_pred[0,0],box_pred[0,1]),(box_pred[0,2],box_pred[0,3]),(0,0,255),5,8)
   # win_name="show_img"
    #cv2.namedWindow('win_name')
    #img3=np.asarray(img2)
    imshow(img3)
  #  cv2.waitKey(0)
    #cv2.destroyWindow(win_name)
   # f_roi='./tensorflow_h5/box_pred.h5'
   # h5_write(f_roi,box_pred)
        
        
def h5_show(fil):
    #np.set_printoptions(threshold = 'nan')
    f=h5py.File(fil,'r')
    for name in f:
        print name
    a=np.asarray(f[name])
    f.close
    #a=np.transpose(a,(1,2,0))
    print np.shape(a)
#    (r,n,row,col)=np.shape(a)
    (row,col)=np.shape(a)
    b=np.zeros([col])
    f=open('./tensorflow_h5/box_pred.txt','w')
#    for chal in range(r):
#        for k in range(n):
#            for i in range(row):
#                b=a[chal,k,i,:]
#                s=map(str,b)
#                for j in range(col):
#                    f.write(s[j] + '    ')
#                f.write('\n')
#    for chal in range(n):
#        for i in range(row):
#            b=a[i,:,chal]
#            s=map(str,b)
#            for j in range(col):
#                f.write(s[j] + '    ')
#            f.write('\n')
    for i in range(row):
        b=a[i,:]
        s=map(str,b)
        for j in range(col):
            f.write(s[j] + '    ')
        f.write('\n')
    f.close()
