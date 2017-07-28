#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:02:19 2017

@author: lxy
"""

import numpy as np

def Roi_pooling(conv,proposal,pad_ratio):
    '''conv=[h,w,channels],proposal=[x1,y1,x2,y2,score]'''
    pool_w=4
    pool_h=8
#    pad_ratio=0.25
    spatial_scale=0.125
    (conv_h,conv_w,conv_chal)=np.shape(conv)
    pad_w=(proposal[2]-proposal[0]+1)*pad_ratio
    pad_h=(proposal[3]-proposal[1]+1)*pad_ratio
    roi_start_w=round((proposal[0]-pad_w)*spatial_scale)
    roi_start_h=round((proposal[1]-pad_h)*spatial_scale)
    roi_end_w=round((proposal[2]+pad_w)*spatial_scale)
    roi_end_h=round((proposal[3]+pad_h)*spatial_scale)
    roi_h=max(roi_end_h - roi_start_h +1,1)
    roi_w=max(roi_end_w - roi_start_w +1,1)
    bin_h=roi_h/pool_h
    bin_w=roi_w/pool_w
    top=np.zeros([pool_h,pool_w,conv_chal])
    for c in range(conv_chal):
        for ph in range(pool_h):
            for pw in range(pool_w):
                hstart=np.floor(ph*bin_h)
                wstart=np.floor(pw*bin_w)
                hend=np.ceil((ph+1)*bin_h)
                wend=np.ceil((pw+1)*bin_w)
                hstart=int(min(max(hstart+roi_start_h,0),conv_h))
                hend=int(min(max(hend+roi_start_h,0),conv_h))
                wstart=int(min(max(wstart+roi_start_w,0),conv_w))
                wend=int(min(max(wend+roi_start_w,0),conv_w))
                is_empty=((hend<=hstart) or (wend<=wstart))
                if not is_empty:
                    top[ph,pw,c]=np.max(conv[hstart:hend,wstart:wend,c])
    return top                
