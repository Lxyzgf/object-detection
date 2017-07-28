# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def Gen_boxs(propal):
    ''' input is proposal layer of Mscnn. propal=[height,width,chal];output is 
    boxes=[,x1,y1,x2,y2,score]; chal=[N,P,x,y,bw,bh]'''
    (row,col,chal)=np.shape(propal)
    assert chal==6
    fg_thr =-5
    iou_thr=0.65
    #nms_type= "IOU"
    field_w=160
    field_h=320
    downsample_rate=64
    field_whr=2
    field_xyr=2
    max_nms_num=2000
    min_whr=np.log(1./field_whr)
    max_whr=np.log(field_whr)
    min_xyr=-1./field_xyr
    max_xyr=1./field_xyr
    img_wid=field_w*downsample_rate
    img_height=field_h*downsample_rate
    fg_score=np.zeros([row,col])
    fg_score=propal[:,:,1]-propal[:,:,0]
    bbox=[]
    box_vector=np.zeros(5)
    for i in range(row):
        for j in range(col):
            if fg_score[i,j] >fg_thr:
                bbx=propal[i,j,2]
                bby=propal[i,j,3]
                bbw=propal[i,j,4]
                bbh=propal[i,j,5]
                bbx=max(min_xyr,bbx)
                bbx=min(max_xyr,bbx)
                bby=max(min_xyr,bby)
                bby=min(max_xyr,bby)
                bbw=max(min_whr,bbw)
                bbw=min(max_whr,bbw)
                bbh=max(min_whr,bbh)
                bbh=min(max_whr,bbh)
                bbx=bbx*field_w+(j+0.5)*downsample_rate
                bby=bby*field_h+(i+0.5)*downsample_rate
                bbw=field_w*np.exp(bbw)
                bbh=field_h*np.exp(bbh)
                bbx=bbx-bbw/2
                bby=bby-bbh/2
                bbx=max(bbx,0)
                bby=max(bby,0)
                bbw=min(bbw,img_wid-bbx)
                bbh=min(bbh,img_height-bby)
                box_vector=[bbx,bby,bbw,bbh,fg_score[i,j]]
                bbox+=[box_vector]
    box_len=len(bbox)
    box_new=np.zeros([box_len,5])
    print 'the lenth of box is %d' %box_len 
    #print bbox[0]
    for i in range(box_len):
        #print 'the box shape is %d' % (np.shape(bbox[i]))
        box_new[i,:]=bbox[i]
    #print box_new[0,0]
    box_new=np.stack((box_new[:,0],box_new[:,1],box_new[:,0]+box_new[:,2],box_new[:,1]+box_new[:,3],box_new[:,4]),axis=-1) 
    print np.shape(box_new)
    for i in range(box_len):
        print box_new[i,:]        
    box=non_max_suppression(box_new,iou_thr)
    (r,c)=np.shape(box)
    print 'the out box shape is %d %d' %(r,c)
    print box[0,0:5]
    #box[:,1:5]=box[:,0:4]
    #box[:,0]=0
    box_out=box
    #box_out=np.expand_dims(box,0)
    #print box_out.shape
    return box_out
#    score_vector=box_new[:,4]
#    idx=np.argsort(score_vector)
#    box_out=box_new
#    j=0
#    for i in idx:
#        box_out[j,:]=box_new[i,:]
#        j+=1
def non_max_suppression(boxes,overlapthresh):
    if len(boxes)==0:
        return []
#    if boxes.dtype.kind=="i":
#        boxes=boxes.astype("float")
    pick=[]
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]
    score=boxes[:,4]
    area=(x2-x1+1)*(y2-y1+1)
    idx=np.argsort(score)
    while len(idx)>0:
        last=len(idx)-1
        i=idx[last]
        pick.append(i)
        xx1=np.maximum(x1[i],x1[idx[:last]])
        yy1=np.maximum(y1[i],y1[idx[:last]])
        xx2=np.minimum(x2[i],x2[idx[:last]])
        yy2=np.minimum(y2[i],y2[idx[:last]])
        w=np.maximum(0,xx2-xx1+1)
        h=np.maximum(0,yy2-yy1+1)
        overlap=(w*h)/(area[idx[:last]] + area[idx[last]] - w*h)
        idx=np.delete(idx,np.concatenate(([last],np.where(overlap >overlapthresh)[0])))
    return boxes[pick]
#    return boxes[pick].astype("int")
                
