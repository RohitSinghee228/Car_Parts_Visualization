import cv2
import pandas as pd
import json
import numpy as np
import random

def meta_data(file_path):
    choice=None
    with open(file_path, 'r') as json_file:
        json_work = json.load(json_file)
        df2 = pd.json_normalize(json_work)
        list=[]
        for index,rows in df2.iterrows():
            if index==0 and 'value.choices' in df2.columns:
                choice=rows['value.choices']

            else:

                poly_list = rows['value.points']
                img_width = rows['original_width']
                img_height = rows['original_height']
                list.append([denormalize(poly_list,img_width,img_height),rows['value.polygonlabels']])
    return(list,choice)

def denormalize(list,img_width,img_height):

    for i in list:
        i[0]=(i[0]/100)*img_width
        i[1]=(i[1]/100)*img_height
    return list

def bounding_box(image,arr,label,color):
    peri=cv2.arcLength(arr,True)
    approx=cv2.approxPolyDP(arr,0.02*peri,True)
    x,y,w,h=cv2.boundingRect(approx)
    cv2.rectangle(image,(x,y),(x+w,y+h),color,1)
    #cv2.rectamgle(image,(x,y)),(x+3,y+3)

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    image = cv2.rectangle(image, (x, y ), (x + w, y+h), (0,0,0), -1)
    image = cv2.putText(image, label, (x, y +h),cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def draw_fill_polygons(img2,list,shade=60):
    col_list=[]
    img_temp=img2.copy()

    for i in list:
        arr = np.array(i[0], np.int32)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        col_list.append(color)
        #Shade variable to set darker shade of bounding Boxes
        color_dark = (color[0] - shade, color[1] - shade, color[2] - shade)
        #to draw the polygon fills
        cv2.drawContours(img_temp, [arr], -1, color, -1)
        #to draw the polygon boundary with darker shade
        cv2.drawContours(img_temp, [arr], -1, color_dark, 1)
    return img_temp,col_list
def draw_bounding_box(base,list,col_list,shade=60):
    img_btemp=base.copy()
    for index,i in enumerate(list):
        arr = np.array(i[0], np.int32)
        #print('index',index,'col_list[index]')
        color = col_list[index]
        color_dark = (color[0] - shade, color[1] - shade, color[2] - shade)
        bounding_box(img_btemp, arr, i[1][0], color_dark)
    return img_btemp

