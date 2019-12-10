# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:34:12 2019

@author: 융
"""

#import flask_test 
'''
import numpy as np
import cv2 as cv

img_w = 300
img_h = 300
bpp = 3

center_x = int(img_w / 2.0)
center_y = int(img_h / 2.0)

thickness = 2 
font = cv.FONT_HERSHEY_SCRIPT_SIMPLEX
fontScale = 1
img = np.zeros((img_h, img_w, bpp), np.uint8)
  

def showing(what):
    image=img
    location = (center_x - 200, center_y - 100)
    cv.putText(image, what, location, font, fontScale, (0, 255, 255), thickness)
    cv.imshow("drawing", image)
    print("그려지고있음")
    cv.waitKey(0)
'''    
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
 
#img = np.zeros((100,600,3),np.uint8)#까만 판
b,g,r,a = 0,0,0,0#까만색 글씨

fontpath = "fonts/gulim.ttc"
font = ImageFont.truetype(fontpath, 20)

#what=["미안해 내가 말실수를 한 것 같아ㅠㅠㅠㅠ용서해줄래?ㅠㅠㅠ","니가 슬프니 나도 슬퍼졌어","미안해"] 
def showing(what,reaction,img):
    #global img
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    draw.text((300, 30),  "$이렇게 말해보세요!$ ("+reaction+")\n"+what, font=font, fill=(b,g,r,a))#여기 가로세로 좌표 맞음
     
    image = np.array(img_pil)
    #cv2.putText(img,  "by Dip2K", (250,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
    #print("잉?")
    cv2.imshow("Coaching Bot", image)
    cv2.waitKey(50)
    return image

#for i in range(3):
#showing(what[1])
