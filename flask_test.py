# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:07:55 2019

@author: 융
"""

# -*- encoding: utf-8 -*-

import requests
import json
#import realtime_emotion_detect
from flask import Flask, request, jsonify  
import numpy as np
import cv2 as cv
import sys
'''
img_w = 900
img_h = 500
bpp = 3

center_x = int(img_w / 2.0)
center_y = int(img_h / 2.0)

thickness = 2 
font = cv.FONT_HERSHEY_SCRIPT_SIMPLEX
fontScale = 1
img = np.zeros((img_h, img_w, bpp), np.uint8)
'''
def get_answer(text, user_key):
    data_send = { 
        'query': text,
        'sessionId': user_key,
        'lang': 'ko',
    }
    
    data_header = {
        'Authorization': 'Bearer fb99c4b77b9648a484307839a879e941',
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    dialogflow_url = 'https://api.dialogflow.com/v1/query?v=20150910'    

    res = requests.post(dialogflow_url, data=json.dumps(data_send), headers=data_header)
    if res.status_code != requests.codes.ok:
        return '오류가 발생했습니다.'    

    data_receive = res.json()
    answer = data_receive['result']['fulfillment']['speech'] 
    #if text=='시작':
    #    realtime_emotion_detect.run()#감정 디텍트 시작해서 get_answer함수의 text매개변수에 ans값넣어줘서 answer값을 유저한테 보여줘야지
    #######showing()
    return answer

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
'''
@app.route('/message', methods=['POST','GET'])
def send_query():
    
    content=realtime_emotion_detect.answ
    userid='a'
    if content!="":
        what=get_answer(content,userid)
        what_res=realtime_emotion_detect.answ+what#지금 문제: 페이지를 요청 해야만 프린트가 됨. 
        print(what_res)#get_answer값을 gui로 보여주자
        return showing()   
'''
'''
def showing():
    image=img
    location = (center_x - 200, center_y - 100)
    cv.putText(image, "i don't know what i said plz tell me what was my fault", location, font, fontScale, (0, 255, 255), thickness)
    cv.imshow("drawing", image)
    cv.waitKey(4000)
    
 '''
'''   
    
@app.route('/', methods=['POST', 'GET'])
def webhook():
    content = request.args.get('content')
    userid = request.args.get('userid')
    return get_answer(content, userid)
'''
if __name__ == '__main__':    
    app.run(host='0.0.0.0')    

    
