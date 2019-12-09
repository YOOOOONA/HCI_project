# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 01:02:54 2019

@author: 융
"""
#가상환경 tensor
# usage
#이거 말고 python emotion_detectot.py --cascade haarcascade_frontalface_default.xml --model output/epoch_75.hdf5
#python 실시간감정인식코드.py --cascade haarcascade_frontalface_default.xml --model model_2layer_2_2_pool.json --weight model_2layer_2_2_pool.h5

from keras.preprocessing.image import img_to_array
#from keras.models import load_model
from keras.models import model_from_json
import numpy as np
#import argparse
import imutils
import cv2
import flask_test
import cv_prototype as proto
import tkinter as tk
answ=""

def run():
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cascade', required = True)
    ap.add_argument('-m', '--model', required = True)
    ap.add_argument('-w', '--weight', required = True)
    ap.add_argument('-v', '--video')
    args = vars(ap.parse_args())
    '''
    args={'cascade':'haarcascade_frontalface_default.xml', 'model':'model_2layer_2_2_pool.json', 'weight':'model_2layer_2_2_pool.h5'}
    
    
    #load face detector cascade
    detector = cv2.CascadeClassifier(args['cascade'])
    #######model = load_model(args['model'])
    json_file = open(args['model'],"r")
    loaded_model_json=json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    #로드한 모델에 웨이트 로드하기
    loaded_model.load_weights(args['weight'])
    print("Loaded model from disk!")
    
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise','Neutral']
    
    if not args.get('video', False):
        camera = cv2.VideoCapture(0)
    
    else:
        camera = cv2.VideoCapture(args['video'])
    
    count={'Angry':0, 'Disgust':0, 'Fear':0, 'Happy':0, 'Sad':0, 'Surprise':0,'Neutral':0}
    labels=[]
    while True:
        (grabbed, frame) = camera.read()
    
        if args.get('video') and not grabbed:
            break
        # resize the frame and convert it to grayscale
        frame = imutils.resize(frame, width=900)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        #initialize the canvas for the visualization, clone
        # the frame so we can draw on it
        canvas = np.zeros((260, 400, 3), dtype= 'uint8')#세로,가로,?
        frameClone = frame.copy()
    
        rects = detector.detectMultiScale(gray, scaleFactor = 1.1,
                                            minNeighbors = 5, minSize = (30, 30),
                                            flags = cv2.CASCADE_SCALE_IMAGE)
    
        if len(rects) >0:
            #face area
            rect = sorted(rects, reverse=True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = rect
    
            roi = gray[fY:fY+fH, fX:fX+fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis = 0)
    
            #preds
            preds = loaded_model.predict(roi)[0]
            #print(preds)
            #print("젤 큰값?",preds.argmax(),"이건데 인덱스는",len(EMOTIONS),"까지")
            label = EMOTIONS[preds.argmax()]
            print(label)
            #EMOTIONS=['','angry', 'scared', 'happy', 'sad', 'surprised', 'neutral']
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                #print((emotion,prob))
                text = "{}: {:.2f}%".format(emotion, prob*100)
    
                w= int(prob * 300)
                cv2.rectangle(frameClone, (5, (i*35) + 5),(w, (i*35)+35), (220,255,0), -1)#원래 canvas위에서 하던거 영상이미지 받는 화면(frameCLone)에 출력
                cv2.putText(frameClone, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
    
                cv2.putText(frameClone, label, (fX, fY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,255,0), 2)#frame사각형에 감정멘트 부분
                cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH),(220,255,0), 2)#사각형부분
    
                cv2.imshow("face", frameClone)
                cv2.imshow("prob", canvas)
    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if label!="Neutral":#뉴트럴은 무시한다.
                #3프레임이상 나온 감정만 리스트에 넣는다.근데 리스트에 넣고나서도 반복되면 넣지 않는다.
                count[label]+=1
                if count[label]==3:#세번 나타나면
                    labels.append(label)#리스트에 넣고
                    count[label]=0#카운트를0으로 만듦..다시 세번 나와야 리스트에 넣어질 수 있고
            emt3=labels[-4:-1]#``그렇게 거른 감정 중 최근 세개를 받아서 처리하자
            what=flask_test.get_answer(mention(emt3),'a')
            print("상대방: "+mention(emt3)+" 챗봇이 추천하는 내 대답: "+what)
            #print("상대방: "+mention(emt3))
            proto.showing(what)
    
    camera.release()
    cv2.destroyAllWindows()
    
    
root=tk.Tk()
root.geometry("900x650+400+150")
# 웹캠 화면.
mv_label = tk.Label(root)
mv_label.pack(fill = tk.X, side = tk.TOP)

def mention(emt3):
    global answ
    if emt3==['Happy','Happy','Sad']:
        answ="나 행복했는데 너 때문에 슬퍼졌어"
    elif emt3==['Happy','Happy','Angry']:
        answ="나 행복했는데 너 때문에 화가났어"
    elif emt3==['Angry','Angry','Happy']:
        answ="나 화가 났었는데 너 덕분에 행복해졌어"
    elif emt3==['Sad','Sad','Happy']:
        answ="나 슬펐는데 너 덕분에 행복해졌어"
    elif emt3==['Angry','Angry','Happy']:
        answ="나 화났었는데 너 덕분에 행복해졌어"
    print(emt3)
    return answ

run()