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
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cascade', required = True)
ap.add_argument('-m', '--model', required = True)
ap.add_argument('-w', '--weight', required = True)
ap.add_argument('-v', '--video')
args = vars(ap.parse_args())

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

EMOTIONS = ['angry', 'scared', 'happy', 'sad', 'surprised', 'neutral']

if not args.get('video', False):
    camera = cv2.VideoCapture(0)

else:
    camera = cv2.VideoCapture(args['video'])

while True:
    (grabbed, frame) = camera.read()

    if args.get('video') and not grabbed:
        break
    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #initialize the canvas for the visualization, clone
    # the frame so we can draw on it
    canvas = np.zeros((220, 400, 3), dtype= 'uint8')
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
        print("젤 큰값?",preds.argmax(),"이건데 인덱스는",len(EMOTIONS),"까지")
        label = EMOTIONS[preds.argmax()-1]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob*100)

            w= int(prob * 300)
            cv2.rectangle(canvas, (5, (i*35) + 5),(w, (i*35)+35), (0,0,225), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)

            cv2.putText(frameClone, label, (fX, fY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH),(255,0,0), 2)

            cv2.imshow("face", frameClone)
            cv2.imshow("prob", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

camera.release()
cv2.destroyAllWindows()
