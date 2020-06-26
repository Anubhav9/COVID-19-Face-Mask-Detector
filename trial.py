import cv2
import numpy as np
from keras.models import load_model
from PIL.Image import *
model=load_model('detect.h5')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
##hog=cv2.HOGDescriptor()

source=cv2.VideoCapture(0)
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
while(True):
    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ##gray=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    faces=face_clsfr.detectMultiScale(gray,1.15,5)
     

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(224,224))
        resized= cv2.cvtColor(resized,cv2.COLOR_GRAY2RGB)
        normalized=resized/255.0
        reshape=np.expand_dims(normalized,axis=0)
        ##reshaped=np.reshape(normalized,(1,100,100,3))
        result=model.predict_classes(reshape)
        print(result)
        
        if(result==0):
            label=0
        else:
            label=1



        ##label=np.argmax(result,axis=1)[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()