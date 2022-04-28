import cv2
import numpy as np
from time import sleep

largura_min=80 #minimum rectangle width
altura_min=80 #minimum height of rectangle

offset=6 #Allowed error between pixels 

BLOB_SIZE = 300
pos_linha=550 #

#Thickness of the drawing line
LINE_THICKNESS = 1

#Font for the drawing text
font = cv2.FONT_HERSHEY_SIMPLEX

delay= 60 #FPS do vídeo

detec = []
carros= 0
#Blob Detector Function

def make_blobs(f_gray):
    fgmask = fgbg.apply(f_gray)
    blur = cv2.GaussianBlur(fgmask, (5,5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    dialation = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernel)
    ret, thresholded_img = cv2.threshold(dialation, 100, 255, cv2.THRESH_BINARY)
    return(thresholded_img)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
	
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy
#frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture('video.mp4')
#Backgroud Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = fgbg.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    frame_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    dilated = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilated = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #new start from here
    
    #Blob Formation

    blob = make_blobs(frame_gray)

    #Extract foreground by multiplying thresholded image with orignal image
    frame_gray = np.uint8(blob*frame_gray)
    #cv.imshow('blob', blob)
    
    #Drawing contours of Blob
    #im, contours, hierarchy = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    #Filter blobs with area < 500 sq pixels
    blobs = list(filter(lambda c: cv2.contourArea(c) > BLOB_SIZE, contours))


    if blobs:
        for c in blobs:
            # Find the bounding rectangle and center for each blob
            (x, y, w, h) = cv2.boundingRect(c)
            center = (int(x + w/2), int(y + h/2))
            
            aspect_ratio = w/h
            blob_label = None
            label_width = 0
            color_bb = None
            if aspect_ratio <= 0.65:
                blob_label = 'Light'
                label_width = 30
                color_bb = (255,255,0)
            #default aspect_ratio <= 1.2
            elif aspect_ratio <= 2.2:
                blob_label = 'Medium'
                label_width = 50
                color_bb = (0,255,255)
            else:
                blob_label = 'Heavy'
                label_width = 30
                color_bb = (255,0,255)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), color_bb, LINE_THICKNESS)
            cv2.rectangle(frame1, (x, y-10), (x+label_width,y ), color_bb, -1)
            cv2.putText(frame1, blob_label, (x+2,y), font, 0.3,(255,255,255))
            
    #orginal end from here
    
    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255,127,0), 3) 
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(pos_linha+offset) and y>(pos_linha-offset):
                carros+=1
                cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0,127,255), 3)  
                detec.remove((x,y))
                print("car is detected : "+str(carros))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detector",dilated)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
