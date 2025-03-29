import cv2
import numpy as np


WIDTH = 320
HEIGHT = 240

def empty(x):
    pass
def nothing(a):
    pass

def drawPoints(img,points):
    for x in range( 0,4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img

def warpImg (img,points,w,h,inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def initializeTrackbars(intialTracbarVals,wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", WIDTH, HEIGHT)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)

def valTrackbars(wT=480):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points

def reproduzir_video(caminho_video):
    cap = cv2.VideoCapture(caminho_video)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", WIDTH, HEIGHT)
    cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
    cv2.createTrackbar("HUE Max", "HSV", 112, 179, empty)
    cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
    cv2.createTrackbar("SAT Max", "HSV", 90, 255, empty)
    cv2.createTrackbar("VALUE Min", "HSV", 127, 255, empty)
    cv2.createTrackbar("VALUE Max", "HSV", 188, 255, empty)
    # wT HT Wb Hb
    intialTracbarVals = [75,76,37,216]
    initializeTrackbars(intialTracbarVals,WIDTH, HEIGHT)
    
    # Loop para reproduzir o vídeo
    while True:
        ret, frame = cap.read()
        
        # Se chegou ao final do vídeo, reiniciar
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        image = cv2.resize(frame, (WIDTH, HEIGHT))
        
        #cv2.imshow('Video', frame)
        
        imgHsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        h_min = cv2.getTrackbarPos("HUE Min", "HSV")
        h_max = cv2.getTrackbarPos("HUE Max", "HSV")
        s_min = cv2.getTrackbarPos("SAT Min", "HSV")
        s_max = cv2.getTrackbarPos("SAT Max", "HSV")
        v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
        v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)
    
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        points = valTrackbars(WIDTH)
        imgWarp = warpImg(mask, points, w=WIDTH, h=HEIGHT)
        hStack = np.hstack([image, mask, result])
        cv2.imshow('Horizontal Stacking', hStack)
        imgWarpPoints = drawPoints(image, points)
        hStack = np.hstack([imgWarp, imgWarpPoints])
        cv2.imshow('Points', hStack)
        
        if cv2.waitKey(25) & 0xFF == 27:
            break
    
 
    cap.release()
    cv2.destroyAllWindows()

 
if __name__ == "__main__":
 
    caminho_do_video = "/home/djoker/code/pyhon/udacity/project_video.mp4"
    caminho_do_video = "/home/djoker/code/pyhon/udacity/video_final.mp4"
    reproduzir_video(caminho_do_video)
