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

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

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

def getHistogram(img, display=True, minPer=0.1, region=4):
    histValues = np.sum(img, axis=0)
    maxValue = np.max(histValues)  # FIND THE MAX VALUE
    minValue = minPer * maxValue
    indexArray = np.where(histValues >= minValue) # ALL INDICES WITH MIN VALUE OR ABOVE
    basePoint = int(np.average(indexArray)) # AVERAGE ALL MAX INDICES VALUES
    
    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            # Fix: Handle the intensity value as a scalar
            if isinstance(intensity, np.ndarray):
                # If it's an array, take the first value or use sum/mean
                intensity_value = intensity[0]  # or np.sum(intensity) or np.mean(intensity)
            else:
                intensity_value = intensity
                
            if intensity_value > minValue:
                color = (255, 0, 255)
            else:
                color = (0, 0, 255)
                
            cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - (int(intensity_value // 255 // region))), color, 1)
        cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist
    
    return basePoint

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(img, lines):

    line_image = np.zeros_like(img)
    
    # Se não houver linhas, retorna imagem em branco
    if lines is None:
        return line_image
    
    # Itera sobre as linhas, verificando se são válidas
    for line in lines:
        if line is not None:  # Verifica se a linha não é None
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    return line_image
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Evitar divisão por zero - quando x1 == x2, a linha é vertical
            if x2 - x1 == 0:
                continue
                
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            
            # Classificar as linhas como esquerda ou direita com base na inclinação
            # Em uma imagem transformada, a inclinação pode ser diferente do esperado
            if slope < 0:  # Linha da esquerda (na imagem, x aumenta da esquerda para a direita)
                left_fit.append((slope, intercept))
            else:  # Linha da direita
                right_fit.append((slope, intercept))
    
    # Verificar se encontramos linhas em cada lado
    if len(left_fit) == 0 or len(right_fit) == 0:
        return None
    
    # Calcular médias
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    
    # Gerar pontos
    try:
        left_line = make_points(image, left_fit_average)
        right_line = make_points(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines
    except:
        # Se ocorrer algum erro ao gerar os pontos, retorne None
        return None
    
def make_points(image, line):
    try:
        # Verificar se line é um array ou tupla com pelo menos 2 elementos
        if not hasattr(line, "__len__") or len(line) < 2:
            return None
            
        slope, intercept = line
        
        # Evitar inclinações muito pequenas que podem causar valores x muito grandes
        if abs(slope) < 0.1:
            return None
            
        height = image.shape[0]
        y1 = height  # Parte inferior da imagem
        y2 = int(height * 3/5)  # Um pouco abaixo do meio
        
        # Calcular coordenadas x
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        
        # Verificar se os pontos estão dentro dos limites da imagem
        width = image.shape[1]
        if x1 < 0 or x1 >= width or x2 < 0 or x2 >= width:
            return None
            
        return [[x1, y1, x2, y2]]
    except Exception as e:
        print(f"Erro em make_points: {e}")
        return None    
def draw_polygon(img, points, color=(0, 255, 0), thickness=2):
    """Desenha um polígono fechado usando os pontos fornecidos"""
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, thickness)
    return img
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
        #cv2.imshow('Horizontal Stacking', hStack)
        imgWarpPoints = drawPoints(image, points)

        middlePoint, imgHist = getHistogram(imgWarp, display=True, minPer=0.5, region=1)
       

        canny_image = canny(result)
        lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
        averaged_lines = average_slope_intercept(imgWarp, lines)
        line_image = display_lines(imgWarp, averaged_lines)
        combo_image = cv2.addWeighted(imgWarp, 0.8, line_image, 1, 1)
        
        imgStack = stackImages(1, 
        [
            [canny_image, result, imgHist],
            [combo_image, imgWarp, imgWarpPoints]
        ])
        
        cv2.imshow('Processamento', imgStack)
     
        
        if cv2.waitKey(25) & 0xFF == 27:
            break
    
 
    cap.release()
    cv2.destroyAllWindows()

 
if __name__ == "__main__":
 
    caminho_do_video = "/home/djoker/code/pyhon/udacity/project_video.mp4"
    caminho_do_video = "/home/djoker/code/pyhon/udacity/video_final.mp4"
    #caminho_do_video = "/home/djoker/code/pyhon/udacity/pt1.mp4"
    reproduzir_video(caminho_do_video)
