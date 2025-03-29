import cv2
import numpy as np

WIDTH = 320
HEIGHT = 240

def empty(x):
    pass
def nothing(a):
    pass

def drawPoints(img, points):
    img_copy = img.copy()
    for x in range(0, 4):
        cv2.circle(img_copy, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img_copy

def warpImg(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
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

def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", WIDTH, HEIGHT+100)  # Aumentado para acomodar mais trackbars
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)
    cv2.createTrackbar("Shift Left", "Trackbars", intialTracbarVals[4], wT//2, nothing)
    cv2.createTrackbar("Shift Right", "Trackbars", intialTracbarVals[5], wT//2, nothing)

def valTrackbars(wT=480):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    shiftLeft = cv2.getTrackbarPos("Shift Left", "Trackbars")
    shiftRight = cv2.getTrackbarPos("Shift Right", "Trackbars")
    
    # Aplicar deslocamentos aos pontos
    leftTopX = widthTop + shiftLeft
    rightTopX = wT - widthTop + shiftLeft
    leftBottomX = widthBottom + shiftRight
    rightBottomX = wT - widthBottom + shiftRight
    
    # Garantir que os pontos estejam dentro dos limites da imagem
    leftTopX = max(0, min(leftTopX, wT-1))
    rightTopX = max(0, min(rightTopX, wT-1))
    leftBottomX = max(0, min(leftBottomX, wT-1))
    rightBottomX = max(0, min(rightBottomX, wT-1))
    
    points = np.float32([
        (leftTopX, heightTop),           # Topo Esquerdo
        (rightTopX, heightTop),          # Topo Direito
        (leftBottomX, heightBottom),     # Inferior Esquerdo
        (rightBottomX, heightBottom)     # Inferior Direito
    ])
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
            # Handle the intensity value as a scalar
            if isinstance(intensity, np.ndarray):
                intensity_value = np.mean(intensity)
            else:
                intensity_value = intensity
                
            if intensity_value > minValue:
                color = (255, 0, 255)
            else:
                color = (0, 0, 255)
                
            # Ensure the height calculation doesn't cause errors (limit to avoid negatives)
            height = min(img.shape[0], int(intensity_value // 255 // region))
            cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - height), color, 1)
        cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist
    
    return basePoint

def improved_edge_detection(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection with improved thresholds
    edges = cv2.Canny(blur, 50, 150)
    
    # Apply morphological operations to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    return edges

def improved_line_detection(edges, img_shape):
    # Use HoughLinesP with parameters tuned for lane detection
    lines = cv2.HoughLinesP(
        edges,
        rho=2,
        theta=np.pi/180,
        threshold=50,  # Reduced threshold to detect more lines
        minLineLength=30,  # Minimum line length
        maxLineGap=10  # Maximum gap between line segments
    )
    return lines

def filter_lines(lines, min_slope=0.3):
    """Filter lines by slope to remove horizontal lines and noise"""
    if lines is None:
        return None
    
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Skip if line is too horizontal (small slope)
            if x2 - x1 == 0:  # Avoid division by zero
                continue
            
            slope = abs((y2 - y1) / (x2 - x1))
            if slope > min_slope:  # Only keep lines with significant slope
                filtered_lines.append([[x1, y1, x2, y2]])
    
    return filtered_lines if filtered_lines else None

def separate_lines(lines, img_width):
    """Separate lines into left and right categories based on position and slope"""
    if lines is None:
        return None, None
    
    left_lines = []
    right_lines = []
    
    # Midpoint of the image width
    mid_x = img_width // 2
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate slope
            if x2 - x1 == 0:  # Avoid division by zero
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Determine line position (left/right) based on position and slope
            line_center_x = (x1 + x2) // 2
            
            if slope < 0 and line_center_x < mid_x:
                left_lines.append([(x1, y1, x2, y2, slope)])
            elif slope > 0 and line_center_x > mid_x:
                right_lines.append([(x1, y1, x2, y2, slope)])
    
    return left_lines, right_lines

def average_lane_lines(lines, img_height):
    """Average the lines to get a single line"""
    if not lines:
        return None
    
    x_coords = []
    y_coords = []
    
    for line in lines:
        for x1, y1, x2, y2, _ in line:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
    
    # Fit a polynomial to the points
    if len(x_coords) > 0 and len(y_coords) > 0:
        try:
            z = np.polyfit(y_coords, x_coords, 1)
            polynomial = np.poly1d(z)
            
            # Generate line points
            y1 = img_height
            y2 = int(img_height * 0.6)  # Stop line at 60% of image height
            x1 = int(polynomial(y1))
            x2 = int(polynomial(y2))
            
            return [[x1, y1, x2, y2]]
        except:
            return None
    return None

def draw_lane_lines(img, left_line, right_line, color=(0, 255, 0), thickness=5):
    """Draw the lane lines on the image"""
    line_img = np.zeros_like(img)
    
    if left_line is not None:
        for x1, y1, x2, y2 in left_line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    
    if right_line is not None:
        for x1, y1, x2, y2 in right_line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    
    # Create a filled polygon between the lines (if both exist)
    if left_line is not None and right_line is not None:
        left_points = [(left_line[0][0], left_line[0][1]), (left_line[0][2], left_line[0][3])]
        right_points = [(right_line[0][0], right_line[0][1]), (right_line[0][2], right_line[0][3])]
        
        # Form a polygon (bottom-left, top-left, top-right, bottom-right)
        polygon_pts = np.array([
            left_points[0],    # Bottom-left
            left_points[1],    # Top-left
            right_points[1],   # Top-right
            right_points[0]    # Bottom-right
        ], np.int32)
        
        cv2.fillPoly(line_img, [polygon_pts], (0, 200, 100, 50))  # Semi-transparent green
    
    # Blend the lane markings with the original image
    result = cv2.addWeighted(img, 1, line_img, 0.7, 0)
    return result

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
    
    # wT HT Wb Hb ShiftL ShiftR
    intialTracbarVals = [75, 76, 37, 216, 0, 0]
    initializeTrackbars(intialTracbarVals, WIDTH, HEIGHT)
    
    # Loop para reproduzir o vídeo
    while True:
        ret, frame = cap.read()
        
        # Se chegou ao final do vídeo, reiniciar
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        image = cv2.resize(frame, (WIDTH, HEIGHT))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Conversão HSV e filtro de cor
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
    
        # Conversão para BGR (para visualização)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Transformação de perspectiva
        points = valTrackbars(WIDTH)
        imgWarp = warpImg(mask_bgr, points, w=WIDTH, h=HEIGHT)
        imgWarpOriginal = warpImg(image, points, w=WIDTH, h=HEIGHT)
        
        # Detecção de bordas melhorada
        edges = improved_edge_detection(imgWarp)
        
        # Detecção de linhas melhorada
        lines = improved_line_detection(edges, imgWarp.shape)
        
        # Filtragem de linhas
        filtered_lines = filter_lines(lines)
        
        # Separar linhas em esquerda e direita
        left_lines, right_lines = separate_lines(filtered_lines, WIDTH)
        
        # Calcular linhas médias
        avg_left_line = average_lane_lines(left_lines, HEIGHT) if left_lines else None
        avg_right_line = average_lane_lines(right_lines, HEIGHT) if right_lines else None
        
        # Desenhar linhas na imagem original
        lane_img = draw_lane_lines(imgWarpOriginal, avg_left_line, avg_right_line)
        
        # Histograma para análise
        try:
            middlePoint, imgHist = getHistogram(imgWarp, display=True, minPer=0.5, region=1)
        except:
            imgHist = np.zeros_like(imgWarp)
            middlePoint = WIDTH // 2
        
        # Desenhar os pontos da transformação de perspectiva
        imgWarpPoints = drawPoints(image, points)
        
        # Visualização de todas as etapas
        imgStack = stackImages(1, [
            [imgWarpPoints, imgHsv, mask_bgr,result],
            [imgWarpOriginal, edges, lane_img, imgHist]
        ])
        
        cv2.imshow('Linhas', imgStack)
        
        # Mostrar também o histograma
        #cv2.imshow('Histograma', imgHist)
        
        # Mostrar os valores numéricos dos pontos de transformação
        # pts_info = np.zeros((150, WIDTH, 3), np.uint8)
        # cv2.putText(pts_info, f"Top Left: ({int(points[0][0])},{int(points[0][1])})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(pts_info, f"Top Right: ({int(points[1][0])},{int(points[1][1])})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(pts_info, f"Bottom Left: ({int(points[2][0])},{int(points[2][1])})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(pts_info, f"Bottom Right: ({int(points[3][0])},{int(points[3][1])})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # # Adicionar informações sobre deslocamentos
        # shiftLeft = cv2.getTrackbarPos("Shift Left", "Trackbars")
        # shiftRight = cv2.getTrackbarPos("Shift Right", "Trackbars")
        # cv2.putText(pts_info, f"Shift Left: {shiftLeft}  Shift Right: {shiftRight}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # cv2.imshow('Transform', pts_info)
        
        if cv2.waitKey(25) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #caminho_do_video = "/home/djoker/code/pyhon/udacity/project_video.mp4"
    #caminho_do_video = "/home/djoker/code/pyhon/udacity/video_final.mp4"
    caminho_do_video = "/home/djoker/code/pyhon/udacity/pt1.mp4"
    reproduzir_video(caminho_do_video)