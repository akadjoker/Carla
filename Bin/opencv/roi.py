import cv2
import numpy as np
import json
import os

def nothing(x):
    pass

def draw_polygon(img, points, color=(0, 255, 0), thickness=2):
    """Desenha um polígono fechado usando os pontos fornecidos"""
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, thickness)
    return img

def create_mask(shape, points):
    """Cria uma máscara baseada nos pontos do polígono"""
    mask = np.zeros(shape, dtype=np.uint8)
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    return mask

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

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
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

def region_of_interest(img, vertices):
    """
    Aplica uma máscara de região de interesse à imagem.
    
    Parameters:
    img: A imagem original
    vertices: Array de pontos que define a região de interesse
    
    Returns:
    Imagem mascarada
    """
    # Defining a blank mask to start with
    mask = np.zeros_like(img)
    
    # Definindo a cor de preenchimento
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # Cores i.e. 3 ou 4 dependendo do formato da imagem
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    # Preenchendo pixels dentro do polígono definido pelos "vertices" com a cor preenchida    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # Retornando a imagem apenas onde a máscara de pixels é diferente de zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def salvar_roi_config(config, nome_arquivo="roi_config.json"):
    """Salva a configuração do ROI em um arquivo JSON"""
    with open(nome_arquivo, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuração do ROI salva em {nome_arquivo}")

def carregar_roi_config(nome_arquivo="roi_config.json"):
    """Carrega a configuração do ROI de um arquivo JSON"""
    if os.path.exists(nome_arquivo):
        with open(nome_arquivo, 'r') as f:
            config = json.load(f)
        print(f"Configuração do ROI carregada de {nome_arquivo}")
        return config
    else:
        print(f"Arquivo {nome_arquivo} não encontrado. Usando configuração padrão.")
        return None

def reproduzir_video_com_perspectiva(caminho_video):
    # Abrir o arquivo de vídeo
    cap = cv2.VideoCapture(caminho_video)
    
    # Verificar se o vídeo foi aberto com sucesso
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    # Criar janela para os sliders
    cv2.namedWindow('Controles ROI')
    cv2.resizeWindow('Controles ROI', 640, 400)
    
    # Obter um frame para determinar tamanho
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler o primeiro frame.")
        return
    
    altura, largura = frame.shape[0], frame.shape[1]
    
    # Carregar configuração do ROI salva anteriormente
    roi_config = carregar_roi_config()
    
    # Valores padrão para os pontos do ROI (ou use os valores carregados)
    x1 = 200 if roi_config is None else roi_config['x1']
    y1 = altura if roi_config is None else roi_config['y1']
    x2 = 550 if roi_config is None else roi_config['x2']
    y2 = 250 if roi_config is None else roi_config['y2']
    x3 = 1100 if roi_config is None else roi_config['x3']
    y3 = altura if roi_config is None else roi_config['y3']
    
    # Criar sliders para os pontos do ROI
    # Ponto inferior esquerdo (x1, y1)
    cv2.createTrackbar('X1', 'Controles ROI', x1, largura, nothing)
    cv2.createTrackbar('Y1', 'Controles ROI', y1, altura, nothing)
    
    # Ponto superior (x2, y2)
    cv2.createTrackbar('X2', 'Controles ROI', x2, largura, nothing)
    cv2.createTrackbar('Y2', 'Controles ROI', y2, altura, nothing)
    
    # Ponto inferior direito (x3, y3)
    cv2.createTrackbar('X3', 'Controles ROI', x3, largura, nothing)
    cv2.createTrackbar('Y3', 'Controles ROI', y3, altura, nothing)
    
    # Parâmetros de Canny
    cv2.createTrackbar('Canny Min', 'Controles ROI', 50, 255, nothing)
    cv2.createTrackbar('Canny Max', 'Controles ROI', 150, 255, nothing)
    
    # Parâmetros de HoughLinesP
    cv2.createTrackbar('Min Length', 'Controles ROI', 40, 200, nothing)
    cv2.createTrackbar('Max Gap', 'Controles ROI', 5, 50, nothing)
    
    # Resetar o vídeo para o início
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Loop para reproduzir o vídeo
    while True:
        # Ler um frame do vídeo
        ret, frame = cap.read()
        
        # Se chegou ao final do vídeo, reiniciar
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Redimensionar o frame para 640x480
        frame_redimensionado = cv2.resize(frame, (640, 480))
        altura_redim, largura_redim = frame_redimensionado.shape[0], frame_redimensionado.shape[1]
        
        # Obter valores atuais dos sliders do ROI
        x1 = cv2.getTrackbarPos('X1', 'Controles ROI')
        y1 = cv2.getTrackbarPos('Y1', 'Controles ROI')
        x2 = cv2.getTrackbarPos('X2', 'Controles ROI')
        y2 = cv2.getTrackbarPos('Y2', 'Controles ROI')
        x3 = cv2.getTrackbarPos('X3', 'Controles ROI')
        y3 = cv2.getTrackbarPos('Y3', 'Controles ROI')
        
        # Ajustar valores para a resolução redimensionada
        x1_redim = int(x1 * largura_redim / largura)
        y1_redim = int(y1 * altura_redim / altura)
        x2_redim = int(x2 * largura_redim / largura)
        y2_redim = int(y2 * altura_redim / altura)
        x3_redim = int(x3 * largura_redim / largura)
        y3_redim = int(y3 * altura_redim / altura)
        
        # Parâmetros de processamento de imagem
        canny_min = cv2.getTrackbarPos('Canny Min', 'Controles ROI')
        canny_max = cv2.getTrackbarPos('Canny Max', 'Controles ROI')
        min_length = cv2.getTrackbarPos('Min Length', 'Controles ROI')
        max_gap = cv2.getTrackbarPos('Max Gap', 'Controles ROI')
        
        # Definir os vértices da região de interesse
        vertices = np.array([[(x1_redim, y1_redim), (x2_redim, y2_redim), (x3_redim, y3_redim)]], dtype=np.int32)
        
        # Visualizar a região de interesse
        roi_visualization = frame_redimensionado.copy()
        cv2.polylines(roi_visualization, vertices, True, (0, 255, 0), 2)
        
        # Adicionar pontos nos vértices do ROI
        for i, (x, y) in enumerate(vertices[0]):
            cv2.circle(roi_visualization, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(roi_visualization, f"P{i+1}", (x+10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Processar a imagem
        canny_image = canny(frame_redimensionado)
        
        # Aplicar a região de interesse usando os novos vértices
        cropped_canny = region_of_interest(canny_image, vertices)
        
        # Detectar linhas
        try:
            lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=min_length, maxLineGap=max_gap)
            averaged_lines = average_slope_intercept(frame_redimensionado, lines)
            line_image = display_lines(frame_redimensionado, averaged_lines)
            combo_image = cv2.addWeighted(frame_redimensionado, 0.8, line_image, 1, 1)
        except Exception as e:
            print(f"Erro na detecção de linhas: {e}")
            combo_image = frame_redimensionado.copy()
        
        # Mostrar imagens
        cv2.imshow('ROI', roi_visualization)
        #cv2.imshow('Canny', canny_image)
        cv2.imshow('Cropped', cropped_canny)
        cv2.imshow('Resultado', combo_image)
        #cv2.imshow('Lines', line_image)
        
        # Mostrar configuração atual na janela de controle
        config_text = np.zeros((200, 640, 3), dtype=np.uint8)
        config_str = f"Configuracao ROI Atual:\n"
        config_str += f"P1: ({x1}, {y1})\n"
        config_str += f"P2: ({x2}, {y2})\n"
        config_str += f"P3: ({x3}, {y3})\n"
        config_str += f"Canny: Min={canny_min}, Max={canny_max}\n"
        config_str += f"Linhas: MinLength={min_length}, MaxGap={max_gap}\n"
        config_str += f"S: Salvar, L: Carregar\n"
        
        y0, dy = 30, 20
        for i, line in enumerate(config_str.split('\n')):
            y = y0 + i*dy
            cv2.putText(config_text, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Config ROI', config_text)
        
        # Verificar teclas pressionadas
        key = cv2.waitKey(25) & 0xFF
        
        # ESC para sair
        if key == 27:
            break
        # 'S' para salvar configuração
        elif key == ord('s') or key == ord('S'):
            roi_config = {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'x3': x3,
                'y3': y3,
                'canny_min': canny_min,
                'canny_max': canny_max,
                'min_length': min_length,
                'max_gap': max_gap
            }
            salvar_roi_config(roi_config)
        # 'L' para carregar configuração
        elif key == ord('l') or key == ord('L'):
            roi_config = carregar_roi_config()
            if roi_config is not None:
                cv2.setTrackbarPos('X1', 'Controles ROI', roi_config['x1'])
                cv2.setTrackbarPos('Y1', 'Controles ROI', roi_config['y1'])
                cv2.setTrackbarPos('X2', 'Controles ROI', roi_config['x2'])
                cv2.setTrackbarPos('Y2', 'Controles ROI', roi_config['y2'])
                cv2.setTrackbarPos('X3', 'Controles ROI', roi_config['x3'])
                cv2.setTrackbarPos('Y3', 'Controles ROI', roi_config['y3'])
                if 'canny_min' in roi_config:
                    cv2.setTrackbarPos('Canny Min', 'Controles ROI', roi_config['canny_min'])
                if 'canny_max' in roi_config:
                    cv2.setTrackbarPos('Canny Max', 'Controles ROI', roi_config['canny_max'])
                if 'min_length' in roi_config:
                    cv2.setTrackbarPos('Min Length', 'Controles ROI', roi_config['min_length'])
                if 'max_gap' in roi_config:
                    cv2.setTrackbarPos('Max Gap', 'Controles ROI', roi_config['max_gap'])
    
    # Liberar os recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    caminho_do_video = "/home/djoker/code/pyhon/udacity/video_final.mp4"
    #caminho_do_video = "/home/djoker/code/pyhon/udacity/project_video.mp4"
    reproduzir_video_com_perspectiva(caminho_do_video)