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

def perspective_transform(img, src=None, dst=None):
    """
    Aplica transformação de perspectiva para obter visão de cima da estrada
    """
    img_size = (img.shape[1], img.shape[0])
    if src is None:
        # Pontos de origem padrão
        src = np.float32([
            [185, 431],  # Inferior esquerdo
            [282, 297],  # Superior esquerdo
            [376, 295],  # Superior direito
            [510, 433]   # Inferior direito
        ])
    if dst is None:
        # Pontos de destino padrão
        dst = np.float32([
            [171, 480],  # Inferior esquerdo
            [171, 0],    # Superior esquerdo
            [571, 0],    # Superior direito
            [571, 480]   # Inferior direito
        ])
    # Calcular matriz de transformação e sua inversa
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Aplicar transformação de perspectiva
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def salvar_configuracao(config, nome_arquivo="perspective_config.json"):
    """Salva a configuração atual em um arquivo JSON"""
    with open(nome_arquivo, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuração salva em {nome_arquivo}")

def carregar_configuracao(nome_arquivo="perspective_config.json"):
    """Carrega a configuração de um arquivo JSON"""
    if os.path.exists(nome_arquivo):
        with open(nome_arquivo, 'r') as f:
            config = json.load(f)
        print(f"Configuração carregada de {nome_arquivo}")
        return config
    else:
        print(f"Arquivo {nome_arquivo} não encontrado. Usando configuração padrão.")
        return None
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



def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    triangle = np.array([[
    (200, height),
    (550, 250),
    (1100, height),]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def reproduzir_video_com_perspectiva(caminho_video):
    # Abrir o arquivo de vídeo
    cap = cv2.VideoCapture(caminho_video)
    
    # Verificar se o vídeo foi aberto com sucesso
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    # Criar janela para os sliders
    cv2.namedWindow('Controles')
    cv2.resizeWindow('Controles', 640, 400)
    
    # Obter um frame para determinar tamanho
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler o primeiro frame.")
        return
    
    altura, largura = frame.shape[0], frame.shape[1]
    
    # Tentar carregar configuração salva
    config = carregar_configuracao()
    
    # Valores iniciais para os sliders
    inf_esq_x = 185 if config is None else config['inf_esq_x']
    inf_esq_y = 431 if config is None else config['inf_esq_y']
    sup_esq_x = 282 if config is None else config['sup_esq_x']
    sup_esq_y = 297 if config is None else config['sup_esq_y']
    sup_dir_x = 376 if config is None else config['sup_dir_x']
    sup_dir_y = 295 if config is None else config['sup_dir_y']
    inf_dir_x = 510 if config is None else config['inf_dir_x']
    inf_dir_y = 433 if config is None else config['inf_dir_y']
    dst_esq = 171 if config is None else config['dst_esq']
    dst_dir = 571 if config is None else config['dst_dir']
    
    # Criar sliders para os pontos de origem
    # Inferior esquerdo (x, y)
    cv2.createTrackbar('Inf_Esq_X', 'Controles', inf_esq_x, largura, nothing)
    cv2.createTrackbar('Inf_Esq_Y', 'Controles', inf_esq_y, altura, nothing)
    
    # Superior esquerdo (x, y)
    cv2.createTrackbar('Sup_Esq_X', 'Controles', sup_esq_x, largura, nothing)
    cv2.createTrackbar('Sup_Esq_Y', 'Controles', sup_esq_y, altura, nothing)
    
    # Superior direito (x, y)
    cv2.createTrackbar('Sup_Dir_X', 'Controles', sup_dir_x, largura, nothing)
    cv2.createTrackbar('Sup_Dir_Y', 'Controles', sup_dir_y, altura, nothing)
    
    # Inferior direito (x, y)
    cv2.createTrackbar('Inf_Dir_X', 'Controles', inf_dir_x, largura, nothing)
    cv2.createTrackbar('Inf_Dir_Y', 'Controles', inf_dir_y, altura, nothing)
    
    # Criar sliders para os pontos de destino
    cv2.createTrackbar('Dst_Esq', 'Controles', dst_esq, largura, nothing)
    cv2.createTrackbar('Dst_Dir', 'Controles', dst_dir, largura, nothing)
    
 
    
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
        
        # Obter valores atuais dos sliders de origem
        inf_esq_x = cv2.getTrackbarPos('Inf_Esq_X', 'Controles')
        inf_esq_y = cv2.getTrackbarPos('Inf_Esq_Y', 'Controles')
        
        sup_esq_x = cv2.getTrackbarPos('Sup_Esq_X', 'Controles')
        sup_esq_y = cv2.getTrackbarPos('Sup_Esq_Y', 'Controles')
        
        sup_dir_x = cv2.getTrackbarPos('Sup_Dir_X', 'Controles')
        sup_dir_y = cv2.getTrackbarPos('Sup_Dir_Y', 'Controles')
        
        inf_dir_x = cv2.getTrackbarPos('Inf_Dir_X', 'Controles')
        inf_dir_y = cv2.getTrackbarPos('Inf_Dir_Y', 'Controles')
        
        # Obter valores atuais dos sliders de destino
        dst_esq = cv2.getTrackbarPos('Dst_Esq', 'Controles')
        dst_dir = cv2.getTrackbarPos('Dst_Dir', 'Controles')
        
        # Verificar se deve aplicar a máscara
        usar_mascara = cv2.getTrackbarPos('Usar Mascara', 'Controles')
        
        # Configurar os pontos de origem para a transformação
        src = np.float32([
            [inf_esq_x, inf_esq_y],  # Inferior esquerdo
            [sup_esq_x, sup_esq_y],  # Superior esquerdo
            [sup_dir_x, sup_dir_y],  # Superior direito
            [inf_dir_x, inf_dir_y]   # Inferior direito
        ])
        
        # Configurar os pontos de destino para a transformação
        dst = np.float32([
            [dst_esq, 480],    # Inferior esquerdo
            [dst_esq, 0],      # Superior esquerdo
            [dst_dir, 0],      # Superior direito
            [dst_dir, 480]     # Inferior direito
        ])
        
        # Desenhar os pontos e o polígono de origem no frame original
        frame_com_pontos = frame_redimensionado.copy()
        
        # Desenhar os pontos de origem
        for i, ponto in enumerate(src):
            x, y = int(ponto[0]), int(ponto[1])
            cv2.circle(frame_com_pontos, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame_com_pontos, f"P{i+1}", (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Desenhar o polígono de origem (região de interesse)
        draw_polygon(frame_com_pontos, src, color=(0, 255, 0), thickness=2)
        
        # Aplicar a transformação de perspectiva
        warped, M, Minv = perspective_transform(frame_redimensionado, src, dst)
        
        # Criar uma imagem para mostrar o retângulo de destino
        dst_visualization = np.zeros_like(warped)
        
        # Desenhar o retângulo de destino
        for i, ponto in enumerate(dst):
            x, y = int(ponto[0]), int(ponto[1])
            cv2.circle(dst_visualization, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(dst_visualization, f"D{i+1}", (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Desenhar o retângulo de destino
        draw_polygon(dst_visualization, dst, color=(255, 0, 0), thickness=2)
        
        
        mask = np.zeros_like(warped)
        pts = dst.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        
        # Aplicar máscara usando bitwise AND
        masked_warped = cv2.bitwise_and(warped, mask)
        
        # Criar uma versão com fundo preto fora do retângulo
        result_masked = masked_warped.copy()
 
        
        # Criar uma versão com visualização de área mascarada
        result_with_highlight = warped.copy()
        # Fazer área fora da máscara mais escura
        result_with_highlight[mask == 0] = result_with_highlight[mask == 0] // 2
        
        # Desenhar contorno do retângulo para visualização
        #draw_polygon(result_with_highlight, dst, color=(0, 0, 255), thickness=2)
        
        # Mostrar as imagens com máscara
        #cv2.imshow('Mask', result_masked)
        cv2.imshow('Highlight', result_with_highlight)

        canny_image = canny(result_masked)
        #cropped_canny = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
        averaged_lines = average_slope_intercept(result_masked, lines)
        line_image = display_lines(result_masked, averaged_lines)
        combo_image = cv2.addWeighted(result_masked, 0.8, line_image, 1, 1)
        cv2.imshow('Result', combo_image)   
        cv2.imshow('Canny', canny_image)
        
      
        cv2.putText(frame_com_pontos, "S: salvar, L: carregar, M: alternar mascara", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Mostrar configuração atual na janela de controle
        config_text = np.zeros((200, 640, 3), dtype=np.uint8)
        config_str = f"Configuracao Atual:\n"
        config_str += f"Inf_Esq: ({inf_esq_x}, {inf_esq_y})\n"
        config_str += f"Sup_Esq: ({sup_esq_x}, {sup_esq_y})\n"
        config_str += f"Sup_Dir: ({sup_dir_x}, {sup_dir_y})\n"
        config_str += f"Inf_Dir: ({inf_dir_x}, {inf_dir_y})\n"
        config_str += f"Dst_Esq: {dst_esq}, Dst_Dir: {dst_dir}\n"
    
        
        y0, dy = 30, 20
        for i, line in enumerate(config_str.split('\n')):
            y = y0 + i*dy
            cv2.putText(config_text, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        #cv2.imshow('Config', config_text)
        
        # Mostrar a imagem original com os pontos
        cv2.imshow('Main', frame_com_pontos)
        
        # Verificar teclas pressionadas
        key = cv2.waitKey(25) & 0xFF
        

        if key == 27:
            break

        elif key == ord('s') or key == ord('S'):
            config = {
                'inf_esq_x': inf_esq_x,
                'inf_esq_y': inf_esq_y,
                'sup_esq_x': sup_esq_x,
                'sup_esq_y': sup_esq_y,
                'sup_dir_x': sup_dir_x,
                'sup_dir_y': sup_dir_y,
                'inf_dir_x': inf_dir_x,
                'inf_dir_y': inf_dir_y,
                'dst_esq': dst_esq,
                'dst_dir': dst_dir
            }
            salvar_configuracao(config)

        elif key == ord('l') or key == ord('L'):
            config = carregar_configuracao()
            if config is not None:
                cv2.setTrackbarPos('Inf_Esq_X', 'Controles', config['inf_esq_x'])
                cv2.setTrackbarPos('Inf_Esq_Y', 'Controles', config['inf_esq_y'])
                cv2.setTrackbarPos('Sup_Esq_X', 'Controles', config['sup_esq_x'])
                cv2.setTrackbarPos('Sup_Esq_Y', 'Controles', config['sup_esq_y'])
                cv2.setTrackbarPos('Sup_Dir_X', 'Controles', config['sup_dir_x'])
                cv2.setTrackbarPos('Sup_Dir_Y', 'Controles', config['sup_dir_y'])
                cv2.setTrackbarPos('Inf_Dir_X', 'Controles', config['inf_dir_x'])
                cv2.setTrackbarPos('Inf_Dir_Y', 'Controles', config['inf_dir_y'])
                cv2.setTrackbarPos('Dst_Esq', 'Controles', config['dst_esq'])
                cv2.setTrackbarPos('Dst_Dir', 'Controles', config['dst_dir'])


        elif key == ord('m') or key == ord('M'):
            novo_valor = 1 if usar_mascara == 0 else 0
            cv2.setTrackbarPos('Usar Mascara', 'Controles', novo_valor)
    

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    caminho_do_video = "/home/djoker/code/pyhon/udacity/pt1.mp4"
    caminho_do_video = "/home/djoker/code/pyhon/udacity/video_final.mp4"
    reproduzir_video_com_perspectiva(caminho_do_video)