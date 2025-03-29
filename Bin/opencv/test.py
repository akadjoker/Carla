import cv2
import numpy as np


def nothing(x):
    pass

def perspective_transform(img, src=None, dst=None):
    """
    Aplica transformação de perspectiva para obter visão de cima da estrada
    """
    img_size = (img.shape[1], img.shape[0])
    
    if src is None:
        # Pontos de origem padrão
        src = np.float32([
            [190, 720],   # Inferior esquerdo
            [590, 450],   # Superior esquerdo
            [690, 450],   # Superior direito
            [1130, 720]   # Inferior direito
        ])
    
    if dst is None:
        # Pontos de destino padrão
        dst = np.float32([
            [320, 720],   # Inferior esquerdo
            [320, 0],     # Superior esquerdo
            [960, 0],     # Superior direito
            [960, 720]    # Inferior direito
        ])
    
    # Calcular matriz de transformação e sua inversa
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Aplicar transformação de perspectiva
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv

def reproduzir_video(caminho_video):
    cap = cv2.VideoCapture(caminho_video)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    # Loop para reproduzir o vídeo
    while True:
        ret, frame = cap.read()
        
        # Se chegou ao final do vídeo, reiniciar
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_redimensionado = cv2.resize(frame, (640, 480))
        
        #cv2.imshow('Video', frame)
        
        # Aplicar transformação de perspectiva
        warped, M, Minv = perspective_transform(frame_redimensionado)
        
        cv2.imshow('Perspective Transform', frame_redimensionado)
        
        if cv2.waitKey(25) & 0xFF == 27:
            break
    
    # Liberar os recursos
    cap.release()
    cv2.destroyAllWindows()

# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelo caminho do seu arquivo de vídeo
    caminho_do_video = "/home/djoker/code/pyhon/udacity/project_video.mp4"
    reproduzir_video(caminho_do_video)
