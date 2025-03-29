import cv2
import numpy as np
import time
import threading
import argparse
from collections import deque
import os
import csv
from datetime import datetime

# Importar o cliente do simulador
from simulator_client import SimulatorClient

class AutonomousDriving:
    def __init__(self, simulator_ip='127.0.0.1', data_collection=False, training_directory=None):
        self.simulator_ip = simulator_ip
        self.client = SimulatorClient(simulator_ip=simulator_ip)
        
        # Direção autônoma - parâmetros
        self.lane_detection_active = True
        self.autonomous_mode = False
        self.debug_view = True
        
        # Variáveis para coleta de dados
        self.data_collection = data_collection
        if data_collection:
            self.training_directory = training_directory or f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(os.path.join(self.training_directory, "images"), exist_ok=True)
            self.csv_file = open(os.path.join(self.training_directory, "driving_log.csv"), 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['image_path', 'steering', 'throttle', 'speed', 'timestamp'])
            self.frame_counter = 0
            print(f"Coletando dados de treinamento em: {self.training_directory}")
        
        # Variáveis para detecção de faixas
        self.left_line = deque(maxlen=10)
        self.right_line = deque(maxlen=10)
        self.steering_history = deque(maxlen=5)
        
        # Timestamp da última detecção de faixas bem-sucedida
        self.last_detection_time = 0
        
        # Controles
        self.throttle = 0.0
        self.steering = 0.0
        self.target_speed = 30.0  # km/h
        
    def connect(self):
        """Conecta ao simulador e configura callbacks"""
        if not self.client.connect():
            print("Falha ao conectar ao simulador")
            return False
            
        # Configurar callbacks
        self.client.set_image_callback(self.process_image)
        self.client.set_data_callback(self.process_vehicle_data)
        
        return True
        
    def disconnect(self):
        """Desconecta do simulador e limpa recursos"""
        self.client.disconnect()
        if self.data_collection:
            self.csv_file.close()
        cv2.destroyAllWindows()
        
    def process_image(self, image):
        """Processa a imagem da câmera para detecção de faixas e decisões de direção"""
        original_image = image.copy()
        
        if self.lane_detection_active:
            steering_angle = self.detect_lanes(image)
            
            # Aplicar suavização ao ângulo de direção
            if steering_angle is not None:
                self.steering_history.append(steering_angle)
                avg_steering = sum(self.steering_history) / len(self.steering_history)
                
                # Apenas atualizar a direção se estiver em modo autônomo
                if self.autonomous_mode:
                    self.steering = avg_steering
                    
                    # Ajustar aceleração baseado na curva e velocidade alvo
                    vehicle_data = self.client.get_vehicle_data()
                    current_speed = vehicle_data['speed']
                    steering_factor = abs(avg_steering)
                    
                    # Reduzir velocidade em curvas
                    curve_target_speed = self.target_speed * (1.0 - steering_factor * 0.5)
                    
                    # Ajustar aceleração para atingir a velocidade alvo
                    if current_speed < curve_target_speed - 2.0:
                        self.throttle = min(self.throttle + 0.01, 0.7)
                    elif current_speed > curve_target_speed + 2.0:
                        self.throttle = max(self.throttle - 0.03, 0.0)
                    
                # Mostrar as linhas de faixa detectadas se debug estiver ativado
                if self.debug_view:
                    self.draw_lane_overlay(original_image, avg_steering)
        
        # Armazenar dados para treinamento se a coleta estiver ativada
        if self.data_collection and time.time() - self.last_frame_save > 0.2:  # 5 FPS para dados
            self.save_training_data(original_image)
            self.last_frame_save = time.time()
        
        # Mostrar a imagem com informações
        if self.debug_view:
            self.draw_hud(original_image)
            cv2.imshow("Simulador - Visão Autônoma", original_image)
    
    def process_vehicle_data(self, data):
        """Processa dados do veículo para decisões de controle e telemetria"""
        # Implementar lógica específica para processamento de dados do veículo
        pass
        
    def detect_lanes(self, image):
        """Detecta as faixas de rodagem na imagem e calcula o ângulo de direção"""
        try:
            # Conversão para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Região de interesse (metade inferior da imagem)
            height, width = gray.shape
            roi_height = height // 2
            roi = gray[roi_height:height, 0:width]
            
            # Detecção de bordas
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            
            # Transformação de Hough para detecção de linhas
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=50, maxLineGap=100)
            
            if lines is not None:
                left_lines = []
                right_lines = []
                
                # Separar linhas da esquerda e direita
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calcular inclinação
                    if x2 - x1 == 0:  # Evitar divisão por zero
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Filtrar linhas horizontais
                    if abs(slope) < 0.1:
                        continue
                    
                    # Separar linhas por inclinação
                    if slope < 0:  # Linhas à esquerda têm inclinação negativa
                        left_lines.append(line[0])
                    else:  # Linhas à direita têm inclinação positiva
                        right_lines.append(line[0])
                
                # Encontrar linhas médias
                left_avg_line = self.average_lines(left_lines) if left_lines else None
                right_avg_line = self.average_lines(right_lines) if right_lines else None
                
                # Armazenar linhas detectadas
                if left_avg_line is not None:
                    self.left_line.append(left_avg_line)
                if right_avg_line is not None:
                    self.right_line.append(right_avg_line)
                
                # Usar a média das últimas linhas detectadas
                left_line_avg = self.get_average_line(self.left_line) if self.left_line else None
                right_line_avg = self.get_average_line(self.right_line) if self.right_line else None
                
                # Calcular o ponto central da faixa e o ângulo de direção
                if left_line_avg is not None and right_line_avg is not None:
                    # Ambas as linhas detectadas
                    lane_center = (left_line_avg[0] + right_line_avg[0]) // 2
                    car_center = width // 2
                    steering_angle = (lane_center - car_center) / (width / 2) * 0.5  # Normalizado para [-0.5, 0.5]
                elif left_line_avg is not None:
                    # Apenas linha esquerda detectada
                    # Aproximar o centro da faixa
                    lane_center = left_line_avg[0] + 160  # Valor estimado para largura da faixa
                    car_center = width // 2
                    steering_angle = (lane_center - car_center) / (width / 2) * 0.4
                elif right_line_avg is not None:
                    # Apenas linha direita detectada
                    # Aproximar o centro da faixa
                    lane_center = right_line_avg[0] - 160  # Valor estimado para largura da faixa
                    car_center = width // 2
                    steering_angle = (lane_center - car_center) / (width / 2) * 0.4
                else:
                    # Nenhuma linha detectada
                    steering_angle = None
                
                self.last_detection_time = time.time()
                return steering_angle
                
        except Exception as e:
            print(f"Erro na detecção de faixas: {e}")
        
        # Se chegou aqui, a detecção falhou
        # Se a última detecção bem-sucedida foi há muito tempo, retornar None
        if time.time() - self.last_detection_time > 1.0:
            return None
        
        # Caso contrário, manter o último valor calculado
        return self.steering if self.steering_history else None
    
    def average_lines(self, lines):
        """Calcula a linha média de um conjunto de linhas"""
        if not lines:
            return None
            
        x1_sum = 0
        y1_sum = 0
        x2_sum = 0
        y2_sum = 0
        
        for x1, y1, x2, y2 in lines:
            x1_sum += x1
            y1_sum += y1
            x2_sum += x2
            y2_sum += y2
            
        count = len(lines)
        x1_avg = x1_sum // count
        y1_avg = y1_sum // count
        x2_avg = x2_sum // count
        y2_avg = y2_sum // count
        
        return [x1_avg, y1_avg, x2_avg, y2_avg]
    
    def get_average_line(self, line_queue):
        """Calcula a média móvel das linhas detectadas"""
        if not line_queue:
            return None
            
        x1_sum = 0
        y1_sum = 0
        x2_sum = 0
        y2_sum = 0
        
        for line in line_queue:
            x1_sum += line[0]
            y1_sum += line[1]
            x2_sum += line[2]
            y2_sum += line[3]
            
        count = len(line_queue)
        x1_avg = x1_sum // count
        y1_avg = y1_sum // count
        x2_avg = x2_sum // count
        y2_avg = y2_sum // count
        
        return [x1_avg, y1_avg, x2_avg, y2_avg]
    
    def draw_lane_overlay(self, image, steering_angle):
        """Desenha a sobreposição visual das faixas detectadas"""
        height, width, _ = image.shape
        
        # Desenhar linha central
        center_x = width // 2
        cv2.line(image, (center_x, height), (center_x, height - 20), (255, 0, 0), 2)
        
        # Desenhar ângulo de direção
        if steering_angle is not None:
            direction_x = int(center_x + steering_angle * width / 2)
            cv2.line(image, (center_x, height - 20), (direction_x, height - 50), (0, 255, 0), 3)
        
        # Desenhar linhas de faixa detectadas
        if self.left_line:
            left_line = self.get_average_line(self.left_line)
            x1, y1, x2, y2 = left_line
            y1 += height // 2  # Ajustar para a posição real na imagem
            y2 += height // 2
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
        if self.right_line:
            right_line = self.get_average_line(self.right_line)
            x1, y1, x2, y2 = right_line
            y1 += height // 2  # Ajustar para a posição real na imagem
            y2 += height // 2
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    def draw_hud(self, image):
        """Desenha informações na interface (HUD)"""
        height, width, _ = image.shape
        
        # Dados do veículo
        vehicle_data = self.client.get_vehicle_data()
        speed = vehicle_data['speed']
        
        # Textos informativos
        cv2.putText(image, f"Velocidade: {speed:.1f} km/h", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(image, f"Direção: {self.steering:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                   
        cv2.putText(image, f"Aceleração: {self.throttle:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status do modo autônomo
        mode_text = "AUTÔNOMO" if self.autonomous_mode else "MANUAL"
        mode_color = (0, 255, 0) if self.autonomous_mode else (0, 0, 255)
        cv2.putText(image, mode_text, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    def save_training_data(self, image):
        """Salva dados para treinamento de redes neurais"""
        if not self.data_collection:
            return
            
        # Salvar apenas se o veículo estiver em movimento
        vehicle_data = self.client.get_vehicle_data()
        if abs(vehicle_data['speed']) < 1.0:
            return  # Ignorar se estiver parado
            
        # Salvar imagem
        image_path = os.path.join(self.training_directory, "images", f"img_{self.frame_counter:06d}.jpg")
        cv2.imwrite(image_path, image)
        
        # Salvar dados no CSV
        rel_path = os.path.join("images", f"img_{self.frame_counter:06d}.jpg")
        self.csv_writer.writerow([
            rel_path,
            self.steering,
            self.throttle,
            vehicle_data['speed'],
            time.time()
        ])
        
        self.frame_counter += 1
        
    def run(self):
        """Loop principal de execução"""
        self.last_frame_save = time.time()
        
        try:
            print("Iniciando sistema de direção autônoma. Controles:")
            print("  A - Ativar/Desativar modo autônomo")
            print("  D - Ativar/Desativar visualização de debug")
            print("  L - Ativar/Desativar detecção de faixas")
            print("  R - Resetar variáveis")
            print("  [+]/[-] - Aumentar/Diminuir velocidade alvo")
            print("  ESC - Sair")
            
            while self.client.is_connected():
                key = cv2.waitKey(1) & 0xFF
                
                # ESC para sair
                if key == 27:
                    break
                    
                # A - Alternar modo autônomo
                elif key == ord('a'):
                    self.autonomous_mode = not self.autonomous_mode
                    if not self.autonomous_mode:
                        self.throttle = 0.0
                        self.steering = 0.0
                    print(f"Modo autônomo: {'ATIVADO' if self.autonomous_mode else 'DESATIVADO'}")
                
                # D - Alternar visualização de debug
                elif key == ord('d'):
                    self.debug_view = not self.debug_view
                    print(f"Visualização de debug: {'ATIVADA' if self.debug_view else 'DESATIVADA'}")
                
                # L - Alternar detecção de faixas
                elif key == ord('l'):
                    self.lane_detection_active = not self.lane_detection_active
                    print(f"Detecção de faixas: {'ATIVADA' if self.lane_detection_active else 'DESATIVADA'}")
                
                # R - Resetar variáveis
                elif key == ord('r'):
                    self.steering_history.clear()
                    self.left_line.clear()
                    self.right_line.clear()
                    self.throttle = 0.0
                    self.steering = 0.0
                    print("Variáveis resetadas")
                
                # Aumentar/diminuir velocidade alvo
                elif key == ord('+') or key == ord('='):
                    self.target_speed = min(self.target_speed + 5.0, 100.0)
                    print(f"Velocidade alvo: {self.target_speed:.1f} km/h")
                elif key == ord('-'):
                    self.target_speed = max(self.target_speed - 5.0, 10.0)
                    print(f"Velocidade alvo: {self.target_speed:.1f} km/h")
                
                # Controle manual em modo não-autônomo
                if not self.autonomous_mode:
                    if key == ord('w'):
                        self.throttle = min(self.throttle + 0.1, 1.0)
                    elif key == ord('s'):
                        self.throttle = max(self.throttle - 0.1, -0.5)
                    elif key == ord(' '):  # Espaço para frear
                        self.throttle = 0.0
                        
                    if key == ord('a'):
                        self.steering = max(self.steering - 0.1, -1.0)
                    elif key == ord('d'):
                        self.steering = min(self.steering + 0.1, 1.0)
                    
                # Enviar controles para o simulador
                self.client.send_control(self.throttle, self.steering)
                
                # Pequena pausa para não sobrecarregar
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nEncerrando sistema de direção autônoma...")
        finally:
            self.disconnect()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sistema de Direção Autônoma para Simulador')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Endereço IP do simulador')
    parser.add_argument('--collect', action='store_true', help='Ativar coleta de dados para treinamento')
    parser.add_argument('--dir', type=str, default=None, help='Diretório para salvar dados de treinamento')
    args = parser.parse_args()
    
    # Criar e executar o sistema
    autonomous_system = AutonomousDriving(
        simulator_ip=args.ip,
        data_collection=args.collect,
        training_directory=args.dir
    )
    
    if autonomous_system.connect():
        autonomous_system.run()
    else:
        print("Falha ao iniciar o sistema de direção autônoma")