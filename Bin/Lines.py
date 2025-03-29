import socket
import threading
import time
import cv2
import numpy as np
import struct
from collections import deque
import argparse

class SimulatorClient:
    # Configurações padrão
    IMAGE_PORT = 5000
    DATA_PORT = 5002
    CONTROL_PORT = 5001
    
    # Constantes
    VEHICLE_DATA_PACKET_ID = 0x01
    
    def __init__(self, simulator_ip="127.0.0.1"):
        self.simulator_ip = simulator_ip

        # Sockets para comunicação
        self.image_socket = None
        self.data_socket = None
        self.control_socket = None
        
        # Threads

        self.data_thread = None
        self.running = False
        
        # Buffers e filas
        self.latest_image = None
        self.latest_image_time = 0
        self.vehicle_data = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'rotation': {'y': 0.0},
            'speed': 0.0,
            'steering': 0.0,
            'throttle': 0.0,
            'control_mode': 0
        }
        self.vehicle_data_time = 0
        
        # Callbacks
        self.image_callback = None
        self.data_callback = None
        
        # Status de conexão
        self.connected = False
        self.last_received_time = 0
        
        # Mutex para acesso aos dados
        self.left_line = deque(maxlen=10)
        self.right_line = deque(maxlen=10)
        self.steering_history = deque(maxlen=5)
        
        # Timestamp da última detecção de faixas bem-sucedida
        self.last_detection_time = 0

        self.data_lock = threading.Lock()
        
    def connect(self):
        """Conecta aos sockets UDP para receber imagens e dados do simulador."""
        try:
            # Socket para receber imagens
            self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.image_socket.bind(("0.0.0.0", self.IMAGE_PORT))
            self.image_socket.settimeout(1.0)  # Timeout de 1 segundo
            
            # Socket para receber dados do veículo
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.data_socket.bind(("0.0.0.0", self.DATA_PORT))
            self.data_socket.settimeout(1.0)  # Timeout de 1 segundo
            
            # Socket para enviar comandos de controle
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Iniciar threads
            self.running = True
          
            
            self.data_thread = threading.Thread(target=self._receive_vehicle_data)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            print(f"Aguardando conexão com o simulador em {self.simulator_ip}...")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Erro ao configurar sockets: {e}")
            return False
    
    def disconnect(self):
        """Desconecta do simulador."""
        self.running = False
        
     
        
        if self.data_thread:
            self.data_thread.join(timeout=1.0)
        
        # Fechar sockets
        if self.image_socket:
            self.image_socket.close()
            
        if self.data_socket:
            self.data_socket.close()
            
        if self.control_socket:
            self.control_socket.close()
            
        self.connected = False
        print("Desconectado do simulador")
    
    def receive_images(self):
        try:
                # Receber pacote de imagem
                data, _ = self.image_socket.recvfrom(65507)  
                image_array = np.frombuffer(data, dtype=np.uint8)
                self.latest_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                return self.latest_image    
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Erro ao receber imagem: {e}")
                
    
    def _receive_vehicle_data(self):
        """Thread para receber dados do veículo do simulador."""
        while self.running:
            try:
                # Receber pacote de dados
                data, addr = self.data_socket.recvfrom(1024)
                
                if data and len(data) >= 30:  # Verificar tamanho mínimo do pacote
                    # Verificar ID do pacote
                    if data[0] == self.VEHICLE_DATA_PACKET_ID:
                        # Extrair dados do veículo
                        pos_x = struct.unpack('f', data[1:5])[0]
                        pos_y = struct.unpack('f', data[5:9])[0]
                        pos_z = struct.unpack('f', data[9:13])[0]
                        rot_y = struct.unpack('f', data[13:17])[0]
                        speed = struct.unpack('f', data[17:21])[0]
                        steering = struct.unpack('f', data[21:25])[0]
                        throttle = struct.unpack('f', data[25:29])[0]
                        control_mode = data[29]
                        
                        # Atualizar dados do veículo
                        with self.data_lock:
                            self.vehicle_data['position']['x'] = pos_x
                            self.vehicle_data['position']['y'] = pos_y
                            self.vehicle_data['position']['z'] = pos_z
                            self.vehicle_data['rotation']['y'] = rot_y
                            self.vehicle_data['speed'] = speed
                            self.vehicle_data['steering'] = steering
                            self.vehicle_data['throttle'] = throttle
                            self.vehicle_data['control_mode'] = control_mode
                            self.vehicle_data_time = time.time()
                            
                            # Chamar callback se existir
                            if self.data_callback:
                                self.data_callback(self.vehicle_data)
                        
                        self.last_received_time = time.time()
            
            except socket.timeout:
                # Timeout é normal, continuar
                pass
            except Exception as e:
                print(f"Erro ao receber dados do veículo: {e}")
                time.sleep(0.1)
    
    def send_control(self, throttle, breaking,steering):
        """Envia comandos de controle para o simulador."""
        if not self.connected or self.control_socket is None:
            return False
        
        try:
            # Criar pacote de controle: [throttle(4), breaking(4), steering(4)]
            control_packet = bytearray(12)
            struct.pack_into('f', control_packet, 0, float(throttle))
            struct.pack_into('f', control_packet, 4, float(steering))
            struct.pack_into('f', control_packet, 8, float(breaking))
            

            self.control_socket.sendto(control_packet, (self.simulator_ip, self.CONTROL_PORT))
            return True
        except Exception as e:
            print(f"Erro ao enviar controle: {e}")
            return False
    
    def get_latest_image(self):
        """Retorna a imagem mais recente recebida do simulador."""
        if self.latest_image is not None:
            return self.latest_image.copy()
        return None
    
    def get_vehicle_data(self):
        """Retorna os dados mais recentes do veículo."""
        with self.data_lock:
            return self.vehicle_data.copy()
 
    def set_data_callback(self, callback):
        """Define um callback para processar dados do veículo recebidos."""
        self.data_callback = callback
    
    def is_connected(self):
        """Verifica se o cliente está conectado e recebe dados do simulador."""
        if time.time() - self.last_received_time > 3.0:
            self.connected = False
        return self.connected
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


if __name__ == "__main__":
    
    client = SimulatorClient(simulator_ip='127.0.0.1')
    last_command_time = time.time()
    command_rate_limit = 0.1
    frames_processed = 0
    start_time = time.time()
    last_fps_update = start_time
    current_fps = 0
    throttle = 0.0
    steering = 0.0
    breaking = 0.0
    
    def process_image(image):
        vehicle_data = client.get_vehicle_data()
        

        cv2.putText(image, f"FPS: {current_fps:.1f} ", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(image, f"Velocidade: {vehicle_data['speed']:.1f} km/h", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(image, f"Steering: ({vehicle_data['steering']:.1f}, Throttle:  {vehicle_data['throttle']:.1f})", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        original_image = image.copy()
        
        steering_angle = client.detect_lanes(original_image)
        client.draw_lane_overlay(original_image, steering_angle)
    

        cv2.imshow("Original", original_image)    
        #cv2.imshow("Simulador", image)
        cv2.waitKey(1)
    
    # Callback para processar dados do veículo
    def process_vehicle_data(data):
        pass
    
 
    client.set_data_callback(process_vehicle_data)
 
    try:
    
        if client.connect():
            print("Conectado ao simulador. Pressiona Ctrl+C para sair.")
            


            
            while client.running:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    client.running = False
                    break
                
                frame = client.receive_images()
                if frame is not None:  
                    process_image(frame)
                    frames_processed += 1 

                current_time = time.time()
                if current_time - last_fps_update >= 1.0:
                    current_fps = frames_processed / (current_time - last_fps_update)
                    frames_processed = 0
                    last_fps_update = current_time
                

                
                # Teclas WASD para controle básico
                if key == ord('w'):
                    throttle = min(throttle + 0.05, 1.0)
                    breaking = 0.0
                elif key == ord('s'):
                    breaking = 1.0
                elif key == ord('a'):
                    steering = max(steering - 0.1, -1.0)
                elif key == ord('d'):
                    steering = min(steering + 0.1, 1.0)
                elif key == ord(' '): 
                    throttle = 0.0
                    steering = 0.0
                    breaking = 0.0
                
              
                if current_time - last_command_time >= command_rate_limit:
                    last_command_time = current_time
                    client.send_control(throttle, breaking, steering)
                
   
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nEncerrando cliente...")
    finally:

        client.disconnect()
        cv2.destroyAllWindows()
    
    print("Cliente encerrado.")
