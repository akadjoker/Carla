import socket
import threading
import time
import cv2
import numpy as np
import struct
from collections import deque


import tensorflow as tf
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from tensorflow.keras.models import load_model
from Training import preProcess

def rotate_steering_wheel(image, steering_angle):
    steering_wheel = cv2.imread('steering_wheel_image.jpg', cv2.IMREAD_UNCHANGED)
    
    steering_wheel = cv2.resize(steering_wheel, (200, 200))
    
    height, width = steering_wheel.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, -steering_angle * 45, 1.0)
    
    rotated_wheel = cv2.warpAffine(steering_wheel, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0,0,0,0))
    
    return rotated_wheel

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
    tf.keras.config.enable_unsafe_deserialization()
    model = load_model('best_steering_model.keras')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v', 'XVID'
    out = cv2.VideoWriter("model.mp4", fourcc, 30, (640, 480))

    
    def process_image(image):
        vehicle_data = client.get_vehicle_data()
        

        cv2.putText(image, f"FPS: {current_fps:.1f} ", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(image, f"Velocidade: {vehicle_data['speed']:.1f} km/h", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(image, f"Steering: ({vehicle_data['steering']:.1f}, Throttle:  {vehicle_data['throttle']:.1f})", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        process = preProcess(image)
        frame = np.array([process])
        angulo = float(model.predict(frame)[0]) #+ (0.080-0.015)
        steering_wheel = rotate_steering_wheel(image, angulo)
        steering_atual = vehicle_data['steering']
        erro = angulo - steering_atual

        frame_height, frame_width = image.shape[:2]
        wheel_height, wheel_width = steering_wheel.shape[:2]
        
        
        x_offset = frame_width - wheel_width - 20
        y_offset = frame_height - wheel_height - 20
        
        # Adicionar canal alfa
        for c in range(0, 3):
            image[y_offset:y_offset+wheel_height, x_offset:x_offset+wheel_width, c] = \
                steering_wheel[:,:,c] * (steering_wheel[:,:,3]/255.0) + \
                image[y_offset:y_offset+wheel_height, x_offset:x_offset+wheel_width, c] * (1.0 - steering_wheel[:,:,3]/255.0)

        cv2.putText(image, f"Predict: {angulo:.3f} | Erro: {erro:.3f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        #out.write(image)
        cv2.imshow("Simulador", image)
        cv2.imshow("Aggregated", process)
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
        out.release()
        cv2.destroyAllWindows()
    
    print("Cliente encerrado.")
