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
        self.image_thread = None
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
        self.image_lock = threading.Lock()
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
            self.image_thread = threading.Thread(target=self._receive_images)
            self.image_thread.daemon = True
            self.image_thread.start()
            
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
        
        # Aguardar threads terminarem
        if self.image_thread:
            self.image_thread.join(timeout=1.0)
        
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
    
    def _receive_images(self):
        """Thread para receber imagens do simulador."""
        while self.running:
            try:
                # Receber pacote de imagem
                data, addr = self.image_socket.recvfrom(65536 * 8)  # Buffer grande para imagens
                
                if data:
                    # Atualizar endereço do simulador se necessário
                    if self.simulator_ip == "127.0.0.1" and addr[0] != "127.0.0.1":
                        self.simulator_ip = addr[0]
                        print(f"Simulador detectado em {self.simulator_ip}")
                    
                    # Decodificar imagem JPG
                    image_array = np.frombuffer(data, dtype=np.uint8)
                    
                    with self.image_lock:
                        self.latest_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        self.latest_image_time = time.time()
                        
                        # Chamar callback se existir
                        if self.image_callback and self.latest_image is not None:
                            self.image_callback(self.latest_image)
                    
                    self.last_received_time = time.time()
            
            except socket.timeout:
                # Timeout é normal, continuar
                pass
            except Exception as e:
                print(f"Erro ao receber imagem: {e}")
                time.sleep(0.1)
    
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
    
    def send_control(self, throttle, steering):
        """Envia comandos de controle para o simulador."""
        if not self.connected or self.control_socket is None:
            return False
        
        try:
            # Criar pacote de controle: [throttle(4), steering(4)]
            control_packet = bytearray(8)
            struct.pack_into('f', control_packet, 0, float(throttle))
            struct.pack_into('f', control_packet, 4, float(steering))
            
            # Enviar pacote
            self.control_socket.sendto(control_packet, (self.simulator_ip, self.CONTROL_PORT))
            return True
        except Exception as e:
            print(f"Erro ao enviar controle: {e}")
            return False
    
    def get_latest_image(self):
        """Retorna a imagem mais recente recebida do simulador."""
        with self.image_lock:
            if self.latest_image is not None:
                return self.latest_image.copy()
        return None
    
    def get_vehicle_data(self):
        """Retorna os dados mais recentes do veículo."""
        with self.data_lock:
            return self.vehicle_data.copy()
    
    def set_image_callback(self, callback):
        """Define um callback para processar imagens recebidas."""
        self.image_callback = callback
    
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
    
    def process_image(image):
        vehicle_data = client.get_vehicle_data()
        

        cv2.putText(image, f"Velocidade: {vehicle_data['speed']:.1f} km/h", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(image, f"Pos: ({vehicle_data['position']['x']:.1f}, {vehicle_data['position']['z']:.1f})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

        
        cv2.imshow("Simulador", image)
        cv2.waitKey(1)
    
    # Callback para processar dados do veículo
    def process_vehicle_data(data):
        pass
    
    client.set_image_callback(process_image)
    client.set_data_callback(process_vehicle_data)
    last_command_time = time.time()
    command_rate_limit = 0.1
    frames_processed = 0
    start_time = time.time()
    last_fps_update = start_time
    current_fps = 0
 
    try:
    
        if client.connect():
            print("Conectado ao simulador. Pressione Ctrl+C para sair.")
            

            throttle = 0.0
            steering = 0.0

            
            while client.running:
                key = cv2.waitKey(1) & 0xFF
     
                current_time = time.time()
                if current_time - last_fps_update >= 1.0:
                    current_fps = frames_processed / (current_time - last_fps_update)
                    frames_processed = 0
                    last_fps_update = current_time
                

                if key == 27:
                    break
                
                # Teclas WASD para controle básico
                if key == ord('w'):
                    throttle = min(throttle + 0.1, 1.0)
                elif key == ord('s'):
                    throttle = max(throttle - 0.1, -1.0)
                elif key == ord('a'):
                    steering = max(steering - 0.1, -1.0)
                elif key == ord('d'):
                    steering = min(steering + 0.1, 1.0)
                elif key == ord(' '):  # Espaço para frear
                    throttle = 0.0
                    steering = 0.0
                
                # Enviar comandos de controle
                if current_time - last_command_time >= command_rate_limit:
                    client.send_control(throttle, steering)
                    last_command_time = current_time
                
                # Pequena pausa para não sobrecarregar
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nEncerrando cliente...")
    finally:

        client.disconnect()
        cv2.destroyAllWindows()
    
    print("Cliente encerrado.")
