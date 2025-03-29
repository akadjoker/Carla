#!/usr/bin/env python3
import os
import subprocess
import glob
from datetime import datetime

# Configurações
pasta_imagens = "/home/djoker/Carla/Bin/dataset/session_20250322_114520/images"
video_saida = "video_final.mp4"
fps = 24

 
padrao = os.path.join(pasta_imagens, "frame_*.jpg")
arquivos = glob.glob(padrao)

if not arquivos:
    print(f"Nenhuma imagem encontrada no padrão {padrao}")
    exit(1)

 
def extrair_timestamp(nome_arquivo):
    # Extrair a parte do timestamp do nome do arquivo (frame_20250322_114526_580.jpg)
    partes = os.path.basename(nome_arquivo).split('_')
    if len(partes) >= 4:
        data = partes[1]
        hora = partes[2]
        milis = partes[3].split('.')[0]
        try:
 
            return datetime.strptime(f"{data}_{hora}_{milis}", "%Y%m%d_%H%M%S_%f")
        except ValueError:
  
            return nome_arquivo
    return nome_arquivo

 
arquivos_ordenados = sorted(arquivos, key=extrair_timestamp)

print(f"Encontrados {len(arquivos_ordenados)} arquivos de imagem")

 
lista_temp = "lista_imagens_temp.txt"
with open(lista_temp, "w") as f:
    for arquivo in arquivos_ordenados:
 
        f.write(f"file '{os.path.abspath(arquivo)}'\n")

 
comando = [
    "ffmpeg",
    "-r", str(fps),
    "-f", "concat",
    "-safe", "0",
    "-i", lista_temp,
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    video_saida
]

print("Executando FFmpeg para criar o vídeo...")
subprocess.run(comando)

 
os.remove(lista_temp)

print(f"Vídeo criado com sucesso: {video_saida}")
