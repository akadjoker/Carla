import subprocess

def optimize_for_whatsapp(input_file, output_file):
    """Otimiza o vídeo para WhatsApp usando FFmpeg"""
    command = [
        'ffmpeg', '-i', input_file,
        '-c:v', 'libx264', '-crf', '28',  # Compressão de vídeo (23-28 é bom)
        '-preset', 'medium',  # Equilíbrio entre velocidade e compressão
        '-profile:v', 'baseline',  # Perfil mais compatível
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',  # Formato de pixel compatível 
        '-movflags', '+faststart',  # Otimiza para streaming
        '-r', '30',  # Frame rate
        '-vf', 'scale=720:-2',  # Redimensionar para 720p
        '-y',  # Sobrescrever sem perguntar
        output_file
    ]
    
    subprocess.run(command)
    print(f"Vídeo otimizado para WhatsApp: {output_file}")

 
optimize_for_whatsapp('model.mp4', 'video_whatsapp_otimizado.mp4')
