import os
import pandas as pd
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import tensorflow as tf
import random

# Desativar GPU se necessário
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def find_all_sessions(base_dir="dataset"):
    """
    Localiza todas as sessões válidas no diretório de dataset
    """
    session_dirs = glob.glob(os.path.join(base_dir, "session_*"))
    valid_sessions = []
    for session_dir in session_dirs:
        csv_path = os.path.join(session_dir, "steering_data.csv")
        images_dir = os.path.join(session_dir, "images")
        if os.path.exists(csv_path) and os.path.isdir(images_dir):
            valid_sessions.append(session_dir)
    print(f"Encontradas {len(valid_sessions)} sessões válidas com datasets:")
    for session in valid_sessions:
        print(f" - {os.path.basename(session)}")
    return valid_sessions

def load_all_sessions_data(session_dirs):
    """
    Carrega dados de todas as sessões válidas
    """
    combined_data = pd.DataFrame()
    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        csv_path = os.path.join(session_dir, "steering_data.csv")
        try:
            session_data = pd.read_csv(csv_path)
            required_cols = ['image_path', 'steering']
            if not all(col in session_data.columns for col in required_cols):
                print(f"Aviso: Sessão {session_name} não tem as colunas necessárias, saltamos...")
                continue
                
            session_data['session'] = session_name
            
            # Corrigindo a combinação dos caminhos
            session_data['full_image_path'] = session_data['image_path'].apply(
                lambda x: os.path.join(session_dir, 'images', os.path.basename(x))
            )
            
            # Verificando se os arquivos existem
            valid_rows = []
            for idx, row in session_data.iterrows():
                if os.path.exists(row['full_image_path']):
                    valid_rows.append(True)
                else:
                    print(f"Imagem não encontrada: {row['full_image_path']}")
                    valid_rows.append(False)
                    
            session_data = session_data[valid_rows]
            combined_data = pd.concat([combined_data, session_data], ignore_index=True)
            print(f"Sessão {session_name}: {len(session_data)} amostras válidas")
        except Exception as e:
            print(f"Erro ao carregar sessão {session_name}: {e}")
    
    print(f"Total de amostras combinadas: {len(combined_data)}")
    return combined_data

def enhancedPreProcess(img):
    """
    Pré-processamento aprimorado das imagens com correção para diferentes tipos de dados
    """
    # Verificar e converter o tipo de dados da imagem se necessário
    if img.dtype == np.float64:  # CV_64F
        # Converter para uint8 (escala 0-255) se for float64
        # Primeiro verificamos se os valores estão entre 0-1 ou 0-255
        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    elif img.dtype != np.uint8:
        # Converter outros tipos para uint8 também
        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Agora continua com o processamento normal
    # Recorte mais preciso para focar na estrada
    img = img[90:440, :, :]
    
    # Conversão para YUV (agora com certeza em formato uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Equalização adaptativa de histograma no canal Y
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[:,:,0] = clahe.apply(img[:,:,0])
    
    # Desfoque Gaussiano
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Redimensionamento
    img = cv2.resize(img, (200, 66))
    
    # Normalização final para o intervalo 0-1
    img = img/255.0
    
    return img

def curveAwareAugmentation(imgPath, steering):
    """
    Função de aumento de dados com maior ênfase em curvas acentuadas
    Com correção para tipos de dados de imagem
    """
    img = mpimg.imread(imgPath)
    
    # Criar uma cópia modificável da imagem
    img = np.array(img, dtype=np.float32).copy()
    
    # Verificar e converter o tipo de dados se necessário
    # Matplotlib geralmente carrega como float64 (0-1), então vamos padronizar
    original_dtype = img.dtype
    
    # Probabilidade de aumento baseada no ângulo de direção
    aug_probability = min(0.9, 0.5 + abs(steering) * 0.5)
    
    # Pan (deslocamento) com maior probabilidade em curvas
    if np.random.rand() < aug_probability:
        # Deslocamento horizontal mais intenso em curvas
        pan_x = np.random.uniform(-0.1, 0.1) * (1 + abs(steering))
        pan = iaa.Affine(translate_percent={"x": pan_x, "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    
    # Zoom com foco na parte da curva
    if np.random.rand() < aug_probability:
        zoom_factor = np.random.uniform(1.0, 1.2 + abs(steering) * 0.1)
        zoom = iaa.Affine(scale=zoom_factor)
        img = zoom.augment_image(img)
    
    # Ajuste de brilho para simular condições variadas de luz
    if np.random.rand() < 0.6:
        brightness = iaa.Multiply(np.random.uniform(0.4, 1.3))
        img = brightness.augment_image(img)
    
    # Adição de sombras aleatórias para melhorar robustez
    if np.random.rand() < 0.3:
        # Converter para uint8 se for float64 para operações com OpenCV
        if img.dtype == np.float64:
            if np.max(img) <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)
        else:
            img_uint8 = img
        
        # Cria uma máscara de sombra aleatória
        h, w = img_uint8.shape[0], img_uint8.shape[1]
        top_x, top_y = int(w * np.random.uniform()), 0
        bot_x, bot_y = int(w * np.random.uniform()), h
        shadow_mask = np.zeros_like(img_uint8[:,:,0])
        cv2.fillPoly(shadow_mask, np.array([[(top_x, top_y), (bot_x, bot_y), (0, h), (0, 0)]]), 1)
        
        # Aplica a sombra com intensidade variável
        shadow_factor = np.random.uniform(0.6, 0.9)
        
        # Converter de volta para o tipo original após a operação
        if original_dtype == np.float64 and np.max(img) <= 1.0:
            # Se a imagem estava em float64 com valores 0-1
            for c in range(3):
                img[:,:,c] = img[:,:,c] * (1 - shadow_mask * (1 - shadow_factor))
        else:
            for c in range(3):
                img_uint8[:,:,c] = img_uint8[:,:,c] * (1 - shadow_mask * (1 - shadow_factor))
            if original_dtype == np.float64:
                img = img_uint8 / 255.0
            else:
                img = img_uint8
    
    # Espelhamento (flip horizontal) para balancear curvas esquerda/direita
    if np.random.rand() < 0.5:
        if img.dtype == np.float64 and np.max(img) <= 1.0:
            # Converter para uint8 para operações com OpenCV
            img_uint8 = (img * 255).astype(np.uint8)
            img_uint8 = cv2.flip(img_uint8, 1)
            # Converter de volta para float64
            img = img_uint8 / 255.0
        else:
            img = cv2.flip(img, 1)
        steering = -steering
    
    # Pequena adição de ruído para aumentar robustez
    if np.random.rand() < 0.2:
        noise = np.random.normal(0, 0.01, img.shape)
        
        if original_dtype == np.float64 and np.max(img) <= 1.0:
            # Se a imagem estava em float64 com valores 0-1
            img = np.clip(img + noise, 0, 1)
        else:
            # Se a imagem estava em outro formato
            if original_dtype == np.uint8:
                img = np.clip(img + noise * 255, 0, 255).astype(np.uint8)
            else:
                img = np.clip(img + noise * np.max(img), 0, np.max(img)).astype(original_dtype)
    
    return img, steering

def advancedCurveBalancing(data, display=False):
    """
    Balanceamento de dados avançado com foco em preservar mais amostras de curvas
    
    Este algoritmo mantém mais amostras nas regiões de curvas difíceis e
    reduz mais agressivamente amostras de linha reta, que são super-representadas.
    """
    nBin = 31  # Número de bins para distribuição
    
    # Calculando histograma inicial
    hist, bins = np.histogram(data['steering'], nBin)
    
    # Visualização antes do balanceamento
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.figure(figsize=(10, 6))
        plt.bar(center, hist, width=0.03)
        plt.title('Distribuição Original dos Ângulos de Direção')
        plt.xlabel('Ângulo de Direção')
        plt.ylabel('Número de Amostras')
        plt.show()
    
    # Definição adaptativa de amostras por bin
    # Bins próximos a zero (linha reta) recebem menos amostras que bins de curvas
    samplesPerBin = []
    for i in range(nBin):
        bin_center = (bins[i] + bins[i+1]) / 2
        # Função de ponderação que dá mais peso a curvas (ângulos maiores)
        # Equação: base + adicional com base no valor absoluto do ângulo
        weight = 1.0 + 2.0 * min(1.0, abs(bin_center) * 3)
        # Aplicamos esse peso à contagem base
        samplesPerBin.append(int(200 * weight))
    
    # Limitamos para não exceder o número original de amostras
    for i in range(nBin):
        if samplesPerBin[i] > hist[i]:
            samplesPerBin[i] = hist[i]
    
    # Remoção de amostras em excesso
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        
        # Número de amostras a manter para este bin
        samplesToKeep = samplesPerBin[j]
        
        # Se temos mais amostras que o desejado
        if len(binDataList) > samplesToKeep:
            binDataList = shuffle(binDataList)
            # Mantemos apenas as amostras desejadas
            binDataList = binDataList[samplesToKeep:]
            removeindexList.extend(binDataList)
    
    print('Amostras Removidas:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Amostras Restantes:', len(data))
    
    # Visualização após o balanceamento
    if display:
        hist, _ = np.histogram(data['steering'], nBin)
        plt.figure(figsize=(10, 6))
        plt.bar(center, hist, width=0.03)
        
        # Plotando a linha de referência que mostra o número desejado de amostras por bin
        for i in range(len(center)):
            plt.plot([center[i], center[i]], [0, samplesPerBin[i]], 'r-', linewidth=1)
        
        plt.title('Distribuição Balanceada dos Ângulos de Direção')
        plt.xlabel('Ângulo de Direção')
        plt.ylabel('Número de Amostras')
        plt.show()
    
    return data

def createImprovedModel():
    """
    Cria um modelo CNN melhorado para previsão de ângulo de direção
    """
    model = Sequential()
    model.add(Input(shape=(66, 200, 3)))
    
    # Bloco 1 - Convolução inicial com mais filtros
    model.add(Convolution2D(32, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(96, (3, 3), activation='elu'))
    model.add(Convolution2D(96, (3, 3), activation='elu'))
    
    # Adição de dropout para evitar overfitting
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    # FCLs com dropout
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1))
    
    # Otimizador com taxa de aprendizado adaptável
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model

def enhancedDataGen(imagesPath, steeringList, batchSize, trainFlag):
    """
    Gerador de dados aprimorado com tratamento de tipo de dados de imagem
    """
    while True:
        imgBatch = []
        steeringBatch = []
        
        # Quantidade de ângulos de curva a incluir no batch
        minCurveSamples = int(batchSize * 0.4)
        curveCount = 0
        
        # Lista de índices para acompanhar quais amostras foram selecionadas
        selectedIndices = []
        
        # Primeiro, selecionamos amostras de curvas acentuadas
        while curveCount < minCurveSamples and len(selectedIndices) < batchSize:
            index = random.randint(0, len(imagesPath) - 1)
            
            # Evitar usar o mesmo índice duas vezes
            if index in selectedIndices:
                continue
                
            # Verificamos se é uma curva significativa
            if abs(steeringList[index]) > 0.1:
                selectedIndices.append(index)
                curveCount += 1
        
        # Completamos o batch com amostras aleatórias
        while len(selectedIndices) < batchSize:
            index = random.randint(0, len(imagesPath) - 1)
            if index not in selectedIndices:
                selectedIndices.append(index)
        
        # Agora processamos as amostras selecionadas
        for index in selectedIndices:
            try:
                if trainFlag:
                    # Use o aumento de dados avançado para treino
                    img, steering = curveAwareAugmentation(imagesPath[index], steeringList[index])
                else:
                    # Para validação, apenas carregamos a imagem
                    img = mpimg.imread(imagesPath[index])
                    steering = steeringList[index]
                
                # Pré-processamento aprimorado para todas as imagens
                # Com tratamento de exceções para depuração
                img = enhancedPreProcess(img)
                
                imgBatch.append(img)
                steeringBatch.append(steering)
            except Exception as e:
                print(f"Erro ao processar imagem {imagesPath[index]}: {e}")
                # Em caso de erro, substituir por uma amostra diferente
                continue
        
        # Verificar se o batch tem amostras suficientes
        if len(imgBatch) < batchSize:
            # Se tiverem ocorrido muitos erros, preencher com duplicatas das amostras bem-sucedidas
            while len(imgBatch) < batchSize:
                idx = random.randint(0, len(imgBatch) - 1)
                imgBatch.append(imgBatch[idx])
                steeringBatch.append(steeringBatch[idx])
        
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))

def loadData(path, data):
    """
    Carrega caminhos de imagens e ângulos de direção correspondentes
    """
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        full = path + os.path.sep + indexed_data.iloc[2] + os.path.sep + indexed_data.iloc[0]
        imagesPath.append(full)
        steering.append(float(indexed_data.iloc[1]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

def optimizedTraining(xTrain, yTrain, xVal, yVal):
    """
    Configuração de treinamento otimizada para melhor desempenho em curvas
    """
    # Criação do modelo aprimorado
    model = createImprovedModel()
    
    # Configuração de callbacks mais robusta
    callbacks = [
        # Monitorar e salvar o melhor modelo
        tf.keras.callbacks.ModelCheckpoint(
            'best_steering_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Parar se não houver melhoria por um número de épocas
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,  # Aumentado para permitir mais tempo de aprendizado
            restore_best_weights=True,
            verbose=1
        ),
        # Redução adaptativa da taxa de aprendizado
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Log para TensorBoard para monitoramento avançado
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Configurações de treinamento otimizadas
    # - Mais épocas para permitir convergência
    # - Mais passos por época para melhor exposição aos dados
    # - Batch size balanceado para estabilidade/velocidade
    history = model.fit(
        enhancedDataGen(xTrain, yTrain, 64, True),  # Batch size reduzido para 64
        steps_per_epoch=len(xTrain) // 64,  # Garantir que todos os dados sejam usados
        epochs=30,  # Mais épocas, o early stopping vai interromper se necessário
        validation_data=enhancedDataGen(xVal, yVal, 64, False),
        validation_steps=len(xVal) // 64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Salvar o modelo final
    model.save('curve_optimized_model.keras')
    
    # Visualização mais detalhada do histórico de treinamento
    plt.figure(figsize=(12, 8))
    
    # Plot da perda (loss)
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Curva de Aprendizado do Modelo')
    plt.ylabel('Erro Médio Quadrático (MSE)')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True)
    
    # Plot da taxa de aprendizado se disponível
    if 'lr' in history.history:
        plt.subplot(2, 1, 2)
        plt.plot(history.history['lr'])
        plt.title('Taxa de Aprendizado Dinâmica')
        plt.ylabel('Taxa de Aprendizado')
        plt.xlabel('Época')
        plt.yscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_detailed.png')
    plt.show()
    
    return model, history

def evaluateModelPerformance(model, xVal, yVal):
    """
    Avalia o desempenho do modelo em diferentes categorias de curvas
    """
    # Gerar predições para o conjunto de validação
    val_generator = enhancedDataGen(xVal, yVal, 32, False)
    val_steps = len(xVal) // 32
    
    # Coletar todas as predições e valores reais
    y_true = []
    y_pred = []
    
    for _ in range(val_steps):
        x_batch, y_batch = next(val_generator)
        batch_pred = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch)
        y_pred.extend(batch_pred.flatten())
    
    # Converter para arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcular MSE global
    mse_global = np.mean((y_true - y_pred) ** 2)
    print(f"MSE Global: {mse_global:.4f}")
    
    # Definir categorias de curvas
    categories = [
        ("Linha reta", lambda x: abs(x) < 0.05),
        ("Curva suave", lambda x: 0.05 <= abs(x) < 0.15),
        ("Curva média", lambda x: 0.15 <= abs(x) < 0.25),
        ("Curva acentuada", lambda x: abs(x) >= 0.25)
    ]
    
    # Avaliar desempenho por categoria
    for name, condition in categories:
        mask = condition(y_true)
        if np.sum(mask) > 0:
            category_true = y_true[mask]
            category_pred = y_pred[mask]
            mse = np.mean((category_true - category_pred) ** 2)
            print(f"MSE para {name} ({np.sum(mask)} amostras): {mse:.4f}")
    
    # Visualizar predições vs valores reais
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('Ângulo Real')
    plt.ylabel('Ângulo Previsto')
    plt.title('Comparação entre Ângulos Reais e Previstos')
    plt.grid(True)
    plt.savefig('prediction_vs_true.png')
    plt.show()

def main():
    print("Iniciando processo de treinamento otimizado para curvas...")
    
    # Configuração para debug de erros
    np.seterr(all='print')  # Para capturar avisos de numpy
    
    # Carregamento de dados
    sessions = find_all_sessions()
    if not sessions:
        print("Nenhuma sessão válida encontrada. Verifique o diretório dataset.")
        return
    
    # Carregamento e combinação de dados
    combined_data = load_all_sessions_data(sessions)
    if len(combined_data) < 10:
        print("Dados insuficientes para treinar. É necessário pelo menos 10 amostras.")
        return
    
    # Teste inicial de processamento de imagem para verificar compatibilidade
    print("Testando processamento de imagem com uma amostra...")
    test_img_path = combined_data.iloc[0]['full_image_path']
    try:
        test_img = mpimg.imread(test_img_path)
        print(f"Formato da imagem lida: {test_img.shape}, tipo: {test_img.dtype}, min: {np.min(test_img)}, max: {np.max(test_img)}")
        
        processed_img = enhancedPreProcess(test_img)
        print(f"Pré-processamento bem-sucedido. Formato: {processed_img.shape}, tipo: {processed_img.dtype}")
        
        # Visualizar a imagem processada para verificação
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img)
        plt.title('Imagem Original')
        
        plt.subplot(1, 2, 2)
        # Converter de YUV para RGB para visualização
        processed_img_rgb = cv2.cvtColor((processed_img*255).astype(np.uint8), cv2.COLOR_YUV2RGB)
        plt.imshow(processed_img_rgb)
        plt.title('Imagem Processada')
        
        plt.savefig('preprocessing_test.png')
        plt.show()
    except Exception as e:
        print(f"Erro no teste de processamento de imagem: {e}")
        print("Verificando problemas de tipo de dados e continuando...")
    
    # Visualização da distribuição original dos dados
    plt.figure(figsize=(10, 6))
    plt.hist(combined_data['steering'], bins=31)
    plt.title('Distribuição Original dos Ângulos de Direção')
    plt.xlabel('Ângulo de Direção')
    plt.ylabel('Frequência')
    plt.savefig('original_distribution.png')
    plt.show()
    
    # Balanceamento avançado focado em curvas
    print("Aplicando balanceamento otimizado para curvas...")
    balanced_data = advancedCurveBalancing(combined_data, display=True)
    
    # Carregamento dos dados balanceados
    print("Carregando dados processados...")
    imagesPath, steerings = loadData("./dataset", balanced_data)
    
    # Divisão treinamento/validação estratificada para garantir representação de curvas
    print("Dividindo dados em conjuntos de treino e validação...")
    
    # Usamos stratify para garantir que tanto o conjunto de treino quanto o de validação
    # tenham distribuições similares de ângulos de direção
    bins = np.linspace(-1, 1, 11)  # 10 bins para estratificação
    binned_steering = np.digitize(steerings, bins)
    
    xTrain, xVal, yTrain, yVal = train_test_split(
        imagesPath, 
        steerings, 
        test_size=0.2,
        random_state=42,
        stratify=binned_steering  # Estratificação para manter distribuição
    )
    
    print('Total de Imagens de Treino:', len(xTrain))
    print('Total de Imagens de Validação:', len(xVal))
    
    # Verificação da distribuição nos conjuntos de treino e validação
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(yTrain, bins=31)
    plt.title('Distribuição no Conjunto de Treino')
    plt.xlabel('Ângulo de Direção')
    
    plt.subplot(1, 2, 2)
    plt.hist(yVal, bins=31)
    plt.title('Distribuição no Conjunto de Validação')
    plt.xlabel('Ângulo de Direção')
    
    plt.tight_layout()
    plt.savefig('train_val_distribution.png')
    plt.show()
    
    # Treinamento otimizado
    print("Iniciando treinamento otimizado para curvas...")
    model, history = optimizedTraining(xTrain, yTrain, xVal, yVal)
    
    # Avaliação do modelo em diferentes faixas de ângulos
    print("\nAvaliação do modelo em diferentes categorias de curvas:")
    evaluateModelPerformance(model, xVal, yVal)
    
    print("\nTreinamento concluído! Modelo salvo como 'curve_optimized_model.keras'")
    


if __name__ == "__main__":
    main()
