import os
import pandas as pd
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from functools import partial
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense,Dropout,MaxPooling2D,BatchNormalization,LeakyReLU,Conv2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

import random

 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def find_all_sessions(base_dir="dataset"):
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

def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering



# O primeiro valor (84) representa o início do recorte no eixo vertical (altura).
# O segundo valor (440) representa o fim do recorte no eixo vertical (altura).
# Os dois pontos (:) nos outros eixos significam que todas as colunas (:) e todos os canais de cor (:) serão mantidos.
def preProcess(img):
    """
    Preprocesses the input image for further analysis.

    This function performs the following steps:
    1. Crops the image to remove unnecessary parts, focusing on the region of interest.
    2. Converts the color space from RGB to YUV, which is more suitable for machine learning tasks.
    3. Applies Gaussian blur to reduce noise and detail in the image.
    4. Resizes the image to a standard size of 200x66 pixels.
    5. Normalizes the pixel values to a range of 0 to 1 for better model performance.
    """
    img = img[130:480,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def balanceData(data,display=False):
    nBin = 31
    samplesPerBin =  300
    hist, bins = np.histogram(data['steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    if display:
        hist, _ = np.histogram(data['steering'], (nBin))
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    return data


def createModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=0.0001),loss='mse')
    model.summary()
    return model

def dataGen(imagesPath, steeringList, batchSize, trainFlag):
    
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))




def loadData(path, data):
  imagesPath = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    full =path + os.path.sep + indexed_data.iloc[2] + os.path.sep + indexed_data.iloc[0]
    imagesPath.append( full)
    steering.append(float(indexed_data.iloc[1]))
  imagesPath = np.asarray(imagesPath)
  steering = np.asarray(steering)
  return imagesPath, steering

def main():
    sessions = find_all_sessions()
    if not sessions:
        print("Nenhuma sessão válida encontrada. Verifica o diretório dataset.")
        return
    
    combined_data = load_all_sessions_data(sessions)
    
    if len(combined_data) < 10:
        print("Dados insuficientes para treinar. É necessário pelo menos 10 amostras.")
        return
    
    data = balanceData(combined_data)
    imagesPath, steerings = loadData("./dataset",data)

    
   
    xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.25,random_state=54)
    print('Total Training Images: ',len(xTrain))
    print('Total Validation Images: ',len(xVal))

    # #/home/djoker/Carla/Bin/dataset/session_20250322_114520/images/frame_20250322_114727_393.jpg
    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation="relu", padding="SAME")

    model = Sequential(
        [
            DefaultConv2D(
                filters=64, kernel_size=7, input_shape=[66, 200, 3]
                
            ),  # was 28, 28, 1
            MaxPooling2D(pool_size=2),
            DefaultConv2D(filters=128),
            DefaultConv2D(filters=128),
            MaxPooling2D(pool_size=2),
            DefaultConv2D(filters=256),
            DefaultConv2D(filters=256),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(units=128, activation="relu"),
            Dropout(0.2),  # lower less regularization
            Dense(units=64, activation="relu"),
            Dropout(0.2),
            Dense(units=1, activation="softmax"),
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    tensorboard_cb = tf.keras.callbacks.TensorBoard(histogram_freq=1)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=False)
    history = model.fit(dataGen(xTrain, yTrain,100, 1),
                                    epochs=30,
                                    validation_data=dataGen(xVal, yVal, 50, 0),
                                    callbacks=[tensorboard_cb, early_stopping_cb])
    

        

    #mse_test = model.evaluate(yTrain, yVal)
    #print(f"Test Data - MSE: {mse_test}")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.savefig('training_history.png') 
    plt.show()
    model.save('model.keras')

    
    #train_data, val_data = train_test_split(combined_data, test_size=0.2, random_state=42)


    
    # imgRe,st = augmentImage('/home/djoker/Carla/Bin/dataset/session_20250322_114520/images/frame_20250322_114526_580.jpg',0)
    # mpimg.imsave('Result.jpg',imgRe)
    # plt.imshow(imgRe)
    # plt.show()
    
    
    # imgRe = preProcess(mpimg.imread('/home/djoker/Carla/Bin/dataset/session_20250322_114520/images/frame_20250322_114725_182.jpg'))    
    # mpimg.imsave('Result.jpg',imgRe)
    # plt.imshow(imgRe)
    # plt.show()

if __name__ == "__main__":
     main()