
## Gerekli kütüphanelerin import edilmesi
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import time
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


 # tanımlı duygular ve duyguların çizileceği kutuların renk değerleri
emotions = {
    0: ['sinirli', (0,0,255), (255,255,255)],
    1: ['igrenmis', (0,102,0), (255,255,255)],
    2: ['korkmus', (255,255,153), (0,51,51)],
    3: ['mutlu', (153,0,153), (255,255,255)],
    4: ['mutlu', (255,0,0), (255,255,255)],
    5: ['saskin', (0,255,0), (255,255,255)],
    6: ['dogal', (160,160,160), (255,255,255)]
}

 # modele tanımlanacak sınıf sayısı ve modele beslenecek görüntünün boyutu
num_classes = len(emotions)
input_shape = (48, 48, 1)

# daha önce eğitilmiş model ağırlıklarının yüklenmesi
weights_1 = r'C:\Users\ubeyt\OneDrive\Masaüstü\emotion_detection\facial_emotion_recognition\saved_models\vggnet.h5'
weights_2 = r'C:\Users\ubeyt\OneDrive\Masaüstü\emotion_detection\facial_emotion_recognition\saved_models\vggnet_up.h5'



# model mimarisinin model girdisine göre tanımlanması

class VGGNet(Sequential):
    def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
        super().__init__()
        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Flatten())
        
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(256, activation='relu'))

        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer=Adam(learning_rate=lr),
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])
        
        self.checkpoint_path = checkpoint_path



# model ağırlıklarının yüklenmesi

model_1 = VGGNet(input_shape, num_classes, weights_1)
model_1.load_weights(model_1.checkpoint_path)

model_2 = VGGNet(input_shape, num_classes, weights_2)
model_2.load_weights(model_2.checkpoint_path)



# ÇIKARIM

# ilk olarak yüz tespiti yapılır.
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)






# model boyutuna uygun olacak şekilde görüntüyü yeniden boyutlandırır.
def detection_preprocessing(image, h_max=360):
    h, w, _ = image.shape
    if h > h_max:
        ratio = h_max / h
        w_ = int(w * ratio)
        image = cv2.resize(image, (w_,h_max))
    return image

 # görüntüyü tensöre getirip modele uygun girişe ölçekler.
def resize_face(face):
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, (48,48))

 # yüz noktaları modele verilmek için tensöre çevrilir.
def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x

 # input olarak verilen görüntü matrisinden duygu tespiti ve koordinat tespiti yapılır.
def inference(image):
    H, W, _ = image.shape
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # yüz tespiti yapılır.
    results = face_detection.process(rgb_image)

    # görüntüde yüz varsa
    if results.detections:
        faces = []
        pos = []
        
        # her yüz için
        for detection in results.detections:

            # yüzün koordinatları alınır.
            box = detection.location_data.relative_bounding_box
            # mp_drawing.draw_detection(image, detection)

            x = int(box.xmin * W)
            y = int(box.ymin * H)
            w = int(box.width * W)
            h = int(box.height * H)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, W)
            y2 = min(y + h, H)

            face = image[y1:y2,x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces.append(face)
            pos.append((x1, y1, x2, y2))

        # tespit edilen yüzlerin duygusunu analiz etmek için duygu tespiti modeline girdi olarak verilir.
        x = recognition_preprocessing(faces)

        y_1 = model_1.predict(x)
        y_2 = model_2.predict(x)
        l = np.argmax(y_1+y_2, axis=1)

        #  duygu tespiti model çıktısına göre görüntüdeki yüzlerin üzerine yüzü çevreleyen dikdörtgen ve tespit edilen duygu yazılır.
        for i in range(len(faces)):
            cv2.rectangle(image, (pos[i][0],pos[i][1]),
                            (pos[i][2],pos[i][3]), emotions[l[i]][1], 2, lineType=cv2.LINE_AA)
            
            cv2.rectangle(image, (pos[i][0],pos[i][1]-20),
                            (pos[i][2]+20,pos[i][1]), emotions[l[i]][1], -1, lineType=cv2.LINE_AA)
            
            cv2.putText(image, f'{emotions[l[i]][0]} ({(y_1[i][l[i]]+y_2[i][l[i]])*100:.2f}%)', (pos[i][0],pos[i][1]-5),
                            0, 0.6, emotions[l[i]][2], 2, lineType=cv2.LINE_AA)

            
            
            
            
            
            
    return image


 # dosya yolu verilen tek görüntünün  duygu tespiti yapılır 
def infer_single_image(path):
    image = cv2.imread(path)
    image = detection_preprocessing(image)
    result = inference(image)
     # Yüz tespiti yap
    results = face_detection.process(image)
    
    # Yüz algılanmazsa veya birden fazla yüz algılanırsa
    if not results.detections :
        # Hata mesajını yaz
        cv2.putText(image, "Algilanamayan resim", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Hatalı resmi göster
        cv2.imshow("Hatali Resim", image)
        cv2.waitKey(0)
        return
    return result
    #cv2.imwrite('run/inference/out.jpg', result)


 # dosya yolu verilen birden fazla görüntünün  duygu tespiti yapılır 
def infer_multi_images(paths):
    for i, path in enumerate(paths):
        image = cv2.imread(path)
        image = detection_preprocessing(image)
        result = inference(image)
        cv2.imwrite('run/inference/out_'+str(i)+'.jpg', result)
    
    # Duygu tespiti yap
    image = detection_preprocessing(image)
    result = inference(image)
    cv2.imshow("Tahmin", result)
    cv2.waitKey(0)


# dosya yolu verilen görüntünün duygu tespiti yapılıp run/inference/out.jpg dosyasına kaydedilir.
path = r"C:\test_images\saskin2.jpg"
out = infer_single_image(path)
if __name__ == "__main__":
    cv2.imshow("prediction",out)
    cv2.waitKey(0)


