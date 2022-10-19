from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import pygame

# Load the model
model = load_model('./models/keras_model.h5')

# labels.txt 파일 읽어오기
# load class_name
class_names=[]
with open("./models/labels.txt", "r") as f:
    print(f)
    for line in f:
        #print(line.strip())
        class_names.append(line)

#소리파일
pygame.mixer.init()
pygame.mixer.music.load('./audio/nomask-warning.mp3')
#clock = pygame.time.Clock()

#비디오 준비 함수
def videoReady():
    # 0번 카메라 객체화 
    video_cap = cv2.VideoCapture(0)

    # 카메라 on 체크
    if not video_cap.isOpened:
        print('카메라 에러')
        exit(0)

    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return video_cap

#비디오 표시, 마스크/노마스크 ML 체크 함수
def videoDisplayDetection(image):

    # 테스트 이미지를 저장할 변수의 타입과 shape 정의 
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Teachable Machine의 기본 이미지 사이즈 정의 및 리사이즈
    size = (224, 224)
    #이미지 1장을 리사이즈 하기
    #image = ImageOps.fit(image, size, Image.ANTIALIAS)
 
    

    # 동영상 frame 리사이즈 하기
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_AREA)
    # 이미지를 numpy array로 변경
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # 정규화한 이미지를 data 변수에 대입
    data[0] = normalized_image_array

    # run the inference(추론)
    return model.predict(data)


  
#카메라 준비 함수 실행
video_cap = videoReady()
#비디오 frame 읽기
while True:

    ret, frame = video_cap.read()
    #print(ret, frame)
    
    if frame is None:
        print("영상이 없음")
        video_cap.release()
        break;

    prediction = videoDisplayDetection(frame)

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    if index == 0: # index 0 : mask
        msg = 'm'
        print('send m')
        font_color = (255,0,0) #BGR
    elif index == 1: # index 1 : nomask
        msg = 'n'
        print('send n')
        #play sound 1
        if(pygame.mixer.music.get_busy() == False): #소리가 끝까지 출력되기 전에는 다음 소리를 출력하지 않음
            pygame.mixer.music.play()
        #경고화면
        font_color = (0, 0, 255)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    else:       # index 2
        msg = 'b'
        print('send b')
        font_color = (255,0,0) #BGR
        
    #글자 출력
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    
    #cv2.putText(img, text, 글자위치, 폰트, 폰트스케일, 폰트컬러)
    frame = cv2.putText(frame, class_name, (10,40), font, font_scale, font_color)
    # 영상 보이기
    cv2.imshow("video test", frame)
    
    #print("prediction : ", prediction)
    #print("np.argmax(prediction) : ", index)
    print("Class: ", class_name)
    #print("Confidence Score: ", confidence_score)


    # 아무키나 누를때까지 기다림
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

# 창 종료
video_cap.release()
cv2.destroyAllWindows()
