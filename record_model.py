import pyaudio
import wave
import json
import serial
from array import array
import librosa
import numpy as np
# import librosa.display as display
import torch
import torch.nn as nn

# 오디오 녹음 설정
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=44100 # 초당 44100개의 샘플링 캡쳐 
CHUNK=1024  # 한 번에 처리되는 데이터의 양
RECORD_SECONDS=3 # 녹음 시간
Position = 1

def readFile(filepath):
    # 음성 파일 읽음
    y,sr=librosa.load(filepath) 

    # 스펙트로그램 계산
    D=librosa.stft(y)
    D_real, D_imag = np.real(D), np.imag(D)
    D_energy = np.sqrt(D_real**2+D_imag**2)
    
    # 정규화
    norm = librosa.util.normalize(D_energy)
    
    # 패딩
    result=np.pad(norm,([(0,0),(0,8196-len(norm[0]))]),'constant')

    return result

class CNN_Ball(nn.Module):
    def __init__(self): # 클래스 초기화 함수
        super(CNN_Ball,self).__init__()
        self.fc1=nn.Linear(1025*8196,2) # 레이어의 입력 크기 1025*3200, 출력 크기 2
    def forward(self,x): # 데이터를 신경망에 통과시킴
        out=self.fc1(x) 
        return out # 입력데이터 x를 fc1 레이어에 통과시킨 결과를 반환

# 아두이노로부터 젯슨나노가 플래그를 받는 코드
connection = serial.Serial(port="/dev/ttyACM0", baudrate=115200) # 아두이노 port, 통신속도 115200
connection.reset_input_buffer() # 입력 버퍼름 비움
while True: 
    data = connection.readline().decode("utf-8") # 시리얼 포트로부터 데이터를 읽어옴
    try:
        dict_json = json.loads(data) # 데이터를 JSON 형식으로 파싱
        if 'STEP' in dict_json and dict_json['STEP'] == 1: # 데이터 안에 'STEP' 키가 존재하고 값이 1인 경우에 실행
            audio=pyaudio.PyAudio() #PyAudio 인스턴스 생성
            stream=audio.open(format=FORMAT,channels=CHANNELS, 
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK) # 오디오 스트림을 열어서 설정된 값을 매개변수로 넣음
            # 녹음 시작
            print("start recording")
            frames=[] # 음성 데이터를 저장할 frames 리스트 초기화
            for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)): # 설정한 시간동안 반복하여 데이터를 읽어옴
                data=stream.read(CHUNK)
                data_chunk=array('h',data)
                vol=max(data_chunk)
                frames.append(data) # frames 리스트에 append 
            print("finish recording")
            # 녹음 중지
            stream.stop_stream() # 스트림 중지
            stream.close() # 스트림 close
            audio.terminate() # PyAudio 인스턴스 종료
            # 녹음된 데이터 파일을 wav 파일로 저장
            wavfile=wave.open("NORMAL_{}.wav".format(Position),'wb')
            wavfile.setnchannels(CHANNELS)
            wavfile.setsampwidth(audio.get_sample_size(FORMAT))
            wavfile.setframerate(RATE)
            wavfile.writeframes(b''.join(frames))#append frames recorded to file
            wavfile.close()
            
            # 모델 연산 코드 
            cnn = CNN_Ball()

            classes = ('NORMAL', 'BROKEN')

            test_net = CNN_Ball()
            test_net.load_state_dict(torch.load('./test_neural_network_20.pt')) # 사전에 학습된 가중치 파일을 불러옴
            test_file_name = "NORMAL_{}.wav".format(Position) # 테스트할 오디오 파일명 설정
            test_wav=readFile(test_file_name) # 오디오 파일을 읽고 전처리
            outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*8196)) # 이미지 사이즈는 1025 * 8196
            _, predicted = torch.max(outputs, 1) # 모델에 데이터를 입력하고 예측 결과를 얻음, 가장 높은 확률을 가진 클래스를 예측 결과로 선택
            
            # 테스트된 파일의 정보 출력(이름, 확률값, 예측값)
            print('filename:', test_file_name)
            print('Conf. score: ', torch.max(torch.softmax(outputs, dim=1)).item())
            print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                                        for j in range(1)))
            
            flag_string = ' '.join(f'{classes[predicted[j]]:1s}' for j in range(1))  # 예측된 클래스를 문자열로 변환하여 flag_string에 저장
             
            accuracy = torch.max(torch.softmax(outputs, dim=1)).item()  # 예측 결과의 정확도 값을 accuracy에 저장
            
            if flag_string in ['NORMAL']: # 예측된 클래스 결과값이 NORMAL이라면 flag_value = 1
                flag_value = 1
            elif flag_string in ['BROKEN']: # 예측된 클래스 결과값이 BROKEN이라면 flag_value = 2
                flag_value = 2

            if flag_value == 1: # normal ball
                doc = {"act": flag_value, "accuracy": round(accuracy, 4)} # act 키에 flag_value, accuracy 키에 확률값을 포함하는 딕셔너리 생성
                
            elif flag_value == 2: # broken ball
                doc = {"act": flag_value, "accuracy": round(accuracy, 4)}
            Position+=1 # 음성 파일의 이름을 구분하기 위한 position 변수 +1
            
        connection.write(json.dumps(doc).encode('utf-8')) # 딕셔너리를 JSON 형식으로 변환하여 시리얼 포트로 전송(젯슨나노 -> 아두이노)
        # JSON 디코딩 에러가 발생한 경우 에러메시지 출력
    except json.JSONDecodeError as e: 
        print("JSON:", e)
        
    