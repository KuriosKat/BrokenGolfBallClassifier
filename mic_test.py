import pyaudio
# import numpy as np
audio = pyaudio.PyAudio()

import pyaudio # 침묵이 있는지 여부를 확인하는 간단한 스크립트
import wave
from array import array

FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=44100 
CHUNK=1024  # 음성데이터를 불러올 때 한번에 몇개의 정수를 불러올지
RECORD_SECONDS=2
FILE_NAME="SILENT_test.wav"

audio=pyaudio.PyAudio() #instantiate the pyaudio

#recording prerequisites
stream=audio.open(format=FORMAT,channels=CHANNELS,input_device_index=1,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)

#starting recording
print("Start to record the audio.")

frames=[]

for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
    data=stream.read(CHUNK)
    data_chunk=array('h',data)
    vol=max(data_chunk)
    frames.append(data)

print("Recording is finished.")
#end of recording
stream.stop_stream()
stream.close()
audio.terminate()
#writing to file
wavfile=wave.open(FILE_NAME,'wb')
wavfile.setnchannels(CHANNELS)
wavfile.setsampwidth(audio.get_sample_size(FORMAT))
wavfile.setframerate(RATE)
wavfile.writeframes(b''.join(frames))#append frames recorded to file
wavfile.close()
