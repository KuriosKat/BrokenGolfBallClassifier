import platform
import io
import os



import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.utils.data as data_utils

import torchvision
import torchvision.transforms as transforms

import pandas as pd
import numpy as np

import torch.nn.functional as F 
torch.manual_seed(1234) # 시드값 1234를 사용하여 재구현성 확보

import librosa
import librosa.display as display
import pdb

import sklearn
import sklearn.model_selection as ms

dataset=[]
'''---------------------------------------------------------------------'''

# feature_scaling
y_feature = 4096

use_gpu=torch.cuda.is_available() #CUDA 설정
print("GPU Available:{}".format(use_gpu)) #CUDA가 가능한가 (사용환경 -> GPUx 노트북이라 9750H 6core 12thr, 40epoch 기준 학습시간 최대 5분)

def readFile(filepath):
    y,sr=librosa.load(filepath)

    D=librosa.stft(y)

    D_real, D_imag = np.real(D), np.imag(D) #real:imag = 실수:허수
    #print(D_imag)
    # D_energy = np.real(D)
    a=D_real**2+D_imag**2
    #print(a)
    D_energy = np.sqrt(D_real**2+D_imag**2) # 루트값 씌워서 정렬

    '''i^2 = -1, if sure that..'''
    # a=D_real**2+D_imag**2
    # if a>=0:
    #     D_energy = np.sqrt(D_real**2+D_imag**2)
    # else:
    #     print(a)
    #     D_energy=0
    
    result=np.log(D_energy) #dB 단위는 log scale
    norm = librosa.util.normalize(D_energy) #평활화
    display.specshow(norm, y_axis='log', x_axis='time')

    # plt.plot(result)
    # plt.show()
    result=np.pad(norm,([(0,0),(0,y_feature-len(norm[0]))]),'constant', constant_values=0)
    return result


def import_data(folder,i):
    for file in os.listdir(folder):
        try:
            temp=[]
            f=folder+"/"+file
            temp.append(torch.tensor(readFile(f)))
            temp.append(torch.tensor(i))
            dataset.append(temp)
        except Exception:
            continue


"""Import Data, insert folder name fulled sound file like 'test1.wav'"""
########################
import_data("NORMAL_0525_5",0)
#print(numItr)
import_data("BROKEN_0525_5",1)
#print(numItr)
import_data("SILENT_0601",2)


"""dataset split stack"""
########################
train_set,test_set=ms.train_test_split(dataset, train_size=0.7) # Train:Test = 70:30(%)
#train_set, test_set = train_test_split(features, labels, test_size=0.33, random_state=42)
BATCH_SIZE=1 #attention to ur RAM memory size


"""dataset DEFINE"""
train_loader=torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True) #shuffle을 켜 섞는다 -> 일관성 배재 -> On is good
test_loader=torch.utils.data.DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

class GolfBallClassifier_ML(nn.Module):
    def __init__(self):
        super(GolfBallClassifier_ML,self).__init__()
        # self.cv1=nn.Conv2d(1, 16, 2, 1)
        # self.cv2=nn.Conv2d(16, 32, 2, 1)
        # #self.fc=nn.Linear(322875,4)
        # self.fc=nn.Linear(12800,10)
        self.fc1=nn.Linear(1025*y_feature,3)
        #self.fc2=nn.Linear(128,32)
        #self.fc3=nn.Linear(32,2)

        # self.fc2=nn.Linear(500,100)
        # self.fc3=nn.Linear(500,4)

    def forward(self,x):
        # out=F.relu(self.cv1(x))
        # out=F.relu(self.cv2(x))
        # out=out.view(BATCH_SIZE,-1)
        out = self.fc1(x)
        #out = F.relu(self.fc2(out))
        #out = self.fc3(out)
        # out=self.cv2(x)
        # out=F.relu(self.fc2(out))
        #out=F.relu(self.fc3(out))
        return out


class GolfBallClassifier_CNN(nn.Module):
    def __init__(self):
        super(GolfBallClassifier_CNN,self).__init__()
        # self.fc=nn.Linear(322875,4)
        # self.fc=nn.Linear(12800,10)
        self.fc1=nn.Linear(1025*y_feature,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(32,2)

        # self.fc2=nn.Linear(500,100)
        # self.fc3=nn.Linear(500,4)

    def forward(self,x):
        # out=out.view(BATCH_SIZE,-1)
        out = F.relu(self.fc1(x)) # activation_func
        out = F.relu(self.fc2(out)) # activation_func
        out = self.fc3(out)
        # out=self.cv2(x)
        # out=F.relu(self.fc2(out))
        #out=F.relu(self.fc3(out))
        return out

cnn = GolfBallClassifier_ML() #Neural net callback
# print(cnn) # just test print

'''For using cuda(nvidia) setting'''
if use_gpu:
    cnn.cuda()

cnn.eval() #do evalation callback func.
criterion = nn.CrossEntropyLoss() # because categorical
optimizer = optim.Adam(cnn.parameters(), lr=0.001) #Smaller learing_rate -> good train, but processing hard 

def train_baby(epoch,model,train_loader,optimizer):
    model.train()

    total_loss=0
    correct=0

    for i, (image,label) in enumerate(train_loader):

        optimizer.zero_grad()

        # image=image.view(-1,1025*3200)
        image = image.view(-1, 1025*y_feature)
        #image=image.float()
        #image.reshape(BATCH_SIZE,1025,315)
        #image=image.squeeze(0)
        #print(image.shape)
        
        if use_gpu:
            image=image.cuda()
            label=label.cuda()

        prediction=model(image)
        label=label.long()
        loss=criterion(prediction,label)

        loss.backward()

        optimizer.step()

        total_loss+=loss
        pred_classes = prediction.data.max(1,keepdim=True)[1]
        correct += pred_classes.eq(label.data.view_as(pred_classes)).sum().double()

    mean_loss=total_loss/len(train_loader.dataset)
    acc=correct/len(train_loader.dataset)

    print('Train Epoch: {}   Avg_Loss: {:.5f}   Acc: {}/{} ({:.3f}%)'.format(
        epoch, mean_loss, correct, len(train_loader.dataset),
        100. * acc)) # traing Epoch, Avg_loss, Acc

    return mean_loss, acc

def eval_baby(model,eval_loader):
    
    model.eval()

    total_loss=0
    correct=0

    for i, (image,label) in enumerate(eval_loader):

        optimizer.zero_grad()

        # image=image.view(-1,1025*3200)
        image = image.view(-1, 1025*y_feature)
        #image=image.float()
        #image.squeeze(0)
        

        if use_gpu:
            image=image.cuda()
            label=label.cuda()

        prediction=model(image)
        label=label.long()

        loss=criterion(prediction,label)

        loss.backward()

        optimizer.step()

        total_loss+=loss

        pred_classes=prediction.data.max(1,keepdim=True)[1]

        correct+=pred_classes.eq(label.data.view_as(pred_classes)).sum().double()

    mean_loss=total_loss/len(eval_loader.dataset)
    acc=correct/len(eval_loader.dataset)

    print('Eval:  Avg_Loss: {:.5f}   Acc: {}/{} ({:.3f}%)'.format(
        mean_loss, correct, len(eval_loader.dataset),
        100. *acc)) 

    return mean_loss, acc

def save_model(epoch, model, path='./'): #To save .pt file
    
    # file name and path 
    filename = path + 'realball_0526_try1_test_neural_network_3s_{}.pt'.format(epoch) #할때마다 주의할것
    
    # load the model parameters 
    torch.save(model.state_dict(), filename) #save file
    print(model.state_dict()) #print 
    
    
    return model

def load_model(epoch, model, path='./'):
    
    # file name and path 
    filename = path + 'realball_0526_try1_test_neural_network_3s_{}.pt'.format(epoch) #할때마다 주의할것
    
    # load the model parameters 
    model.load_state_dict(torch.load(filename))
    
    
    return model

# epoch define
numEpochs = 20

# feature_scaling
# y_feature = 8196

# checkpoint define
checkpoint_freq = 10

# path to save the data, 현재경로 저장용
path = './'

# empty lists 
train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

# traininng 
for epoch in range(1, numEpochs + 1):
    
    # train() function
    train_loss, train_acc = train_baby(epoch, cnn, train_loader, optimizer)
    
    # eval() function
    test_loss, test_acc = eval_baby(cnn, test_loader)    
    
    # append lists for plotting and printing 
    train_losses.append(train_loss)    
    test_losses.append(test_loss)
    
    train_accuracies.append(train_acc)    
    test_accuracies.append(test_acc)
    
    # Checkpoint
    if epoch % checkpoint_freq == 0:
        save_model(epoch, cnn, path)

# Last checkpoint
save_model(numEpochs, cnn, path)
    
print("\n\n\nOptimization ended.\n")  # checkpoint print out

classes = ('NORMAL', 'BROKEN', 'SILENT') #일범주 분류

test_net = GolfBallClassifier()
test_net.load_state_dict(torch.load('./realball_0526_try1_test_neural_network_3s_20.pt')) #.pt파일명에 주의


'''IF train finished, test with record file'''
valid_file_name = "NORMAL_111.wav" #임시 테스트 사운드파일
test_wav=readFile(valid_file_name)
outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*y_feature)) #16000*16 = 4000*64
_, predicted = torch.max(outputs, 1) #첫 1번이 곧 high한 conf를 가지는 값임, 곧 예측값임
print("Input_file_name: ", valid_file_name)
print("Conf. score: ", torch.max(torch.softmax(outputs, dim=1)).item())
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                              for j in range(1)))

valid_file_name = "NORMAL_222.wav" #임시 테스트 사운드파일
test_wav=readFile(valid_file_name)
outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*y_feature)) #16000*16 = 4000*64
_, predicted = torch.max(outputs, 1) #첫 1번이 곧 high한 conf를 가지는 값임, 곧 예측값임
print("Input_file_name: ", valid_file_name)
print("Conf. score: ", torch.max(torch.softmax(outputs, dim=1)).item())
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                              for j in range(1)))

valid_file_name = "NORMAL_333.wav" #임시 테스트 사운드파일
test_wav=readFile(valid_file_name)
outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*y_feature)) #16000*16 = 4000*64
_, predicted = torch.max(outputs, 1) #첫 1번이 곧 high한 conf를 가지는 값임, 곧 예측값임
print("Input_file_name: ", valid_file_name)
print("Conf. score: ", torch.max(torch.softmax(outputs, dim=1)).item())
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                              for j in range(1)))

valid_file_name = "BROKEN_111.wav" #임시 테스트 사운드파일
test_wav=readFile(valid_file_name)
outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*y_feature)) #16000*16 = 4000*64
_, predicted = torch.max(outputs, 1) #첫 1번이 곧 high한 conf를 가지는 값임, 곧 예측값임
print("Input_file_name: ", valid_file_name)
print("Conf. score: ", torch.max(torch.softmax(outputs, dim=1)).item())
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                              for j in range(1)))

valid_file_name = "BROKEN_222.wav" #임시 테스트 사운드파일
test_wav=readFile(valid_file_name)
outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*y_feature)) #16000*16 = 4000*64
_, predicted = torch.max(outputs, 1) #첫 1번이 곧 high한 conf를 가지는 값임, 곧 예측값임
print("Input_file_name: ", valid_file_name)
print("Conf. score: ", torch.max(torch.softmax(outputs, dim=1)).item())
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                              for j in range(1)))

valid_file_name = "BROKEN_333.wav" #임시 테스트 사운드파일
test_wav=readFile(valid_file_name)
outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*y_feature)) #16000*16 = 4000*64
_, predicted = torch.max(outputs, 1) #첫 1번이 곧 high한 conf를 가지는 값임, 곧 예측값임
print("Input_file_name: ", valid_file_name)
print("Conf. score: ", torch.max(torch.softmax(outputs, dim=1)).item())
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                              for j in range(1)))

valid_file_name = "SILENT_111.wav" #임시 테스트 사운드파일
test_wav=readFile(valid_file_name)
outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*y_feature)) #16000*16 = 4000*64
_, predicted = torch.max(outputs, 1) #첫 1번이 곧 high한 conf를 가지는 값임, 곧 예측값임
print("Input_file_name: ", valid_file_name)
print("Conf. score: ", torch.max(torch.softmax(outputs, dim=1)).item())
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                              for j in range(1)))

valid_file_name = "SILENT_222.wav" #임시 테스트 사운드파일
test_wav=readFile(valid_file_name)
outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*y_feature)) #16000*16 = 4000*64
_, predicted = torch.max(outputs, 1) #첫 1번이 곧 high한 conf를 가지는 값임, 곧 예측값임
print("Input_file_name: ", valid_file_name)
print("Conf. score: ", torch.max(torch.softmax(outputs, dim=1)).item())
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                              for j in range(1)))

valid_file_name = "SILENT_333.wav" #임시 테스트 사운드파일
test_wav=readFile(valid_file_name)
outputs = test_net(torch.tensor(test_wav).reshape(-1,1025*y_feature)) #16000*16 = 4000*64
_, predicted = torch.max(outputs, 1) #첫 1번이 곧 high한 conf를 가지는 값임, 곧 예측값임
print("Input_file_name: ", valid_file_name)
print("Conf. score: ", torch.max(torch.softmax(outputs, dim=1)).item())
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:1s}'
                              for j in range(1)))
                              


