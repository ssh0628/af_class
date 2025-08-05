import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import os, random, math
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from config import opt
from DBprocessing import MyDataset, CropAverage, ToTensor
from VGG import vgg16_bn
from sklearn.metrics import accuracy_score

# Mean filtering function (이동 평균 필터)
def MeanFilter(ppg,M): # ppg(필터링 대상 1차원 numpy 배열), M(윈도우 길이)
    if M%2 ==0:
        M = M+1 # 평균 필터는 중심 값 기준(대칭이 필요), 길이를 항상 홀수로 유지
    Len = math.floor((M-1)/2) # 윈도우 절반 길이
    
    # 패딩
    x = np.zeros((1,Len+len(ppg)+Len)) # 신호를 필터링 하기 위해 좌우로 len 길이 만큼 패딩한 배열
    x = x[0,:]
    x[0:Len] = ppg[0:Len] # 앞 부분 패딩
    x[Len:Len+len(ppg)] = ppg # 본문
    x[Len+len(ppg):len(x)] = ppg[-Len:,] # 뒷 부분 패딩
    
    ppgs = ppg.astype(float) # 필터링된 결과를 저장할 배열
    
    #이동 평균 계산
    for i in range(Len,len(x)-Len): # 중심 위치 i 기준으로 양쪽 len 만큼 포함해 평균을 계산 (3이 아닌, Len이 맞음/하드코딩 일 가능성)
        ppgs[i-Len] = np.mean(x[i-Len:i+Len]) # 원래 신호에 대응하는 인덱스 i-Len에 결과 저장
    return ppgs

def main():
    # 체크 포인트 설정
    try:
        os.makedirs(opt.outf) # 모델 체크 포인트 저장 폴더
    except OSError:
        pass    # 이미 존재하면 패스
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)       # 사용자가 수동 시드를 지정하지 않으면, 랜덤 숫자를 시드로 선택(1~10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)                   # 파이썬의 기본 random 모듈과 파이토치의 랜덤 시드를 모두 고정해 결과 재현성을 높임
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)      # CUDA 사용 시, GPU 연산의 랜덤 시드도 동일하게 고정 / GPU 연산도 동일한 결과 재현 가능
    cudnn.benchmark = True                              # cuDNN 벤치마크 모드 활성화 / cuDNN 라이브러리에서 최적의 연산 알고리즘을 자동으로 설정
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
    # Loading training set
    train_dataset = MyDataset(dir=opt.dataroot,
                                    ext='train',
                                    transform=transforms.Compose([
                                        CropAverage(opt.recordLength),
                                        ToTensor()
                                    ]))
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                            shuffle=True,  num_workers=int(opt.workers))

    # Loading validation set
    valid_dataset = MyDataset(dir=opt.dataroot,
                                    ext='valid',
                                    transform=transforms.Compose([
                                        CropAverage(opt.recordLength),
                                        ToTensor()
                                    ]))
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batchSize,
                            shuffle=False,  num_workers=int(opt.workers))

    # Model
    opt.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    
    model = vgg16_bn(ngpu=opt.ngpu,num_classes=2).to(device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model is on GPU: {next(model.parameters()).is_cuda}")


    EpochAcc = np.empty((0, 2))         # epoch, train loss
    EpochValidAcc = np.empty((0, 2))    # epoch, validation loss
    ValidAcc = 0.0                      # 최고 validation accuracy

    # Loss function
    loss_fun = nn.CrossEntropyLoss().to(device)
        
    optimizer = optim.Adam(list(model.parameters()), lr = opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.004)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Model training
    for epoch in range(opt.niter):
        
        losses = []
        predictions = []
        labelsall = []
        
        model.train() # 학습 모드 활성화
        
        for i, train_data in enumerate(train_loader, 0):
            model.zero_grad() # 기울기 초기화 (매 학습 스텝마다 이전 기울기가 누적 되는걸 방지)
            
            inputs, labels = train_data['signal'], train_data['labels']
            
            inputs = inputs.to(device)
            labels = labels[:, 0].long().to(device)# 라벨을 1차원 정수형으로 변환
            # print(f"[DEBUG] Input device: {inputs.device}")
            # print(f"[DEBUG] Model device: {next(model.parameters()).device}")


                    
            output = model(inputs) # 모델이 input 전달 
            loss = loss_fun(output, labels) # 손실 계산

            loss.backward() # 역전파
            optimizer.step() # 가중치 갱신
            
            losses.append(loss.item()) # 손실 저장
            
            _, predicted = torch.max(output.data, 1) # 출력 중 확률이 가장 높은 클래스 인덱스
            predicted = predicted.cpu()
            labels = labels.cpu()
            
            predictions.extend(predicted.numpy()) # 예측 결과 리스트에 추가
            labelsall.extend(labels.numpy()) # 실제 라벨 리스트에 추가
            
        scheduler.step() # epoch가 끝난 후 학습률 조절
        
        # 학습 결과 출력 및 기록
        print('[%d/%d] Loss: %.4f Training Accuracy: %.4f' %(epoch, opt.niter, np.average(losses), 
                                                            accuracy_score(labelsall, predictions))) 
        EpochAcc = np.vstack((EpochAcc,np.array((epoch, np.average(losses))))) # EpochAcc 배열에 기록 (손실 기록용)
        
        
        # Model validation (5 epoch마다 수행)
        model.eval() # 평가 모드 활성화
        if (epoch+1) % 5 == 0:
            # validate
            predictions = []
            labelsall = []
            losses = []
            
            for i, valid_data in enumerate(valid_loader, 0):
                inputs, labels = valid_data['signal'], valid_data['labels']
                    
                inputs = inputs.to(device)
                labels = labels[:, 0].long().to(device)
                #print(f"[DEBUG] Input device: {inputs.device}")
                #print(f"[DEBUG] Model device: {next(model.parameters()).device}")

  
                output = model(inputs)
                loss = loss_fun(output, labels)
                
                losses.append(loss.item())  
                
                _,predicted = torch.max(output.data, 1)
                predicted = predicted.cpu()
                labels = labels.cpu()
                
                predictions.extend(predicted.numpy())
                labelsall.extend(labels.numpy())
            val_acc = accuracy_score(labelsall, predictions)
            val_loss = np.average(losses)
            print('[%d/%d] Validation Loss: %.4f Accuracy: %.4f' % (epoch, opt.niter, val_loss, val_acc))
            # print('[%d/%d] Validation Accuracy: %.4f' %(epoch, opt.niter, accuracy_score(labelsall, predictions)))
            EpochValidAcc = np.vstack((EpochValidAcc,np.array((epoch, np.average(losses)))))           
                
            if accuracy_score(labelsall, predictions) > ValidAcc:
                ValidAcc = accuracy_score(labelsall, predictions)
                labelsAll = labelsall
                predictionsAll = predictions
                torch.save(model, 'modelPPG.pkl')
                print("[Debug] save best model.")
                 
    np.save(f"{opt.outf}/train_loss.npy", EpochAcc)
    np.save(f"{opt.outf}/valid_acc.npy", EpochValidAcc)
    
    print("[INFO] Training logs saved.")

if __name__ == '__main__':
    main()