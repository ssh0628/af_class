import os, torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

class MyDataset(Dataset): # Pytorch의 Dataset을 상속 받은 커스텀 class
    
    def __init__(self, dir, ext='train', transform=None):
        """
        Args:
            dir (string): Directory  # 실제로는 사용X, 하드 코딩된 경로를 사용
            ext (string): the extension used to claim training or testing data # 'train', 'test' 같은 접미사
            transform 전처리 함수 묶음(CropAverage, ToTensor 등)
        """
        
        labels = np.zeros((1,1)) # (n, 1) 라벨 배열
        ppgseg = np.zeros((1,1000)) #(n, 1000) PPG segment
        with open(ext+'_list.txt','r') as f: # train_list.txt 파일을 열고 읽음
            for line in f.readlines():
                line = line.strip('\n')

                path1 = os.path.join(r"C:\Users\cream\OneDrive\Desktop\AF_Emu\Data\MAT",line[1:-1]) # 두 경로 중 하나에 데이터가 존재
                path2 = os.path.join(r"C:\Users\cream\OneDrive\Desktop\AF_Emu\Data\MAT",line[1:-1])
                if os.path.exists(path1):
                    data = sio.loadmat(path1)
                else:
                    data = sio.loadmat(path2)
                    
                if 'label' in data and 'labels' not in data:
                    data['labels'] = data['label']
                
                if 'ppg' in data and 'ppgseg' not in data:
                    data['ppgseg'] = data['ppg']
                    
                labels = np.vstack((labels,data['labels'])) # (1,1) 라벨(0~1)
                ppgseg = np.vstack((ppgseg,data['ppgseg'])) # (1, 1000) PPG segment / 누적 해 쌓음
        
        labels = labels[1:len(labels),:] # 더미 zero 행 삭제
        ppgseg = ppgseg[1:len(ppgseg),:] # 더미 zero 행 삭제
        # labels
        # 0 : NON_AF
        # 1 : AF
        print(ppgseg.shape)
        
        self.labels = labels
        self.ppgseg = ppgseg
        self.transform = transform
        
    def __len__(self):
        return len(self.labels) # 전체 데이터의 길이

    def __getitem__(self, idx): # 샘플 하나 반환

        ppgseg = self.ppgseg[idx]
        labels = self.labels[idx]
    
        sample = { 'ppgseg': ppgseg, 'labels': labels} # 딕셔너리 형태로 반환
        
        if self.transform:
            sample = self.transform(sample) # tranform이 지정돼 있으면 적용

        return sample
    
class CropAverage(object):
    """
    Crop the signal to certain length, and provide the average sbp as the target

    Args:
        length (int): the length to be cropped to
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, sample):

        ppgseg, labels= sample['ppgseg'], sample['labels']
        signal = np.array([ppgseg]) # 1차원 신호를 (1, length)로 reshape

        return {'signal': signal, 'labels': labels}
    
class ToTensor(object): # numpy 배열을 Pytorch 텐서로 변환
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        signal = sample['signal']
        labels = sample['labels']

        return {
                'signal': torch.from_numpy(signal).float(),
                'labels': torch.from_numpy(labels).float()
               }
        
