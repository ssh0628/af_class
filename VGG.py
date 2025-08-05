import torch.nn as nn
import torch

# Conv 1d 기반 VGG
# input = (batch_size, 1, sequence_length) 형태의 1D 시계열 데이터
# output = num_classes개의 클래스에 대한 예측 (default = 4)
class VGG(nn.Module):

    def __init__(self, features, ngpu, num_classes=4, init_weights=True):
        super(VGG, self).__init__()
        self.features = features # make_layers로 생성된 CNN 구조
        self.ngpu = ngpu # 다중 GPU 병렬 처리를 위한 옵션
        
        self.classifier = nn.Sequential( # Conv1d를 Flatten 후 1024차원의 feature가 나올 것을 기대하고 설계
            nn.Linear(1024, 256), # 1024를 256으로
            nn.ReLU(True),        
            nn.Dropout(),        
            nn.Linear(256, num_classes) # 256을 num_classes로 분류
        )
        if init_weights: # True일 경우
            self._initialize_weights() # 가중치 초기화

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.features, x, range(self.ngpu))
            x = x.view(x.size(0), -1)
            x = nn.parallel.data_parallel(self.classifier, x, range(self.ngpu))
        else:
            x = self.features(x) # Conv1d layers 통과
            x = x.view(x.size(0), -1) # Flatten (배치 마다 벡터로 변환)
            x = self.classifier(x) # FC layer (분류 결과)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d): 
                nn.init.kaiming_normal_(m.weight.data,mode='fan_out') # kaiming 초기화로 ReLU 계열 활성화 함수와 잘 맞도록 설정
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d): # BatchNorm1d는 weight를 1로, bias를 0으로
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,mode='fan_out') # Linear도 동일하게 Kaiming 초기화
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=True): 
    # cfg를 호출 (Conv1d, ReLU, BatchNorm1d, MaxPool1d를 순차적으로 쌓음)
    # M은 MaxPool1d(kernel_size=3, stride=3)
    # 각 숫자는 Conv1d의 out_channels 값
    
    layers = []
    in_channels = 1 
    for v in cfg:
        if v == 'M': 
            layers += [nn.MaxPool1d(kernel_size=3,stride=3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'E': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
}

def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model