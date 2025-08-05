import torch
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
from DBprocessing import MyDataset, CropAverage, ToTensor
from config import opt

def main():
    # Loading test set
    test_dataset = MyDataset(dir=opt.dataroot,
                                    ext='test',
                                    transform=transforms.Compose([
                                        CropAverage(opt.recordLength),
                                        ToTensor()
                                    ]))
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                            shuffle=False,  num_workers=int(opt.workers))

    # Loading model
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    model = torch.load('modelPPG.pkl', map_location=device, weights_only=False)
        
    model.to(device)
    model.eval()  


    predictions = []
    labelsall = []
    pred_score = []

    # Fea_all = np.zeros((1,256), dtype = float)
    pred_score = np.zeros((1,2), dtype = float)
    for i, test_data in enumerate(test_loader, 0):
        inputs, labels = test_data['signal'], test_data['labels']
        if opt.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        output = model(inputs)
        labels = labels[:,0].long()
        score = output.data.cpu().numpy()
        pred_score = np.vstack((pred_score,score))       

        _,predicted = torch.max(output.data, 1)
        predicted = predicted.cpu()
        labels = labels.cpu()  
        predictions.extend(predicted.numpy())
        labelsall.extend(labels.numpy())
        
    labelsAll = labelsall
    predictionsAll = predictions
    pred_score = pred_score[1:len(pred_score),:]

    assert set(labelsAll).issubset({0,1})

    print(classification_report(labelsAll,predictionsAll,digits=4))

    # Confusion Matrix
    confusion = confusion_matrix(labelsAll,predictionsAll)
    classes = ['NON_AF', 'AF']
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.imshow(confusion,cmap=plt.cm.OrRd)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index-0.1, second_index, confusion[second_index][first_index])
            
    print('Sensitivity, Specificity, Positive predictive value,Negative predictive values,Accuracy')
    for i in range(len(confusion)):
        TP = confusion[i,i]
        FN = sum(confusion[i,:])-TP
        FP = sum(confusion[:,i])-TP
        TN = sum(sum(confusion))-TP-FP-FN
        
        print(np.array([TP/(TP+FN), TN/(FP+TN), TP/(TP+FP), TN/(FN+TN), (TP+TN)/(TP+FP+FN+TN)]))
        
    plt.show()
    
    # ROC, AUC
    """ 
    y_true = label_binarize(np.array(labelsAll), classes=[0, 1])
    n_classes = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], pred_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],
            label='Average ROC curve (AUC = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linewidth=2)
    
    colors = cycle(['blue','red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                label='ROC curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC to Binary-Class')
    plt.legend(loc="lower right")
    plt.show()
    """

if __name__ == '__main__':
    main()