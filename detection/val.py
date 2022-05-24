from model import Net
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import LambdaLR
from myDataset import MyDataset

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 5
    path3 = "../dataset/test/gt/"
    path4 = "../dataset/test/moire/"  
    test_dataset = MyDataset(path3, path4, train=False, size = 160)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = torch.load('./models/model_0.96.pkl',map_location={'cuda:2': 'cuda:0','cuda:1': 'cuda:0'}).to(device)

    all_correct_num = 0
    all_sample_num = 0

    for idx, (test_x, test_label) in enumerate(test_loader):
        if idx%10==0:
            print (idx*batch_size)
        test_x, test_label = test_x.to(device), test_label.to(device)
        predict_y0 = model(test_x.float()).detach()
        predict_y = torch.Tensor.cpu(predict_y0)
        predict_y = torch.nn.functional.softmax(predict_y,dim=1)
        test_label = torch.Tensor.cpu(test_label)
        predict_y = np.argmax(predict_y, axis=-1)
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]
    acc = all_correct_num/all_sample_num
    print('accuracy: {:.5f}'.format(acc))
