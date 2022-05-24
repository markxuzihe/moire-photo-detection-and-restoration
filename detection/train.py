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
    batch_size = 10
    init_lr = 0.001
    path1 = "../dataset/train/gt/"
    path2 = "../dataset/train/moire/"
    path3 = "../dataset/test/gt/"
    path4 = "../dataset/test/moire/"    
    train_dataset = MyDataset(path1, path2, train=True, size = 3000)
    test_dataset = MyDataset(path3, path4, train=False, size = 160)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # dict to define model parameters
    params_model = {"input_shape": (3, 1024, 1024),
                    "initial_filters": 8,
                    "num_fc1": 100,
                    "dropout_rate": 0.25,
                    "num_classes": 2
                    }
    model = Net(params_model).to(device)
    #model = torch.load('./models/model_0.96.pkl').to(device)
    sgd = SGD(model.parameters(), lr=init_lr)
    scheduler_1 = LambdaLR(sgd, lr_lambda=lambda epoch: 1/(epoch+1))
    loss_fn = CrossEntropyLoss()
    all_epoch = 50
    for current_epoch in range(all_epoch):
        print('=======================================================')
        print('epoch: {}, lr: {}'.format(current_epoch, sgd.param_groups[0]['lr']))
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x, train_label = train_x.to(device), train_label.to(device)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()
        scheduler_1.step()
        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x, test_label = test_x.to(device), test_label.to(device)
            predict_y0 = model(test_x.float()).detach()
            predict_y = torch.Tensor.cpu(predict_y0)
            test_label = torch.Tensor.cpu(test_label)
            predict_y = np.argmax(predict_y, axis=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num/all_sample_num
        print('accuracy: {:.4f}'.format(acc))
        torch.save(model, 'models/model_{:.4f}.pkl'.format(acc))
