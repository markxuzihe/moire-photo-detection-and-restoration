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
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset

target_path = "../../graph_for_classification/moire/10.jpg"

def default_loader(target_path):
    labels = []
    images = []
    images.append(Image.open(target_path).resize((1024,1024)))
    labels.append(1)
    return labels,images

class TestData(Dataset):

    def __init__(self, path):
        super(TestData, self).__init__()
        self.labels, self.images = default_loader(path)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image = np.asarray(image)
        image = np.transpose(image,(2,0,1))
        image = torch.from_numpy(image).div(255.0)
        label = self.labels[index]
        label = int(label)
        return image, label


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1


    test_dataset = TestData(target_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = torch.load('./models/model_0.96.pkl',map_location={'cuda:2': 'cuda:0','cuda:1': 'cuda:0'}).to(device)


    for idx, (test_x, test_label) in enumerate(test_loader):
        if idx%10==0:
            print (idx*batch_size)
        test_x, test_label = test_x.to(device), test_label.to(device)
        predict_y0 = model(test_x.float()).detach()
        predict_y = torch.Tensor.cpu(predict_y0)
        predict_y = torch.nn.functional.softmax(predict_y,dim=1)
        print(predict_y)
        test_label = torch.Tensor.cpu(test_label)
        predict_y = np.argmax(predict_y, axis=-1)
        current_correct_num = predict_y == test_label
        res = np.sum(current_correct_num.numpy(), axis=-1)
        print(res)
