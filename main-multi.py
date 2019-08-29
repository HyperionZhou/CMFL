from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pickle
import copy

DATA_LEN = 60000

class custom_MNIST_dset(Dataset):
    def __init__(self,
                 image_path,
                 label_path,
                 img_transform = None):
        with open(image_path, 'rb') as image_file:
            self.image_list = pickle.load(image_file)
        with open(label_path, 'rb') as label_file:
            self.label_list = pickle.load(label_file)
        
        self.img_transform = img_transform
    
    def __getitem__(self, index):
        image = self.image_list[index]
        label = self.label_list[index]
        
        if self.img_transform is not None:
            image = self.img_transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.image_list)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def cli_train(args, old_model, device, train_loader, epoch, last_update):
    model = copy.deepcopy(old_model)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.zero_grad()
    
    cli_ite_num = 3
    for batch_idx, (data, target) in enumerate(train_loader):
        if cli_ite_num == 0:
            break
        cli_ite_num -= 1
        
        data, target = data.type('torch.FloatTensor').to(device), target.to(device, dtype=torch.int64)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
    relv, updates = check_relevance(model, old_model)
        
    return relv, updates

def glo_train(args, model, device, train_loaders, optimizer, epoch, commu, flag):
    model.train()
    for i in range(DATA_LEN // (args.client_num * args.batch_size)):
        optimizer.zero_grad()
        new_para_grad_list = []
        
        if flag:
            tmp_flag = True
            
            for i in range(args.client_num):
                relv, grad = cli_train(args, model, device, train_loaders[i], epoch, None)
                new_para_grad_list.append(grad)
                
            cur_commu = args.client_num
            flag = False
            
        else:
            cur_commu = 0
            last_update = []
            
            for item in model.parameters():
                last_update.append(item.grad)

            for i in range(args.client_num):
                relv, grad = cli_train(args, model, device, train_loaders[i], epoch, last_update)
                if relv:
                    cur_commu += 1
                    new_para_grad_list.append(grad)
        
        # Merge model grad
        merge(model, new_para_grad_list)
        optimizer.step()
        commu.append(cur_commu)

        
        print('Train Epoch: {}'.format(epoch))
            
def test(args, model, device, test_loader, commu):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Draw picture acc vs commu
    pass
    #
    
def check_relevance(model, old_model):
    if model is None or old_model is None:
        return True
    
    sign_sum = 0
    sign_size = 0
    rel_threshold = 0.8
    
    for cur_para, old_para in zip(model, old_model):
        cur_sign = torch.sign(cur_para)
        old_sign = torch.sign(old_para)
        
        sign = cur_sign * old_sign
        sign[sign < 0] = 0
        sign_sum += torch.sum(sign)
        sign_size += sign.numel()
    
    return e >= rel_threshold

def merge(model, new_para_grad_list):
    para_ind = 0
    for item in model.parameters():
        item.grad = new_para_grad_list[0][para_ind]
        for i in range(1, len(new_para_grad_list)):
            item.grad += new_para_grad_list[i][para_ind]
        
        para_ind += 1
        item.grad /= len(new_para_grad_list)
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--client-num', type=int, default=10)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loaders = []
    for i in range(args.client_num):
        train_loaders.append(torch.utils.data.DataLoader(
                            custom_MNIST_dset('MNIST_data/train_data-' + str(i), 'MNIST_data/train_label-' + str(i),
                                           img_transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ])),
                            batch_size=args.batch_size, shuffle=False, **kwargs))
    
    test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
                    batch_size=args.test_batch_size, shuffle=True, **kwargs)

    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    commu = []
    flag = True
    for epoch in range(1, args.epochs + 1):
        glo_train(args, model, device, train_loaders, optimizer, epoch, commu, flag)
        test(args, model, device, test_loader, commu)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()