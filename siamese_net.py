import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
import mlflow

class SiameseNetwork(nn.Module):
    """
    Siamese network for image similarity estimation.
    The network is composed of two identical networks, one for each input.
    The output of each network is concatenated and passed to a linear layer. 
    The output of the linear layer passed through a sigmoid function.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # use resnet18 without pretrained weights
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the classifier (linear layers)
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output

class DatasetLoader(Dataset):
    """
    PyTorch dataset loader for generating input dataset
    """
    def __init__(self, data, labels): 
        super(DatasetLoader, self).__init__()

        self.data = torch.from_numpy(data).to(torch.float).unsqueeze(1).clone()
        lbl_groups = np.unique(labels, return_counts=False)
        self.label_map = dict(zip(lbl_groups, range(len(lbl_groups))))
        self.grouped_examples = {self.label_map[grp]: np.where((grp==labels))[0] for grp in lbl_groups}
        
    def show_map(self):
        return self.label_map

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        """
        Pick 2 images each time, and make them into 2 cases, positive and negative ones.
        positive: 2 images from the same class
        negative: 2 images from the diff class
        -----
        Given an index
        index is even: pick the second image from the same class.
        index is odd : pick the second image from other classes.
        """

        # pick some random class for the first image
        selected_class = random.randint(0, len(self.label_map) - 1)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
        
        # pick the index to get the first image
        index_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        image_1 = self.data[index_1].clone().float()

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
            
            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
            
            # pick the index to get the second image
            index_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)
        
        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, len(self.label_map) - 1)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, len(self.label_map) - 1)
            
            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0] - 1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target


def dataSpliter(data, labels, test_ratio=0.2, seed=100, shuffle=True):
    """
    The data spliter which will split the data with the label distribution
    ---
    Params:
    data(object): a python object to store data
    labels(object): a python object to store labels
    test_ratio(float): a spliting ratio for test dataset
    seed(int): the random seed
    shuffle(bool): indicator the shuffle the data
    ---
    Returns:
    train_data(object): training data
    train_labels(object): training labels
    test_data(object): testing data
    test_labels(object): testing labels
    """
    np.random.seed(seed)
    grps = np.unique(labels)
    test_idx = []
    for _grp in grps:
        idx = np.where((_grp==labels))[0]
        if shuffle:
            np.random.shuffle(idx)
        test_idx.append(idx[:int(len(idx) * test_ratio)])
    test_idx = np.concatenate(test_idx)
    train_idx = np.setdiff1d(np.arange(len(labels)), test_idx)
    train_data, train_labels = data[train_idx], labels[train_idx]
    test_data, test_labels = data[test_idx], labels[test_idx]

    return train_data, train_labels, test_data, test_labels


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def test(model, device, loader):
    model.eval()
    loss = 0
    correct = 0

    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs>0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    total = len(loader.dataset)
    acc = correct / total
    loss /= total

    return total, acc, loss
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network on Dynamic Malware Analysis')
    parser.add_argument('--data-path', type=str, 
                        help='the path to the data in npy format')
    parser.add_argument('--label-path', type=str, 
                        help='the path to the label in npy format')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--sched-step', type=int, default=1, metavar='N',
                        help='scheduler step for learning decay (default: 1)')
    parser.add_argument('--sched-gamma', type=float, default=0.7, metavar='M',
                        help='scheduler gamma for learning decay (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-best-model', action='store_true', default=False,
                        help='save the model with best accuracy')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    data = np.load(args.data_path, allow_pickle=True)

    max_data = data.max()
    data = data / max_data
    labels = np.load(args.label_path, allow_pickle=True)
    train_data, train_labels, test_data, test_labels = dataSpliter(data, labels)

    train_dataset = DatasetLoader(train_data, train_labels)
    test_dataset = DatasetLoader(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)
    best_acc = 0.0

    from datetime import datetime
       
    ExperimentName = str(datetime.strftime(datetime.now(), format='%Y-%m-%d_%H%M%S'))

    ## mlflow
    mlflow.set_experiment("Siamese_Net")
    mlflow.start_run(run_name = ExperimentName)       
    mlflow.log_param('train_batch_size', args.batch_size)
    mlflow.log_param('test_batch_size', args.test_batch_size)
    mlflow.log_param('Learning Rate', args.lr)
    mlflow.log_param('Scheduler Step', args.sched_step)
    mlflow.log_param('Scheduler Gamma', args.sched_gamma)
    mlflow.log_param('Optimizer', type(optimizer).__name__)
    mlflow.log_param('max_data', max_data)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train_samples, train_acc, train_loss = test(model, device, train_loader)
        test_samples, test_acc, test_loss = test(model, device, test_loader)
        scheduler.step()

        ## mlflow 
        mlflow.log_metric("train_acc", train_acc, step = epoch)
        mlflow.log_metric("train_loss", train_loss, step = epoch)
        mlflow.log_metric("test_acc", test_acc, step = epoch)
        mlflow.log_metric("test_loss", test_loss, step = epoch)

        if args.save_best_model and test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "./model/SN_" + ExperimentName + ".pt") 
    mlflow.log_param('best_acc', best_acc)
    mlflow.end_run()

if __name__ == '__main__':
    # python siamese_net.py --data-path img_data_all.npy --label-path img_label_all.npy --batch-size 16 --test-batch-size 32 --epoch 15 --lr 0.001 --sched-step 1 --sched-gamma 0.8 --save-best-model
    main()
