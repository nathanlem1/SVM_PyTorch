import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import argparse
import copy


class SVM(nn.Module):
    """
    Using fully connected neural network to implement linear SVM and Logistic regression with hinge loss and
    cross-entropy loss which computes softmax internally, respectively.
    """
    def __init__(self, input_size, num_classes):
        super(SVM, self).__init__()    # Call the init function of nn.Module
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, other_args):
    """
    This function trains the chosen model (SVM or logistic regression) and with other parameters.

    Arguments:
        model:  model to be trained
        dataloaders:  data loader of train and val
        dataset_sizes:  data set sizes of train and val data sets
        criterion:  loss function
        optimizer:  optimization algorithm
        scheduler:  learning scheduler
        other_args:  required parse arguments such as args.device, args.c, etc
    Return:
        model:  the trained model
    """

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(other_args.num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for i, (images, labels) in enumerate(dataloaders[phase]):
                # Reshape images to (batch_size, input_size) and then move to device
                images = images.reshape(-1, other_args.input_size).to(other_args.device)
                labels = labels.to(other_args.device)

                # Forward pass - track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)  # data loss

                    # Add regularization i.e.  Full loss = data loss + regularization loss
                    weight = model.fc.weight.squeeze()
                    if other_args.rg_type == 'L1':  # add L1 (LASSO - Least Absolute Shrinkage and Selection Operator)
                                                    # loss which leads to sparsity.
                        loss += other_args.c * torch.sum(torch.abs(weight))
                    elif other_args.rg_type == 'L2':   # add L2 (Ridge) loss
                        loss += other_args.c * torch.sum(weight * weight)
                    elif other_args.rg_type == 'L1L2':   # add Elastic net (beta*L2 + L1) loss
                        loss += other_args.c * torch.sum(other_args.beta * weight * weight + torch.abs(weight))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Collect statistics
                running_loss += loss.item() * images.size(0)  # images.size(0) is batch size.
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('Epoch [{}/{}], {} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch + 1, other_args.num_epochs, phase, epoch_loss, epoch_acc*100.))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Acc in percentage: {:.4f}'.format(best_acc*100.))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model



def main():
    """
    Main function to run the linear SVM using hinge loss or logistic regression using softmax loss (cross-entropy loss)
    implemented using PyTorch.
    """
    parser = argparse.ArgumentParser(
        description='Linear SVM implementation using PyTorch (option for using Logistic regression is also included).')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate for training')
    parser.add_argument('--c', type=float, default=0.01,
                        help= 'Regularization parameter')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Mixing parameter for Elastic net regularization')
    parser.add_argument('--rg_type', type=str, default='', choices=['L1', 'L2', 'L1L2'],
                        help='Regularization type to use: L1 (LASSO), L2 (Ridge), Elastic net (beta*L2 + L1) or None')
    parser.add_argument('--classification_type', type=str, default='svm', choices=['svm', 'logisticR'],
                        help='Classification type to use: SVM or Logistic regression')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers to use in data loading')
    parser.add_argument('--input_size', type=int, default=784,
                        help='Number of input size for training and validation dataset')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(args)

    # MNIST dataset (images and labels)
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    val_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader

    dataset_sizes = {}
    dataset_sizes['train'] = len(train_dataset)
    dataset_sizes['val'] = len(val_dataset)

    num_classes = len(dataloaders['train'].dataset.classes)  # 10 for MNIST
    input_size = train_loader.dataset.data[0].reshape(1,-1).size()[1] # input_size = 28*28 = 784 for MNIST
                                                                      # Vectorize the input for fully connected network
    args.input_size = input_size

    # Initialized the model to be trained: SVM or Logistic regression
    model = SVM(input_size, num_classes)
    model.to(args.device)

    # Loss and optimizer
    if args.classification_type == 'svm':
        criterion = nn.MultiMarginLoss()  # Multi-class classification hinge loss (margine-based loss); SVM
    elif args.classification_type == 'logisticR':
        criterion = nn.CrossEntropyLoss()  # Cross-entropy loss which computes softmax internally; logistic regression

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the model
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, args)

    # Save the model
    torch.save(model.state_dict(), 'model.pth')


# Execute from the interpreter
if __name__ == "__main__":
    main()



