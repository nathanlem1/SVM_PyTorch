import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# SVM model
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


# MNIST dataset (images and labels) for testing a trained model
val_dataset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         transform=transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=4,
                                         shuffle=False)

num_classes = len(val_dataset.classes)  # 10 for MNIST
input_size = val_loader.dataset.data[0].reshape(1,-1).size()[1] # input_size = 28*28 = 784 for MNIST
                                                                      # Vectorize the input for fully connected network

# Load the trained model
learned_model = SVM(input_size,num_classes)
learned_model.load_state_dict(torch.load('./model.pth'))
learned_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learned_model.to(device)

# Analyze the model performance on validation set treating it as a testing set by using confusion matrix
gt_all = []
predicted_all = []
labels_total = list(val_dataset.class_to_idx.values())
with torch.no_grad(): # For memory efficiency, it is not necessary to compute gradients in test phase.
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = learned_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        for i in range(images.shape[0]):
            gt_all.append(labels[i].item())
            predicted_all.append(predicted[i].item())

    print('Accuracy of the model on the 10000 test images is {} %'.format(100 * correct / total))


# Draw confusion matrices
cn_matrix = confusion_matrix(
    y_true=gt_all,
    y_pred=predicted_all,
    labels=labels_total,
    normalize='true')
ConfusionMatrixDisplay(cn_matrix).plot(include_values=False, xticks_rotation='vertical')
plt.title("Confusion matrix")
plt.tight_layout()
plt.show()