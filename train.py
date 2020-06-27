from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
import time
path = 'C:/Users/Arthur/Desktop/sum/testone'
#save the location of the datasets
files = glob(os.path.join(path,'*/*.png'))
#read all samples of 
if torch.cuda.is_available():
    is_cuda = True
simple_transform = transforms.Compose([transforms.Resize((40,40)),
                                       transforms.RandomHorizontalFlip(p=0.5)
                                       ,transforms.ToTensor()
                                       ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train = ImageFolder('C:/Users/Arthur/Desktop/sum/tempu/train/',simple_transform)
valid = ImageFolder('C:/Users/Arthur/Desktop/sum/tempu/valid/',simple_transform) 
train_data_gen = torch.utils.data.DataLoader(train,shuffle=True,batch_size=64,num_workers=0,drop_last=True)
valid_data_gen = torch.utils.data.DataLoader(valid,shuffle=True,batch_size=64,num_workers=0,drop_last=True)
dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}
dataloaders = {'train':train_data_gen,'valid':valid_data_gen}
model_ft = models.resnet101(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 5)

if torch.cuda.is_available():
    model_ft = model_ft.cuda()
# Loss and Optimizer
learning_rate = 0.002
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.002, momentum=0.9,weight_decay=1e-5)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                
                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
             
                # forward
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
               
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).float()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            Loss_list[phase].append(epoch_loss*100)
            Accuracy_list[phase].append(epoch_acc*100)
            #Accuracy_list[phase]
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,Loss_list,Accuracy_list
model_ft,Loss_list,Accuracy_list = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)

#draw the plot
x = range(0, 50)
y1 = np.array(Loss_list["valid"])
y2 = np.array(Loss_list["train"])
plt.figure(figsize=(18,14))
plt.subplot(211)
plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="valid")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss vs. epoches')
plt.ylabel('loss')

plt.subplot(212)
y3 = Accuracy_list["train"]
y4 = Accuracy_list["valid"]
plt.plot(x, y3, color="y", linestyle="-", marker=".", linewidth=1, label="train_acc")
plt.plot(x, y4, color="g", linestyle="-", marker=".", linewidth=1, label="valid_acc")
plt.legend()
plt.title('train and valid accuracy')
plt.ylabel('accuracy')

#save the model
torch.save(model_ft, 'model.pth')

