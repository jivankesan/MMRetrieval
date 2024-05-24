import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('--save_model', default='weights/', type=str)
parser.add_argument('--root', default='', type=str)

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import videotransforms
from dataset import *
from timm.models import create_model

# Function to get the available device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()

def run(init_lr=0.01, max_steps=100, mode='rgb', root='', batch_size=16, save_model='weights/'):
    # setup dataset
    train_transforms = transforms.Compose([
        videotransforms.CenterCrop(224),
        transforms.ToTensor(),  # Add ToTensor transform to convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per Swin requirements
    ])
    
    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(224),
        transforms.ToTensor(),  # Add ToTensor transform to convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per Swin requirements
    ])

    num_classes = 31

    dataset = Dataset('./splits/train_cs.txt', 'train', root, 'rgb', train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    val_dataset = Dataset('./splits/validation_cs.txt', 'val', root, 'rgb', test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    
    # setup the model
    model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    model.to(device)
    model = nn.DataParallel(model)

    lr = init_lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

    steps = 0
    # train it
    while steps < max_steps:
        print(f'Step {steps}/{max_steps}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_cls_loss = 0.0
            tot_acc = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))

                outputs = model(inputs)
                criterion = nn.CrossEntropyLoss().to(device)
                cls_loss = criterion(outputs, torch.max(labels, dim=1)[1].long())
                tot_cls_loss += cls_loss.data

                loss = cls_loss
                tot_loss += loss.data
                loss.backward()
                acc = calculate_accuracy(outputs, torch.max(labels, dim=1)[1])
                tot_acc += acc
                if phase == 'train':
                    optimizer.step()
                    optimizer.zero_grad()

            if phase == 'train':
                print(f'{phase} Cls Loss: {tot_cls_loss/num_iter:.4f} Tot Loss: {tot_loss/num_iter:.4f}, Acc: {tot_acc/num_iter:.4f}')
                # save model
                torch.save(model.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                tot_loss = tot_cls_loss = tot_acc = 0.0
                steps += 1
            if phase == 'val':
                lr_sched.step(tot_cls_loss/num_iter)
                print(f'{phase} Cls Loss: {tot_cls_loss/num_iter:.4f} Tot Loss: {tot_loss/num_iter:.4f}, Acc: {tot_acc/num_iter:.4f}')

if __name__ == '__main__':
    run(mode=args.mode, root=args.root, batch_size=16, save_model=args.save_model)