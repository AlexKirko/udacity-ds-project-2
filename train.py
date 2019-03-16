import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

import json
import time

import math
import numpy as np


import argparse
import os

#Import my modules
from netsettings import net_archs, net_classes, net_prefixes, net_params
from classifier import get_classifier

# main training code
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Neural net training',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', type=str, help="""Name of the model to be trained. 
    Supported models: '""" + "', '".join(list(net_archs.keys())) + "'")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="""Learning rate to use for the Adam optimizer.""")
    parser.add_argument('--hidden_units', type=int, default=2,
                        help="""Number of hidden units for the new classifier. 
                        Recommended values: 2 or 3.""")
    parser.add_argument('--training_epochs', type=int, default=15,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('--save_filename', type=str,
                        help="""Name for the checkpoint file.""")
    parser.add_argument('--device', type=str, default='cpu',
                        help="""Device on which to evaluate the model.""")
    args = parser.parse_args()
    
    model_name = args.model_name
    learning_rate = args.learning_rate
    classifier_hidden = args.hidden_units
    epochs = args.training_epochs
    device = args.device
    if args.save_filename is None:
        save_filename = net_prefixes[model_name] + 'checkpoint.pth'
    else:
        save_filename = args.save_filename
    
    
    #Validate supplied args
    if model_name not in list(net_archs.keys()):
        raise ValueError(model_name + ' is not a supported model.')
    
    if device not in ['cuda','cpu']:
        raise ValueError(device + ' is not a supported device.')
    
    try:
        f = open(save_filename, 'w+')
        f.close()
        os.remove(save_filename)
    except:
        raise ValueError(save_filename + ' is not a valid file path.')
    
    
    #Set up remaining settings
    model_class = net_classes[model_name]
    model_load = net_archs[model_name]

    features_count = net_params[model_class]['features_count']
    batch_size = net_params[model_class]['batch_size']
    classifier_name = net_params[model_class]['classifier_name']
    img_resize = net_params[model_class]['img_resize']
    img_crop = net_params[model_class]['img_crop']
    
    #For local training, use the data on an M.2 SSD

    data_dir = 'C:/udacity/data/flowers'
    #data_dir = 'data/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    # Alex Kirko: using the transforms suggested in Part 8 of the PyTorch lesson
    # since flowers are symmetrical, it won't hurt to rotate them more
    # since we'll be using an ImageNet pre-trained network, it's important to
    # normalize the color channels the same way they were normalized when these
    # models were trained
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(img_crop),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    #Alex Kirko: for validation and test there is no need to introduce randomness to the data
    valid_transforms = test_transforms = transforms.Compose([transforms.Resize(img_resize),
                                          transforms.CenterCrop(img_crop),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #Alex Kirko: only the training set benefits from shuffling at the end of each epoch
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    model = model_load(pretrained=True)
    
    # Remove gradient tracking from the network parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create a new classifier
    # Use dropout probability that the VGG models use - 
    # authors likely knew what they were doing

    classifier = get_classifier(features_count, classifier_hidden=classifier_hidden, dropout_p=0.5)

    #Attach the new classifier (it has random weights now)
    if classifier_name == 'classifier':
        model.classifier = classifier
    elif classifier_name == 'fc':
        model.fc = classifier
        
    # Move the model to GPU before constructing the optimizer
    model.to(device)

    # Set up negative likelihood loss and the adam optimizer
    # (momentum is useful)
    criterion = nn.NLLLoss()
    if classifier_name == 'classifier':
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif classifier_name == 'fc':
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    print('Training start.')
    
    train_losses, test_losses = [], []
    for e in range(epochs):
        start = time.time()
        running_loss = 0
        train_r_acc = 0
        for images, labels in train_loader:
            
            images, labels = images.to(device), labels.to(device)
                    
            optimizer.zero_grad()
            
            log_ps = model.forward(images)
            
            #Inception calculates two losses: auxillary and main
            if model_name != 'Inception V3':
                loss = criterion(log_ps, labels)
            else:
                loss = criterion(log_ps[0], labels) + criterion(log_ps[1], labels)
            loss.backward()
            optimizer.step()
            
            #print('Current time: {:.1f}'.format(time.time() - start))
            
            running_loss += loss.item()
            #print('Batch training time: {:.1f}'.format(time.time() - start))
            
            #For debugging. Testing accuracy
            if model_name != 'Inception V3':
                top_log_p, predictions = log_ps.topk(1, dim = 1)
            else:
                top_log_p, predictions = log_ps[0].topk(1, dim = 1)
                #print(predictions)
            equals = predictions == labels.view(*predictions.shape)
            train_r_acc += torch.mean(equals.type(torch.FloatTensor)).item()
            
        else:
            with torch.no_grad():
                ## TODO: Implement the validation pass and print out the validation accuracy
                
                model.eval()
                
                running_acc = 0
                test_loss = 0
                for images, labels in valid_loader:
                    
                    images, labels = images.to(device), labels.to(device)
                    
                    log_ps = model.forward(images)
                    top_log_p, predictions = log_ps.topk(1, dim = 1)
                    equals = predictions == labels.view(*predictions.shape)
                    running_acc += torch.mean(equals.type(torch.FloatTensor)).item()
                    test_loss += criterion(log_ps, labels)
                train_loss = running_loss / len(train_loader)
                test_loss = test_loss / len(valid_loader)
                accuracy = running_acc / len(valid_loader)
                t_accuracy = train_r_acc / len(train_loader)
                
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                
                model.train()
                
                print('Epoch: {}/{}'.format(e+1, epochs))
                print('Training loss: {0:.2f}'.format(train_loss))
                print('Validation loss: {0:.2f}'.format(test_loss))
                print('Validation Accuracy: {0:.2f}'.format(accuracy*100))
                print('Training accuracy: {0:.2f}'.format(t_accuracy*100))
                print('Epoch training time: {:.1f}'.format(time.time() - start))
                
    # TODO: Do validation on the test set
    model.to(device)
    
    print('Testing on the test set.')
    with torch.no_grad():
        model.eval()

        running_acc = 0
        test_loss = 0
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            log_ps = model.forward(images)
            top_log_p, predictions = log_ps.topk(1, dim = 1)
            equals = predictions == labels.view(*predictions.shape)
            running_acc += torch.mean(equals.type(torch.FloatTensor)).item()
            test_loss += criterion(log_ps, labels)
        test_loss = test_loss / len(test_loader)
        accuracy = running_acc / len(test_loader)

        model.train()

        print('Epoch: {}/{}'.format(e+1, epochs))
        print('Test loss: {0:.2f}'.format(test_loss))
        print('Accuracy: {0:.2f}'.format(accuracy*100))
    
    # TODO: Save the checkpoint 

    class_to_idx = train_data.class_to_idx
    optim_state_dict = optimizer.state_dict()

    #print("Our model: \n\n", model, '\n')
    #print("The state dict keys: \n\n", model.state_dict().keys())

    #Based on https://pytorch.org/tutorials/beginner/saving_loading_models.html
    checkpoint = {
        'epoch': e,
        'optim_state_dict': optim_state_dict,
        'model_name': model_name,
        'optim_class' : 'Adam',
        'loss_class' : 'NLLLoss',
        'model_device' : device,
        'model_state_dict': model.state_dict(),
        'class_to_idx' : class_to_idx,
        'learning_rate' : learning_rate,
        'classifier_hidden' : classifier_hidden
    }

    torch.save(checkpoint, save_filename)
    
    print('Model saved to ' + save_filename)