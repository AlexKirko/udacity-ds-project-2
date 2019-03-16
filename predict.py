import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

import json
import time

import math
import numpy as np

from PIL import Image

import argparse
import os

#Import my modules
from netsettings import net_archs, net_classes, net_prefixes, net_params
from classifier import get_classifier

def model_load(path):
    ''' Load a model from checkpoint
    
    In:
    path : path to the checkpoint file
    
    Out:
    model : model object
    criterion : loss function
    optimizer : initialized optimizer
    epoch : epoch at which the model finished training
    device : device from which the model was saved
    model_name : model name 
                 (used to get settings for the model)
    '''
    checkpoint = torch.load(path)
    
    model_name = checkpoint['model_name']
    loss_class = checkpoint['loss_class']
    optim_class = checkpoint['optim_class']
    model_device = checkpoint['model_device']
    class_to_idx = checkpoint['class_to_idx']
    learning_rate = checkpoint['learning_rate']
    classifier_hidden = checkpoint['classifier_hidden']
    
    model_class = net_classes[model_name]
    model_load = net_archs[model_name]
    net_prefix = net_prefixes[model_name]
    
    #Initialize parameters needed to build the network
    features_count = net_params[model_class]['features_count']
    batch_size = net_params[model_class]['batch_size']
    classifier_name = net_params[model_class]['classifier_name']
    img_resize = net_params[model_class]['img_resize']
    img_crop = net_params[model_class]['img_crop']
        
    model = model_load(pretrained=False)
    
    for param in model.parameters():
        param.requires_grad = False

    # Create a new classifier
    classifier = get_classifier(features_count, classifier_hidden=classifier_hidden, dropout_p=0.5)
    #Attach the new classifier (it has random weights now)
    if classifier_name == 'classifier':
        model.classifier = classifier
    elif classifier_name == 'fc':
        model.fc = classifier
        
    model.to(model_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = class_to_idx
    # What we need to get human-readable labels
    # is a mapping from indices to class labels
    model.idx_to_class = {idx : c for c, idx in model.class_to_idx.items()}
    
    
    if loss_class == 'NLLLoss':
        criterion = nn.NLLLoss()        
    else:
        raise ValueError(loss_class + ' loss function is not supported. Your model should use NLLLoss.')
    
    if optim_class == 'Adam':
        if classifier_name == 'classifier':
            optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        elif classifier_name == 'fc':
            optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        raise ValueError(optim_class + ' optimizer is not supported. Your model should use Adam.')
    
    #optimizer.load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint['epoch']
    
    device = model_device
    
    return model, criterion, optimizer, epoch, device, model_name
    
def process_image(img, model_name, net_classes, net_params):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a torch.Tensor
        
        In:
        img : a PIL image created with Image.open
        model_name : model_name (used to get settings)
        net_classes : mapping of model names to classes
        net_params : mapping of model classes to
                     their default parameters
        
        Out:
        img : tensor ready for the model
    '''
    
    model_class = net_classes[model_name]
    img_resize = net_params[model_class]['img_resize']
    img_crop = net_params[model_class]['img_crop']
    
    width, height = img.size
    if width < height:
        size = (img_resize, math.floor(height * img_resize / width))
    else:
        size = (math.floor(width * img_resize / height), img_resize)
    
    img = img.resize(size)
                
    width, height = size
    # Crop based on this stackoverflow article
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    
    left = (width - img_crop)/2
    top = (height - img_crop)/2
    right = (width + img_crop)/2
    bottom = (height + img_crop)/2

    img = img.crop((left, top, right, bottom))
    
    img = np.array(img)
    
    img = img / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img)
    
    # Sometimes the tensor gets cast to DoubleTensor.
    # This can break a model.
    img = img.type(torch.FloatTensor)
    
    return img

#Make sure class_to_idx is recoverable from checkpoint
def predict(image_path, model, model_name, 
            net_classes, net_params, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model
    based on filepath.
    
    In:
    image_path : path to the image file
    model : model used to predict the classes    
    model_name : model_name (used to get settings)
    net_classes : mapping of model names to classes
    net_params : mapping of model classes to
                 their default parameters
    topk : number of top classes to get
    device : device on which to predict
    '''
    model_class = net_classes[model_name]
    img_crop = net_params[model_class]['img_crop']
    
    img = Image.open(image_path)

    img = process_image(img, model_name, net_classes, net_params)
    img.resize_(1, 3, img_crop, img_crop)
    
    model.to(device)
    img = img.to(device)
    model.eval()
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)
    
    
    model.train()
    
    ps = torch.exp(output)
    top_p, predictions = ps.topk(topk, dim = 1)
    idxs = [model.idx_to_class[x.item()] for x in predictions[0]]
    #print(idxs)
    return top_p.cpu(), idxs
    
# main prediction code
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Neural net prediction',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image_path', type=str, help="""Path to the image file""")
    parser.add_argument('checkpoint_path', type=str, help="""Path to the checkpoint file""")
    parser.add_argument('--class_name_map_path', type=str, 
                        help="""Path to the json file that maps class labels to class names.
                        If this argument is provided, the script displays human-readable names.""")
    parser.add_argument('--topk', type=int, default=1,
                        help="""Number of top classes to show.""")
    parser.add_argument('--device', type=str, default='cpu',
                        help="""Device on which to evaluate the model.""")
    args = parser.parse_args()
    
    #Validate device
    if args.device not in ['cuda','cpu']:
        raise ValueError(device + ' is not a supported device.')
    
    model, criterion, optimizer, epoch, device, model_name = model_load(args.checkpoint_path)
    
    img = Image.open(args.image_path)
    
    img = process_image(img, model_name, net_classes, net_params)
    
    top_p, idxs = predict(args.image_path, model, model_name, 
                      net_classes, net_params, topk=args.topk, device=args.device)
    
    if args.class_name_map_path is None:
        print('Most likely class labels (folder names)')
        print(idxs)
    else:
        with open(args.class_name_map_path, 'r') as f:
            cat_to_name = json.load(f)
        cat_names = [cat_to_name[x] for x in idxs]
        print('Most likely class names')
        print(cat_names)