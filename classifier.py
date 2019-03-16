import torch
from torch import nn
from collections import OrderedDict

import math
import numpy as np

def get_classifier(features_count, classifier_hidden=0, dropout_p=0.5):
    ''' Get a new classifier for the neural network
    
    In:
    features_count : number of inputs to the classifier
    classifier_hidden : number of hidden layers in
                        the classifier
    dropout_p : dropout probability for the hidden layers
    
    Out:
    classifier : new classifier (nn.Sequential)
    '''
    
    
    # Return a classifier based on number of features,
    # number of hidden layers, and dropout probability
    output = ('output', nn.LogSoftmax(dim=1))
    
    classifier_spec = OrderedDict([])
    
    in_count = features_count
    
    #Number of final outputs
    out_final = 102
    
    i = 0
    
    if classifier_hidden > 0:
        #Transform the node count range to log2 scale
        feat_log = math.floor(np.log2(features_count))
        out_fin_log = math.ceil(np.log2(out_final))
        # Linearly interpolate hidden layer counts in log2
        # calculate distance
        dist = ((feat_log - out_fin_log) / (classifier_hidden + 1))
        # Build hidden layer output counts in log2 and 
        # return tonormal scale by taking exp
        hidden_outputs = [int(np.exp2(math.floor(feat_log - i * dist))) for i in range(1, classifier_hidden + 1)]
        
        # Check for duplicates
        if len(hidden_outputs) != len(set(hidden_outputs)):
            raise ValueError(str(classifier_hidden) + ' is too many hidden layers. Try 2 or 3.')
        
        
        for i in range(classifier_hidden):
            indeces = ['fc' + str(i + 1), 'relu' + str(i + 1), 'dropout' + str(i + 1)]
            classifier_spec.update({indeces[0] : nn.Linear(in_count, hidden_outputs[i])})
            classifier_spec.update({indeces[1] : nn.ReLU()})
            classifier_spec.update({indeces[2] : nn.Dropout(p=dropout_p)})
            in_count = hidden_outputs[i]
    
    # Getting final class labels
    if i > 0:
        i += 1
    classifier_spec.update({'fc' + str(i + 1) : nn.Linear(in_count, 102)})
    classifier_spec.update({'output' : nn.LogSoftmax(dim=1)})
            
    classifier = nn.Sequential(classifier_spec)
    
    return classifier