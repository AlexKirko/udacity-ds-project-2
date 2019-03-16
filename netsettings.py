#Set up project settings as dictionaries
from torchvision import models

net_archs = {
    'AlexNet' : models.alexnet,
    'VGG-11 BN' : models.vgg11_bn,
    'VGG-13 BN' : models.vgg13_bn,
    'VGG-16 BN' : models.vgg16_bn,
    'VGG-19 BN' : models.vgg19_bn,
    'Inception V3' : models.inception_v3
}

net_classes = {
    'VGG-11 BN' : 'VGG',
    'VGG-13 BN' : 'VGG',
    'VGG-16 BN' : 'VGG',
    'VGG-19 BN' : 'VGG',
    'AlexNet' : 'AlexNet',
    'Inception V3' : 'Inception'
}

net_prefixes = {
    'VGG-11 BN' : 'vgg11_bn_',
    'VGG-13 BN' : 'vgg13_bn_',
    'VGG-16 BN' : 'vgg16_bn_',
    'VGG-19 BN' : 'vgg19_bn_',
    'AlexNet' : 'alexnet_',
    'Inception V3' : 'inception_v3_'
}

net_params = {
    'AlexNet' : {
        'learning_rate' : 0.001,
        'features_count' : 9216,
        'classifier_hidden' : 2,
        'batch_size' : 64,
        'classifier_name' : 'classifier',
        'img_resize' : 255,
        'img_crop' : 224
    },
    'Inception' : {
        'learning_rate' : 0.001,
        'features_count' : 2048,
        'classifier_hidden' : 0,
        'batch_size' : 64,
        'classifier_name' : 'fc',
        'img_resize' : 400,
        'img_crop' : 299
    },
    'VGG' : {
        'learning_rate' : 0.001,
        'features_count' : 25088,
        'classifier_hidden' : 2,
        'batch_size' : 64,
        'classifier_name' : 'classifier',
        'img_resize' : 255,
        'img_crop' : 224
    }
}