import torch
import torch.nn as nn

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    """ """

    def __init__(self, features, num_classes=3, init_weights=True, **kwargs):
        """

        Parameters
        ----------
        features :
            
        num_classes :
            (Default value = 3)
        init_weights :
            (Default value = True)
        **kwargs :
            

        Returns
        -------

        
        """
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2*2*2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        """
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7* 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        """
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """

        Parameters
        ----------
        x :
            

        Returns
        -------

        
        """
        x = self.features(x)
        #print("VGG",x.shape)
        #x = self.avgpool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """ """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    """

    Parameters
    ----------
    cfg :
        
    batch_norm :
        (Default value = False)

    Returns
    -------

    
    """
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    """

    Parameters
    ----------
    arch :
        
    cfg :
        
    batch_norm :
        
    pretrained :
        
    progress :
        
    **kwargs :
        

    Returns
    -------

    
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet (Default value = False)
    progress : bool
        If True, displays a progress bar of the download to stderr (Default value = True)
    **kwargs :
        

    Returns
    -------

    
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)



def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet (Default value = False)
    progress : bool
        If True, displays a progress bar of the download to stderr (Default value = True)
    **kwargs :
        

    Returns
    -------

    
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)



def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet (Default value = False)
    progress : bool
        If True, displays a progress bar of the download to stderr (Default value = True)
    **kwargs :
        

    Returns
    -------

    
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)



def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet (Default value = False)
    progress : bool
        If True, displays a progress bar of the download to stderr (Default value = True)
    **kwargs :
        

    Returns
    -------

    
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)



def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet (Default value = False)
    progress : bool
        If True, displays a progress bar of the download to stderr (Default value = True)
    **kwargs :
        

    Returns
    -------

    
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)



def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet (Default value = False)
    progress : bool
        If True, displays a progress bar of the download to stderr (Default value = True)
    **kwargs :
        

    Returns
    -------

    
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)



def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet (Default value = False)
    progress : bool
        If True, displays a progress bar of the download to stderr (Default value = True)
    **kwargs :
        

    Returns
    -------

    
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)



def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet (Default value = False)
    progress : bool
        If True, displays a progress bar of the download to stderr (Default value = True)
    **kwargs :
        

    Returns
    -------

    
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)