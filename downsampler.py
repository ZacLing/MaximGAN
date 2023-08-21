import torch
from kornia.geometry.transform import resize

def bicubic_downsample(x, scale_factor=2):
    """
    Bicubic downsampling 
    """ 
    return resize(x, (x.shape[2]//scale_factor, 
                      x.shape[3]//scale_factor), 
                 interpolation='bicubic', 
                 align_corners=False)