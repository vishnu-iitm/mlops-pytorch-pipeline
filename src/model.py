import torch.nn as nn
from torchvision.models import resnet18

def get_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == 'resnet18':
        # get standard resnet18
        model = resnet18(weights='DEFAULT')
        
        # tweak for small cifar images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        
        # swap out the final layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        return model
    
    # fallback for unsupported archs
    raise ValueError(f"architecture {architecture} not supported yet")
