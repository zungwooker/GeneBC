from .mlp import *
from torchvision.models import resnet18, ResNet18_Weights

# cmnist: 10 classes | MLP
# bffhq: 2 classes | no-pretrained resnet18
# bar: 6 classes | pretrained resnet18
# dogs_and_cats: 2 classes | no-pretrained resnet18

def get_model(dataset: str):
    if dataset == 'cmnist':
        model = MLP(num_classes=10)
    elif dataset == 'bffhq':
        model = resnet18()
        model.fc = nn.Linear(512, 2)
    elif dataset == 'bar':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(512, 6)
    elif dataset == 'dogs_and_cats':
        model = resnet18()
        model.fc = nn.Linear(512, 2)
    else:
        raise KeyError("Choose one of the four datasets: cmnist, bffhq, bar, dogs_and_cats")

    return model