import torch
from torch import nn


from torchvision.models import resnet50, densenet169, vgg16_bn
from torchvision.models import ResNet50_Weights, DenseNet169_Weights, VGG16_BN_Weights


# Ensemble learning network
class Ensemble(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Ensemble, self).__init__()

        # Backbone networks
        self.backbone_vgg = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        self.backbone_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone_densenet = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)

        self.backbone_vgg = nn.Sequential(*list(self.backbone_vgg.children())[:-1])
        self.backbone_resnet = nn.Sequential(*list(self.backbone_resnet.children())[:-1])
        self.backbone_densenet = nn.Sequential(*list(self.backbone_densenet.children())[:-1])

        # Feature projection modules
        self.linear_vgg = nn.Linear(in_features=25088, out_features=n_features)
        self.linear_resnet = nn.Linear(in_features=2048, out_features=n_features)
        self.linear_densenet = nn.Linear(in_features=1664, out_features=n_features)

        # MLP layers
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(3 * n_features, n_features)
        self.linear2 = nn.Linear(n_features, n_features // 2)
        self.linear3 = nn.Linear(n_features // 2, n_features // 4)
        self.fc = nn.Linear(n_features // 4, n_classes)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)


    # Forward function implementation
    def forward(self, x):
        
        # Feature extraction and projection
        vgg_features = self.linear_vgg(self.backbone_vgg(x).view(-1, 25088))
        resnet_features = self.linear_resnet(self.backbone_resnet(x).view(-1, 2048))
        densenet_features = self.linear_densenet(nn.AdaptiveAvgPool2d((1, 1))(self.backbone_densenet(x)).view(-1, 1664))

        # MLP prediction
        x = self.relu(torch.cat([vgg_features, resnet_features, densenet_features], dim=1))
        x = self.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.relu(self.linear3(x))
        x = self.dropout3(x)
        out = self.fc(x)
        return out