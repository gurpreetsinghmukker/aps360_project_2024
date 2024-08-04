import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

class GTZANModel(nn.Module):
    def __init__(self, num_classes):
        super(GTZANModel, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.kernel_size = 3

        self.conv_layers = []
        self.conv1 = nn.Conv2d(1, 32, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv1, self.relu, nn.BatchNorm2d(32), self.pool])
        self.conv2 = nn.Conv2d(32, 64, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv2, self.relu, nn.BatchNorm2d(64), self.pool])
        self.conv3 = nn.Conv2d(64, 128, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv3, self.relu, nn.BatchNorm2d(128), self.pool])
        self.conv4 = nn.Conv2d(128, 64, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv4, self.relu, nn.BatchNorm2d(64), self.pool])

        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6656, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.name = f"GTZANModel_KS({self.kernel_size})"

    def forward(self, x):
        x = self.conv_layers(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AudioClassifier (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        conv_layers = []
        self.name = "AudioClassifier"
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        # init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        # init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        # init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        # init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=num_classes)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


# ======== Contrastive Models ======== #
class GTZANContrastiveModel(nn.Module):
    def __init__(self, contrastive_dim=128):
        super(GTZANContrastiveModel, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.kernel_size = 3

        self.conv_layers = []
        self.conv1 = nn.Conv2d(1, 32, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv1, self.relu, nn.BatchNorm2d(32), self.pool])
        self.conv2 = nn.Conv2d(32, 64, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv2, self.relu, nn.BatchNorm2d(64), self.pool])
        self.conv3 = nn.Conv2d(64, 128, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv3, self.relu, nn.BatchNorm2d(128), self.pool])
        self.conv4 = nn.Conv2d(128, 64, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv4, self.relu, nn.BatchNorm2d(64), self.pool])

        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6656, 512)

        self.fc2 = nn.Linear(512, contrastive_dim)
        self.name = f"GTZANContrastiveModel_CONT_DIM({contrastive_dim})_({self.kernel_size})"

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GTZANContrastiveModelLarge(nn.Module):
    def __init__(self, contrastive_dim=128):
        super(GTZANContrastiveModelLarge, self).__init__()
        self.name = "GTZANContrastiveModelLarge"
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.kernel_size = 3

        self.conv_layers = []
        self.conv1 = nn.Conv2d(1, 32, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv1, self.relu, nn.BatchNorm2d(32), self.pool])
        self.conv2 = nn.Conv2d(32, 64, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv2, self.relu, nn.BatchNorm2d(64), self.pool])
        self.conv3 = nn.Conv2d(64, 128, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv3, self.relu, nn.BatchNorm2d(128), self.pool])
        self.conv4 = nn.Conv2d(128, 256, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv4, self.relu, nn.BatchNorm2d(256), self.pool])
        self.conv5 = nn.Conv2d(256, 512, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv5, self.relu, nn.BatchNorm2d(512), self.pool])

        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13312, 1024)

        self.backbone = nn.Sequential(
            self.conv_layers,
            self.flatten,
            self.fc1,
            self.relu,
            self.dropout
        )

        self.projective_head = nn.Linear(1024, contrastive_dim)
        self.name = f"GTZANContrastiveModelLarge_CONT_DIM({contrastive_dim})_({self.kernel_size})"


    def forward(self, x):
        x = torch.cat(x, dim=0)
        x = self.backbone(x)
        x = self.projective_head(x)
        x1, x2 = torch.split(x, x.shape[0] // 2, dim=0)
        print(x1.shape)
        print(x2.shape)
        return(x1, x2)
        
    def inference(self, x):
        return self.projective_head(self.backbone(x))
            

# Construct a new model that has the contrastive embedding model as a frozen base
class ContrastiveClassificationModel(nn.Module):
    def __init__(self, base_model, num_classes, embedding_dim=128):
        super(ContrastiveClassificationModel, self).__init__()
        self.name = "ContrastiveClassificationModel"
        self.base_model = base_model
        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x
    
# Construct a new model that has the contrastive embedding model as a frozen base
class ContrastiveClassificationModel_2(nn.Module):
    def __init__(self, base_model, num_classes, embedding_dim=128):
        super(ContrastiveClassificationModel_2, self).__init__()
        self.name = "ContrastiveClassificationModel_2"
        self.base_model = base_model
        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.base_model.inference(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) 
        return x
    


class BarlowTwinContrastive(nn.Module):
    def __init__(self, contrastive_dim=128):
        super(BarlowTwinContrastive, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.kernel_size = 3

        self.conv_layers = []
        self.conv1 = nn.Conv2d(1, 32, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv1, self.relu, nn.BatchNorm2d(32), self.pool])
        self.conv2 = nn.Conv2d(32, 64, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv2, self.relu, nn.BatchNorm2d(64), self.pool])
        self.conv3 = nn.Conv2d(64, 128, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv3, self.relu, nn.BatchNorm2d(128), self.pool])
        self.conv4 = nn.Conv2d(128, 256, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv4, self.relu, nn.BatchNorm2d(256), self.pool])
        self.conv5 = nn.Conv2d(256, 512, kernel_size = self.kernel_size, stride=1, padding=1)
        self.conv_layers.extend([self.conv5, self.relu, nn.BatchNorm2d(512), self.pool])
        
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13312, 1024)

        self.backbone = nn.Sequential(
            self.conv_layers,
            self.flatten,
            self.fc1,
            self.relu,
            self.dropout
        )

        self.projective_head = nn.Linear(1024, contrastive_dim)
        self.name = f"BarlowTwinContrastive_CONT_DIM({contrastive_dim})_KS({self.kernel_size})"

    def forward(self, x):
            
        x1 = x[0]
        x2 = x[1]
        z1 = self.projective_head(self.backbone(x1))
        z2 = self.projective_head(self.backbone(x2))
        print(z1.shape)
        print(z2.shape)
        return z1,z2
        
    def inference(self, x):
        return self.projective_head(self.backbone(x))