import torch
import torch.nn as nn

# Efficient Channel Attention (ECA) Module
class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((torch.log(torch.tensor(channels, dtype=torch.float32)) / torch.log(torch.tensor(2.0))) + b) / gamma)
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling to capture channel-wise statistics
        y = torch.mean(x, dim=(2, 3), keepdim=True)  # Global pooling across H, W
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# ResNet Bottleneck Block with ECA Module
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.eca = ECAModule(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# CCNet Model
class CCNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(CCNet, self).__init__()
        self.in_planes = 64

        # Initial convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers with Bottleneck blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(256, 128, 2, stride=2)
        self.layer3 = self._make_layer(512, 256, 2, stride=2)
        self.layer4 = self._make_layer(1024, 512, 2, stride=2)

        # Adaptive average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Function to instantiate the CCNet model
def CCNet(pretrained=False, num_classes=1000):
    model = CCNet(num_classes=num_classes)
    return model
