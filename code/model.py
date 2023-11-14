import torch.nn as nn


# MLP model
class KimchiMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes = [1024, 800, 400], num_classes = 11):
        super(KimchiMLP, self).__init__()
        
        # Define the layers for the MLP
        layers = []
        in_features = input_size # initial size: 3*64*64 = 12288
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, num_classes))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        # Forward pass of the MLP model
        x = x.view(x.shape[0], -1) # flatten
        
        out = self.mlp(x)
        return out

# CNN model
class KimchiCNN(nn.Module):
    def __init__(self, num_classes = 11):
        # Define the layers and architecture of the CNN model
        super(KimchiCNN, self).__init__()
        
        # Define the layers for the CNN
        # 3*64*64 -> 16*62*62
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16*31*31
        
        # 16*31*31 -> 32*29*29
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32*14*14
        
        # Depending on the size of dataset, more layer can be added. Check for overfitting by cross-validation.
        # # 32*14*14 -> 64*12*12
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        # self.relu3 = nn.ReLU()
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 64*6*6
        
        # # 64*6*6 -> 128*4*4
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        # self.relu4 = nn.ReLU()
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 128*2*2
        
        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=0)
        # self.relu5 = nn.ReLU()
        
        # fix the value after tuning
        self.fc1 = nn.Linear(32*14*14, 3140)  # (input_features, output_features)  
        self.fc2 = nn.Linear(3140, num_classes)
        
    def forward(self, x):
        # Forward pass of the CNN model
        x = self.conv1(x) # [batch size, feature, width, height] = [64, 16, 62, 62]
        x = self.relu1(x)
        x = self.maxpool1(x) # [64, 16, 31, 31]
        
        x = self.conv2(x) # [64, 32, 29, 29]
        x = self.relu2(x)
        x = self.maxpool2(x) # [64, 32, 14, 14]
        
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.maxpool3(x)
        
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.maxpool4(x)
        
        # x = self.conv5(x)
        # x = self.relu5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x