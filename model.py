import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Add padding to retain spatial information and keep output sizes
        # consistent after the convolutions.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout2d(0.25)  # Dropout after conv layers
        # With padding the output feature map of the second pooling layer is
        # 7x7. Adjust the linear layer input dimension accordingly.
        self.fc1 = nn.Linear(3136, 128)
        self.dropout2 = nn.Dropout(0.5)       # Dropout after first fully connected layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.dropout1(x)
        # 64 channels * 7 * 7 from the padded convolutions
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)