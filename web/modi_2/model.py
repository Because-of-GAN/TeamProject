import torch
import torch.nn as nn
import torch.nn.functional as F


# conv function
def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))  # convolution 레이어입니다.
    if bn:
        layers.append(nn.BatchNorm2d(c_out))  # batch normalization 레이어를 추가해줍니다.
    return nn.Sequential(*layers)


# deconv function
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        # Unet encoder
        self.conv1 = conv(3, 64, 4, bn=False)
        self.conv2 = conv(64, 128, 4)
        self.conv3 = conv(128, 256, 4)
        self.conv4 = conv(256, 512, 4)
        self.conv5 = conv(512, 512, 4)
        self.conv6 = conv(512, 512, 4)
        self.conv7 = conv(512, 512, 4)
        self.conv8 = conv(512, 512, 4, bn=False)

        # Unet decoder
        self.deconv1 = deconv(512, 512, 4)
        self.deconv2 = deconv(1024, 512, 4)
        self.deconv3 = deconv(1024, 512, 4)
        self.deconv4 = deconv(1024, 512, 4)
        self.deconv5 = deconv(1024, 256, 4)
        self.deconv6 = deconv(512, 128, 4)
        self.deconv7 = deconv(256, 64, 4)
        self.deconv8 = deconv(128, 3, 4)

    # forward method
    def forward(self, input):
        # Unet encoder
        e1 = self.conv1(input)
        e2 = self.conv2(F.leaky_relu(e1, 0.2))
        e3 = self.conv3(F.leaky_relu(e2, 0.2))
        e4 = self.conv4(F.leaky_relu(e3, 0.2))
        e5 = self.conv5(F.leaky_relu(e4, 0.2))
        e6 = self.conv6(F.leaky_relu(e5, 0.2))
        e7 = self.conv7(F.leaky_relu(e6, 0.2))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))

        # Unet decoder
        d1 = F.dropout(self.deconv1(F.relu(e8)), 0.5, training=True)
        d2 = F.dropout(self.deconv2(F.relu(torch.cat([d1, e7], 1))), 0.5, training=True)
        d3 = F.dropout(self.deconv3(F.relu(torch.cat([d2, e6], 1))), 0.5, training=True)
        d4 = self.deconv4(F.relu(torch.cat([d3, e5], 1)))
        d5 = self.deconv5(F.relu(torch.cat([d4, e4], 1)))
        d6 = self.deconv6(F.relu(torch.cat([d5, e3], 1)))
        d7 = self.deconv7(F.relu(torch.cat([d6, e2], 1)))
        d8 = self.deconv8(F.relu(torch.cat([d7, e1], 1)))
        output = torch.tanh(d8)

        return output
