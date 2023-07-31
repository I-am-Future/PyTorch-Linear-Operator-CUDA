# Implement a three layer MLP to classify MNIST digits.
#

# The MLP has 784 input features, 256 hidden features, and 10 output features.

import torch
import torch.nn as nn
import mylinearops
import torchvision

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = mylinearops.LinearLayer(784, 256, bias=True).cuda()
        self.linear2 = mylinearops.LinearLayer(256, 256, bias=True).cuda()
        self.linear3 = mylinearops.LinearLayer(256, 10, bias=True).cuda()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.softmax(self.linear3(x))
        return x

# Load the MNIST dataset.
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
mnist_dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=128, shuffle=True, num_workers=4)

# Create the MLP model.
mlp = MLP().cuda()

# Create the optimizer.
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)

# Create the loss function.
critertion = nn.CrossEntropyLoss()

# Train the MLP for 10 epochs. Record the accuracy and loss.
for epoch in range(10):
    for i, (images, labels) in enumerate(mnist_dataloader):
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = mlp(images)
        
        logits = torch.argmax(outputs, dim=1)
        acc = torch.sum(logits == labels).item() / 128

        loss = critertion(outputs, labels)
        loss.backward()

        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Acc: %.4f'
                  % (epoch+1, 10, i+1, len(mnist_dataset)//128, loss.item(), acc))



