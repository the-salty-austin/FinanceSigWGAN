import torch
import torch.nn as nn
from torchviz import make_dot

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Create an instance of the model
model = SimpleNet()

# Dummy input data
dummy_input = torch.randn(1, 4)  # Input data with shape (batch_size, input_dim)

# Generate the visualization
output = model(dummy_input)
dot = make_dot(output, params=dict(model.named_parameters()))

# Save the visualization to a file (e.g., PNG format)
dot.render("simple_net", format="png")

# Alternatively, you can display the visualization directly in Jupyter Notebook
# dot.view()
