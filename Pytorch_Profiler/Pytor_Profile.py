import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity

# Define the SimpleModel class
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_data = torch.randn(64, 1, 28, 28)
dataset = TensorDataset(input_data)
dataloader = DataLoader(dataset, batch_size=64)

# here we are taking model for normal CPU Profiler Results
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# here we are taking model1 for CUDA Profiler Results
model1 = SimpleModel()
model1.cuda() 
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model.parameters(), lr=0.001)

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof_cpu:
    with record_function("cpu_model_inference"):
        for data in dataloader:
            inputs = data[0]  
            outputs = model(inputs)
            loss = criterion(outputs, torch.zeros(64, dtype=torch.long))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof_cuda:
    with record_function("cuda_model_inference"):
        for data in dataloader:
            inputs = data[0].cuda()  
            outputs = model1(inputs)
            loss = criterion1(outputs, torch.zeros(64, dtype=torch.long).cuda())  
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()

print("CPU Profiler Results:")
print(prof_cpu.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=5))

print("\nCUDA Profiler Results:")
print(prof_cuda.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=5))

prof_cpu.export_chrome_trace("cpu_trace.json")
prof_cuda.export_chrome_trace("cuda_trace.json")



with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=lambda trace: trace.export_chrome_trace("/tmp/trace_{}.json".format(trace.step_num))
) as p:
    for idx in range(10):  
        inputs = input_data.cuda()  
        model1(inputs)

print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))

