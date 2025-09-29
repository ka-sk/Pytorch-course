import torch
import numpy as np
from random import seed

### seed is important
# https://docs.pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(23)
np.random.seed(23)
seed(23)

### basics

tensor1 = torch.randint(2, 10, [3,4])
tensor2 = torch.randint(2, 10, [3,4])

print(tensor1 + tensor2)
print(tensor1 * tensor2)
print(tensor1 - tensor2)
#shape mismach
#print(torch.matmul(tensor1, tensor2))

#check if transposition shares memory cells
print(tensor2)
tensor3 = tensor2.transpose(dim0=0, dim1=1)

tensor2[0,0] = 0
print(tensor3)
print(torch.matmul(tensor1, tensor2.transpose(dim0=0, dim1=1)))

print(tensor1.min())
print(tensor2.min())
print(tensor1.max())
print(tensor2.max())
#print(tensor1.mean())
#print(tensor2.mean())
print(tensor1.float().mean())
print(tensor2.float().mean())
print(tensor1.median())
print(tensor2.median())
print(tensor1.sum())
print(tensor2.sum())

tensor1.argmin()

'''result1, result1_idx = tensor1.min(dim=0, keepdim=False)
print(tensor1)
print(result1)
print(result1_idx)'''
tensor1 = torch.tensor(([[2, 7, 9, 3],
                         [3, 2, 0, 7],
                         [5, 8, 8, 7]]))
print(tensor1)
print(tensor1.view(-1))
min_val = tensor1.min()
min_idx = torch.argmin(tensor1)
coords = torch.unravel_index(min_idx, tensor1.shape)
print(min_idx, coords)

### Reshaping, squeezing, unsqueezing, stacking

tensor = torch.rand([3,4,5])
print(tensor.view(5,3,2,2))
print(tensor.reshape(5,2,2,3))

print("###########################################################")
print(tensor)
tensor2 = tensor.unsqueeze(dim=3)
print(tensor2)
tensor2 = tensor.unsqueeze(dim=2)
print(tensor2)
tensor2 = tensor.unsqueeze(dim=1)
print(tensor2)
tensor2 = tensor.unsqueeze(dim=0)
print(tensor2)
print(tensor2.squeeze())
print("######################################################")
tensor1 = torch.rand([5,4,3])

tensor3 = torch.stack((tensor1, tensor1))
print(tensor3.size())
tensor3 = torch.stack((tensor1, tensor1), dim=1)
print(tensor3.size())
tensor3 = torch.stack((tensor1, tensor1), dim=2)
print(tensor3.size())
tensor3 = torch.stack((tensor1, tensor1), dim=3)
print(tensor3.size())

### indexing

tensor = torch.rand(size=(2, 3, 4))
print(tensor)
print(tensor[1,1,1]) #working
#print(tensor[2,1,1]) # not working
print(tensor[1,1,2]) #working

# indexing order is corresponding to dimention order

print(tensor[1,0:2,0:2]) #2D tensor
print(tensor[0, 0, 0:2]) #1D tensor
print(tensor[0, 0]) # 1D
print(tensor[0, 0, :]) # the same 1D 
print(tensor[:, 0, 0]) # 1D
print(tensor[:, :, 0]) # 2D

### Pytorch and Numpy
array = np.random.randint(0, 20, [2,3,5])
print(type(array))
array = torch.from_numpy(array)
print(type(array))
array = array.numpy()
print(type(array))

print(array)


### Accesing GPU
# Could computing options: Google Cloud, AWS, Azure

print(torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.cuda.device_count())

tensor = torch.randint(1, 10, 
                       [2, 3, 4], 
                       device=device)
# changing the divice
tensor1.to(device)

# to change tensor into numpy device must be changed to cpu
tensor1.cpu().numpy()