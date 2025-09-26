import torch
import numpy as np

#print(torch.__version__)

scalar = torch.tensor(7)

tensor = torch.tensor(range(10))

tensor = torch.tensor([[1, 6, 8],[88,4,1]])

tensor = torch.zeros([2,3,4])

tensor = torch.ones([5,3,2])

#print(tensor)
#print(tensor[1,2])

tensor = torch.rand([2,4])
#tensor = torch.randint(1, 20, [3,4])
#tensor2 = torch.rand_like(tensor)

tensor = torch.randint(1, 20, [3,4])
print(tensor.to(torch.float64))

tensor2 = torch.rand_like(tensor.float())

print(tensor)
print(tensor2)

print(tensor.ndim)
print(tensor.size())
tensor = torch.arange(2,14, 
                      dtype=torch.float16,
                      device=None,
                      requires_grad=False)
tensor = tensor.reshape_as(tensor2)
print(tensor)
print(tensor.device)
print(tensor * tensor2)
