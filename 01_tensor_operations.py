import torch

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

