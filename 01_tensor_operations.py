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
#print(torch.matmul(tensor1, tensor2.transpose(dim0=0, dim1=1)))
