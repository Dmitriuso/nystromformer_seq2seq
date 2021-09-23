import torch
import torch.nn as nn
import geotorch

# Singular-value decomposition
from numpy import array
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# SVD
U, s, VT = svd(A)


input_ones = torch.ones([5, 5])
specific_input = torch.tensor([[1, -0.5], [-0.5, 1]], dtype=torch.float32)
transposed = torch.transpose(specific_input, 0, 1)

lowrank = torch.linalg.svd(input_ones)

input_ones_neg_pow = torch.pow(specific_input, -1)
back = lowrank[0] * torch.pow(lowrank[1], -1) * lowrank[2]


if __name__ == '__main__':
    # print(f'the input matrix:\n {input_ones}')
    # print(f'the input matrix powered -1:\n {input_ones_neg_pow}')
    # print(f'the transposed matrix:\n {transposed}')
    # print(f'size of the lowrank matrix:\n {[i.shape for i in lowrank]}')
    # print(f'here is the lowrank matrix:\n {lowrank}')
    # print(f'SVD matrices multiplied:\n {back}')
    print(f'the input matrix A:\n {A}')
    print(f'the first SVD matrix:\n {U}')
    print(f'the diagonal SVD matrix:\n {s}')
    print(f'the transposed SVD matrix:\n {VT}')


