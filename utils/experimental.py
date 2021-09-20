import torch
import torch.nn as nn
import geotorch

input_ones = torch.ones([10, 10])
specific_input = torch.tensor([[1, 2, 3], [4, 5, 6]])
transposed = torch.transpose(specific_input, 0, 1)

lowrank = torch.svd_lowrank(input_ones, 10)

back = torch.transpose(lowrank[0], 0, 1) * lowrank[1] * lowrank[2]


if __name__ == '__main__':
    print(f'the input matrix:\n {input_ones}')
    print(f'the transposed matrix:\n {transposed}')
    print(f'size of the lowrank matrix:\n {[i.shape for i in lowrank]}')
    print(f'here is the lowrank matrix:\n {lowrank}')
    print(f'SVD matrices multiplied:\n {back}')

