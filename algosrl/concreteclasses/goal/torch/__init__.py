from torch import device as torch_device
from torch import cuda as torch_cuda

device = torch_device('cuda' if torch_cuda.is_available() else 'cpu')
