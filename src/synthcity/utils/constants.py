# third party
import torch

from synthcity.utils.torch import disable_torch_dynamo

disable_torch_dynamo()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
