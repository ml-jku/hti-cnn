import math

from torch.autograd import Variable
from torch.nn.init import _calculate_fan_in_and_fan_out


def selu_uniform(tensor, gain=1):
    """SELU initialization drawn from uniform distribution"""
    if isinstance(tensor, Variable):
        selu_uniform(tensor.data, gain=gain)
        return tensor

    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1.0 / fan_in)
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-a, a)


def selu_normal(tensor, gain=1):
    """SELU initialization drawn from normal distribution"""
    if isinstance(tensor, Variable):
        selu_normal(tensor.data, gain=gain)
        return tensor

    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1.0 / fan_in)
    return tensor.normal_(0, std)
