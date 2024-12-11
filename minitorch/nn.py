from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Reshape and permute to create blocks
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Flatten the kernel blocks
    tiled = permuted.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width



def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to input tensor"""
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=-1)
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    return max_reduce(input, dim) == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max"""
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max"""
        input, dim = ctx.saved_values
        return argmax(input, int(dim.item())) * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input, tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor"""
    exp = input.exp()
    return exp / exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor"""
    # exp = input.exp()
    # return input - exp.sum(dim).log()
    max_values = max(input, dim)
    subtracted_input = input - max_values
    log_sum_exp = (subtracted_input.exp().sum(dim)).log()
    return subtracted_input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to input tensor"""
    tiled, new_height, new_width = tile(input, kernel)
    pooled = max_reduce(tiled, -1)
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise"""
    if ignore:
        return input
    random_drop = rand(input.shape) > p
    return input * random_drop


# minitorch.max
# minitorch.softmax
# minitorch.logsoftmax
# minitorch.maxpool2d
# minitorch.dropout
