import torch


def detach_and_clone(x: torch.Tensor, requires_grad: bool) -> torch.Tensor:
    """In some cases, a derived scalar may be computed by applying a function to
    some activations, and running .backward() on the output, with some tensors
    desired to be backprop'ed through and some not. This function is for that:
    it detaches and clones the input tensor such that it doesn't interfere with
    other places those activations are used, and so that the gradient information
    is cleared. It then sets requires_grad to the desired value based on whether this
    activation should be backprop'ed through."""
    return x.detach().clone().requires_grad_(requires_grad)
