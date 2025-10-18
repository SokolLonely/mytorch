from ... import Tensor
def l1_loss(pred: Tensor, target: Tensor) -> Tensor: # aka MAE loss
    diff = pred - target
    abs_diff = diff.relu() + (-diff).relu()  # |x| = ReLU(x) + ReLU(-x)
    return abs_diff.mean()
