from ... import Tensor
def MSELoss( logits :Tensor, targets:Tensor):
    out = (logits-targets)*(logits-targets)
    return out#.mean