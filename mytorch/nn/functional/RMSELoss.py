from ... import Tensor
def RMSELoss( logits :Tensor, targets:Tensor):
    out = ((logits-targets)*(logits-targets))**0.5
    return out#.mean