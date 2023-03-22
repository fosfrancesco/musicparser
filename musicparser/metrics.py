from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.classification.stat_scores import Metric
import torch
import re

class VariableMulticlassAccuracy(Metric):
    r"""Computes `Accuracy`_ for multiclass tasks where every example can have a variable number of classes
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.add_state("accuracy", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        mask = target != self.ignore_index
        preds = preds[mask]
        target = target[mask]

        self.accuracy += torch.sum(preds == target).float() / target.numel()
        self.samples += 1

    def compute(self):
        return self.accuracy.float() / self.samples
    

class CTreeSpanSimilarity(Metric):
    r"""Computes CTreeSpanSimilarity
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("span_sim", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        self.span_sim += ctree_span_similarity(preds, target)
        self.samples += 1

    def compute(self):
        return self.span_sim / self.samples
    

############### Evaluation of c_tree, functionals ####################

def get_all_spans(tree):
    all_spans = []
    def _recursive_get_all_spans(node):
        # if node is a leaf, end recursion
        if len(node.children) == 0:
            return 
        else:
            # we traverse tree1, and match string with tree2             
            for child in node.children:
                _recursive_get_all_spans(child)
            # add information of the current node
            # compute the span of the subtree rooted at the current node
            digits = [i for i in re.split(r'(\D+)', str(node)) if i.isdigit()]
            spans = (digits[0], digits[-1])
            all_spans.append(spans)
        
    _recursive_get_all_spans(tree)
    return all_spans

def ctree_span_similarity(pred, truth, return_intersection=False):
    # compute all spans of truth and pred
    spans_truth = get_all_spans(truth)
    spans_pred = get_all_spans(pred)
    
    # find matching spans
    set_span_pred = set(spans_pred) # for computational reason, faster if it's a set
    intersection = [span for span in spans_truth if span in set_span_pred]
    
    if return_intersection:
        return len(intersection)/len(spans_truth), intersection
    else:
        return len(intersection)/len(spans_truth)