from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.classification.stat_scores import Metric
import torch
import re
import numpy as np

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
    
class ArcsAccuracy(Metric):
    r"""Computes `Arcs Accuracy`
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("accuracy", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if len(preds) == 0:
            self.accuracy += 0
            self.samples += 1
        else:
            view_preds = preds.numpy()
            view_target = target.numpy()
            view_preds = np.char.array(view_preds.astype(str))
            view_target = np.char.array(view_target.astype(str))
            view_preds = view_preds[:, 0] + "-" + view_preds[:, 1]
            view_target = view_target[:, 0] + "-" + view_target[:, 1]
            self.accuracy += np.sum(np.isin(view_preds,view_target)) / len(view_preds)
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
    
class CTreeNodeSimilarity(Metric):
    r"""Computes CTreeSpanSimilarity
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("node_sim", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        self.node_sim += ctree_node_similarity(preds, target)
        self.samples += 1

    def compute(self):
        return self.node_sim / self.samples
    

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

def get_all_internal_node(tree):
    all_internal_node = []
    def _recursive_get_all_internal_node(node):
        # if node is a leaf, end recursion
        if len(node.children) == 0:
            return 
        else:
            # we traverse tree1, and match string with tree2             
            for child in node.children:
                _recursive_get_all_internal_node(child)
            # add information of the current node
            # compute the span of the subtree rooted at the current node
            all_internal_node.append(node)
        
    _recursive_get_all_internal_node(tree.children[0])
    _recursive_get_all_internal_node(tree.children[1])
    return all_internal_node

def ctree_node_similarity(pred, truth, return_intersection=False):
    # compute all internal node of truth and pred
    inode_truth = get_all_internal_node(truth)
    inode_pred = get_all_internal_node(pred)
    # convert them to a numeric representation for easier comparison
    inode_truth = [n.child_parent_repr() for n in inode_truth]
    inode_pred = [n.child_parent_repr() for n in inode_pred]
    
    # find matching spans
    set_node_pred = set(inode_pred) # for computational reason, faster if it's a set
    intersection = [node for node in inode_truth if node in set_node_pred]
    
    if return_intersection:
        return len(intersection)/len(inode_truth), intersection
    else:
        return len(intersection)/len(inode_truth)