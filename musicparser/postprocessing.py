

import numpy as np
from collections import defaultdict
import re
import torch


############### From d_tree to c_tree ####################

class Node:
    def __init__(self,label):
        self.label = label
        self.children = []
        self.parent = None

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def set_children(self, node1, node2):
        self.children = [node1, node2]
        node1.parent = self
        node2.parent = self

    def unlabeled_repr(self):
        if len(self.children) == 0:
            return self.label
        else:
            return [self.children[0].unlabeled_repr(),self.children[1].unlabeled_repr()]

    def __str__(self):
        return str(self.unlabeled_repr())

    def children_repr(self):
        if len(self.children) == 0:
            return [-1,-1]
        else:
            return [self.children[0].label,self.children[1].label]

    def parent_repr(self):
        if self.parent is None:
            return -1
        else:
            return self.parent.label

    def child_parent_repr(self):
        return f"P{self.parent_repr()}C{[self.children_repr()]}"



def dtree2unlabeled_ctree(d_arcs, check_single_root = False):
    # build head dictionary
    d = defaultdict(list)
    for arc in d_arcs.tolist():
        if not(arc[0] ==0 and arc[1]==0): # don't append self loop 0 to 0
            d[arc[0]].append(arc[1])
    # ensure the lists in the dictionary are sorted
    for v in d.values():
        v.sort()
    # find root
    potential_roots = d[0]
    assert len(potential_roots) == 1
    # if check_single_root:
    #     # assert(len(potential_roots) == 1)
    #     if len(potential_roots)!=1:
    #         print("MORE THAN ONE ROOT")
    root = potential_roots[0]
    # build ctree
    node_root = Node(root)

    def _recursive_topdown_build(d, node):
        # stopping condition
        if node.label not in d or len(d[node.label]) == 0:
            return
        else:
        # the indices of dependent will be all smaller or all bigger than root, so we can check the first one
            if d[node.label][0] < node.label: # dependent on the left
                left_node_child = Node(d[node.label][0])
                right_node_child = Node(node.label)
                # remove the value from the list
                d[node.label].pop(0)
            else: # dependent on the right
                left_node_child = Node(node.label)
                right_node_child = Node(d[node.label][-1])
                # remove the value from the list
                d[node.label].pop()
            # set childrens and iterate on them
            node.set_children(left_node_child, right_node_child)
            _recursive_topdown_build(d, left_node_child)
            _recursive_topdown_build(d, right_node_child)
    
        
    _recursive_topdown_build(d, node_root)
    return node_root




############### Eisner algorithm for projective directed graphs ####################
# Adapted from https://github.com/HMJW/biaffine-parser
def eisner_one_root(scores):
    roots_to_try = np.arange(1,len(scores))
    for root in roots_to_try:
        scores = scores.copy()

    pass




def eisner(scores, return_probs = False):
    """Parse using Eisner's algorithm.
    The matrix follows the following convention:
        scores[i][j] = p(i=head, j=dep) = p(i --> j)
    """
    rows, collumns = scores.shape
    assert rows == collumns, 'scores matrix must be square'

    num_words = rows - 1  # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([num_words+1, num_words+1, 2])  # s, t, direction (right=1).
    incomplete = np.zeros([num_words+1, num_words+1, 2])  # s, t, direction (right=1).
    complete_backtrack = -np.ones([num_words+1, num_words+1, 2], dtype=int)  # s, t, direction (right=1).
    incomplete_backtrack = -np.ones([num_words+1, num_words+1, 2], dtype=int)  # s, t, direction (right=1).

    incomplete[0, :, 0] -= np.inf

    # Loop from smaller items to larger items.
    for k in range(1, num_words+1):
        for s in range(num_words-k+1):
            t = s + k

            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s]
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t]
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

    value = complete[0][num_words][1]
    heads = -np.ones(num_words + 1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, num_words, 1, 1, heads)

    value_proj = 0.0
    for m in range(1, num_words+1):
        h = heads[m]
        value_proj += scores[h, m]
    if return_probs:
        return heads, value_proj
    else:
        return heads


def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    """
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
    head of each word.
    """
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
        



################# Eisner fast algorithm for directed graphs ####################
# Readapted from https://github.com/HMJW/biaffine-parser/blob/e779bb1e5a12b8caf1237702465882a3ce41fe68/parser/utils/alg.py#L47

def eisner_fast(scores, mask):
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    p_i = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = scores.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i, r) + C(j, r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        ilr = ilr.permute(2, 0, 1)
        il = ilr + scores.diagonal(-w).unsqueeze(-1)
        # I(j, i) = max(C(i, r) + C(j, r+1) + S(j, i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span)
        p_i.diagonal(-w).copy_(il_path + starts)
        ir = ilr + scores.diagonal(w).unsqueeze(-1)
        # I(i, j) = max(C(i, r) + C(j, r+1) + S(i, j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span)
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j, i) = max(C(r, i) + I(j, r)), i <= r < j
        cl = stripe(s_c, n, w, dim=0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i, j) = max(I(i, r) + C(r, j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    predicts = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_ones(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads.to(mask.device))

    return predicts


def backtrack(p_i, p_c, heads, i, j, complete):
    if i == j:
        return
    if complete:
        r = p_c[i, j]
        backtrack(p_i, p_c, heads, i, r, False)
        backtrack(p_i, p_c, heads, r, j, True)
    else:
        r, heads[j] = p_i[i, j], i
        i, j = sorted((i, j))
        backtrack(p_i, p_c, heads, i, r, True)
        backtrack(p_i, p_c, heads, j, r + 1, True)


def stripe(x, n, w, offset=(0, 0), dim=1):
    r'''Returns a diagonal stripe of the tensor.
    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.
    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    '''
    seq_len = x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)


############### MST algorithm for directed graphs ####################
# Adapted from the stanza parser 
# https://github.com/stanfordnlp/stanza/blob/b18e6e80fae7cefbfed7e5255c7ba4ef6f1adae5/stanza/models/common/chuliu_edmonds.py

def tarjan(tree):
    """"""

    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    _index = [0]
    cycles = []
    #-------------------------------------------------------------
    def strong_connect(i):
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True
        dependents = np.where(np.equal(tree, i))[0]
        for j in dependents:
            if indices[j] == -1:
                strong_connect(j)
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            elif onstack[j]:
                lowlinks[i] = min(lowlinks[i], indices[j])

        # There's a cycle!
        if lowlinks[i] == indices[i]:
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)
        return
    #-------------------------------------------------------------
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles

def process_cycle(tree, cycle, scores):
    """
    Build a subproblem with one cycle broken
    """
    # indices of cycle in original tree; (c) in t
    cycle_locs = np.where(cycle)[0]
    # heads of cycle in original tree; (c) in t
    cycle_subtree = tree[cycle]
    # scores of cycle in original tree; (c) in R
    cycle_scores = scores[cycle, cycle_subtree]
    # total score of cycle; () in R
    cycle_score = cycle_scores.sum()

    # locations of noncycle; (t) in [0,1]
    noncycle = np.logical_not(cycle)
    # indices of noncycle in original tree; (n) in t
    noncycle_locs = np.where(noncycle)[0]
    #print(cycle_locs, noncycle_locs)

    # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
    metanode_head_scores = scores[cycle][:,noncycle] - cycle_scores[:,None] + cycle_score
    # scores of cycle's potential dependents; (n x c) in R
    metanode_dep_scores = scores[noncycle][:,cycle]
    # best noncycle head for each cycle dependent; (n) in c
    metanode_heads = np.argmax(metanode_head_scores, axis=0)
    # best cycle head for each noncycle dependent; (n) in c
    metanode_deps = np.argmax(metanode_dep_scores, axis=1)

    # scores of noncycle graph; (n x n) in R
    subscores = scores[noncycle][:,noncycle]
    # pad to contracted graph; (n+1 x n+1) in R
    subscores = np.pad(subscores, ( (0,1) , (0,1) ), 'constant')
    # set the contracted graph scores of cycle's potential heads; (c x n)[:, (n) in n] in R -> (n) in R
    subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
    # set the contracted graph scores of cycle's potential dependents; (n x c)[(n) in n] in R-> (n) in R
    subscores[:-1,-1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]
    return subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps


def expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps):
    """
    Given a partially solved tree with a cycle and a solved subproblem
    for the cycle, build a larger solution without the cycle
    """
    # head of the cycle; () in n
    #print(contracted_tree)
    cycle_head = contracted_tree[-1]
    # fixed tree: (n) in n+1
    contracted_tree = contracted_tree[:-1]
    # initialize new tree; (t) in 0
    new_tree = -np.ones_like(tree)
    #print(0, new_tree)
    # fixed tree with no heads coming from the cycle: (n) in [0,1]
    contracted_subtree = contracted_tree < len(contracted_tree)
    # add the nodes to the new tree (t)[(n)[(n) in [0,1]] in t] in t = (n)[(n)[(n) in [0,1]] in n] in t
    new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[contracted_tree[contracted_subtree]]
    #print(1, new_tree)
    # fixed tree with heads coming from the cycle: (n) in [0,1]
    contracted_subtree = np.logical_not(contracted_subtree)
    # add the nodes to the tree (t)[(n)[(n) in [0,1]] in t] in t = (c)[(n)[(n) in [0,1]] in c] in t
    new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
    #print(2, new_tree)
    # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
    new_tree[cycle_locs] = tree[cycle_locs]
    #print(3, new_tree)
    # root of the cycle; (n)[() in n] in c = () in c
    cycle_root = metanode_heads[cycle_head]
    # add the root of the cycle to the new tree; (t)[(c)[() in c] in t] = (c)[() in c]
    new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
    #print(4, new_tree)
    return new_tree

def prepare_scores(scores):
    """
    Alter the scores matrix to avoid self loops and handle the root
    """
    # prevent self-loops, set up the root location
    np.fill_diagonal(scores, -float('inf')) # prevent self-loops
    scores[0] = -float('inf')
    scores[0,0] = 0

def chuliu_edmonds(scores):
    subtree_stack = []

    prepare_scores(scores)
    tree = np.argmax(scores, axis=1)
    cycles = tarjan(tree)

    #print(scores)
    #print(cycles)

    # recursive implementation:
    #if cycles:
    #    # t = len(tree); c = len(cycle); n = len(noncycle)
    #    # cycles.pop(): locations of cycle; (t) in [0,1]
    #    subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps = process_cycle(tree, cycles.pop(), scores)
    #    # MST with contraction; (n+1) in n+1
    #    contracted_tree = chuliu_edmonds(subscores)
    #    tree = expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps)
    # unfortunately, while the recursion is simpler to understand, it can get too deep for python's stack limit
    # so instead we make our own recursion, with blackjack and (you know how it goes)

    while cycles:
        # t = len(tree); c = len(cycle); n = len(noncycle)
        # cycles.pop(): locations of cycle; (t) in [0,1]
        subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps = process_cycle(tree, cycles.pop(), scores)
        subtree_stack.append((tree, cycles, scores, subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps))

        scores = subscores
        prepare_scores(scores)
        tree = np.argmax(scores, axis=1)
        cycles = tarjan(tree)

    while len(subtree_stack) > 0:
        contracted_tree = tree
        (tree, cycles, scores, subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps) = subtree_stack.pop()
        tree = expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps)

    return tree

#===============================================================
def chuliu_edmonds_one_root(scores):
    """
    Find the maximum spanning tree using Chu-Liu-Edmonds algorithm, without knowing the root.
    This takes an adgency matrix with log probs, encoding edges that point toward the parent. 
    """
    assert((scores[0]==-float('inf')).all())
    assert((np.diagonal(scores)== -float('inf')).all())  
    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0]+1
    if len(roots_to_try) == 1:
        return tree

    #-------------------------------------------------------------
    def set_root(scores, root):
        root_score = scores[root,0]
        scores = np.array(scores)
        scores[1:,0] = -float('inf')
        scores[root] = -float('inf')
        scores[root,0] = 0
        return scores, root_score
    #-------------------------------------------------------------

    best_score, best_tree = -np.inf, None # This is what's causing it to crash
    for root in roots_to_try:
        _scores, root_score = set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = (tree_probs).sum()+(root_score) if (tree_probs > -np.inf).all() else -np.inf
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree
    try:
        assert best_tree is not None
    except:
        with open('debug.log', 'w') as f:
            f.write('{}: {}, {}\n'.format(tree, scores, roots_to_try))
            f.write('{}: {}, {}, {}\n'.format(_tree, _scores, tree_probs, tree_score))
        raise
    return best_tree