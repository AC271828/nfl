class ProbTree:
    def __init__(self, triads):
        self.triads = triads
        if triads:
            self.leaf_value = sum([triad[1] for triad in self.triads])/(sum([triad[2] for triad in self.triads]))
        else: self.leaf_value = 0
        self.trip = None
        self.d_idx = None
        self.under = None
        self.over = None
        self.end = False
    def passthrough(self):
        unders, overs = [], []
        for triad in self.triads:
            if triad[0][self.d_idx] < self.trip: unders.append(triad)
            else: overs.append(triad)
        return unders, overs
    def reset(self):
        self.trip = None
        self.d_idx = None
        self.under = None
        self.over = None
        self.end = False
    def branch(self, ops=False):
        if not self.triads or self.end:
            return
        if self.under and self.over:
            self.under.branch()
            self.over.branch()
            return
        
        if ops:
            step = len(self.triads)//ops
        else: step = 0

        no_split = True
        best_d_idx = None
        best_trip = None
        best_variance = float('inf')
        for d_idx in range(len(self.triads[0][0])):
            sorted_triads = sorted(list(self.triads), key=lambda x: x[0][d_idx])
            sorted_feats = np.array([t[0][d_idx] for t in sorted_triads])
            sorted_vals = np.array([t[1] for t in sorted_triads])
            variances = []
            for split_idx in range(0, len(sorted_feats), step):
                unders = sorted_vals[:split_idx]
                overs = sorted_vals[split_idx:]
                u_mean = np.mean(unders) if len(unders) else 0
                o_mean = np.mean(overs) if len(overs) else 0
                variances.append((split_idx, np.sum((unders - u_mean)**2) + np.sum((overs - o_mean)**2)))
            
            best_variance_tuple = min(variances, key=lambda x: x[1])
            low_index = best_variance_tuple[0]
            if best_variance_tuple[1] < best_variance:
                best_variance = best_variance_tuple[1]
                best_d_idx = d_idx
                if low_index != 0:
                    best_trip = (sorted_feats[low_index] + (sorted_feats[low_index - 1])) / 2
                    no_split = False
                else: best_trip = float('-inf')

        if not no_split:
            self.d_idx = best_d_idx
            self.trip = best_trip

            split = self.passthrough()
            
            self.under = ProbTree(split[0])
            self.over = ProbTree(split[1])
        else: self.end = True
    def leaf(self, features):
        if self.under and self.over:
            if features[self.d_idx] < self.trip: return self.under.leaf(features)
            else: return self.over.leaf(features)
        else: return self.leaf_value
    def listsave(self):
        if self.under and self.over:
            return [(self.d_idx, self.trip), self.under.listsave(), self.over.listsave()]
        else: return (self.leaf_value)

sig = lambda x: 1/(1+np.e**(-x))
logit = lambda x: np.log(x/(1-x))

class ProbBoostTree:
    def __init__(self, pairs):
        self.pairs = pairs
        self.trees = []
        self.functions = [lambda x: logit(np.mean([pair[1] for pair in pairs]))]
    def residuals(self):
        residuals = []
        for pair in self.pairs:
            pred = self.evaluate(pair[0])
            residuals.append((pair[0],pair[1]-pred, pred * (1-pred)))
        return residuals
    def build(self, branch_iters, lr, ops=1): 
        new_tree = ProbTree(self.residuals())
        for _ in range(branch_iters):
            new_tree.branch(ops=ops)
        self.trees.append((lr, new_tree))
        self.functions.append(lambda x: (lr * new_tree.leaf(x)))
    def evaluate(self, features):
        return sig(sum([func(features) for func in self.functions]))
    def listsave(self):
        return [self.functions[0](0), [(tree[0],tree[1].listsave()) for tree in self.trees]]

def boost_list_evaluate(boost_list, features):
    total = boost_list[0] + sum([branch_list[0] * branch_list_leaf(branch_list[1], features) for branch_list in boost_list[1]])
    return sig(total)
    