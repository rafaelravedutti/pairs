from pairs.ir.ast_node import ASTNode
from pairs.ir.assign import Assign
from pairs.ir.bin_op import BinOp, Decl, ASTTerm, VectorAccess
from pairs.ir.layouts import Layouts
from pairs.ir.lit import Lit
from pairs.ir.types import Types
from pairs.ir.vector_expr import VectorExpression


class Features:
    def __init__(self, sim):
        self.sim = sim
        self.features = []
        self.feature_properties = []

    def add(self, f_name):
        f = Feature(self.sim, f_name)
        self.features.append(f)
        return p

    def nfeatures(self):
        return len(self.features)

    def find(self, f_name):
        feature = [f for f in self.features if f.name() == f_name]
        if feature:
            return feature[0]

        return None

    def __iter__(self):
        yield from self.features


class Feature(ASTNode):
    last_feature_id = 0

    def __init__(self, sim, name):
        super().__init__(sim)
        self.feature_id = Feature.last_feature_id
        self.feature_name = name
        self.feature_count = self.sim.add_var(f"count_{self.feature_name}", Types.Int32)
        Feature.last_feature_id += 1

    def __str__(self):
        return f"Feature<{self.feature_name}>"

    def id(self):
        return self.feature_id

    def name(self):
        return self.feature_name

    def count(self):
        return self.feature_count


class FeatureAccess(ASTTerm):
    last_feat_acc = 0

    def new_id():
        PropertyAccess.last_feat_acc += 1
        return PropertyAccess.last_feat_acc - 1

    def __init__(self, sim, feature, index):
        super().__init__(sim)
        self.acc_id = FeatureAccess.new_id()
        self.feature = feature
        self.index = Lit.cvt(sim, index)
        self.inlined = False
        self.terminals = set()

    def __str__(self):
        return f"FeatureAccess<{self.feature}, {self.index}>"

    def copy(self):
        return FeatureAccess(self.sim, self.feature, self.index)

    def inline_rec(self):
        self.inlined = True
        return self

    def propagate_through(self):
        return []

    def set(self, other):
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def sub(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self - other))

    def id(self):
        return self.acc_id

    def type(self):
        return self.prop.type()

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.prop, self.index]
