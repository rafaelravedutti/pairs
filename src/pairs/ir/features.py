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

    def add_property(self, feature, prop):
        self.feature_properties.append([feature, prop])
        return prop

    def nfeatures(self):
        return len(self.features)

    def find(self, f_name):
        prop = [f for f in self.features if f.name() == f_name]
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
        Feature.last_feature_id += 1

    def __str__(self):
        return f"Feature<{self.feature_name}>"

    def id(self):
        return self.feature_id

    def name(self):
        return self.feature_name

    #def __getitem__(self, expr):
    #    return PropertyAccess(self.sim, self, expr)


class FeaturePropertyAccess(ASTTerm, VectorExpression):
    last_prop_acc = 0

    def new_id():
        PropertyAccess.last_prop_acc += 1
        return PropertyAccess.last_prop_acc - 1

    def __init__(self, sim, prop, index):
        super().__init__(sim)
        self.acc_id = PropertyAccess.new_id()
        self.prop = prop
        self.index = Lit.cvt(sim, index)
        self.inlined = False
        self.terminals = set()

    def __str__(self):
        return f"PropertyAccess<{self.prop}, {self.index}>"

    def vector_index(self, v_index):
        sizes = self.prop.sizes()
        layout = self.prop.layout()
        index = self.index * sizes[0] + v_index if layout == Layouts.AoS else \
                v_index * sizes[1] + self.index if layout == Layouts.SoA else \
                None

        assert index is not None, "Invalid data layout"
        return index

    def inline_rec(self):
        self.inlined = True
        return self

    def propagate_through(self):
        return []

    def id(self):
        return self.acc_id

    def type(self):
        return self.prop.type()

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.prop, self.index] + list(super().children())

    def __getitem__(self, index):
        super().__getitem__(index)
        return VectorAccess(self.sim, self, Lit.cvt(self.sim, index))


class RegisterProperty(ASTNode):
    def __init__(self, sim, prop, sizes):
        super().__init__(sim)
        self.prop = prop
        self.sizes_list = [Lit.cvt(sim, s) for s in sizes]
        self.sim.add_statement(self)

    def property(self):
        return self.prop

    def sizes(self):
        return self.sizes_list

    def __str__(self):
        return f"RegisterProperty<{self.prop.name()}>"


class ReallocProperty(ASTNode):
    def __init__(self, sim, prop, sizes):
        super().__init__(sim)
        self.prop = prop
        self.sizes_list = [Lit.cvt(sim, s) for s in sizes]
        self.sim.add_statement(self)

    def property(self):
        return self.prop

    def sizes(self):
        return self.sizes_list

    def __str__(self):
        return f"ReallocProperty<{self.prop.name()}>"
