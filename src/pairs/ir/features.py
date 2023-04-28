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

    def add(self, f_name, f_nkinds):
        f = Feature(self.sim, f_name, f_nkinds)
        self.features.append(f)
        return f

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

    def __init__(self, sim, name, nkinds):
        super().__init__(sim)
        self.feature_id = Feature.last_feature_id
        self.feature_name = name
        self.feature_prop = self.sim.add_property(self.feature_name, Types.Int32)
        self.feature_nkinds = nkinds
        Feature.last_feature_id += 1

    def __str__(self):
        return f"Feature<{self.feature_name}>"

    def id(self):
        return self.feature_id

    def name(self):
        return self.feature_name

    def prop(self):
        return self.feature_prop

    def nkinds(self):
        return self.feature_nkinds

    def __getitem__(self, expr):
        return FeatureAccess(self.sim, self, expr)


class FeatureProperties:
    def __init__(self, sim):
        self.sim = sim
        self.feature_properties = []

    def add(self, fp_feature, fp_name, fp_type, fp_data, fp_layout=Layouts.AoS):
        fp = FeatureProperty(self.sim, fp_feature, fp_name, fp_type, fp_data, fp_layout)
        self.feature_properties.append(fp)
        return fp

    def nfeatures(self):
        return len(self.features)

    def find(self, fp_name):
        feature_prop = [fp for fp in self.feature_properties if fp.name() == fp_name]
        if feature_prop:
            return feature_prop[0]

        return None

    def __iter__(self):
        yield from self.feature_properties


class FeatureProperty(ASTTerm):
    last_feature_prop_id = 0

    def __init__(self, sim, feature, name, dtype, data, layout=Layouts.AoS):
        super().__init__(sim)
        self.feature_prop_id = FeatureProperty.last_feature_prop_id
        self.feature_prop_feature = feature
        self.feature_prop_name = name
        self.feature_prop_type = dtype
        self.feature_prop_data = data
        self.feature_prop_layout = layout
        self.device_flag = False
        FeatureProperty.last_feature_prop_id += 1

    def __str__(self):
        return f"FeatureProperty<{self.feature_prop_name}>"

    def id(self):
        return self.feature_prop_id

    def feature(self):
        return self.feature_prop_feature

    def name(self):
        return self.feature_prop_name

    def type(self):
        return self.feature_prop_type

    def layout(self):
        return self.feature_prop_layout

    def ndims(self):
        return 1 if self.feature_prop_type != Types.Vector else 2

    def sizes(self):
        return [self.feature_prop_feature.nkinds()] if self.feature_prop_type != Types.Vector \
               else [self.sim.ndims(), self.feature_prop_feature.nkinds()]

    def array_size(self):
        nelems = self.feature_prop.feature.nkinds() * \
                 (1 if self.feature_prop_type != Types.Vector else self.sim.ndims())
        return nelems * nelems

    def __getitem__(self, expr):
        return FeaturePropertyAccess(self.sim, self, expr)


class FeaturePropertyAccess(ASTTerm, VectorExpression):
    last_feature_prop_acc = 0

    def new_id():
        PropertyAccess.last_feature_prop_acc += 1
        return PropertyAccess.last_feature_prop_acc - 1

    def __init__(self, sim, feature_prop, index):
        assert isinstance(index, tuple), "Two indexes must be used for feature property access!"
        super().__init__(sim)
        self.acc_id = FeaturePropertyAccess.new_id()
        self.feature_prop = feature_prop
        feature = self.feature_prop.feature()
        self.index = Lit.cvt(sim, feature[index[0]] * feature.nkinds() + feature[index[1]])
        self.inlined = False
        self.terminals = set()

    def __str__(self):
        return f"FeaturePropertyAccess<{self.feature_prop}, {self.index}>"

    def copy(self):
        return FeaturePropertyAccess(self.sim, self.feature_prop, self.index)

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
        return self.feature_prop.type()

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.feature_prop, self.index] + list(super().children())

    def __getitem__(self, index):
        super().__getitem__(index)
        return VectorAccess(self.sim, self, Lit.cvt(self.sim, index))


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
        return Types.Int32

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.prop, self.index]


class RegisterFeatureProperty(ASTNode):
    def __init__(self, sim, feature_prop):
        super().__init__(sim)
        self.feature_prop = feature_prop
        self.sim.add_statement(self)

    def feature_property(self):
        return self.feature_prop

    def __str__(self):
        return f"RegisterFeatureProperty<{self.feature_prop.name()}>"
