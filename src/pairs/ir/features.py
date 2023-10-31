from pairs.ir.accessor_class import AccessorClass
from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.declaration import Decl
from pairs.ir.scalars import ScalarOp
from pairs.ir.layouts import Layouts
from pairs.ir.lit import Lit
from pairs.ir.operator_class import OperatorClass
from pairs.ir.types import Types


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
        return self.feature_prop[expr]


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


class FeatureProperty(ASTNode):
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

    def data(self):
        return self.feature_prop_data

    def layout(self):
        return self.feature_prop_layout

    def ndims(self):
        return 1 if Types.is_scalar(self.prop_type) else 2

    def sizes(self):
        return [self.feature_prop_feature.nkinds()] if Types.is_scalar(self.feature_prop_type) \
               else [Types.number_of_elements(self.sim, self.feature_prop_type),
                     self.feature_prop_feature.nkinds()]

    def array_size(self):
        nelems = self.feature_prop_feature.nkinds() * \
                 Types.number_of_elements(self.sim, self.feature_prop_type)
        return nelems * nelems

    def __getitem__(self, expr):
        return FeaturePropertyAccess(self.sim, self, expr)


class FeaturePropertyAccess(ASTTerm):
    last_feature_prop_acc = 0

    def new_id():
        FeaturePropertyAccess.last_feature_prop_acc += 1
        return FeaturePropertyAccess.last_feature_prop_acc - 1

    def __init__(self, sim, feature_prop, index):
        assert isinstance(index, tuple), "Two indexes must be used for feature property access!"
        super().__init__(sim, OperatorClass.from_type(feature_prop.type()))
        self.acc_id = FeaturePropertyAccess.new_id()
        self.feature_prop = feature_prop
        feature = self.feature_prop.feature()
        self.index = Lit.cvt(sim, feature[index[0]] * feature.nkinds() + feature[index[1]])
        self.inlined = False
        self.terminals = set()
        self.vector_indexes = {}

        if not Types.is_scalar(feature_prop.type()):
            sizes = feature_prop.sizes()
            layout = feature_prop.layout()

            for elem in range(Types.number_of_elements(feature_prop.type())):
                if layout == Layouts.AoS:
                    self.vector_indexes[elem] = self.index * sizes[0] + elem
                elif layout == Layouts.SoA:
                    self.vector_indexes[elem] = elem * sizes[1] + self.index
                else:
                    raise Exception("Invalid data layout.")

    def __str__(self):
        return f"FeaturePropertyAccess<{self.feature_prop}, {self.index}>"

    def copy(self):
        return FeaturePropertyAccess(self.sim, self.feature_prop, self.index)

    def vector_index(self, dimension):
        return self.vector_indexes[dimension]

    def inline_recursively(self):
        self.inlined = True
        return self

    def id(self):
        return self.acc_id

    def name(self):
        return f"feat_prop_acc{self.id()}" + self.label_suffix()

    def type(self):
        return self.feature_prop.type()

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.feature_prop, self.index] + list(self.vector_indexes.values())

    def __getitem__(self, index):
        super().__getitem__(index)
        _acc_class = AccessorClass.from_type(self.feature_prop.type())
        return _acc_class(self.sim, self, Lit.cvt(self.sim, index))


class RegisterFeatureProperty(ASTNode):
    def __init__(self, sim, feature_prop):
        super().__init__(sim)
        self.feature_prop = feature_prop
        self.sim.add_statement(self)

    def feature_property(self):
        return self.feature_prop

    def __str__(self):
        return f"RegisterFeatureProperty<{self.feature_prop.name()}>"
