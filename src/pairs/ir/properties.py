from pairs.ir.ast_node import ASTNode
from pairs.ir.assign import Assign
from pairs.ir.bin_op import BinOp, Decl, ASTTerm, VectorAccess
from pairs.ir.layouts import Layouts
from pairs.ir.lit import Lit
from pairs.ir.types import Types
from pairs.ir.vector_expr import VectorExpression


class Properties:
    def __init__(self, sim):
        self.sim = sim
        self.props = []
        self.capacities = []
        self.defs = {}

    def add(self, p_name, p_type, p_value, p_volatile, p_layout=Layouts.AoS):
        p = Property(self.sim, p_name, p_type, p_value, p_volatile, p_layout)
        self.props.append(p)
        self.defs[p_name] = p_value
        return p

    def add_capacity(self, var):
        self.capacities.append(var)

    def is_capacity(self, var):
        return var in self.capacities

    def defaults(self):
        return self.defs

    def all(self):
        return self.props

    def volatiles(self):
        return [p for p in self.props if p.volatile is True]

    def nprops(self):
        return len(self.props)

    def find(self, p_name):
        prop = [p for p in self.props if p.name() == p_name]
        if prop:
            return prop[0]

        return None

    def __iter__(self):
        yield from self.props


class Property(ASTNode):
    last_prop_id = 0

    def __init__(self, sim, name, dtype, default, volatile, layout=Layouts.AoS):
        super().__init__(sim)
        self.prop_id = Property.last_prop_id
        self.prop_name = name
        self.prop_type = dtype
        self.prop_layout = layout
        self.default_value = default
        self.volatile = volatile
        self.device_flag = False
        Property.last_prop_id += 1

    def __str__(self):
        return f"Property<{self.prop_name}>"

    def id(self):
        return self.prop_id

    def name(self):
        return self.prop_name

    def type(self):
        return self.prop_type

    def layout(self):
        return self.prop_layout

    def default(self):
        return self.default_value

    def ndims(self):
        return 1 if self.prop_type != Types.Vector else 2

    def sizes(self):
        return [self.sim.particle_capacity] if self.prop_type != Types.Vector else [self.sim.ndims(), self.sim.particle_capacity]

    def __getitem__(self, expr):
        return PropertyAccess(self.sim, self, expr)


class PropertyAccess(ASTTerm, VectorExpression):
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

    def copy(self):
        return PropertyAccess(self.sim, self.prop, self.index)

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
        return [self.prop, self.index] + list(super().children())

    def __getitem__(self, index):
        super().__getitem__(index)
        return VectorAccess(self.sim, self, Lit.cvt(self.sim, index))


class PropertyList(ASTNode):
    def __init__(self, sim, properties_list):
        super().__init__(sim)
        self.list = []
        for p in properties_list:
            if isinstance(p, Property):
                self.list.append(p)

            if isinstance(p, str):
                self.list.append(sim.prop(p))

    def __iter__(self):
        yield from self.list

    def length(self):
        return len(self.list)


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
