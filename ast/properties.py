from ast.ast_node import ASTNode
from ast.layouts import Layout_AoS


class Properties:
    def __init__(self, sim):
        self.sim = sim
        self.props = []
        self.capacities = []
        self.defs = {}
        self.nprops = 0

    def add(self, p_name, p_type, p_value, p_volatile, p_layout=Layout_AoS):
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

    def find(self, p_name):
        prop = [p for p in self.props if p.name() == p_name]
        if prop:
            return prop[0]

        return None


class Property(ASTNode):
    def __init__(self, sim, name, dtype, default, volatile, layout=Layout_AoS):
        super().__init__(sim)
        self.prop_name = name
        self.prop_type = dtype
        self.prop_layout = layout
        self.default_value = default
        self.volatile = volatile

    def __str__(self):
        return f"Property<{self.prop_name}>"

    def name(self):
        return self.prop_name

    def type(self):
        return self.prop_type

    def layout(self):
        return self.prop_layout

    def default(self):
        return self.default_value

    def scope(self):
        return self.sim.global_scope

    def __getitem__(self, expr):
        from ast.bin_op import BinOp
        return BinOp(self.sim, self, expr, '[]', True)
