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


class Property:
    def __init__(self, sim, name, dtype, default, volatile, layout=Layout_AoS):
        self.sim = sim
        self.prop_name = name
        self.prop_type = dtype
        self.prop_layout = layout
        self.default_value = default
        self.volatile = volatile
        self.mutable = True
        self.flattened = False

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

    def is_mutable(self):
        return self.mutable

    def scope(self):
        return self.sim.global_scope

    def __getitem__(self, expr_ast):
        from ast.expr import Expr
        return Expr(self.sim, self, expr_ast, '[]', True)

    def children(self):
        return []

    def generate(self, mem=False):
        return self.prop_name

    def transform(self, fn):
        return fn(self)
