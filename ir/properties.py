from ir.ast_node import ASTNode
from ir.layouts import Layout_AoS
from ir.lit import as_lit_ast


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

    def __iter__(self):
        yield from self.props


class Property(ASTNode):
    last_prop_id = 0

    def __init__(self, sim, name, dtype, default, volatile, layout=Layout_AoS):
        super().__init__(sim)
        self.prop_id = Property.last_prop_id
        self.prop_name = name
        self.prop_type = dtype
        self.prop_layout = layout
        self.default_value = default
        self.volatile = volatile
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

    def scope(self):
        return self.sim.global_scope

    def __getitem__(self, expr):
        from ir.bin_op import BinOp
        return BinOp(self.sim, self, expr, '[]', True)


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
        self.sizes_list = [as_lit_ast(sim, s) for s in sizes]
        self.sim.add_statement(self)

    def property(self):
        return self.prop

    def sizes(self):
        return self.sizes_list

    def __str__(self):
        return f"Property<{self.prop.name()}>"
