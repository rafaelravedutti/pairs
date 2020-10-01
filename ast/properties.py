class Properties:
    def __init__(self, sim):
        self.sim = sim
        self.props = []
        self.defs = {}
        self.nprops = 0

    def add(self, p_name, p_type, p_value, p_volatile):
        p = Property(self.sim, p_name, p_type, p_value, p_volatile)
        self.props.append(p)
        self.defs[p_name] = p_value
        return p

    def defaults(self):
        return self.defs

    def all(self):
        return self.props

    def volatiles(self):
        return [p for p in self.props if p.volatile is True]

    def find(self, p_name):
        return [p for p in self.props if p.name() == p_name][0]


class Property:
    def __init__(self, sim, prop_name, prop_type, default_value, volatile):
        self.sim = sim
        self.prop_name = prop_name
        self.prop_type = prop_type
        self.default_value = default_value
        self.volatile = volatile
        self.flattened = False

    def __str__(self):
        return f"Property<{self.prop_name}>"

    def name(self):
        return self.prop_name

    def type(self):
        return self.prop_type

    def __getitem__(self, expr_ast):
        from ast.expr import ExprAST
        return ExprAST(self.sim, self, expr_ast, '[]', True)

    def generate(self, mem=False):
        return self.prop_name

    def transform(self, fn):
        return fn(self)
