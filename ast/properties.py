from ast.data_types import Type_Float, Type_Vector
from code_gen.printer import printer

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

class PropertyDeclAST:
    def __init__(self, sim, prop, size):
        self.sim = sim
        self.prop = prop

    def __str__(self):
        return f"PropertyDecl<{self.prop.name}>"

    def generate(self, mem=False):
        nparticles = self.sim.nparticles.generate()
        if self.prop.prop_type == Type_Float:
            printer.print(f"double {self.prop.prop_name}[{nparticles}]")
        elif self.prop.prop_type == Type_Vector:
            if self.prop.flattened:
                printer.print(f"double {self.prop.prop_name}[{nparticles} * {self.sim.dimensions}];")
            else:
                printer.print(f"double {self.prop.prop_name}[{nparticles}][{self.sim.dimensions}];")
        else:
            raise Exception("Invalid property type!")

    def transform(self, fn):
        return fn(self)
