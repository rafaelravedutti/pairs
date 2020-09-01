class Property:
    def __init__(self, sim, prop_name, prop_type, default_value, volatile):
        self.sim = sim
        self.prop_name = prop_name
        self.prop_type = prop_type
        self.default_value = default_value
        self.volatile = volatile

    def name(self):
        return self.prop_name

    def type(self):
        return self.prop_type

    def __getitem__(self, expr_ast):
        from expr import ExprAST
        return ExprAST(self.sim, self, expr_ast, '[]', True)
