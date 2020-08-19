class Property:
    def __init__(self, prop_name, prop_type, default_value, volatile):
        self.prop_name = prop_name
        self.prop_type = prop_type
        self.default_value = default_value
        self.volatile = volatile

    def __getitem__(self, expr_ast):
        from ast import ExprAST
        return ExprAST(self, expr_ast, '[]', True)
