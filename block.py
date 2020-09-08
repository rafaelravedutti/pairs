from expr import ExprAST, ExprVecAST

class BlockAST:
    def __init__(self, stmts):
        self.stmts = stmts

    def generate(self):
        for stmt in self.stmts:
            stmt.generate()

    def transform(self, fn):
        new_stmts = []
        for stmt in self.stmts:
            new_stmts.append(stmt.transform(fn))

        self.stmts = new_stmts
        return fn(self)

class Transform:
    def flatten(ast):
        if isinstance(ast, ExprVecAST):
            if ast.expr.op == '[]':
                return ExprAST(ast.expr.sim, ast.expr.lhs, ast.expr.rhs * ast.expr.sim.dimensions + ast.index, '[]', ast.expr.mem) 

        return ast
