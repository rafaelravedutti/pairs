from data_types import Type_Vector
from expr import ExprAST, ExprVecAST
from properties import Property

class Transform:
    def flatten(ast):
        if isinstance(ast, ExprVecAST):
            if ast.expr.op == '[]' and ast.expr.type() == Type_Vector:
                return ExprAST(ast.expr.sim, ast.expr.lhs, ast.expr.rhs * ast.expr.sim.dimensions + ast.index, '[]', ast.expr.mem) 

        if isinstance(ast, Property):
            ast.flattened = True

        return ast
