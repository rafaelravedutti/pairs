from data_types import Type_Int, Type_Vector
from expr import ExprAST, ExprVecAST
from lit import LitAST
from properties import Property

class Transform:
    def flatten(ast):
        if isinstance(ast, ExprVecAST):
            if ast.expr.op == '[]' and ast.expr.type() == Type_Vector:
                return ExprAST(ast.expr.sim, ast.expr.lhs, ast.expr.rhs * ast.expr.sim.dimensions + ast.index, '[]', ast.expr.mem) 

        if isinstance(ast, Property):
            ast.flattened = True

        return ast

    def simplify(ast):
        if isinstance(ast, ExprAST):
            if ast.op in ['+', '-'] and ast.rhs == 0:
                return ast.lhs

            if ast.op in ['+', '-'] and ast.lhs == 0:
                return ast.rhs

            if ast.op in ['*', '/'] and ast.rhs == 1:
                return ast.lhs

            if ast.op == '*' and ast.lhs == 1:
                return ast.rhs

            if ast.op == '*' and ast.lhs == 0:
                return LitAST(0 if ast.type() == Type_Int else 0.0)

        return ast
