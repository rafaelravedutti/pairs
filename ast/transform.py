from ast.data_types import Type_Int, Type_Vector
from ast.expr import ExprAST, ExprVecAST
from ast.lit import LitAST
from ast.properties import Property

class Transform:
    flattened_list = []

    def flatten(ast):
        if isinstance(ast, ExprVecAST):
            if ast.expr.op == '[]' and ast.expr.type() == Type_Vector:
                item = [f for f in Transform.flattened_list if f[0] == ast.index and f[1] == ast.expr.rhs]
                if item:
                    return item[0][2]

                new_expr = ExprAST(ast.expr.sim, ast.expr.lhs, ast.expr.rhs * ast.expr.sim.dimensions + ast.index, '[]', ast.expr.mem)
                Transform.flattened_list.append((ast.index, ast.expr.rhs, new_expr))
                return new_expr

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
