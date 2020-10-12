from ast.arrays import ArrayAccess
from ast.data_types import Type_Int, Type_Vector
from ast.expr import ExprAST, ExprVecAST
from ast.lit import LitAST
from ast.loops import IterAST
from ast.properties import Property


class Transform:
    flattened_list = []
    reuse_expressions = {}

    def apply(ast, fn):
        ast.transform(fn)
        Transform.flattened_list = []
        Transform.reuse_expressions = {}

    def flatten(ast):
        if isinstance(ast, ExprVecAST):
            if ast.expr.op == '[]' and ast.expr.type() == Type_Vector:
                item = [f for f in Transform.flattened_list if
                        f[0] == ast.expr.lhs and
                        f[1] == ast.index and
                        f[2] == ast.expr.rhs]
                if item:
                    return item[0][3]

                new_expr = ExprAST(
                    ast.expr.sim,
                    ast.expr.lhs,
                    ast.expr.rhs * ast.expr.sim.dimensions + ast.index,
                    '[]',
                    ast.expr.mem)

                Transform.flattened_list.append(
                    (ast.expr.lhs, ast.index, ast.expr.rhs, new_expr))
                return new_expr

        if isinstance(ast, Property):
            ast.flattened = True

        return ast

    def simplify(ast):
        if isinstance(ast, ExprAST):
            sim = ast.lhs.sim

            if ast.op in ['+', '-'] and ast.rhs == 0:
                return ast.lhs

            if ast.op in ['+', '-'] and ast.lhs == 0:
                return ast.rhs

            if ast.op in ['*', '/'] and ast.rhs == 1:
                return ast.lhs

            if ast.op == '*' and ast.lhs == 1:
                return ast.rhs

            if ast.op == '*' and ast.lhs == 0:
                return LitAST(sim, 0 if ast.type() == Type_Int else 0.0)

        return ast

    def reuse_index_expressions(ast):
        if isinstance(ast, ExprAST):
            iter_id = None

            if isinstance(ast.lhs, IterAST):
                iter_id = ast.lhs.iter_id

            if isinstance(ast.rhs, IterAST):
                iter_id = ast.rhs.iter_id

            if iter_id is not None:
                if iter_id in Transform.reuse_expressions:
                    item = [e for e in Transform.reuse_expressions[iter_id]
                            if ast.match(e)]
                    if item:
                        return item[0]

                else:
                    Transform.reuse_expressions[iter_id] = []

                Transform.reuse_expressions[iter_id].append(ast)

        return ast

    def reuse_expr_expressions(ast):
        if isinstance(ast, ExprAST):
            expr_id = None

            if isinstance(ast.lhs, ExprAST):
                expr_id = ast.lhs.expr_id

            if isinstance(ast.rhs, ExprAST):
                expr_id = ast.rhs.expr_id

            if expr_id is not None:
                if expr_id in Transform.reuse_expressions:
                    item = [e for e in Transform.reuse_expressions[expr_id]
                            if ast.match(e)]
                    if item:
                        return item[0]

                else:
                    Transform.reuse_expressions[expr_id] = []

                Transform.reuse_expressions[expr_id].append(ast)

        return ast

    def reuse_array_access_expressions(ast):
        if isinstance(ast, ExprAST):
            acc_id = None

            if isinstance(ast.lhs, ArrayAccess):
                acc_id = ast.lhs.acc_id

            if isinstance(ast.rhs, ArrayAccess):
                acc_id = ast.rhs.acc_id

            if acc_id is not None:
                if acc_id in Transform.reuse_expressions:
                    item = [e for e in Transform.reuse_expressions[acc_id]
                            if ast.match(e)]
                    if item:
                        return item[0]

                else:
                    Transform.reuse_expressions[acc_id] = []

                Transform.reuse_expressions[acc_id].append(ast)

        return ast

    def move_loop_invariant_expressions(ast):
        if isinstance(ast, ExprAST):
            scope = ast.scope()
            if scope.level() > 0:
                scope.block.add_expression(ast)

        return ast
