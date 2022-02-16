from pairs.ir.lit import Lit
from pairs.ir.mutator import Mutator
from pairs.ir.types import Types


class SimplifyExpressions(Mutator):
    def __init__(self, ast):
        super().__init__(ast)

    def mutate_BinOp(self, ast_node):
        sim = ast_node.lhs.sim
        ast_node.lhs = self.mutate(ast_node.lhs)
        ast_node.rhs = self.mutate(ast_node.rhs)
        ast_node.expressions = {i: self.mutate(e) for i, e in ast_node.expressions.items()}

        if ast_node.op in ['+', '-'] and ast_node.rhs == 0:
            return ast_node.lhs

        if ast_node.op in ['+'] and ast_node.lhs == 0:
            return ast_node.rhs

        if ast_node.op in ['*', '/'] and ast_node.rhs == 1:
            return ast_node.lhs

        if ast_node.op == '*' and ast_node.lhs == 1:
            return ast_node.rhs

        if ast_node.op == '*' and ast_node.lhs == 0:
            return Lit(sim, 0 if Types.is_integer(ast_node.type()) else 0.0)

        return ast_node


def simplify_expressions(ast_node):
    simplify = SimplifyExpressions(ast_node)
    simplify.mutate()
