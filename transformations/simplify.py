from ir.data_types import Type_Int
from ir.lit import Lit
from ir.mutator import Mutator


class SimplifyExpressions(Mutator):
    def __init__(self, ast):
        super().__init__(ast)

    def mutate_BinOp(self, ast_node):
        sim = ast_node.lhs.sim
        ast_node.lhs = self.mutate(ast_node.lhs)
        ast_node.rhs = self.mutate(ast_node.rhs)
        ast_node.vector_index_mapping = {i: self.mutate(e) for i, e in ast_node.vector_index_mapping.items()}

        if ast_node.op in ['+', '-'] and ast_node.rhs == 0:
            return ast_node.lhs

        if ast_node.op in ['+'] and ast_node.lhs == 0:
            return ast_node.rhs

        if ast_node.op in ['*', '/'] and ast_node.rhs == 1:
            return ast_node.lhs

        if ast_node.op == '*' and ast_node.lhs == 1:
            return ast_node.rhs

        if ast_node.op == '*' and ast_node.lhs == 0:
            return Lit(sim, 0 if ast_node.type() == Type_Int else 0.0)

        return ast_node


def simplify_expressions(ast_node):
    simplify = SimplifyExpressions(ast_node)
    simplify.mutate()
