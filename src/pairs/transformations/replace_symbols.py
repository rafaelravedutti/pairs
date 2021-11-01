from pairs.ir.mutator import Mutator


class ReplaceSymbols(Mutator):
    def __init__(self, ast):
        super().__init__(ast)

    def mutate_Symbol(self, ast_node):
        return ast_node.assign_to


def replace_symbols(ast):
    replace = ReplaceSymbols(ast)
    replace.mutate()
