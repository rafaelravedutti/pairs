from pairs.ir.module import Module_Call
from pairs.ir.mutator import Mutator


class ReplaceModulesByCalls(Mutator):
    def __init__(self, ast):
        super().__init__(ast)

    def mutate_Module(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)
        return Module_Call(ast_node.sim, ast_node) if ast_node.name != 'main' else ast_node


def replace_modules_by_calls(ast):
    replace = ReplaceModulesByCalls(ast)
    replace.mutate()
