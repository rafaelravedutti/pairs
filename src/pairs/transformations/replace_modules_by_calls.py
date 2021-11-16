from pairs.ir.module import Module_Call
from pairs.ir.mutator import Mutator


class ReplaceModulesByCalls(Mutator):
    def __init__(self, ast):
        super().__init__(ast)

    def mutate_Module(self, ast_node):
        return Module_Call(ast_node.sim, ast_node)


def replace_modules_by_calls(ast):
    replace = ReplaceModulesByCalls(ast)
    replace.mutate()
