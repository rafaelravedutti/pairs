from pairs.ir.block import Block
from pairs.ir.functions import Call_Void
from pairs.ir.module import ModuleCall
from pairs.ir.mutator import Mutator
from pairs.ir.timers import Timers


class AddModulesInstrumentation(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)

    def mutate_ModuleCall(self, ast_node):
        ast_node._module = self.mutate(ast_node._module)
        module = ast_node._module
        if module.name == 'main':
            return ast_node

        timer_id = module.module_id + Timers.Offset
        start_timer = Call_Void(ast_node.sim, "pairs::start_timer", [timer_id])
        stop_timer = Call_Void(ast_node.sim, "pairs::stop_timer", [timer_id])

        if module.must_profile():
            start_marker = Call_Void(ast_node.sim, "LIKWID_MARKER_START", [module.name])
            stop_marker = Call_Void(ast_node.sim, "LIKWID_MARKER_STOP", [module.name])
            return Block(ast_node.sim, [start_timer, start_marker, ast_node, stop_marker, stop_timer])

        return Block(ast_node.sim, [start_timer, ast_node, stop_timer])
