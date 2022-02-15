from pairs.ir.device import DeviceCopy
from pairs.ir.module import ModuleCall
from pairs.ir.mutator import Mutator
from pairs.ir.visitor import Visitor


class AddDeviceCopies(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.synchronized_props = set()
        self.props_to_copy = {}

    def mutate_Block(self, ast_node):
        new_stmts = []
        stmts = [self.mutate(s) for s in ast_node.stmts]

        for s in stmts:
            if s is not None:
                s_id = id(s)
                if isinstance(s, ModuleCall) and s_id in self.props_to_copy:
                    new_stmts = new_stmts + [DeviceCopy(ast_node.sim, p) for p in self.props_to_copy[s_id]]

                new_stmts.append(s)

        ast_node.stmts = new_stmts
        return ast_node

    def mutate_ModuleCall(self, ast_node):
        copying_properties = {p for p in ast_node.module.properties_to_synchronize() if p not in self.synchronized_props}
        self.props_to_copy[id(ast_node)] = copying_properties
        self.synchronized_props.update(copying_properties)
        self.synchronized_props -= ast_node.module.write_properties()
        return ast_node


def add_device_copies(ast):
    add_copies = AddDeviceCopies(ast)
    add_copies.mutate()
