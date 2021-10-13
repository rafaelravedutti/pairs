from pairs.ir.block import KernelBlock
from pairs.ir.device import DeviceCopy
from pairs.ir.mutator import Mutator
from pairs.ir.visitor import Visitor


class AddAccessedProperties(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.current_kernel_block = None
        self.writing = False

    def visit_Assign(self, ast_node):
        for s in ast_node.sources():
            self.visit(s)
        self.writing = True

        for d in ast_node.destinations():
            self.visit(d)
        self.writing = False

    def visit_KernelBlock(self, ast_node):
        self.current_kernel_block = ast_node
        self.visit_children(ast_node)

    def visit_PropertyAccess(self, ast_node):
        if self.current_kernel_block is not None:
            self.current_kernel_block.add_property_access(ast_node.prop, 'w' if self.writing else 'r')


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
                if isinstance(s, KernelBlock) and s_id in self.props_to_copy:
                    for p in self.props.to_copy[s_id]:
                        new_stmts = new_stmts + DeviceCopy(ast_node.sim, p)

                new_stmts.append(s)

        ast_node.stmts = new_stmts
        return ast_node

    def mutate_KernelBlock(self, ast_node):
        copying_properties = {p for p in ast_node.properties_to_synchronize() if p not in synchronized_props}
        self.props_to_copy[id(ast_node)] = copying_properties
        self.synchronized_props.update(copying_properties)
        self.synchronized_props -= ast_node.writing_properties()


def add_device_copies(ast):
    add_accessed_props = AddAccessedProperties(ast)
    add_accessed_props.visit()
    add_copies = AddDeviceCopies(ast)
    add_copies.mutate()
