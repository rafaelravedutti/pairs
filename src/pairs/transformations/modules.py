from pairs.ir.arrays import Array, ArrayAccess
from pairs.ir.assign import Assign
from pairs.ir.bin_op import BinOp
from pairs.ir.block import Block
from pairs.ir.branches import Branch, Filter
from pairs.ir.lit import Lit
from pairs.ir.loops import While
from pairs.ir.memory import Realloc
from pairs.ir.module import Module, ModuleCall
from pairs.ir.mutator import Mutator
from pairs.ir.properties import UpdateProperty
from pairs.ir.types import Types
from pairs.ir.variables import Var, Deref
from functools import reduce
import operator


class DereferenceWriteVariables(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.module_stack = []

    def mutate_Module(self, ast_node):
        self.module_stack.append(ast_node)
        ast_node._block = self.mutate(ast_node._block)
        self.module_stack.pop()
        return ast_node

    def mutate_Var(self, ast_node):
        parent_module = self.module_stack[-1]
        if parent_module.name != 'main' and ast_node in parent_module.write_variables():
            return Deref(ast_node.sim, ast_node)

        return ast_node


class AddResizeLogic(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.block_stack = []
        self.module_stack = []
        self.module_resizes = {}
        self.resizes_to_check = {}
        self.check_properties_resize = False
        self.nresize_buffers = 0

    def get_capacity_for_size(self, size):
        for _capacity, _size in self.resizes_to_check.items():
            if _size.name() == size.name():
                return _capacity

        return None

    def lookup_capacity(self, nodes):
        capacity = None
        for node in nodes:
            if isinstance(node, (Array, Var)):
                capacity = self.get_capacity_for_size(node)
            elif isinstance(node, ArrayAccess):
                # We just want to look into mutable elements, not indexes
                capacity = self.lookup_capacity([node.array])
            else:
                capacity = self.lookup_capacity(node.children())

            if capacity is not None:
                return capacity

        return None

    def mutate_Assign(self, ast_node):
        for dest, src in ast_node.assignments:
            if not isinstance(src, Lit):
                match_capacity = None

                if isinstance(dest, (ArrayAccess, Var)):
                    match_capacity = self.lookup_capacity([dest])

                # Resize var is used in index, this statement should be checked for safety
                if match_capacity is not None:
                    module = self.module_stack[-1]
                    resizes = list(self.module_resizes[module].keys())
                    capacities = list(self.module_resizes[module].values())
                    resize_id = resizes[capacities.index(match_capacity)]
                    return Branch(ast_node.sim, src + 1 >= match_capacity,
                                  blk_if=Block(ast_node.sim, ast_node.sim.resizes[resize_id].set(src)),
                                  blk_else=Block(ast_node.sim, ast_node))

        return ast_node

    def mutate_Block(self, ast_node):
        self.block_stack.append(ast_node)
        ast_node.stmts = [self.mutate(s) for s in ast_node.stmts]
        self.block_stack.pop()
        return ast_node

    def mutate_Module(self, ast_node):
        # Save current state
        saved_resizes_to_check = self.resizes_to_check
        saved_check_properties_resize = self.check_properties_resize
        saved_nresize_buffers = self.nresize_buffers

        # Update state and keep traversing tree
        self.module_resizes[ast_node] = {}
        self.module_stack.append(ast_node)
        for capacity in ast_node._resizes_to_check.keys():
            self.module_resizes[ast_node][self.nresize_buffers] = capacity
            self.nresize_buffers += 1

        self.resizes_to_check = ast_node._resizes_to_check
        self.check_properties_resize = ast_node._check_properties_resize
        ast_node._block = self.mutate(ast_node._block)
        self.module_stack.pop()

        # Restore saved state
        self.resizes_to_check = saved_resizes_to_check
        self.check_properties_resize = saved_check_properties_resize
        self.nresize_buffers = saved_nresize_buffers
        return ast_node


class ReplaceModulesByCalls(Mutator):
    def __init__(self, ast, grow_fn=None):
        super().__init__(ast)
        self.module_resizes = None
        self.grow_fn = grow_fn if grow_fn is not None else (lambda x: x * 2)

    def set_module_resizes(self, module_resizes):
        self.module_resizes = module_resizes

    def mutate_Module(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)
        if ast_node.name == 'main':
            return ast_node

        sim = ast_node.sim
        call = ModuleCall(sim, ast_node)
        if self.module_resizes[ast_node]:
            properties = sim.properties
            init_stmts = []
            reset_stmts = []
            resize_stmts = []
            branch_cond = None

            for r, c in self.module_resizes[ast_node].items():
                init_stmts.append(Assign(sim, sim.resizes[r], 1))
                reset_stmts.append(Assign(sim, sim.resizes[r], 0))
                cond = BinOp.inline(sim.resizes[r] > 0)
                branch_cond = cond if branch_cond is None else BinOp.or_op(cond, branch_cond)
                props_realloc = []

                if properties.is_capacity(c):
                    for p in properties.all():
                        new_capacity = sum(properties.capacities)
                        sizes = [new_capacity, sim.ndims()] if p.type() == Types.Vector else [new_capacity]
                        props_realloc += [Realloc(sim, p, reduce(operator.mul, sizes)), UpdateProperty(sim, p, sizes)]

                resize_stmts.append(
                    Filter(sim, sim.resizes[r] > 0, Block(sim,
                        [Assign(sim, c, self.grow_fn(sim.resizes[r]))] +
                        [a.realloc() for a in c.bonded_arrays()] +
                        props_realloc)))

            return Block(sim, init_stmts + [While(sim, branch_cond, Block(sim, reset_stmts + [call] + resize_stmts))])

        return call
