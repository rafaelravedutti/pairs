from pairs.ir.assign import Assign
from pairs.ir.bin_op import BinOp
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.data_types import Type_Vector
from pairs.ir.loops import While
from pairs.ir.memory import Realloc
from pairs.ir.module import Module, Module_Call
from pairs.ir.mutator import Mutator
from pairs.ir.properties import UpdateProperty
from pairs.ir.variables import Deref
from pairs.ir.visitor import Visitor
from functools import reduce
import operator


class FetchModulesReferences(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.module_stack = []
        self.writing = False

    def visit_Assign(self, ast_node):
        self.writing = True
        for c in ast_node.destinations():
            self.visit(c)

        self.writing = False
        for c in ast_node.sources():
            self.visit(c)

    def visit_Module(self, ast_node):
        self.module_stack.append(ast_node)
        self.visit_children(ast_node)
        self.module_stack.pop()

    def visit_Array(self, ast_node):
        for m in self.module_stack:
            m.add_array(ast_node)

    def visit_Property(self, ast_node):
        for m in self.module_stack:
            m.add_property(ast_node)

    def visit_Var(self, ast_node):
        for m in self.module_stack:
            m.add_variable(ast_node, self.writing)


class AddDereferencesToWriteVariables(Mutator):
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
        self.match_capacity = None
        self.update = {}
        self.nresize_buffers = 0

    def get_capacity_for_size(self, size):
        for _capacity, _size in self.resizes_to_check.items():
            if _size.name() == size.name():
                return _capacity

        return None

    def look_for_match_capacity(self, size):
        capacity = self.get_capacity_for_size(size)
        if capacity is not None:
            self.match_capacity = capacity

    def mutate_Array(self, ast_node):
        self.look_for_match_capacity(ast_node)
        return ast_node

    def mutate_Assignment(self, ast_node):
        for dest, src in ast_node.assignments.items():
            if isinstance(dest, ArrayAccess):
                self.match_capacity = None
                ast_node.indexes = [self.mutate(i) for i in ast_node.indexes]
                if ast_node.index is not None:
                    ast_node.index = self.mutate(ast_node.index)

                # Resize var is used in index, this statement should be checked for safety
                if self.match_capacity is not None:
                    module = self.module_stack[-1]
                    size = self.resizes_to_check[match_capacity]
                    check_value = self.update[size] if size in self.update else size
                    resize_id = self.module_resizes[module].keys()[self.module_resizes[module].values().index(match_capacity)]
                    return Branch(ast_node.sim, check_value < match_capacity,
                                  Block(ast_node.sim, ast_node),
                                  Block(ast_node.sim, sim.resizes[resize_id].set(check_value)))


                # Size is changed here, assigned value must be used for further checkings
                # When size is of type array (i.e. neighbor list size), just use last assignment to it
                # without checking accessed index (maybe this has to be changed at some point)
                self.update[dest.array] = src

            if isinstance(dest, Var):
                # Size is changed here, assigned value must be used for further checkings
                self.update[dest] = src

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
        saved_update = self.update
        saved_nresize_buffers = self.nresize_buffers

        # Update state and keep traversing tree
        self.module_resizes[ast_node] = {}
        self.module_stack.append(ast_node)
        for capacity in ast_node._resizes_to_check.keys():
            self.module_resizes[ast_node][self.nresize_buffers] = capacity
            self.nresize_buffers += 1

        self.resizes_to_check = ast_node._resizes_to_check
        self.check_properties_resize = ast_node._check_properties_resize
        self.update = {}
        ast_node._block = self.mutate(ast_node._block)
        self.module_stack.pop()

        # Restore saved state
        self.resizes_to_check = saved_resizes_to_check
        self.check_properties_resize = saved_check_properties_resize
        self.update = saved_update
        self.nresize_buffers = saved_nresize_buffers
        return ast_node

    def mutate_Var(self, ast_node):
        self.look_for_match_capacity(ast_node)
        return ast_node


class ReplaceModulesByCalls(Mutator):
    def __init__(self, ast, module_resizes, grow_fn=None):
        super().__init__(ast)
        self.module_resizes = module_resizes
        self.grow_fn = grow_fn if grow_fn is not None else (lambda x: x * 2)

    def mutate_Module(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)
        if ast_node.name == 'main':
            return ast_node

        sim = ast_node.sim
        call = Module_Call(sim, ast_node)
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
                        sizes = [c, sim.ndims()] if p.type() == Type_Vector else [c]
                        props_realloc += [Realloc(sim, p, reduce(operator.mul, sizes)), UpdateProperty(sim, p, sizes)]

                resize_stmts.append(
                    Filter(sim, sim.resizes[r] > 0, Block(sim,
                        [Assign(sim, c, self.grow_fn(sim.resizes[r]))] +
                        [a.realloc() for a in c.bonded_arrays()] +
                        props_realloc)))

            return Block(sim, init_stmts + [While(sim, branch_cond, Block(sim, reset_stmts + [call] + resize_stmts))])

        return call


def modularize(ast):
    add_resize_logic = AddResizeLogic(ast)
    add_resize_logic.mutate()
    fetch_refs = FetchModulesReferences(ast)
    fetch_refs.visit()
    add_derefs_to_write_vars = AddDereferencesToWriteVariables(ast)
    add_derefs_to_write_vars.mutate()
    replace = ReplaceModulesByCalls(ast, add_resize_logic.module_resizes)
    replace.mutate()
