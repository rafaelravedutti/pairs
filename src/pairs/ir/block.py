from pairs.ir.ast_node import ASTNode
from pairs.ir.module import Module


def pairs_block(func):
    def inner(*args, **kwargs):
        sim = args[0].sim # self.sim
        sim.clear_block()
        func(*args, **kwargs)
        return sim.block

    return inner


def pairs_device_block(func):
    def inner(*args, **kwargs):
        sim = args[0].sim # self.sim
        sim.clear_block()
        func(*args, **kwargs)
        module = Module(sim, block=KernelBlock(sim, sim.block))
        sim.add_module(module)
        return module

    return inner


# TODO: Is this really useful? Or just pairs_block is enough?
def pairs_host_block(func):
    def inner(*args, **kwargs):
        sim = args[0].sim # self.sim
        sim.clear_block()
        func(*args, **kwargs)
        return KernelBlock(sim, sim.block, run_on_host=True)

    return inner


class Block(ASTNode):
    def __init__(self, sim, stmts):
        super().__init__(sim)
        self.variants = set()

        if isinstance(stmts, Block):
            self.stmts = stmts.statements()
        else:
            self.stmts = [stmts] if not isinstance(stmts, list) else stmts

    def add_statement(self, stmt):
        if isinstance(stmt, list):
            self.stmts = self.stmts + stmt
        elif isinstance(stmt, Block):
            self.stmts = self.stmts + stmt.statements()
        else:
            self.stmts.append(stmt)

    def add_variant(self, variant):
        for v in variant if isinstance(variant, list) else [variant]:
            self.variants.add(v)

    def statements(self):
        return self.stmts

    def children(self):
        return self.stmts

    def merge_blocks(block1, block2):
        assert isinstance(block1, Block), "First block type is not Block!"
        assert isinstance(block2, Block), "Second block type is not Block!"
        assert not isinstance(block1, KernelBlock), "Kernel blocks cannot be merged!"
        assert not isinstance(block2, KernelBlock), "Kernel blocks cannot be merged!"
        return Block(block1.sim, block1.statements() + block2.statements())

    def from_list(sim, block_list):
        assert isinstance(block_list, list), "Given argument is not a list!"
        result_block = Block(sim, [])

        for block in block_list:
            if isinstance(block, Block):
                result_block = Block.merge_blocks(result_block, block)
            else:
                result_block.add_statement(block)

        return result_block


class KernelBlock(ASTNode):
    def __init__(self, sim, block, run_on_host=False):
        super().__init__(sim)
        self.block = block if isinstance(block, Block) else Block(sim, block)
        self.run_on_host = run_on_host
        self.props_accessed = {}

    def add_property_access(self, prop, oper):
        prop_key = prop.name()
        if prop_key not in self.props_accessed:
            self.props_accessed[prop_key] = oper

        elif oper not in self.props_accessed[prop_key]:
            self.props_accessed[prop_key] += oper

    def children(self):
        return [self.block]

    def properties_to_synchronize(self):
        return {p for p in self.props_accessed if self.props_accessed[p][0] == 'r'}

    def writing_properties(self):
        return {p for p in self.props_accessed if 'w' in self.props_accessed[p][0]}
