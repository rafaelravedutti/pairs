from pairs.ir.ast_node import ASTNode
from pairs.ir.module import Module


def pairs_inline(func):
    def inner(*args, **kwargs):
        sim = args[0].sim # self.sim
        sim.init_block()
        func(*args, **kwargs)
        return sim._block

    return inner


def pairs_host_block(func):
    def inner(*args, **kwargs):
        sim = args[0].sim # self.sim
        sim.init_block()
        func(*args, **kwargs)
        return Module(sim,
            name=sim._module_name,
            block=Block(sim, sim._block),
            resizes_to_check=sim._resizes_to_check,
            check_properties_resize=sim._check_properties_resize,
            run_on_device=False)

    return inner


def pairs_device_block(func):
    def inner(*args, **kwargs):
        sim = args[0].sim # self.sim
        sim.init_block()
        func(*args, **kwargs)
        return Module(sim,
            name=sim._module_name,
            block=Block(sim, sim._block),
            resizes_to_check=sim._resizes_to_check,
            check_properties_resize=sim._check_properties_resize,
            run_on_device=True)

    return inner


class Block(ASTNode):
    def __init__(self, sim, stmts):
        super().__init__(sim)
        self.variants = set()

        if isinstance(stmts, Block):
            self.stmts = stmts.statements()
        else:
            self.stmts = [stmts] if not isinstance(stmts, list) else stmts

    def __str__(self):
        return "Block<>"

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

    def clear(self):
        self.stmts = []

    def statements(self):
        return self.stmts

    def children(self):
        return self.stmts

    def merge_blocks(block1, block2):
        assert isinstance(block1, Block), "First block type is not Block!"
        assert isinstance(block2, Block), "Second block type is not Block!"
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
