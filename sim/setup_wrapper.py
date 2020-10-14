from ast.block import Block

class SetupWrapper():
    def __init__(self):
        self.setups = Block(self, [])

    def add_setup_block(self, block):
        self.setups = Block.merge_blocks(self.setups, block)

    def lower(self):
        return self.setups

