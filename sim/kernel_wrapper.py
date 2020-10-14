from ast.block import Block

class KernelWrapper():
    def __init__(self):
        self.kernels = Block(self, [])

    def add_kernel_block(self, block):
        self.kernels = Block.merge_blocks(self.kernels, block)

    def lower(self):
        return self.kernels

