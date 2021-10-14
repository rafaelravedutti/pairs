from pairs.ir.block import Block, KernelBlock


class KernelWrapper():
    def __init__(self, sim):
        self.sim = sim
        self.kernels = []

    def add_kernel_block(self, block):
        self.kernels.append(KernelBlock(self.sim, block))

    def lower(self):
        return self.kernels
