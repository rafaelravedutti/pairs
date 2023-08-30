from pairs.ir.ast_node import ASTNode
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class Assign(ASTNode):
    def __init__(self, sim, dest, src):
        super().__init__(sim)
        self._dest = dest
        self._src = Lit.cvt(sim, src)

        # When vector assignments occur, all indexes for the dest
        # and source terms must be generated
        if self._dest.type() == Types.Vector:
            for dim in range(sim.ndims()):
                self._dest.add_index_to_generate(dim)

                if self._src.type() == Types.Vector:
                    self._src.add_index_to_generate(dim)

    def __str__(self):
        return f"Assign<{self._dest, self._src}>"

    def children(self):
        return [self._dest, self._src]
