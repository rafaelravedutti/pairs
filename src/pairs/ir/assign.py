from pairs.ir.ast_node import ASTNode
from pairs.ir.lit import Lit


class Assign(ASTNode):
    def __init__(self, sim, dest, src):
        super().__init__(sim)
        self._dest = dest
        self._src = Lit.cvt(sim, src)

    def __str__(self):
        return f"Assign<{self._dest, self._src}>"

    def children(self):
        return [self._dest, self._src]
