from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class Assign(ASTNode):
    def __init__(self, sim, dest, src):
        super().__init__(sim)
        self._dest = dest
        self._src = Lit.cvt(sim, src)

        # When non-scalar assignments occur, all indexes for the dest
        # and source terms must be generated
        if isinstance(self._dest, ASTTerm) and not Types.is_scalar(self._dest.type()):
            for elem in range(Types.number_of_elements(self.sim, self._dest.type())):
                self._dest.add_index_to_generate(elem)

                if isinstance(self._src, ASTTerm) and not Types.is_scalar(self._src.type()):
                    assert self._dest.type() == self._src.type(), \
                        "Non-scalar types must match for assignments."
                    self._src.add_index_to_generate(elem)

        sim.add_statement(self)

    def __str__(self):
        return f"Assign<{self._dest, self._src}>"

    def children(self):
        return [self._dest, self._src]
