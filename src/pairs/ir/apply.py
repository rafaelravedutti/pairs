from pairs.ir.assign import Assign
from pairs.ir.ast_node import ASTNode
from pairs.ir.block import pairs_inline
from pairs.ir.branches import Filter
from pairs.ir.lit import Lit
from pairs.ir.properties import Property
from pairs.ir.scalars import ScalarOp
from pairs.ir.vectors import Vector
from pairs.ir.types import Types
from pairs.sim.flags import Flags
from pairs.sim.lowerable import FinalLowerable, Lowerable


class Apply(Lowerable):
    def __init__(self, sim, prop, expr, j):
        assert isinstance(prop, Property), "Apply(): Destination must of Property type."
        assert prop.type() == expr.type(), "Apply(): Property and expression must be of same type."
        assert sim.current_apply_list() is not None, "Apply(): Not used within particle interaction."
        super().__init__(sim)
        self._prop = prop
        self._expr = Lit.cvt(sim, expr)
        self._j = j
        self._reduction_variable = None
        sim.current_apply_list().add(self)
        sim.add_statement(self)

    def __str__(self):
        return f"Apply<{self._prop, self._expr}>"

    def prop(self):
        return self._prop

    def expression(self):
        return self._expr

    def add_reduction_variable(self):
        self._reduction_variable = self.sim.add_temp_var([0.0, 0.0, 0.0])

    def reduction_variable(self):
        return self._reduction_variable

    def children(self):
        return [self._prop, self._expr, self._reduction_variable, self._j]

    @pairs_inline
    def lower(self):
        if self.sim._compute_half:
            for _ in Filter(self.sim,
                            ScalarOp.and_op(self._j < self.sim.nlocal,
                                            ScalarOp.cmp(self.sim.particle_flags[self._j] & Flags.Fixed, 0))):

                Assign(self.sim, self._prop[self._j], self._prop[self._j] - self._expr)

        Assign(self.sim, self._reduction_variable, self._reduction_variable + self._expr)
