from pairs.ir.assign import Assign
from pairs.ir.ast_node import ASTNode
from pairs.ir.atomic import AtomicInc
from pairs.ir.block import pairs_inline
from pairs.ir.branches import Filter
from pairs.ir.lit import Lit
from pairs.ir.mutator import Mutator
from pairs.ir.properties import Property, PropertyAccess
from pairs.ir.scalars import ScalarOp
from pairs.ir.vectors import Vector, VectorAccess, VectorOp
from pairs.ir.types import Types
from pairs.sim.flags import Flags
from pairs.sim.lowerable import FinalLowerable, Lowerable


class Apply(Lowerable):
    def __init__(self, sim, prop, expr, i, j):
        assert isinstance(prop, Property), "Apply(): Destination must of Property type."
        assert prop.type() == expr.type(), "Apply(): Property and expression must be of same type."
        assert sim.current_apply_list() is not None, "Apply(): Not used within particle interaction."
        super().__init__(sim)
        self._prop = prop
        self._expr = Lit.cvt(sim, expr)
        self._i = i
        self._j = j
        self._expr_i = self.build_expression_with_index(self._expr, self._i)
        self._expr_j = self.build_expression_with_index(self._expr, self._j) if sim._compute_half else None
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
        return [self._prop, self._i, self._j] + \
               [self._reduction_variable] if self._reduction_variable is not None else [] + \
               [self._expr] if self._expr is not None else [] + \
               [self._expr_i] if self._expr_i is not None else [] + \
               [self._expr_j] if self._expr_j is not None else []

    def build_expression_with_index(self, expr, index):
        return self._build_expression_with_index(expr, index)[0]

    # TODO: This method should comprise all operators and dynamic data types, it would also be
    # better to provide a way to implement it with a Mutator or Visitor
    def _build_expression_with_index(self, expr, index):
        if isinstance(expr, (ScalarOp, VectorOp)):
            new_lhs, changed_lhs = self._build_expression_with_index(expr.lhs, index)
            changed_rhs = False

            if not expr.operator().is_unary():
                new_rhs, changed_rhs = self._build_expression_with_index(expr.rhs, index)

            if changed_lhs or changed_rhs:
                if isinstance(expr, ScalarOp):
                    return (ScalarOp(self.sim, new_lhs, new_rhs, expr.operator(), expr.mem), True)

                if isinstance(expr, VectorOp):
                    return (VectorOp(self.sim, new_lhs, new_rhs, expr.operator(), expr.mem), True)

            return (expr, False)

        if isinstance(expr, Vector):
            values = []
            changed = False

            for value in expr._values:
                new_value, changed_value = self._build_expression_with_index(value, index)
                values.append(new_value)
                changed = changed or changed_value

            if changed:
                return (Vector(self.sim, values), True)

            return (expr, False)

        if isinstance(expr, VectorAccess):
            new_expr, changed = self._build_expression_with_index(expr.expr, index)

            if changed:
                return (VectorAccess(self.sim, new_expr, expr.index), True)

            return (expr, False)

        if isinstance(expr, PropertyAccess):
            return (expr, False)

        if isinstance(expr, Property):
            return (expr[index], True)

        changed = False
        for child in expr.children():
            _, changed_child = self._build_expression_with_index(child, index)
            changed = changed or changed_child

        return (expr, changed)

    @pairs_inline
    def lower(self):
        Assign(self.sim, self._reduction_variable, self._reduction_variable + self._expr_i)

        if self.sim._compute_half:
            for _ in Filter(self.sim,
                            ScalarOp.and_op(self._j < self.sim.nlocal,
                                            ScalarOp.cmp(self.sim.particle_flags[self._j] & Flags.Fixed, 0))):

                if Types.is_scalar(self._prop.type()):
                    AtomicInc(self.sim, self._prop[self._j], -self._expr_j)

                else:
                    for d in range(Types.number_of_elements(self.sim, self._prop.type())):
                        AtomicInc(self.sim, self._prop[self._j][d], -(self._expr_j[d]))
