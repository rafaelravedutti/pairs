from ast.ast_node import ASTNode
from ast.assign import Assign
from ast.data_types import Type_Float, Type_Bool, Type_Vector
from ast.lit import as_lit_ast
from ast.properties import Property


class BinOpDef(ASTNode):
    def __init__(self, bin_op):
        super().__init__(bin_op.sim)
        self.bin_op = bin_op
        self.bin_op.sim.add_statement(self)

    def __str__(self):
        return f"BinOpDef<bin_op: self.bin_op>"

    def children(self):
        return [self.bin_op]

    def transform(self, fn):
        self.bin_op = self.bin_op.transform(fn)
        return fn(self)


class BinOp(ASTNode):
    # BinOp kinds
    Kind_Scalar = 0
    Kind_Vector = 1

    last_bin_op = 0

    def new_id():
        BinOp.last_bin_op += 1
        return BinOp.last_bin_op - 1

    def inline(op):
        if not isinstance(op, BinOp):
            return op

        return op.inline_rec()

    def __init__(self, sim, lhs, rhs, op, mem=False):
        super().__init__(sim)
        self.bin_op_id = BinOp.new_id()
        self.lhs = as_lit_ast(sim, lhs)
        self.rhs = as_lit_ast(sim, rhs)
        self.op = op
        self.mem = mem
        self.inlined = False
        self.generated = False
        self.bin_op_type = BinOp.infer_type(self.lhs, self.rhs, self.op)
        self.bin_op_scope = None
        self.bin_op_vector_indexes = set()
        self.bin_op_vector_index_mapping = {}
        self.bin_op_def = BinOpDef(self)

    def __str__(self):
        return f"BinOp<a: {self.lhs.id()}, b: {self.rhs.id()}, op: {self.op}>"

    def match(self, bin_op):
        return self.lhs == bin_op.lhs and \
               self.rhs == bin_op.rhs and \
               self.op == bin_op.operator()

    def x(self):
        return self.__getitem__(0)

    def y(self):
        return self.__getitem__(1)

    def z(self):
        return self.__getitem__(2)

    def map_vector_index(self, index, expr):
        self.bin_op_vector_index_mapping[index] = expr

    def mapped_vector_index(self, index):
        mapping = self.bin_op_vector_index_mapping
        return mapping[index] if index in mapping else as_lit_ast(self.sim, index)

    def vector_indexes(self):
        return self.bin_op_vector_indexes

    def propagate_vector_access(self, index):
        self.bin_op_vector_indexes.add(index)

        if isinstance(self.lhs, BinOp) and self.lhs.kind() == BinOp.Kind_Vector:
            self.lhs.propagate_vector_access(index)

        if isinstance(self.rhs, BinOp) and self.rhs.kind() == BinOp.Kind_Vector:
            self.rhs.propagate_vector_access(index)

    def __getitem__(self, index):
        assert self.type() == Type_Vector, "Cannot use operator [] on specified type!"
        self.propagate_vector_access(index)
        return BinOp(self.sim, self, as_lit_ast(self.sim, index), '[]', self.mem)

    def is_property_access(self):
        return isinstance(self.lhs, Property) and self.operator() == '[]'

    def is_vector_property_access(self):
        return self.is_property_access() and self.type() == Type_Vector

    def set(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def sub(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.add_statement(Assign(self.sim, self, self - other))

    def infer_type(lhs, rhs, op):
        lhs_type = lhs.type()
        rhs_type = rhs.type()

        if op in ['>', '<', '>=', '<=', '==', '!=']:
            return Type_Bool

        if op == '[]':
            if isinstance(lhs, Property):
                return lhs_type

            if lhs_type == Type_Vector:
                return Type_Float

            return lhs_type

        if lhs_type == rhs_type:
            return lhs_type

        if lhs_type == Type_Vector or rhs_type == Type_Vector:
            return Type_Vector

        if lhs_type == Type_Float or rhs_type == Type_Float:
            return Type_Float

        return None

    def inline_rec(self):
        self.inlined = True

        if isinstance(self.lhs, BinOp):
            self.lhs.inline_rec()

        if isinstance(self.rhs, BinOp):
            self.rhs.inline_rec()

        return self

    def id(self):
        return self.bin_op_id

    def type(self):
        return self.bin_op_type

    def definition(self):
        return self.bin_op_def

    def operator(self):
        return self.op

    def kind(self):
        return BinOp.Kind_Vector if self.type() == Type_Vector else BinOp.Kind_Scalar

    def scope(self):
        if self.bin_op_scope is None:
            lhs_scp = self.lhs.scope()
            rhs_scp = self.rhs.scope()
            self.bin_op_scope = lhs_scp if lhs_scp > rhs_scp else rhs_scp

        return self.bin_op_scope

    def children(self):
        return [self.lhs, self.rhs]

    def transform(self, fn):
        self.lhs = self.lhs.transform(fn)
        self.rhs = self.rhs.transform(fn)
        self.bin_op_vector_index_mapping = {i: e.transform(fn) for i, e in self.bin_op_vector_index_mapping.items()}
        return fn(self)

    def __add__(self, other):
        return BinOp(self.sim, self, other, '+')

    def __radd__(self, other):
        return BinOp(self.sim, other, self, '+')

    def __sub__(self, other):
        return BinOp(self.sim, self, other, '-')

    def __mul__(self, other):
        return BinOp(self.sim, self, other, '*')

    def __rmul__(self, other):
        return BinOp(self.sim, other, self, '*')

    def __truediv__(self, other):
        return BinOp(self.sim, self, other, '/')

    def __rtruediv__(self, other):
        return BinOp(self.sim, other, self, '/')

    def __lt__(self, other):
        return BinOp(self.sim, self, other, '<')

    def __le__(self, other):
        return BinOp(self.sim, self, other, '<=')

    def __gt__(self, other):
        return BinOp(self.sim, self, other, '>')

    def __ge__(self, other):
        return BinOp(self.sim, self, other, '>=')

    def and_op(self, other):
        return BinOp(self.sim, self, other, '&&')

    def cmp(lhs, rhs):
        return BinOp(lhs.sim, lhs, rhs, '==')

    def neq(lhs, rhs):
        return BinOp(lhs.sim, lhs, rhs, '!=')

    def inv(self):
        return BinOp(self.sim, 1.0, self, '/')

    def __mod__(self, other):
        return BinOp(self.sim, self, other, '%')


class ASTTerm(ASTNode):
    def __init__(self, sim):
        super().__init__(sim)

    def __add__(self, other):
        return BinOp(self.sim, self, other, '+')

    def __radd__(self, other):
        return BinOp(self.sim, other, self, '+')

    def __sub__(self, other):
        return BinOp(self.sim, self, other, '-')

    def __mul__(self, other):
        return BinOp(self.sim, self, other, '*')

    def __rmul__(self, other):
        return BinOp(self.sim, other, self, '*')

    def __truediv__(self, other):
        return BinOp(self.sim, self, other, '/')

    def __rtruediv__(self, other):
        return BinOp(self.sim, other, self, '/')

    def __lt__(self, other):
        return BinOp(self.sim, self, other, '<')

    def __le__(self, other):
        return BinOp(self.sim, self, other, '<=')

    def __gt__(self, other):
        return BinOp(self.sim, self, other, '>')

    def __ge__(self, other):
        return BinOp(self.sim, self, other, '>=')

    def and_op(self, other):
        return BinOp(self.sim, self, other, '&&')

    def cmp(lhs, rhs):
        return BinOp(lhs.sim, lhs, rhs, '==')

    def neq(lhs, rhs):
        return BinOp(lhs.sim, lhs, rhs, '!=')

    def inv(self):
        return BinOp(self.sim, 1.0, self, '/')

    def __mod__(self, other):
        return BinOp(self.sim, self, other, '%')
