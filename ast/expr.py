from ast.assign import Assign
from ast.data_types import Type_Float, Type_Bool, Type_Vector
from ast.lit import as_lit_ast
from ast.properties import Property


class BinOpDef:
    def __init__(self, bin_op):
        self.bin_op = bin_op
        self.bin_op.sim.add_statement(self)

    def __str__(self):
        return f"BinOpDef<bin_op: self.bin_op>"

    def children(self):
        return [self.bin_op]

    def generate(self, mem=False):
        bin_op = self.bin_op

        if not isinstance(bin_op, BinOp):
            return None

        if bin_op.inlined is False and bin_op.operator() != '[]' and bin_op.generated is False:
            if bin_op.kind() == BinOp.Kind_Scalar:
                lhs = bin_op.lhs.generate(bin_op.mem)
                rhs = bin_op.rhs.generate()
                bin_op.sim.code_gen.generate_expr(bin_op.id(), bin_op.type(), lhs, rhs, bin_op.op)

            elif bin_op.kind() == BinOp.Kind_Vector:
                for i in bin_op.vector_indexes():
                    lhs = bin_op.lhs.generate(bin_op.mem, index=i)
                    rhs = bin_op.rhs.generate(index=i)
                    bin_op.sim.code_gen.generate_vec_expr(bin_op.id(), i, lhs, rhs, bin_op.operator(), bin_op.mem)

            else:
                raise Exception("Invalid BinOp kind!")

            bin_op.generated = True

    def transform(self, fn):
        self.bin_op = self.bin_op.transform(fn)
        return fn(self)


class BinOp:
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
        self.sim = sim
        self.bin_op_id = BinOp.new_id()
        self.lhs = as_lit_ast(sim, lhs)
        self.rhs = as_lit_ast(sim, rhs)
        self.op = op
        self.mem = mem
        self.mutable = self.lhs.is_mutable() or self.rhs.is_mutable() # Value can change accross references
        self.inlined = False
        self.generated = False
        self.bin_op_type = BinOp.infer_type(self.lhs, self.rhs, self.op)
        self.bin_op_scope = None
        self.bin_op_vector_indexes = set()
        self.bin_op_vector_index_mapping = {}
        self.bin_op_def = BinOpDef(self)

    def __str__(self):
        return f"BinOp<a: {self.lhs.id()}, b: {self.rhs.id()}, op: {self.op}>"

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

    def operator(self):
        return self.op

    def kind(self):
        return BinOp.Kind_Vector if self.type() == Type_Vector else BinOp.Kind_Scalar

    def is_mutable(self):
        return self.mutable

    def scope(self):
        if self.bin_op_scope is None:
            lhs_scp = self.lhs.scope()
            rhs_scp = self.rhs.scope()
            self.bin_op_scope = lhs_scp if lhs_scp > rhs_scp else rhs_scp

        return self.bin_op_scope

    def children(self):
        return [self.lhs, self.rhs]

    def generate(self, mem=False, index=None):
        if isinstance(self.lhs, BinOp) and self.lhs.kind() == BinOp.Kind_Vector and self.op == '[]':
            return self.lhs.generate(self.mem, self.rhs.generate())

        lhs = self.lhs.generate(mem, index)
        rhs = self.rhs.generate(index=index)

        if self.op == '[]':
            idx = self.mapped_vector_index(index).generate() if self.is_vector_property_access() else rhs
            return self.sim.code_gen.generate_expr_access(lhs, idx, self.mem)

        if self.inlined is True:
            assert self.bin_op_type != Type_Vector, "Vector operations cannot be inlined!"
            return self.sim.code_gen.generate_inline_expr(lhs, rhs, self.op)

        # Some expressions can be defined on-the-fly during transformations, hence they do not have
        # a definition statement, so we generate them right before usage
        if not self.generated:
            self.bin_op_def.generate()

        if self.kind() == BinOp.Kind_Vector:
            assert index is not None, "Index must be set for vector reference!"
            return self.sim.code_gen.generate_vec_expr_ref(self.id(), index, self.mem)

        return self.sim.code_gen.generate_expr_ref(self.id())

    def transform(self, fn):
        self.lhs = self.lhs.transform(fn)
        self.rhs = self.rhs.transform(fn)
        return fn(self)

