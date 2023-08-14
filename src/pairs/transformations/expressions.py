from pairs.ir.bin_op import BinOp, Decl
from pairs.ir.lit import Lit
from pairs.ir.mutator import Mutator
from pairs.ir.operators import Operators
from pairs.ir.types import Types


class ReplaceSymbols(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)

    def mutate_Symbol(self, ast_node):
        return ast_node.assign_to

    def mutate_SymbolAccess(self, ast_node):
        access = ast_node.symbol()
        for i in ast_node.indexes():
            access = access[i]

        return access


class LowerNeighborIndexes(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)
        self._lower_to_relative = False

    def mutate_ContactPropertyAccess(self, ast_node):
        ast_node.contact_prop = self.mutate(ast_node.contact_prop)
        self._lower_to_relative = True
        ast_node.index = self.mutate(ast_node.index)
        ast_node.expressions = {i: self.mutate(e) for i, e in ast_node.expressions.items()}
        self._lower_to_relative = False
        return ast_node

    def mutate_Neighbor(self, ast_node):
        return ast_node.neighbor_index() if self._lower_to_relative else ast_node.particle_index()


class SimplifyExpressions(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)

    def mutate_BinOp(self, ast_node):
        sim = ast_node.lhs.sim
        ast_node.lhs = self.mutate(ast_node.lhs)
        if not ast_node.operator().is_unary():
            ast_node.rhs = self.mutate(ast_node.rhs)

        ast_node.expressions = {i: self.mutate(e) for i, e in ast_node.expressions.items()}

        if not ast_node.operator().is_unary():
            if ast_node.op in [Operators.Add, Operators.Sub] and ast_node.rhs == 0:
                return ast_node.lhs

            if ast_node.op in [Operators.Add] and ast_node.lhs == 0:
                return ast_node.rhs

            if ast_node.op in [Operators.Mul, Operators.Div] and ast_node.rhs == 1:
                return ast_node.lhs

            if ast_node.op == Operators.Mul and ast_node.lhs == 1:
                return ast_node.rhs

            if ast_node.op == Operators.Mul and ast_node.lhs == 0:
                return Lit(sim, 0 if Types.is_integer(ast_node.type()) else 0.0)

        return ast_node


class PrioritizeScalarOps(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)

    def can_rearrange(op1, op2):
        return op1 == op2 and op1 in [Operators.Add, Operators.Mul]

    def mutate_BinOp(self, ast_node):
        sim = ast_node.sim
        ast_node.lhs = self.mutate(ast_node.lhs)

        if not ast_node.operator().is_unary():
            ast_node.rhs = self.mutate(ast_node.rhs)

            if ast_node.type() == Types.Vector:
                lhs = ast_node.lhs
                rhs = ast_node.rhs
                op = ast_node.op

                if( isinstance(lhs, BinOp) and lhs.type() == Types.Vector and
                    Types.is_real(rhs.type()) and PrioritizeScalarOps.can_rearrange(op, lhs.op) ):

                    if lhs.lhs.type() == Types.Vector and Types.is_real(lhs.rhs.type()):
                        ast_node.reassign(lhs.lhs, BinOp(sim, lhs.rhs, rhs, op), op)
                        #return BinOp(sim, lhs.lhs, BinOp(sim, lhs.rhs, rhs, op), op)

                    if lhs.rhs.type() == Types.Vector and Types.is_real(lhs.lhs.type()):
                        ast_node.reassign(lhs.rhs, BinOp(sim, lhs.lhs, rhs, op), op)
                        #return BinOp(sim, lhs.rhs, BinOp(sim, lhs.lhs, rhs, op), op)

                if( isinstance(rhs, BinOp) and rhs.type() == Types.Vector and
                    Types.is_real(lhs.type()) and PrioritizeScalarOps.can_rearrange(op, rhs.op) ):

                    if rhs.lhs.type() == Types.Vector and Types.is_real(rhs.rhs.type()):
                        ast_node.reassign(rhs.lhs, BinOp(sim, rhs.rhs, lhs, op), op)
                        #return BinOp(sim, rhs.lhs, BinOp(sim, rhs.rhs, lhs, op), op)

                    if rhs.rhs.type() == Types.Vector and Types.is_real(rhs.lhs.type()):
                        ast_node.reassign(rhs.rhs, BinOp(sim, rhs.lhs, lhs, op), op)
                        #return BinOp(sim, rhs.rhs, BinOp(sim, rhs.lhs, lhs, op), op)

        return ast_node


class AddExpressionDeclarations(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.declared_exprs = []
        self.params = []
        self.decls = {}
        self.block_stack = []
        self.writing = False

    def set_data(self, data):
        self.declared_exprs = data[0]

    def push_decl(self, decl):
        assert len(self.block_stack) > 0, "push_decl(): Block stack is empty, cannot push declaration!"
        block_id = self.block_stack[-1]
        self.decls[block_id].append(decl)

    def mutate_ArrayAccess(self, ast_node):
        writing = self.writing
        ast_node.array = self.mutate(ast_node.array)

        self.writing = False
        ast_node.partial_indexes = [self.mutate(i) for i in ast_node.partial_indexes]
        if ast_node.flat_index is not None:
            ast_node.flat_index = self.mutate(ast_node.flat_index)

        self.writing = writing
        if self.writing is False and ast_node.inlined is False:
            array_access_id = id(ast_node)
            if array_access_id not in self.declared_exprs and array_access_id not in self.params:
                self.push_decl(Decl(ast_node.sim, ast_node))
                self.declared_exprs.append(array_access_id)

        return ast_node

    def mutate_Assign(self, ast_node):
        for a in ast_node.assignments:
            self.writing = True
            dst = self.mutate(a[0])
            self.writing = False
            src = self.mutate(a[1])

        return ast_node

    def mutate_AtomicAdd(self, ast_node):
        ast_node.elem = self.mutate(ast_node.elem)
        ast_node.value = self.mutate(ast_node.value)
        atomic_add_id = id(ast_node)
        if atomic_add_id not in self.declared_exprs and atomic_add_id not in self.params:
            self.push_decl(Decl(ast_node.sim, ast_node))
            self.declared_exprs.append(atomic_add_id)

        return ast_node

    def mutate_Block(self, ast_node):
        block_id = id(ast_node)
        self.decls[block_id] = []
        self.block_stack.append(block_id)
        new_stmts = []
        for s in ast_node.stmts:
            new_s = self.mutate(s)
            new_stmts = new_stmts + self.decls[block_id] + [new_s]
            self.decls[block_id] = []

        self.block_stack.pop()
        ast_node.stmts = new_stmts
        return ast_node

    def mutate_BinOp(self, ast_node):
        ast_node.lhs = self.mutate(ast_node.lhs)
        if not ast_node.operator().is_unary():
            ast_node.rhs = self.mutate(ast_node.rhs)

        if ast_node.inlined is False:
            bin_op_id = id(ast_node)
            if bin_op_id not in self.declared_exprs and bin_op_id not in self.params:
                self.push_decl(Decl(ast_node.sim, ast_node))
                self.declared_exprs.append(bin_op_id)

        return ast_node

    def mutate_Decl(self, ast_node):
        self.declared_exprs.append(id(ast_node.elem))
        ast_node.elem = self.mutate(ast_node.elem)
        return ast_node

    def mutate_Kernel(self, ast_node):
        _params = self.params
        self.params = self.params + [id(b) for b in ast_node.bin_ops()]
        ast_node._block = self.mutate(ast_node._block)
        self.params = _params
        return ast_node

    def mutate_MathFunction(self, ast_node):
        ast_node._params = [self.mutate(p) for p in ast_node._params]

        if ast_node.inlined is False:
            math_func_id = id(ast_node)
            if math_func_id not in self.declared_exprs and math_func_id not in self.params:
                self.push_decl(Decl(ast_node.sim, ast_node))
                self.declared_exprs.append(math_func_id)

        return ast_node

    def mutate_PropertyAccess(self, ast_node):
        writing = self.writing
        ast_node.prop = self.mutate(ast_node.prop)
        self.writing = False
        ast_node.index = self.mutate(ast_node.index)
        ast_node.expressions = {i: self.mutate(e) for i, e in ast_node.expressions.items()}
        self.writing = writing

        if self.writing is False and ast_node.inlined is False:
            prop_access_id = id(ast_node)
            if prop_access_id not in self.declared_exprs and prop_access_id not in self.params:
                self.push_decl(Decl(ast_node.sim, ast_node))
                self.declared_exprs.append(prop_access_id)

        return ast_node

    def mutate_Select(self, ast_node):
        ast_node.cond = self.mutate(ast_node.cond)
        ast_node.expr_if = self.mutate(ast_node.expr_if)
        ast_node.expr_else = self.mutate(ast_node.expr_else)

        if ast_node.inlined is False:
            select_id = id(ast_node)
            if select_id not in self.declared_exprs and select_id not in self.params:
                self.push_decl(Decl(ast_node.sim, ast_node))
                self.declared_exprs.append(select_id)

        return ast_node

    def mutate_ContactPropertyAccess(self, ast_node):
        writing = self.writing
        ast_node.contact_prop = self.mutate(ast_node.contact_prop)
        self.writing = False
        ast_node.index = self.mutate(ast_node.index)
        ast_node.expressions = {i: self.mutate(e) for i, e in ast_node.expressions.items()}
        self.writing = writing

        if self.writing is False and ast_node.inlined is False:
            contact_prop_access_id = id(ast_node)
            if contact_prop_access_id not in self.declared_exprs and contact_prop_access_id not in self.params:
                self.push_decl(Decl(ast_node.sim, ast_node))
                self.declared_exprs.append(contact_prop_access_id)

        return ast_node

    def mutate_FeaturePropertyAccess(self, ast_node):
        assert self.writing is False, "Cannot change feature property!"
        ast_node.feature_prop = self.mutate(ast_node.feature_prop)
        ast_node.index = self.mutate(ast_node.index)
        ast_node.expressions = {i: self.mutate(e) for i, e in ast_node.expressions.items()}

        if ast_node.inlined is False:
            feature_prop_access_id = id(ast_node)
            if feature_prop_access_id not in self.declared_exprs and feature_prop_access_id not in self.params:
                self.push_decl(Decl(ast_node.sim, ast_node))
                self.declared_exprs.append(feature_prop_access_id)

        return ast_node
