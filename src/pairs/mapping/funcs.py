import ast
import inspect
from pairs.ir.assign import Assign
from pairs.ir.branches import Branch, Filter
from pairs.ir.lit import Lit
from pairs.ir.loops import ParticleFor
from pairs.ir.operators import Operators
from pairs.ir.operator_class import OperatorClass
from pairs.ir.scalars import ScalarOp
from pairs.ir.types import Types
from pairs.mapping.keywords import Keywords
from pairs.sim.flags import Flags
from pairs.sim.interaction import ParticleInteraction


class UndefinedSymbol():
    def __init__(self, symbol_id):
        self.symbol_id = symbol_id

    def type(self):
        return Types.Invalid


class FetchParticleFuncInfo(ast.NodeVisitor):
    def __init__(self):
        self._params = []

    def visit_arg(self, node):
        self._params.append(node.arg)

    def nparams(self):
        return len(self._params)

    def params(self):
        return self._params


class BuildParticleIR(ast.NodeVisitor):
    def get_unary_op(op):
        unary_ops = {
            ast.UAdd: Operators.UAdd,
            ast.USub: Operators.USub,
            ast.Invert: Operators.Invert,
            ast.Not: Operators.Not
        }

        op_type = type(op)
        if op_type in unary_ops:
            return unary_ops[op_type]

        raise Exception("Invalid operator: {}".format(ast.dump(op)))

    def get_binary_op(op):
        binary_ops = {
            ast.Add: Operators.Add,
            ast.Sub: Operators.Sub,
            ast.Mult: Operators.Mul,
            ast.Div: Operators.Div,
            #ast.FloorDiv: Operators.FloorDiv,
            ast.Mod: Operators.Mod,
            ast.Eq: Operators.Eq,
            ast.NotEq: Operators.Neq,
            ast.Gt: Operators.Gt,
            ast.Lt: Operators.Lt,
            ast.GtE: Operators.Geq,
            ast.LtE: Operators.Leq,
            ast.BitAnd: Operators.BinAnd,
            ast.BitOr: Operators.BinOr,
            ast.BitXor: Operators.BinXor,
            #ast.LShift: Operators.LShift,
            #ast.RShift: Operators.RShift,
            ast.And: Operators.And,
            ast.Or: Operators.Or
        }

        op_type = type(op)
        if op_type in binary_ops:
            return binary_ops[op_type]

        raise Exception("Invalid operator: {}".format(ast.dump(op)))

    def __init__(self, sim, ctx_symbols={}):
        self.sim = sim
        self.ctx_symbols = ctx_symbols.copy()
        self.keywords = Keywords(sim)

    def add_symbols(self, symbols):
        self.ctx_symbols.update(symbols)

    def visit_Assign(self, node):
        #print(ast.dump(node))
        assert len(node.targets) == 1, "Only one target is allowed on assignments!"
        lhs = self.visit(node.targets[0])
        rhs = self.visit(node.value)

        if isinstance(lhs, UndefinedSymbol):
            self.add_symbols({lhs.symbol_id: rhs})
            rhs.set_label(lhs.symbol_id)
        else:
            Assign(self.sim, lhs, rhs)

    def visit_AugAssign(self, node):
        lhs = self.visit(node.target)
        rhs = self.visit(node.value)
        op_class = OperatorClass.from_type_list([lhs.type(), rhs.type()])
        bin_op = op_class(self.sim, lhs, rhs, BuildParticleIR.get_binary_op(node.op))

        if isinstance(lhs, UndefinedSymbol):
            self.add_symbols({lhs.symbol_id: bin_op})
            rhs.set_label(lhs.symbol_id)
        else:
            Assign(self.sim, lhs, bin_op)

    def visit_BinOp(self, node):
        #print(ast.dump(node))
        lhs = self.visit(node.left)
        assert not isinstance(lhs, UndefinedSymbol), f"Undefined lhs used in BinOp: {lhs.symbol_id}"
        rhs = self.visit(node.right)
        assert not isinstance(rhs, UndefinedSymbol), f"Undefined rhs used in BinOp: {rhs.symbol_id}"
        operator = BuildParticleIR.get_binary_op(node.op)

        if operator == Operators.Mul:
            if Types.Matrix in (lhs.type(), rhs.type()):
                return self.keywords.keyword_matrix_multiplication([lhs, rhs])

            if Types.Quaternion in (lhs.type(), rhs.type()):
                return self.keywords.keyword_quaternion_multiplication([lhs, rhs])

        op_class = OperatorClass.from_type_list([lhs.type(), rhs.type()])
        return op_class(self.sim, lhs, rhs, BuildParticleIR.get_binary_op(node.op))

    def visit_BoolOp(self, node):
        #print(ast.dump(node))
        lhs = self.visit(node.values[0])
        assert not isinstance(lhs, UndefinedSymbol), f"Undefined lhs used in BoolOp: {lhs.symbol_id}"
        rhs = self.visit(node.values[1])
        assert not isinstance(rhs, UndefinedSymbol), f"Undefined rhs used in BoolOp: {rhs.symbol_id}"
        return ScalarOp(self.sim, lhs, rhs, BuildParticleIR.get_binary_op(node.op))

    def visit_Call(self, node):
        func = self.visit(node.func).symbol_id
        args = [self.visit(a) for a in node.args]

        if self.keywords.exists(func):
            return self.keywords(func, args)

        if func in ['delta', 'squared_distance', 'penetration_depth', 'contact_point', 'contact_normal']:
            return self.ctx_symbols[f"__{func}__"]

        raise Exception(f"Undefined function called: {func}")

    def visit_Constant(self, node):
        return Lit(self.sim, node.value)

    def visit_Compare(self, node):
        #print(ast.dump(node))
        valid_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)
        if len(node.ops) != 1 or not isinstance(node.ops[0], valid_ops):
            raise Exception(f"Chained comparisons or unsupported comparison found: {ast.dump(node)}")

        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        operator = BuildParticleIR.get_binary_op(node.ops[0])
        op_class = OperatorClass.from_type_list([lhs.type(), rhs.type()])
        return op_class(self.sim, lhs, rhs, operator)

    def visit_If(self, node):
        condition = self.visit(node.test)
        one_way = node.orelse is None

        if one_way:
            for _ in Filter(self.sim, condition):
                for stmt in node.body:
                    self.visit(stmt)

        else:
            for test in Branch(self.sim, condition):
                if test:
                    for stmt in node.body:
                        self.visit(stmt)
                else:
                    for stmt in node.orelse:
                        self.visit(stmt)

    def visit_IfExp(self, node):
        condition = self.visit(node.test)
        for test in Branch(self.sim, condition):
            if test:
                for stmt in node.body:
                    self.visit(stmt)
            else:
                for stmt in node.orelse:
                    self.visit(stmt)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        symbol_types = [
            self.ctx_symbols.get,
            self.sim.array,
            self.sim.property,
            self.sim.feature_property,
            self.sim.contact_property,
            self.sim.var
        ]
        
        for symbol_func in symbol_types:
            result = symbol_func(node.id)
            if result is not None:
                return result

        return UndefinedSymbol(node.id)

    def visit_Num(self, node):
        return Lit(self.sim, node.n)

    def visit_Subscript(self, node):
        #print(ast.dump(node))
        return self.visit(node.value)[self.visit(node.slice)]

    def visit_Tuple(self, node):
        #print(ast.dump(node))
        return tuple(self.visit(v) for v in node.elts)

    def visit_UnaryOp(self, node):
        #print(ast.dump(node))
        operand = self.visit(node.operand)
        assert not isinstance(operand, UndefinedSymbol), \
            f"Undefined operand used in UnaryOp: {operand.symbol_id}"
        op_class = OperatorClass.from_type(operand.type())
        return op_class(self.sim, operand, None, BuildParticleIR.get_unary_op(node.op))


def compute(sim, func, cutoff_radius=None, symbols={}):
    src = inspect.getsource(func)
    tree = ast.parse(src, mode='exec')
    #print(ast.dump(ast.parse(src, mode='exec')))

    # Fetch function info
    info = FetchParticleFuncInfo()
    info.visit(tree)
    params = info.params()
    nparams = info.nparams()

    # Compute functions must have parameters
    assert nparams > 0, "Number of parameters from compute functions must be higher than zero!"

    # Convert literal symbols
    symbols = {symbol: Lit.cvt(sim, value) for symbol, value in symbols.items()}

    sim.init_block()
    sim.module_name(func.__name__)

    if nparams == 1:
        for i in ParticleFor(sim):
            for _ in Filter(sim, ScalarOp.cmp(sim.particle_flags[i] & Flags.Fixed, 0)):
                ir = BuildParticleIR(sim, symbols)
                ir.add_symbols({params[0]: i})
                ir.visit(tree)

    else:
        for interaction_data in ParticleInteraction(sim, nparams, cutoff_radius):
            # Start building IR
            ir = BuildParticleIR(sim, symbols)
            ir.add_symbols({
                params[0]: interaction_data.i(),
                params[1]: interaction_data.j(),
                '__delta__': interaction_data.delta(),
                '__squared_distance__': interaction_data.squared_distance(),
                '__penetration_depth__': interaction_data.penetration_depth(),
                '__contact_point__': interaction_data.contact_point(),
                '__contact_normal__': interaction_data.contact_normal()
            })

            ir.visit(tree)

    sim.build_module_with_statements()


def setup(sim, func, symbols={}):
    src = inspect.getsource(func)
    tree = ast.parse(src, mode='exec')

    # Fetch function info
    info = FetchParticleFuncInfo()
    info.visit(tree)
    params = info.params()
    nparams = info.nparams()

    # Compute functions must have parameters
    assert nparams == 1, "Number of parameters from setup functions must be one!"

    # Convert literal symbols
    symbols = {symbol: Lit.cvt(sim, value) for symbol, value in symbols.items()}

    sim.init_block()
    sim.module_name(func.__name__)

    for i in ParticleFor(sim):
        ir = BuildParticleIR(sim, symbols)
        ir.add_symbols({params[0]: i})
        ir.visit(tree)

    sim.build_setup_module_with_statements()
