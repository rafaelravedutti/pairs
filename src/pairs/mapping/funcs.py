import ast
import inspect
from pairs.ir.assign import Assign
from pairs.ir.bin_op import BinOp
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.const_vector import ZeroVector
from pairs.ir.lit import Lit
from pairs.ir.loops import ParticleFor, Continue
from pairs.ir.math import Sqrt
from pairs.ir.operators import Operators
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.interaction import ParticleInteraction


class UndefinedSymbol():
    def __init__(self, symbol_id):
        self.symbol_id = symbol_id


class FetchParticleFuncInfo(ast.NodeVisitor):
    def __init__(self):
        self._params = []

    def visit_arg(self, node):
        self._params.append(node.arg)

    def nparams(self):
        return len(self._params)

    def params(self):
        return self._params


class Keywords:
    def __init__(self, sim):
        self.sim = sim

    def get_method(self, method_name):
        method = getattr(self, method_name, None)
        return method if callable(method) else None

    def __call__(self, keyword, args):
        method = self.get_method(f"keyword_{keyword}")
        assert method is not None, "Invalid keyword: {keyword}"
        return method(args)

    def exists(self, keyword):
        method = self.get_method(f"keyword_{keyword}")
        return method is not None

    def keyword_select(self, args):
        assert len(args) == 3, "select() keyword requires three parameter!"
        return Select(self.sim, args[0], args[1], args[2])

    def keyword_skip_when(self, args):
        assert len(args) == 1, "skip_when() keyword requires one parameter!"
        return Filter(self.sim, args[0], Block(self.sim, [Continue(self.sim)]))

    def keyword_min(self, args):
        e_min = args[0]
        for a in args[1:]:
            e_min = Min(self.sim, e_min, a)

        return e_min

    def keyword_max(self, args):
        e_max = args[0]
        for a in args[1:]:
            e_max = Max(self.sim, e_max, a)

        return e_max

    def keyword_length(self, args):
        assert len(args) == 1, "length() keyword requires one parameter!"
        vector = args[0]
        assert vector.type() == Types.Vector, "length(): Argument must be a vector!"
        return Sqrt(self.sim, sum([vector[d] * vector[d] for d in range(self.sim.ndims())]))

    def keyword_dot(self, args):
        assert len(args) == 2, "dot() keyword requires two parameters!"
        vector1 = args[0]
        vector2 = args[1]
        assert vector1.type() == Types.Vector, "dot(): First argument must be a vector!"
        assert vector2.type() == Types.Vector, "dot(): Second argument must be a vector!"
        return sum([vector1[d] * vector2[d] for d in range(self.sim.ndims())])

    def keyword_normalized(self, args):
        assert len(args) == 1, "normalized() keyword requires one parameter!"
        vector = args[0]
        assert vector.type() == Types.Vector, "length(): Argument must be a vector!"
        length = self.keyword_length([vector])
        inv_length = Lit(self.sim, 1.0) / length
        return Select(self.sim, length > Lit(self.sim, 0.0), vector * inv_length, ZeroVector(self.sim))

    def keyword_square_length(self, args):
        assert len(args) == 1, "length() keyword requires one parameter!"
        vector = args[0]
        assert vector.type() == Types.Vector, "length(): Argument must be a vector!"
        return sum([vector[d] * vector[d] for d in range(self.sim.ndims())])


class BuildParticleIR(ast.NodeVisitor):
    def get_unary_op(op):
        if isinstance(op, ast.UAdd):
            return Operators.UAdd

        if isinstance(op, ast.USub):
            return Operators.USub

        if isinstance(op, ast.Invert):
            return Operators.Invert

        if isinstance(op, ast.Not):
            return Operators.Not

        raise Exception("Invalid operator: {}".format(ast.dump(op)))

    def get_binary_op(op):
        if isinstance(op, ast.Add):
            return Operators.Add

        if isinstance(op, ast.Sub):
            return Operators.Sub

        if isinstance(op, ast.Mult):
            return Operators.Mul

        if isinstance(op, ast.Div):
            return Operators.Div

        raise Exception("Invalid operator: {}".format(ast.dump(op)))

    def parse_function_and_get_return_value(func, args):
        return None

    def __init__(self, sim, ctx_symbols={}, ctx_calls=[]):
        self.sim = sim
        self.ctx_symbols = ctx_symbols
        self.ctx_calls = ctx_calls
        self.keywords = Keywords(sim)

    def add_symbols(self, symbols):
        self.ctx_symbols.update(symbols)

    def visit_Assign(self, node):
        print(ast.dump(node))
        assert len(node.targets) == 1, "Only one target is allowed on assignments!"
        lhs = self.visit(node.targets[0])
        rhs = self.visit(node.value)

        if isinstance(lhs, UndefinedSymbol):
            self.add_symbols({lhs.symbol_id: rhs})
        else:
            lhs.set(rhs)

    def visit_AugAssign(self, node):
        lhs = self.visit(node.target)
        rhs = self.visit(node.value)

        if isinstance(lhs, UndefinedSymbol):
            self.add_symbols({lhs.symbol_id: rhs})
        else:
            lhs.add(rhs)

    def visit_BinOp(self, node):
        #print(ast.dump(node))
        lhs = self.visit(node.left)
        assert not isinstance(lhs, UndefinedSymbol), f"Undefined lhs used in BinOp: {lhs.symbol_id}"
        rhs = self.visit(node.right)
        assert not isinstance(rhs, UndefinedSymbol), f"Undefined rhs used in BinOp: {rhs.symbol_id}"
        return BinOp(self.sim, lhs, rhs, BuildParticleIR.get_binary_op(node.op))

    def visit_Call(self, node):
        func = self.visit(node.func).symbol_id
        args = [self.visit(a) for a in node.args]

        if self.keywords.exists(func):
            return self.keywords(func, args)

        for c in self.ctx_calls:
            if c['func'] == func and len(c['args']) == len(args) and all([c['args'][a] == args[a] for a in range(0, len(args))]):
                return c['value']

        value = BuildParticleIR.parse_function_and_get_return_value(func, args)
        self.ctx_calls.append({'func': func, 'args': args, 'value': value})
        return value

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        as_sym = self.ctx_symbols[node.id] if node.id in self.ctx_symbols else None
        if as_sym is not None:
            return as_sym

        as_array = self.sim.array(node.id)
        if as_array is not None:
            return as_array

        as_prop = self.sim.property(node.id)
        if as_prop is not None:
            return as_prop

        as_feature_prop = self.sim.feature_property(node.id)
        if as_feature_prop is not None:
            return as_feature_prop

        as_contact_prop = self.sim.contact_property(node.id)
        if as_contact_prop is not None:
            return as_contact_prop

        as_var = self.sim.var(node.id)
        if as_var is not None:
            return as_var

        return UndefinedSymbol(node.id)

    def visit_Num(self, node):
        return node.n

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
        return BinOp(self.sim, operand, None, BuildParticleIR.get_unary_op(node.op))


def compute(sim, func, cutoff_radius=None, symbols={}):
    src = inspect.getsource(func)
    tree = ast.parse(src, mode='exec')
    #print(ast.dump(ast.parse(src, mode='exec')))

    # Fetch function info
    info = FetchParticleFuncInfo()
    info.visit(tree)
    params = info.params()
    nparams = info.nparams()

    # Start building IR
    ir = BuildParticleIR(sim, symbols)
    assert nparams > 0, "Number of parameters from compute functions must be higher than zero!"

    sim.init_block()
    sim.module_name(func.__name__)

    if nparams == 1:
        for i in ParticleFor(sim):
            ir.add_symbols({params[0]: i})
            ir.visit(tree)

    else:
        pairs = ParticleInteraction(sim, nparams, cutoff_radius)
        for i, j in pairs:
            ir.add_symbols({params[0]: i, params[1]: j, 'delta': pairs.delta(), 'rsq': pairs.squared_distance()})
            ir.visit(tree)

    sim.build_module_with_statements()
