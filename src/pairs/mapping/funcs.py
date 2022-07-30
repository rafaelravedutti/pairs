import ast
import inspect
from pairs.ir.assign import Assign
from pairs.ir.bin_op import BinOp
from pairs.ir.loops import ParticleFor
from pairs.ir.operators import Operators
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


class BuildParticleIR(ast.NodeVisitor):
    def get_op(op):
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

    def add_symbols(self, symbols):
        self.ctx_symbols.update(symbols)

    def visit_Assign(self, node):
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
        return BinOp(self.sim, lhs, rhs, BuildParticleIR.get_op(node.op))

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(a) for a in node.args]

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

        as_var = self.sim.var(node.id)
        if as_var is not None:
            return as_var

        return UndefinedSymbol(node.id)

    def visit_Num(self, node):
        return node.n

    def visit_Subscript(self, node):
        #print(ast.dump(node))
        return self.visit(node.value)[self.visit(node.slice)]


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
