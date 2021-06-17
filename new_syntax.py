import part_prot as pt
from ir.assign import Assign
from ir.bin_op import BinOp
import ast
import inspect


def delta(i, j):
    return position[i] - position[j]


def rsq(i, j):
    dp = delta(i, j)
    return dp.x() * dp.x() + dp.y() * dp.y() + dp.z() * dp.z()


def lj(i, j):
    sr2 = 1.0 / rsq
    sr6 = sr2 * sr2 * sr2 * sigma6
    #f = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon
    #force[i] += delta * f
    force[i] += delta * 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon


def euler(i):
    velocity[i] += dt * force[i] / mass[i]
    position[i] += dt * velocity[i]


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
            return '+'

        if isinstance(op, ast.Sub):
            return '-'

        if isinstance(op, ast.Mult):
            return '*'

        if isinstance(op, ast.Div):
            return '/'

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


def add_kernel(sim, func, cutoff_radius=None, position=None, symbols={}):
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

    if nparams == 1:
        for i in sim.particles():
            ir.add_symbols({params[0]: i})
            ir.visit(tree)

    elif nparams == 2:
        for i, j, delta, rsq in psim.particle_pairs(cutoff_radius, sim.property(position)):
            ir.add_symbols({params[0]: i, params[1]: j, 'delta': delta, 'rsq': rsq})
            ir.visit(tree)

    else:
        raise Exception(f"Invalid number of parameters: {nparams}")


dt = 0.005
cutoff_radius = 2.5
skin = 0.3
sigma = 1.0
epsilon = 1.0
sigma6 = sigma ** 6

psim = pt.simulation("lj_ns")
psim.add_real_property('mass', 1.0)
psim.add_vector_property('position')
psim.add_vector_property('velocity')
psim.add_vector_property('force', vol=True)
psim.from_file("data/minimd_setup_4x4x4.input", ['mass', 'position', 'velocity'])
psim.create_cell_lists(2.8, 2.8)
psim.periodic(2.8)
psim.vtk_output("output/test")

add_kernel(psim, lj, cutoff_radius, 'position', {'sigma6': sigma6, 'epsilon': epsilon})
add_kernel(psim, euler, symbols={'dt': dt})

psim.generate()
