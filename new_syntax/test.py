import ast
import inspect


def lj(i, j):
    sr2 = 1.0 / rsq(i, j)
    sr6 = sr2 * sr2 * sr2 * sigma6
    force[i] += delta(i, j) * 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon


def euler(i):
    velocity[i] += dt * force[i] / mass[i]
    position[i] += dt * velocity[i]


class BuildParticleAST(ast.NodeVisitor):
    def __init__(self, sim):
        self.sim = sim
        self.block = Block([])
        self.temp_values = {}

    def visit_Assign(self, node):
        print(node.targets[0].id)
        print(node.value)

    def visit_AugAssign(self, node):
        print(node.targets[0].id)
        print(node.value)


lj_src = inspect.getsource(lj)
#print(ast.dump(ast.parse(lj_src, mode='eval'), indent=4))
tree = ast.parse(lj_src, mode='exec')
build = BuildParticleAST()
build.visit(tree)
#print(ast.dump(ast.parse(lj_src, mode='exec')))
