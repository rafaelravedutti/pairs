from pairs.ir.ast_node import ASTNode
from pairs.ir.arrays import Array
from pairs.ir.features import FeatureProperty
from pairs.ir.properties import ContactProperty, Property
from pairs.ir.loops import Iter
from pairs.ir.symbols import Symbol
from pairs.ir.variables import Var
from pairs.sim.interaction import Neighbor


def is_terminal(node):
    terminal_types = (Array, ContactProperty, FeatureProperty, Iter, Neighbor, Property, Symbol, Var)
    return any([isinstance(node, _type) for _type in terminal_types])

