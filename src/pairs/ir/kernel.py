from pairs.ir.arrays import Array
from pairs.ir.ast_node import ASTNode
from pairs.ir.properties import Property
from pairs.ir.variables import Var


class Kernel(ASTNode):
    last_kernel = 0

    def __init__(self, sim, name=None, block=None):
        super().__init__(sim)
        self._id = Kernel.last_kernel
        self._name = name if name is not None else "kernel" + str(Kernel.last_kernel)
        self._variables = {}
        self._arrays = {}
        self._properties = {}
        self._bin_ops = []
        self._block = block
        sim.add_kernel(self)
        Kernel.last_kernel += 1

    @property
    def kernel_id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def block(self):
        return self._block

    def variables(self):
        return self._variables

    def read_only_variables(self):
        return [v for v in self._variables if 'w' not in self._variables[v]]

    def write_variables(self):
        return [v for v in self._variables if 'w' in self._variables[v]]

    def arrays(self):
        return self._arrays

    def properties(self):
        return self._properties

    def properties_to_synchronize(self):
        return {p for p in self._properties if self._properties[p][0] == 'r'}

    def write_properties(self):
        return {p for p in self._properties if 'w' in self._properties[p]}

    def add_array(self, array, write=False):
        array_list = array if isinstance(array, list) else [array]
        character = 'w' if write else 'r'
        for a in array_list:
            assert isinstance(a, Array), "Kernel.add_array(): given element is not of type Array!"
            self._arrays[a] = character if a not in self._arrays else self._arrays[a] + character

    def add_variable(self, variable, write=False):
        variable_list = variable if isinstance(variable, list) else [variable]
        character = 'w' if write else 'r'
        for v in variable_list:
            assert isinstance(v, Var), "Kernel.add_variable(): given element is not of type Var!"
            self._variables[v] = character if v not in self._variables else self._variables[v] + character

    def add_property(self, prop, write=False):
        prop_list = prop if isinstance(prop, list) else [prop]
        character = 'w' if write else 'r'
        for p in prop_list:
            assert isinstance(p, Property), "Kernel.add_property(): given element is not of type Property!"
            self._properties[p] = character if p not in self._properties else self._properties[p] + character

    def add_bin_op(self, bin_op):
        bin_op_list = bin_op if isinstance(bin_op, list) else [bin_op]
        for b in bin_op_list:
            assert isinstance(b, BinOp), "Kernel.add_bin_op(): given element is not of type BinOp!"
            self._bin_ops.append(b)

    def children(self):
        return [self._block]


class KernelLaunch(ASTNode):
    def __init__(self, sim, kernel, iterator, range_min, range_max):
        assert isinstance(module, Kernel), "KernelLaunch(): given parameter is not of type Kernel!"
        super().__init__(sim)
        self._kernel = kernel
        self._iterator = iterator
        self._range_min = range_min
        self._range_max = range_max

    @property
    def kernel(self):
        return self._kernel
