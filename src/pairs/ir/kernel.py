from pairs.ir.arrays import Array, ArrayAccess
from pairs.ir.ast_node import ASTNode
from pairs.ir.bin_op import BinOp
from pairs.ir.lit import Lit
from pairs.ir.properties import Property
from pairs.ir.variables import Var


class Kernel(ASTNode):
    last_kernel = 0

    def __init__(self, sim, name=None, block=None, iterator=None):
        super().__init__(sim)
        self._id = Kernel.last_kernel
        self._name = name if name is not None else "kernel" + str(Kernel.last_kernel)
        self._variables = {}
        self._arrays = {}
        self._properties = {}
        self._array_accesses = set()
        self._bin_ops = []
        self._block = block
        self._iterator = iterator
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

    @property
    def iterator(self):
        return self._iterator

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

    def array_accesses(self):
        return self._array_accesses

    def bin_ops(self):
        return self._bin_ops

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

    def add_array_access(self, array_access):
        array_access_list = array_access if isinstance(array_access, list) else [array_access]
        for a in array_access_list:
            assert isinstance(a, ArrayAccess), "Kernel.add_array_access(): given element is not of type ArrayAccess!"
            self._array_accesses.add(a)

    def add_bin_op(self, bin_op):
        bin_op_list = bin_op if isinstance(bin_op, list) else [bin_op]
        for b in bin_op_list:
            assert isinstance(b, BinOp), "Kernel.add_bin_op(): given element is not of type BinOp!"
            self._bin_ops.append(b)

    def children(self):
        return [self._block]


class KernelLaunch(ASTNode):
    def __init__(self, sim, kernel, iterator, range_min, range_max):
        assert isinstance(kernel, Kernel), "KernelLaunch(): given parameter is not of type Kernel!"
        super().__init__(sim)
        self._kernel = kernel
        self._iterator = iterator
        self._range_min = range_min
        self._range_max = range_max
        self._threads_per_block = Lit.cvt(sim, 32)
        self._nelems = (range_max - range_min) 
        self._nblocks = (self._nelems + self._threads_per_block - 1) / self._threads_per_block

    @property
    def kernel(self):
        return self._kernel

    @property
    def min(self):
        return self._range_min

    @property
    def max(self):
        return self._range_max

    @property
    def threads_per_block(self):
        return self._threads_per_block

    @property
    def nblocks(self):
        return self._nblocks

    def children(self):
        return [self._kernel, self._iterator, self._range_min, self._range_max]
