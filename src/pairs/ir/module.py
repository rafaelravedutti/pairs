from pairs.ir.arrays import Array
from pairs.ir.ast_node import ASTNode
from pairs.ir.properties import Property
from pairs.ir.variables import Var


class Module(ASTNode):
    last_module = 0

    def __init__(self, sim, name=None, block=None, resizes_to_check={}, check_properties_resize=False, run_on_device=False):
        super().__init__(sim)
        self._name = name if name is not None else "module" + str(Module.last_module)
        self._variables = {}
        self._arrays = {}
        self._properties = {}
        self._block = block
        self._resizes_to_check = resizes_to_check
        self._check_properties_resize = check_properties_resize
        self._run_on_device = run_on_device
        sim.add_module(self)
        Module.last_module += 1

    @property
    def name(self):
        return self._name

    @property
    def block(self):
        return self._block

    @property
    def run_on_device(self):
        return self._run_on_device

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
            assert isinstance(a, Array), "Module.add_array(): given element is not of type Array!"
            self._arrays[a] = character if a not in self._arrays else self._arrays[a] + character

    def add_variable(self, variable, write=False):
        variable_list = variable if isinstance(variable, list) else [variable]
        character = 'w' if write else 'r'
        for v in variable_list:
            assert isinstance(v, Var), "Module.add_variable(): given element is not of type Var!"
            self._variables[v] = character if v not in self._variables else self._variables[v] + character

    def add_property(self, prop, write=False):
        prop_list = prop if isinstance(prop, list) else [prop]
        character = 'w' if write else 'r'
        for p in prop_list:
            assert isinstance(p, Property), "Module.add_property(): given element is not of type Property!"
            self._properties[p] = character if p not in self._properties else self._properties[p] + character

    def children(self):
        return [self._block]


class ModuleCall(ASTNode):
    def __init__(self, sim, module):
        assert isinstance(module, Module), "ModuleCall(): given parameter is not of type Module!"
        super().__init__(sim)
        self._module = module

    @property
    def module(self):
        return self._module

    def children(self):
        return [self._module]
