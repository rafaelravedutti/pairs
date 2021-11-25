from pairs.ir.arrays import Array
from pairs.ir.ast_node import ASTNode
from pairs.ir.properties import Property
from pairs.ir.variables import Var


class Module(ASTNode):
    last_module = 0

    def __init__(self, sim, name=None, block=None, resizes_to_check={}, check_properties_resize=False):
        super().__init__(sim)
        self._name = name if name is not None else "module_" + str(Module.last_module)
        self._variables = {}
        self._arrays = set()
        self._properties = set()
        self._block = block
        self._resizes_to_check = resizes_to_check
        self._check_properties_resize = check_properties_resize
        sim.add_module(self)
        Module.last_module += 1

    @property
    def name(self):
        return self._name

    @property
    def block(self):
        return self._block

    def variables(self):
        return self._variables

    def read_only_variables(self):
        return [v for v in self._variables if not self._variables[v]]

    def write_variables(self):
        return [v for v in self._variables if self._variables[v]]

    def arrays(self):
        return self._arrays

    def properties(self):
        return self._properties

    def add_array(self, array, write=False):
        array_list = array if isinstance(array, list) else [array]
        for a in array_list:
            assert isinstance(a, Array), "Module.add_array(): given element is not of type Array!"
            self._arrays.add(a)

    def add_variable(self, variable, write=False):
        variable_list = variable if isinstance(variable, list) else [variable]
        for v in variable_list:
            assert isinstance(v, Var), "Module.add_variable(): given element is not of type Var!"
            if v not in self._variables:
                self._variables[v] = write
            else:
                self._variables[v] = self._variables[v] or write

    def add_property(self, prop, write=False):
        prop_list = prop if isinstance(prop, list) else [prop]
        for p in prop_list:
            assert isinstance(p, Property), "Module.add_property(): given element is not of type Property!"
            self._properties.add(p)

    def children(self):
        return [self._block]


class Module_Call(ASTNode):
    def __init__(self, sim, module):
        assert isinstance(module, Module), "Module_Call(): given parameter is not of type Module!"
        super().__init__(sim)
        self._module = module

    @property
    def module(self):
        return self._module
