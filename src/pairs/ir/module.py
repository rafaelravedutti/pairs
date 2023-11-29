from pairs.ir.arrays import Array
from pairs.ir.actions import Actions
from pairs.ir.ast_node import ASTNode
from pairs.ir.features import FeatureProperty
from pairs.ir.properties import Property, ContactProperty
from pairs.ir.variables import Var


class Module(ASTNode):
    last_module = 0

    def __init__(self, sim, name=None, block=None, resizes_to_check={}, check_properties_resize=False, run_on_device=False):
        super().__init__(sim)
        self._id = Module.last_module
        self._name = name if name is not None else "module" + str(Module.last_module)
        self._variables = {}
        self._arrays = {}
        self._properties = {}
        self._contact_properties = {}
        self._feature_properties = {}
        self._host_references = set()
        self._block = block
        self._resizes_to_check = resizes_to_check
        self._check_properties_resize = check_properties_resize
        self._run_on_device = run_on_device
        self._profile = False
        sim.add_module(self)
        Module.last_module += 1

    def __str__(self):
        return f"Module<{self._name}>"

    @property
    def module_id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def block(self):
        return self._block

    @property
    def run_on_device(self):
        return self._run_on_device

    def profile(self):
        self._profile = True
        self.sim.enable_profiler()

    def must_profile(self):
        return self._profile

    def variables(self):
        return self._variables

    def read_only_variables(self):
        return [var for var in self._variables if self._variables[var] == Actions.ReadOnly]

    def write_variables(self):
        return [var for var in self._variables if self._variables[var] != Actions.ReadOnly]

    def arrays(self):
        return self._arrays

    def properties(self):
        return self._properties

    def contact_properties(self):
        return self._contact_properties

    def feature_properties(self):
        return self._feature_properties

    def host_references(self):
        return self._host_references

    def add_array(self, array, write=False):
        array_list = array if isinstance(array, list) else [array]
        new_op = 'w' if write else 'r'

        for array in array_list:
            assert isinstance(array, Array), \
                "Module.add_array(): given element is not of type Array."

            action = Actions.NoAction if array not in self._arrays else self._arrays[array]
            self._arrays[array] = Actions.update_rule(action, new_op)

    def add_variable(self, variable, write=False):
        variable_list = variable if isinstance(variable, list) else [variable]
        new_op = 'w' if write else 'r'

        for var in variable_list:
            assert isinstance(var, Var), \
                "Module.add_variable(): given element is not of type Var!"

            action = Actions.NoAction if var not in self._variables else self._variables[var]
            self._variables[var] = Actions.update_rule(action, new_op)

    def add_property(self, prop, write=False):
        prop_list = prop if isinstance(prop, list) else [prop]
        new_op = 'w' if write else 'r'

        for prop in prop_list:
            assert isinstance(prop, Property), \
                "Module.add_property(): given element is not of type Property."

            action = Actions.NoAction if prop not in self._properties else self._properties[prop]
            self._properties[prop] = Actions.update_rule(action, new_op)

    def add_contact_property(self, contact_prop, write=False):
        contact_prop_list = contact_prop if isinstance(contact_prop, list) else [contact_prop]
        new_op = 'w' if write else 'r'

        for contact_prop in contact_prop_list:
            assert isinstance(contact_prop, ContactProperty), \
                "Module.add_contact_property(): given element is not of type ContactProperty."

            action = Actions.NoAction if contact_prop not in self._contact_properties else \
                     self._contact_properties[contact_prop]

            self._contact_properties[contact_prop] = Actions.update_rule(action, new_op)

    def add_feature_property(self, feature_prop):
        feature_prop_list = feature_prop if isinstance(feature_prop, list) else [feature_prop]

        for fp in feature_prop_list:
            assert isinstance(fp, FeatureProperty), \
                "Module.add_feature_property(): given element is not of type FeatureProperty."

            # Feature properties cannot be written into
            self._feature_properties[fp] = Actions.ReadOnly

    def add_host_reference(self, elem):
        self._host_references.add(elem)

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
