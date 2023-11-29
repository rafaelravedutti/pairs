from pairs.ir.actions import Actions
from pairs.ir.arrays import Array, ArrayAccess
from pairs.ir.ast_node import ASTNode
from pairs.ir.scalars import ScalarOp
from pairs.ir.features import FeatureProperty
from pairs.ir.lit import Lit
from pairs.ir.matrices import MatrixOp
from pairs.ir.properties import Property, ContactProperty
from pairs.ir.quaternions import QuaternionOp
from pairs.ir.variables import Var
from pairs.ir.vectors import VectorOp


class Kernel(ASTNode):
    last_kernel = 0

    def __init__(self, sim, name=None, block=None, iterator=None):
        super().__init__(sim)
        self._id = Kernel.last_kernel
        self._name = name if name is not None else "kernel" + str(Kernel.last_kernel)
        self._variables = {}
        self._arrays = {}
        self._properties = {}
        self._contact_properties = {}
        self._feature_properties = {}
        self._array_accesses = set()
        self._scalar_ops = []
        self._vector_ops = []
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

    def array_accesses(self):
        return self._array_accesses

    def scalar_ops(self):
        return self._scalar_ops

    def vector_ops(self):
        return self._vector_ops

    def add_array(self, array, write=False):
        array_list = array if isinstance(array, list) else [array]
        new_op = 'w' if write else 'r'

        for array in array_list:
            assert isinstance(array, Array), \
                "Kernel.add_array(): Element is not of type Array."

            action = Actions.NoAction if array not in self._arrays else self._arrays[array]
            self._arrays[array] = Actions.update_rule(action, new_op)

    def add_variable(self, variable, write=False):
        variable_list = variable if isinstance(variable, list) else [variable]
        new_op = 'w' if write else 'r'

        for var in variable_list:
            if not var.temporary():
                assert isinstance(var, Var), \
                    "Kernel.add_variable(): Element is not of type Var."

                action = Actions.NoAction if var not in self._variables else self._variables[var]
                self._variables[var] = Actions.update_rule(action, new_op)

    def add_property(self, prop, write=False):
        prop_list = prop if isinstance(prop, list) else [prop]
        new_op = 'w' if write else 'r'

        for prop in prop_list:
            assert isinstance(prop, Property), \
                "Kernel.add_property(): Element is not of type Property."

            action = Actions.NoAction if prop not in self._properties else self._properties[prop]
            self._properties[prop] = Actions.update_rule(action, new_op)

    def add_contact_property(self, contact_prop, write=False):
        contact_prop_list = contact_prop if isinstance(contact_prop, list) else [contact_prop]
        new_op = 'w' if write else 'r'

        for contact_prop in contact_prop_list:
            assert isinstance(cp, ContactProperty), \
                "Kernel.add_contact_property(): Element is not of type ContactProperty."

            action = Actions.NoAction if contact_prop not in self._contact_properties else \
                     self._contact_properties[contact_prop]

            self._contact_properties[contact_prop] = Actions.update_rule(action, new_op)

    def add_feature_property(self, feature_prop):
        feature_prop_list = feature_prop if isinstance(feature_prop, list) else [feature_prop]

        for fp in feature_prop_list:
            assert isinstance(fp, FeatureProperty), \
                "Kernel.add_feature_property(): Element is not of type FeatureProperty."

            # Feature properties cannot be written into
            self._feature_properties[fp] = Actions.ReadOnly

    def add_array_access(self, array_access):
        array_access_list = array_access if isinstance(array_access, list) else [array_access]
        for a in array_access_list:
            assert isinstance(a, ArrayAccess), \
                "Kernel.add_array_access(): Element is not of type ArrayAccess."

            self._array_accesses.add(a)

    def add_scalar_op(self, scalar_op):
        scalar_op_list = scalar_op if isinstance(scalar_op, list) else [scalar_op]

        for b in scalar_op_list:
            assert isinstance(b, ScalarOp), \
                "Kernel.add_scalar_op(): Element is not of type ScalarOp."

            self._scalar_ops.append(b)

    def add_vector_op(self, vector_op):
        vector_op_list = vector_op if isinstance(vector_op, list) else [vector_op]

        for b in vector_op_list:
            assert isinstance(b, VectorOp), \
                "Kernel.add_vector_op(): Element is not of type VectorOp."

            self._vector_ops.append(b)

    def add_matrix_op(self, matrix_op):
        matrix_op_list = matrix_op if isinstance(matrix_op, list) else [matrix_op]

        for b in matrix_op_list:
            assert isinstance(b, MatrixOp), \
                "Kernel.add_matrix_op(): Element is not of type MatrixOp."

            self._matrix_ops.append(b)

    def add_quaternion_op(self, quat_op):
        quat_op_list = quat_op if isinstance(quat_op, list) else [quat_op]

        for b in quat_op_list:
            assert isinstance(b, QuaternionOp), \
                "Kernel.add_quaternion_op(): Element is not of type QuaternionOp."

            self._quat_ops.append(b)

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
