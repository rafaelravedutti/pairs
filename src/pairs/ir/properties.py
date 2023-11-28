from pairs.ir.accessor_class import AccessorClass
from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.declaration import Decl
from pairs.ir.layouts import Layouts
from pairs.ir.lit import Lit
from pairs.ir.operator_class import OperatorClass
from pairs.ir.scalars import ScalarOp
from pairs.ir.types import Types


class Properties:
    def __init__(self, sim):
        self.sim = sim
        self.props = []
        self.defs = {}

    def add(self, p_name, p_type, p_value, p_volatile, p_layout=Layouts.AoS):
        p = Property(self.sim, p_name, p_type, p_value, p_volatile, p_layout)
        self.props.append(p)
        self.defs[p_name] = p_value
        return p

    def defaults(self):
        return self.defs

    def all(self):
        return self.props

    def volatiles(self):
        return [p for p in self.props if p.volatile is True]

    def non_volatiles(self):
        return [p for p in self.props if p.volatile is False]

    def nprops(self):
        return len(self.props)

    def find(self, p_name):
        prop = [p for p in self.props if p.name() == p_name]
        if prop:
            return prop[0]

        return None

    def __iter__(self):
        yield from self.props


class Property(ASTNode):
    last_prop_id = 0

    def __init__(self, sim, name, dtype, default, volatile, layout=Layouts.AoS):
        super().__init__(sim)
        self.prop_id = Property.last_prop_id
        self.prop_name = name
        self.prop_type = dtype
        self.prop_layout = layout
        self.default_value = default
        self.volatile = volatile
        self.device_flag = False
        Property.last_prop_id += 1

    def __str__(self):
        return f"Property<{self.prop_name}>"

    def __getitem__(self, expr):
        return PropertyAccess(self.sim, self, expr)

    def id(self):
        return self.prop_id

    def name(self):
        return self.prop_name

    def type(self):
        return self.prop_type

    def layout(self):
        return self.prop_layout

    def default(self):
        return self.default_value

    def ndims(self):
        return 1 if Types.is_scalar(self.prop_type) else 2

    def sizes(self):
        return [self.sim.particle_capacity] if Types.is_scalar(self.prop_type) \
               else [Types.number_of_elements(self.sim, self.prop_type), self.sim.particle_capacity]


class PropertyAccess(ASTTerm):
    last_prop_acc = 0

    def new_id():
        PropertyAccess.last_prop_acc += 1
        return PropertyAccess.last_prop_acc - 1

    def __init__(self, sim, prop, index):
        super().__init__(sim, OperatorClass.from_type(prop.type()))
        self.acc_id = PropertyAccess.new_id()
        self.prop = prop
        self.index = Lit.cvt(sim, index)
        self.inlined = False
        self.terminals = set()
        self.vector_indexes = {}

        if not Types.is_scalar(prop.type()):
            sizes = prop.sizes()
            layout = prop.layout()

            for elem in range(Types.number_of_elements(sim, prop.type())):
                if layout == Layouts.AoS:
                    self.vector_indexes[elem] = self.index * sizes[0] + elem
                elif layout == Layouts.SoA:
                    self.vector_indexes[elem] = elem * sizes[1] + self.index
                else:
                    raise Exception("Invalid data layout.")

    def __str__(self):
        return f"PropertyAccess<{self.prop}, {self.index}>"

    def __getitem__(self, index):
        _acc_class = AccessorClass.from_type(self.prop.type())
        return _acc_class(self.sim, self, Lit.cvt(self.sim, index))

    def copy(self, deep=False):
        return PropertyAccess(self.sim, self.prop, self.index)

    def vector_index(self, dimension):
        return self.vector_indexes[dimension]

    def inline_recursively(self):
        self.inlined = True
        return self

    def id(self):
        return self.acc_id

    def name(self):
        return f"prop_acc{self.id()}" + self.label_suffix()

    def type(self):
        return self.prop.type()

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.prop, self.index] + list(self.vector_indexes.values())


class RegisterProperty(ASTNode):
    def __init__(self, sim, prop, sizes):
        super().__init__(sim)
        self.prop = prop
        self.sizes_list = [Lit.cvt(sim, s) for s in sizes]
        self.sim.add_statement(self)

    def __str__(self):
        return f"RegisterProperty<{self.prop.name()}>"

    def property(self):
        return self.prop

    def sizes(self):
        return self.sizes_list


class ReallocProperty(ASTNode):
    def __init__(self, sim, prop, sizes):
        super().__init__(sim)
        self.prop = prop
        self.sizes_list = [Lit.cvt(sim, s) for s in sizes]
        self.sim.add_statement(self)

    def __str__(self):
        return f"ReallocProperty<{self.prop.name()}>"

    def property(self):
        return self.prop

    def sizes(self):
        return self.sizes_list


class ContactProperties:
    def __init__(self, sim):
        self.sim = sim
        self.contact_properties = []

    def __iter__(self):
        yield from self.contact_properties

    def add(self, cp_name, cp_type, cp_layout, cp_default):
        cp = ContactProperty(self.sim, cp_name, cp_type, cp_layout, cp_default)
        self.contact_properties.append(cp)
        return cp

    def find(self, cp_name):
        contact_prop = [cp for cp in self.contact_properties if cp.name() == cp_name]
        if contact_prop:
            return contact_prop[0]

        return None

    def empty(self):
        return len(self.contact_properties) == 0

    def nprops(self):
        return len(self.contact_properties)


class ContactProperty(ASTNode):
    last_contact_prop_id = 0

    def __init__(self, sim, name, dtype, layout, default):
        super().__init__(sim)
        self.contact_prop_id = ContactProperty.last_contact_prop_id
        self.contact_prop_name = name
        self.contact_prop_type = dtype
        self.contact_prop_layout = layout
        self.contact_prop_default = default
        self.device_flag = False
        ContactProperty.last_contact_prop_id += 1

    def __str__(self):
        return f"ContactProperty<{self.contact_prop_name}>"

    def __getitem__(self, expr):
        return ContactPropertyAccess(self.sim, self, expr)

    def id(self):
        return self.contact_prop_id

    def name(self):
        return self.contact_prop_name

    def type(self):
        return self.contact_prop_type

    def layout(self):
        return self.contact_prop_layout

    def default(self):
        return self.contact_prop_default

    def ndims(self):
        return 1 if Types.is_scalar(self.contact_prop_type) else 2

    def sizes(self):
        neighbor_list_sizes = [self.sim.particle_capacity, self.sim.neighbor_capacity]
        return neighbor_list_sizes if Types.is_scalar(self.contact_prop_type) \
               else [Types.number_of_elements(self.sim, self.contact_prop_type)] + neighbor_list_sizes


class ContactPropertyAccess(ASTTerm):
    last_contact_prop_acc = 0

    def new_id():
        ContactPropertyAccess.last_contact_prop_acc += 1
        return ContactPropertyAccess.last_contact_prop_acc - 1

    def __init__(self, sim, contact_prop, index):
        assert isinstance(index, tuple), "Two indexes must be used for contact property access!"
        super().__init__(sim, OperatorClass.from_type(contact_prop.type()))
        self.acc_id = ContactPropertyAccess.new_id()
        self.contact_prop = contact_prop
        self.index = index[0] * self.sim.neighbor_capacity + index[1]
        self.inlined = False
        self.terminals = set()
        self.vector_indexes = {}

        if not Types.is_scalar(contact_prop.type()):
            sizes = contact_prop.sizes()
            layout = contact_prop.layout()

            for elem in range(Types.number_of_elements(sim, contact_prop.type())):
                if layout == Layouts.AoS:
                    self.vector_indexes[elem] = self.index * sizes[0] + elem
                elif layout == Layouts.SoA:
                    self.vector_indexes[elem] = elem * sizes[1] + self.index
                else:
                    raise Exception("Invalid data layout.")

    def __str__(self):
        return f"ContactPropertyAccess<{self.contact_prop}, {self.index}>"

    def __getitem__(self, index):
        _acc_class = AccessorClass.from_type(self.contact_prop.type())
        return _acc_class(self.sim, self, Lit.cvt(self.sim, index))

    def copy(self, deep=False):
        return ContactPropertyAccess(self.sim, self.contact_prop, self.index)

    def vector_index(self, dimension):
        return self.vector_indexes[dimension]

    def inline_recursively(self):
        self.inlined = True
        return self

    def id(self):
        return self.acc_id

    def name(self):
        return f"cont_prop_acc{self.id()}" + self.label_suffix()

    def type(self):
        return self.contact_prop.type()

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.contact_prop, self.index] + list(self.vector_indexes.values())


class RegisterContactProperty(ASTNode):
    def __init__(self, sim, prop, sizes):
        super().__init__(sim)
        self.prop = prop
        self.sizes_list = [Lit.cvt(sim, s) for s in sizes]
        self.sim.add_statement(self)

    def __str__(self):
        return f"RegisterContactProperty<{self.prop.name()}>"

    def property(self):
        return self.prop

    def sizes(self):
        return self.sizes_list


class ReallocContactProperty(ASTNode):
    def __init__(self, sim, prop, sizes):
        super().__init__(sim)
        self.prop = prop
        self.sizes_list = [Lit.cvt(sim, s) for s in sizes]
        self.sim.add_statement(self)

    def __str__(self):
        return f"ReallocContactProperty<{self.prop.name()}>"

    def property(self):
        return self.prop

    def sizes(self):
        return self.sizes_list
