import math
from pairs.ir.actions import Actions
from pairs.ir.assign import Assign
from pairs.ir.atomic import AtomicAdd, AtomicInc
from pairs.ir.arrays import Array, ArrayAccess, DeclareStaticArray, RegisterArray, ReallocArray
from pairs.ir.block import Block
from pairs.ir.branches import Branch
from pairs.ir.cast import Cast
from pairs.ir.contexts import Contexts
from pairs.ir.declaration import Decl
from pairs.ir.scalars import ScalarOp
from pairs.ir.device import CopyArray, CopyContactProperty, CopyProperty, CopyFeatureProperty, CopyVar, DeviceStaticRef, HostRef
from pairs.ir.features import FeatureProperty, FeaturePropertyAccess, RegisterFeatureProperty
from pairs.ir.functions import Call
from pairs.ir.kernel import KernelLaunch
from pairs.ir.layouts import Layouts
from pairs.ir.lit import Lit
from pairs.ir.loops import For, Iter, While, Continue, Break
from pairs.ir.quaternions import Quaternion, QuaternionAccess, QuaternionOp
from pairs.ir.math import MathFunction
from pairs.ir.matrices import Matrix, MatrixAccess, MatrixOp
from pairs.ir.memory import Malloc, Realloc
from pairs.ir.module import ModuleCall
from pairs.ir.particle_attributes import ParticleAttributeList
from pairs.ir.properties import Property, PropertyAccess, RegisterProperty, ReallocProperty, ContactProperty, ContactPropertyAccess, RegisterContactProperty
from pairs.ir.select import Select
from pairs.ir.sizeof import Sizeof
from pairs.ir.types import Types
from pairs.ir.print import Print, PrintCode
from pairs.ir.variables import Var, DeclareVariable, Deref
from pairs.ir.parameters import Parameter
from pairs.ir.vectors import Vector, VectorAccess, VectorOp, ZeroVector
from pairs.ir.ret import Return
from pairs.code_gen.printer import Printer
from pairs.code_gen.accessor import PairsAcessor


class CGen:
    temp_id = 0

    def __init__(self, ref, debug=False):
        self.sim = None
        self.target = None
        self.print = None
        self.kernel_context = False
        self.loop_scope = False
        self.generate_full_object_names = False
        self.ref = ref
        self.debug = debug

    def assign_simulation(self, sim):
        self.sim = sim

    def assign_target(self, target):
        self.target = target

    def real_type(self):
        return Types.c_keyword(self.sim, Types.Real)
    
    def generate_object_reference(self, obj, device=False, index=None):
        if device and (not self.target.is_gpu() or not obj.device_flag):
            # Ideally this should never be called
            return "nullptr"
        
        name = obj.name() if not device else f"{obj.name()}_d"
        t = obj.type()
        if not Types.is_scalar(t) and index is not None:
            name += f"_{index}"

        if isinstance(obj, Var):
            if self.generate_full_object_names:
                if not obj.temporary():
                    if obj.device_flag and self.target.is_gpu() and device:
                        return f"pobj->rv_{obj.name()}"
                    else:
                        return f"pobj->{name}"
            return name

        if isinstance(obj, FeatureProperty) and device and obj.device_flag:
            return name
        
        if isinstance(obj, Array) and device and obj.device_flag:
            if obj.is_static():
                return name
        

        if self.generate_full_object_names:
            return f"pobj->{name}"
        else:
            return name
        

    def generate_object_address(self, obj, device=False, index=None):
        if device and (not self.target.is_gpu() or not obj.device_flag):
            return "nullptr"

        ref = self.generate_object_reference(obj, device, index)
        return f"&({ref})"

    def generate_interfaces(self):
        #self.print = Printer(f"interfaces/{self.ref}.hpp")
        self.print = Printer("internal_interfaces/last_generated.hpp")
        self.print.start()
        self.print("#pragma once")
        self.generate_interface_namespace('pairs_host_interface')

        if self.target.is_gpu():
            self.generate_interface_namespace('pairs_cuda_interface', "__inline__ __device__")
            
        self.print.end()

    def generate_interface_namespace(self, namespace, prefix=None):
        self.print("")
        self.print(f"namespace {namespace} {{")
        self.print("")

        for prop in self.sim.properties.all():
            prop_name = prop.name()
            t = prop.type()
            tkw = Types.c_keyword(self.sim, t)
            func_decl = "" if prefix is None else f"{prefix} "
            if Types.is_scalar(t):
                func_decl += f"{tkw} get_{prop_name}({tkw} *{prop_name}, int i) {{ return {prop_name}[i]; }}"

            else:
                nelems = Types.number_of_elements(self.sim, t)
                func_decl += f"{tkw} get_{prop_name}({tkw} *{prop_name}, int i, int j, int capacity) {{ return {prop_name}["

                if prop.layout() == Layouts.AoS:
                    func_decl += f"i * {nelems} + j"

                else:
                    func_decl += f"j * capacity + i"

                func_decl += "]; }"

            self.print(func_decl)

        self.print("")
        self.print("}")

    def generate_preamble(self):
        if self.target.is_gpu():
            self.print("#include <math_constants.h>")
             
        if self.target.is_openmp():
            self.print("#define PAIRS_TARGET_OPENMP")
            self.print("#include <omp.h>")

        self.print("#include <limits.h>")
        self.print("#include <math.h>")
        self.print("#include <stdbool.h>")
        self.print("#include <stdio.h>")
        self.print("#include <stdlib.h>")
        self.print("//---")
        self.print("#include \"likwid-marker.h\"")
        self.print("#include \"pairs.hpp\"")
        self.print("#include \"utility/copper_fcc_lattice.hpp\"")
        self.print("#include \"utility/create_body.hpp\"")
        self.print("#include \"utility/dem_sc_grid.hpp\"")
        self.print("#include \"utility/read_from_file.hpp\"")
        self.print("#include \"utility/stats.hpp\"")
        self.print("#include \"utility/timing.hpp\"")
        self.print("#include \"utility/thermo.hpp\"")
        self.print("#include \"utility/vtk.hpp\"")
        self.print("")
        self.print("using namespace pairs;")
        self.print("")

    def generate_module_header(self, module, definition=True):
        module_params = []

        if not module.interface:
            module_params += ["PairsRuntime *pairs_runtime", "struct PairsObjects *pobj"]

        module_params += [f"{Types.c_keyword(self.sim, param.type())} {param.name()}" for param in module.parameters()]

        print_params = ", ".join(module_params)
        ending = "{" if definition else ";"
        tkw = Types.c_keyword(self.sim, module.return_type)
        self.print(f"{tkw} {module.name}({print_params}){ending}")

    def generate_module_decls(self):
        self.print("")
        self.print("namespace pairs::internal {")
        self.print.add_indent(4)

        # All internal modules are declared in the pairs::internal scope
        for module in self.sim.modules():
            self.generate_module_header(module, definition=False)
        
        self.print.add_indent(-4)
        self.print("}")
        self.print("")
        
    def generate_pairs_object_structure(self):
        self.print("")
        externkw = "extern "
        if self.target.is_gpu():
            for array in self.sim.arrays.statics():
                if array.device_flag:
                    t = array.type()
                    tkw = Types.c_keyword(self.sim, t)
                    size = self.generate_expression(ScalarOp.inline(array.alloc_size()))
                    self.print(f"{externkw}__constant__ {tkw} {array.name()}_d[{size}];")

            for feature_prop in self.sim.feature_properties:
                if feature_prop.device_flag:
                    t = feature_prop.type()
                    tkw = Types.c_keyword(self.sim, t)
                    size = feature_prop.array_size()
                    self.print(f"{externkw}__constant__ {tkw} {feature_prop.name()}_d[{size}];")

        self.print("")
        self.print("struct PairsObjects {")
        self.print.add_indent(4)

        self.print("// Arrays")
        for a in self.sim.arrays.all():
            ptr = a.name()
            tkw = Types.c_keyword(self.sim, a.type())

            if a.is_static():
                size = self.generate_expression(ScalarOp.inline(a.alloc_size()))
                self.print(f"{tkw} {ptr}[{size}];")

            else:
                self.print(f"{tkw} *{ptr};")

            if self.target.is_gpu() and a.device_flag:
                if a.is_static():
                    continue
                else:
                    self.print(f"{tkw} *{ptr}_d;")

        self.print("// Properties")
        for p in self.sim.properties:
            ptr = p.name()
            tkw = Types.c_keyword(self.sim, p.type())
            self.print(f"{tkw} *{ptr};")

            if self.target.is_gpu() and p.device_flag:
                self.print(f"{tkw} *{ptr}_d;")

        self.print("// Contact properties")
        for cp in self.sim.contact_properties:
            ptr = cp.name()
            tkw = Types.c_keyword(self.sim, cp.type())
            self.print(f"{tkw} *{ptr};")

            if self.target.is_gpu() and cp.device_flag:
                self.print(f"{tkw} *{ptr}_d;")

        self.print("// Feature properties")
        for fp in self.sim.feature_properties:
            ptr = fp.name()
            array_size = fp.array_size()
            tkw = Types.c_keyword(self.sim, fp.type())
            self.print(f"{tkw} {ptr}[{array_size}];")

        self.print("// Variables")
        for v in self.sim.vars.all():
            vname = v.name()
            tkw = Types.c_keyword(self.sim, v.type())
            self.print(f"{tkw} {vname};")

            if self.target.is_gpu() and v.device_flag:
                self.print(f"RuntimeVar<{tkw}> rv_{vname};")

        self.print.add_indent(-4)
        self.print("};")
        self.print("")

    def generate_library(self):
        self.generate_interfaces()
        # Generate CUDA/CPP file with modules
        ext = ".cu" if self.target.is_gpu() else ".cpp"
        self.print = Printer(self.ref + ext)
        self.print.start()
        self.generate_preamble()
        self.print(f"#include \"{self.ref}.hpp\"")
        self.print("")

        if self.target.is_gpu():
            for array in self.sim.arrays.statics():
                if array.device_flag:
                    t = array.type()
                    tkw = Types.c_keyword(self.sim, t)
                    size = self.generate_expression(ScalarOp.inline(array.alloc_size()))
                    self.print(f"__constant__ {tkw} {array.name()}_d[{size}];")

            for feature_prop in self.sim.feature_properties:
                if feature_prop.device_flag:
                    t = feature_prop.type()
                    tkw = Types.c_keyword(self.sim, t)
                    size = feature_prop.array_size()
                    self.print(f"__constant__ {tkw} {feature_prop.name()}_d[{size}];")

        self.print("")
                    
        self.print("namespace pairs::internal {")
        self.print.add_indent(4)

        for kernel in self.sim.kernels():
            self.generate_kernel(kernel)

        # All internal modules are defined in the pairs::internal scope
        for module in self.sim.modules():
            self.generate_module(module)

        self.print.add_indent(-4)
        self.print("}")

        self.print.end()

        # Generate library header
        self.print = Printer(self.ref + ".hpp")
        self.print.start()
        self.print("#pragma once")

        self.generate_preamble()
        self.generate_pairs_object_structure()
        self.generate_module_decls()

        self.generate_full_object_names = True
        self.print("class PairsSimulation {")
        self.print("private:")
        self.print("    PairsRuntime *pairs_runtime;")
        self.print("    struct PairsObjects *pobj;")
        self.print("    friend class PairsAccessor;")
        self.print("")
        self.print("public:")
        self.print.add_indent(4)

        self.print("PairsRuntime* getPairsRuntime() {")
        self.print("    return pairs_runtime;")
        self.print("}")

        # Only interface modules are generated in the PairsSimulation class
        for module in self.sim.interface_modules():
            self.generate_module(module)

        self.print.add_indent(-4)
        self.print("};")

        PairsAcessor(self).generate()
        
        self.print.end()
        self.generate_full_object_names = False

    def generate_module_declerations(self, module):
        device_cond = module.run_on_device and self.target.is_gpu()

        for var in module.read_only_variables():
            type_kw = Types.c_keyword(self.sim, var.type())
            self.print(f"{type_kw} {var.name()} = pobj->{var.name()};")

        for var in module.write_variables():
            type_kw = Types.c_keyword(self.sim, var.type())

            if device_cond and var.device_flag:
                self.print(f"{type_kw} *{var.name()} = pobj->rv_{var.name()}.getDevicePointer();")
            elif var.force_read:
                self.print(f"{type_kw} {var.name()} = pobj->{var.name()};")
            else:
                self.print(f"{type_kw} *{var.name()} = &(pobj->{var.name()});")

        for array in module.arrays():
            type_kw = Types.c_keyword(self.sim, array.type())
            name = array.name() if not device_cond else f"{array.name()}_d"
            if not array.is_static() or (array.is_static() and not device_cond):
                self.print(f"{type_kw} *{array.name()} = pobj->{name};")

            if array in module.host_references():
                self.print(f"{type_kw} *{array.name()}_h = pobj->{array.name()};")


        for prop in module.properties():
            type_kw = Types.c_keyword(self.sim, prop.type())
            name = prop.name() if not device_cond else f"{prop.name()}_d"
            self.print(f"{type_kw} *{prop.name()} = pobj->{name};")

            if prop in module.host_references():
                self.print(f"{type_kw} *{prop.name()}_h = pobj->{prop.name()};")

        for contact_prop in module.contact_properties():
            type_kw = Types.c_keyword(self.sim, contact_prop.type())
            name = contact_prop.name() if not device_cond else f"{contact_prop.name()}_d"
            self.print(f"{type_kw} *{contact_prop.name()} = pobj->{name};")

            if contact_prop in module.host_references():
                self.print(f"{type_kw} *{contact_prop.name()}_h = pobj->{contact_prop.name()};")

        for feature_prop in module.feature_properties():
            type_kw = Types.c_keyword(self.sim, feature_prop.type())
            name = feature_prop.name() if not device_cond else f"{feature_prop.name()}_d"

            if feature_prop.device_flag and device_cond:
                # self.print(f"{type_kw} *{feature_prop.name()} = {self.generate_object_reference(feature_prop, device=device_cond)};")
                continue
            else:
                self.print(f"{type_kw} *{feature_prop.name()} = pobj->{name};")

            if feature_prop in module.host_references():
                self.print(f"{type_kw} *{feature_prop.name()}_h = pobj->{feature_prop.name()};")

    def generate_module(self, module):
        self.generate_module_header(module, definition=True)
        self.print.add_indent(4)

        # if self.debug:
        #     self.print(f"PAIRS_DEBUG(\"\\n{module.name}\\n\");")

        if not module.interface:
            self.generate_module_declerations(module)

        self.print.add_indent(-4)
        self.generate_statement(module.block)
        self.print("}")
        self.print("")

    def generate_kernel(self, kernel):
        kernel_params = "int range_start"
        has_resizes = False
        for param in kernel.parameters():
            type_kw = Types.c_keyword(self.sim, param.type())
            decl = f"{type_kw} {param.name()}"
            kernel_params += f", {decl}"

        for var in kernel.read_only_variables():
            type_kw = Types.c_keyword(self.sim, var.type())
            decl = f"{type_kw} {var.name()}"
            kernel_params += f", {decl}"

        for var in kernel.write_variables():
            type_kw = Types.c_keyword(self.sim, var.type())
            decl = f"{type_kw} *{var.name()}"
            kernel_params += f", {decl}"

        for it in kernel.iters():
            type_kw = Types.c_keyword(self.sim, it.type())
            decl = f"{type_kw} {it.name()}"
            kernel_params += f", {decl}"

        for array in kernel.arrays():
            if array.is_static():
                continue
            type_kw = Types.c_keyword(self.sim, array.type())
            decl = f"{type_kw} *{array.name()}"
            kernel_params += f", {decl}"
            if array.name() == "resizes":
                has_resizes = True

        for prop in kernel.properties():
            type_kw = Types.c_keyword(self.sim, prop.type())
            decl = f"{type_kw} *{prop.name()}"
            kernel_params += f", {decl}"

        for contact_prop in kernel.contact_properties():
            type_kw = Types.c_keyword(self.sim, contact_prop.type())
            decl = f"{type_kw} *{contact_prop.name()}"
            kernel_params += f", {decl}"

        for feature_prop in kernel.feature_properties():
            if feature_prop.device_flag:
                continue
            type_kw = Types.c_keyword(self.sim, feature_prop.type())
            decl = f"{type_kw} *{feature_prop.name()}"
            kernel_params += f", {decl}"

        for array_access in kernel.array_accesses():
            type_kw = Types.c_keyword(self.sim, array_access.type())
            decl = f"{type_kw} {array_access.name()}"
            kernel_params += f", {decl}"

        for scalar_op in kernel.scalar_ops():
            type_kw = Types.c_keyword(self.sim, scalar_op.type())
            decl = f"{type_kw} {scalar_op.name()}"
            kernel_params += f", {decl}"

        self.print(f"__global__ void {kernel.name}({kernel_params}) {{")
        self.print(f"    const int {kernel.iterator.name()} = blockIdx.x * blockDim.x + threadIdx.x + range_start;")
        self.print.add_indent(4)
        self.kernel_context = True

        self.generate_statement(kernel.block)

        self.kernel_context = False
        self.print.add_indent(-4)
        self.print("}")

    def generate_statement(self, ast_node):
        if isinstance(ast_node, DeclareStaticArray):
            t = ast_node.array.type()
            tkw = Types.c_keyword(self.sim, t)
            size = self.generate_expression(ScalarOp.inline(ast_node.array.alloc_size()))

            if ast_node.array.init_value is not None:
                v_str = str(ast_node.array.init_value)
                if t == Types.Int64:
                    v_str += "LL"
                if t == Types.UInt64:
                    v_str += "ULL"

                for i in range(size):
                    self.print(f"{ast_node.array.name()}[{i}] = {v_str};")

        if isinstance(ast_node, Assign):
            if not Types.is_scalar(ast_node._dest.type()):
                for e in range(Types.number_of_elements(self.sim, ast_node._dest.type())):
                    dest = self.generate_expression(ast_node._dest, mem=True, index=e)
                    src = self.generate_expression(ast_node._src, index=e)
                    self.print(f"{dest} = {src};")

            else:
                dest = self.generate_expression(ast_node._dest, mem=True)
                src = self.generate_expression(ast_node._src)
                self.print(f"{dest} = {src};")

        if isinstance(ast_node, AtomicInc):
            elem = self.generate_expression(ast_node.elem, mem=True)
            value = self.generate_expression(ast_node.value)
            prefix = "" if ast_node.device_flag else "host_"

            if ast_node.check_for_resize():
                resize = self.generate_expression(ast_node.resize)
                capacity = self.generate_expression(ast_node.capacity)
                self.print(f"pairs::{prefix}atomic_add_resize_check(&({elem}), {value}, &({resize}), {capacity});")

            else:
                self.print(f"pairs::{prefix}atomic_add(&({elem}), {value});")

        if isinstance(ast_node, Block):
            self.print.add_indent(4)
            for stmt in ast_node.statements():
                self.generate_statement(stmt)
            self.print.add_indent(-4)

        if isinstance(ast_node, Continue):
            if self.loop_scope:
                self.print("continue;")
            else:
                self.print("return;")

        if isinstance(ast_node, Break):
            if self.loop_scope:
                self.print("break;")
            else:
                self.print("return;")

        # TODO: Why there are Decls for other types?
        if isinstance(ast_node, Decl):
            if isinstance(ast_node.elem, ArrayAccess):
                array_access = ast_node.elem
                array_name = self.generate_expression(array_access.array)
                tkw = Types.c_keyword(self.sim, array_access.type())
                acc_index = self.generate_expression(array_access.flat_index)
                acc_ref = array_access.name()
                self.print(f"const {tkw} {acc_ref} = {array_name}[{acc_index}];")

            if isinstance(ast_node.elem, AtomicAdd):
                atomic_add = ast_node.elem
                elem = self.generate_expression(atomic_add.elem)
                value = self.generate_expression(atomic_add.value)
                tkw = Types.c_keyword(self.sim, atomic_add.type())
                acc_ref = atomic_add.name()
                prefix = "" if ast_node.elem.device_flag else "host_"

                if atomic_add.check_for_resize():
                    resize = self.generate_expression(atomic_add.resize)
                    capacity = self.generate_expression(atomic_add.capacity)
                    self.print(f"const {tkw} {acc_ref} = pairs::{prefix}atomic_add_resize_check(&({elem}), {value}, &({resize}), {capacity});")
                else:
                    self.print(f"const {tkw} {acc_ref} = pairs::{prefix}atomic_add(&({elem}), {value});")

            if isinstance(ast_node.elem, ContactPropertyAccess):
                contact_prop_access = ast_node.elem
                contact_prop = contact_prop_access.contact_prop
                prop_name = self.generate_expression(contact_prop)
                acc_ref = contact_prop_access.name()

                if not contact_prop_access.is_scalar():
                    for dim in contact_prop_access.indexes_to_generate():
                        expr = self.generate_expression(contact_prop_access.vector_index(dim))
                        self.print(f"const {self.real_type()} {acc_ref}_{dim} = {prop_name}[{expr}];")

                else:
                    tkw = Types.c_keyword(self.sim, contact_prop_access.type())
                    acc_index = self.generate_expression(contact_prop_access.index)
                    self.print(f"const {tkw} {acc_ref} = {prop_name}[{acc_index}];")

            if isinstance(ast_node.elem, FeaturePropertyAccess):
                feature_prop_access = ast_node.elem
                feature_prop = feature_prop_access.feature_prop
                prop_name = self.generate_expression(feature_prop)
                acc_ref = feature_prop_access.name()

                if not feature_prop_access.is_scalar():
                    for dim in feature_prop_access.indexes_to_generate():
                        expr = self.generate_expression(feature_prop_access.vector_index(dim))
                        self.print(f"const {self.real_type()} {acc_ref}_{dim} = {prop_name}[{expr}];")

                else:
                    tkw = Types.c_keyword(self.sim, feature_prop_access.type())
                    acc_index = self.generate_expression(feature_prop_access.index)
                    self.print(f"const {tkw} {acc_ref} = {prop_name}[{acc_index}];")

            if isinstance(ast_node.elem, PropertyAccess):
                prop_access = ast_node.elem
                prop_name = self.generate_expression(prop_access.prop)
                acc_ref = prop_access.name()

                if not prop_access.is_scalar():
                    for dim in prop_access.indexes_to_generate():
                        expr = self.generate_expression(prop_access.vector_index(dim))
                        self.print(f"const {self.real_type()} {acc_ref}_{dim} = {prop_name}[{expr}];")
                else:
                    tkw = Types.c_keyword(self.sim, prop_access.type())
                    index_g = self.generate_expression(prop_access.index)
                    self.print(f"const {tkw} {acc_ref} = {prop_name}[{index_g}];")

            if isinstance(ast_node.elem, Quaternion):
                quaternion = ast_node.elem
                for i in quaternion.indexes_to_generate():
                    expr = self.generate_expression(quaternion.get_value(i))
                    self.print(f"const {self.real_type()} {quaternion.name()}_{i} = {expr};")

            if isinstance(ast_node.elem, QuaternionOp):
                quat_op = ast_node.elem
                for i in quat_op.indexes_to_generate():
                    lhs = self.generate_expression(quat_op.lhs, quat_op.mem, index=dim)
                    rhs = self.generate_expression(quat_op.rhs, index=dim)
                    operator = quat_op.operator()

                    if operator.is_unary():
                        self.print(f"const {self.real_type()} {quat_op.name()}_{dim} = {operator.symbol()}({lhs});")
                    else:
                        self.print(f"const {self.real_type()} {quat_op.name()}_{dim} = {lhs} {operator.symbol()} {rhs};")

            if isinstance(ast_node.elem, ScalarOp):
                scalar_op = ast_node.elem
                if scalar_op.inlined is False:
                    lhs = self.generate_expression(scalar_op.lhs, scalar_op.mem)
                    rhs = self.generate_expression(scalar_op.rhs)
                    operator = scalar_op.operator()
                    tkw = Types.c_keyword(self.sim, scalar_op.type())

                    if operator.is_unary():
                        self.print(f"const {tkw} {scalar_op.name()} = {operator.symbol()}({lhs});")
                    else:
                        self.print(f"const {tkw} {scalar_op.name()} = {lhs} {operator.symbol()} {rhs};")

            if isinstance(ast_node.elem, Select):
                select = ast_node.elem
                acc_ref = select.name()

                if not select.is_scalar():
                    for dim in select.indexes_to_generate():
                        cond = self.generate_expression(select.cond, index=dim)
                        expr_if = self.generate_expression(select.expr_if, index=dim)
                        expr_else = self.generate_expression(select.expr_else, index=dim)
                        self.print(f"const {self.real_type()} {acc_ref}_{dim} = ({cond}) ? ({expr_if}) : ({expr_else});")
                else:
                    cond = self.generate_expression(select.cond)
                    expr_if = self.generate_expression(select.expr_if)
                    expr_else = self.generate_expression(select.expr_else)
                    tkw = Types.c_keyword(self.sim, select.type())
                    self.print(f"const {tkw} {acc_ref} = ({cond}) ? ({expr_if}) : ({expr_else});")

            if isinstance(ast_node.elem, MathFunction):
                math_func = ast_node.elem
                params = ", ".join([str(self.generate_expression(p)) for p in math_func.parameters()])
                tkw = Types.c_keyword(self.sim, math_func.type())
                self.print(f"const {tkw} {math_func.name()} = {math_func.function_name()}({params});")

            if isinstance(ast_node.elem, Matrix):
                matrix = ast_node.elem
                for i in matrix.indexes_to_generate():
                    expr = self.generate_expression(matrix.get_value(i))
                    self.print(f"const {self.real_type()} {matrix.name()}_{i} = {expr};")

            if isinstance(ast_node.elem, MatrixOp):
                matrix_op = ast_node.elem
                for i in matrix_op.indexes_to_generate():
                    lhs = self.generate_expression(matrix_op.lhs, matrix_op.mem, index=i)
                    rhs = self.generate_expression(matrix_op.rhs, index=i)
                    operator = matrix_op.operator()

                    if operator.is_unary():
                        self.print(f"const {self.real_type()} {matrix_op.name()}_{dim} = {operator.symbol()}({lhs});")
                    else:
                        self.print(f"const {self.real_type()} {matrix_op.name()}_{dim} = {lhs} {operator.symbol()} {rhs};")

            if isinstance(ast_node.elem, Vector):
                vector = ast_node.elem
                for dim in vector.indexes_to_generate():
                    expr = self.generate_expression(vector.get_value(dim))
                    self.print(f"const {self.real_type()} {vector.name()}_{dim} = {expr};")

            if isinstance(ast_node.elem, VectorOp):
                vector_op = ast_node.elem
                for dim in vector_op.indexes_to_generate():
                    lhs = self.generate_expression(vector_op.lhs, vector_op.mem, index=dim)
                    rhs = self.generate_expression(vector_op.rhs, index=dim)
                    operator = vector_op.operator()

                    if operator.is_unary():
                        self.print(f"const {self.real_type()} {vector_op.name()}_{dim} = {operator.symbol()}({lhs});")
                    else:
                        self.print(f"const {self.real_type()} {vector_op.name()}_{dim} = {lhs} {operator.symbol()} {rhs};")

        if isinstance(ast_node, Branch):
            cond = self.generate_expression(ast_node.cond)
            self.print(f"if({cond}) {{")
            self.generate_statement(ast_node.block_if)

            if ast_node.block_else is not None:
                self.print("} else {")
                self.generate_statement(ast_node.block_else)

            self.print("}") 

        if isinstance(ast_node, Call):
            call = self.generate_expression(ast_node)
            self.print(f"{call};")

        if isinstance(ast_node, CopyArray):
            array_id = ast_node.array().id()
            array_name = ast_node.array().name()
            ctx_suffix = "Device" if ast_node.context() == Contexts.Device else "Host"
            action = Actions.c_keyword(ast_node.action())
            size = self.generate_expression(ast_node.size())

            if size is not None:
                self.print(f"pairs_runtime->copyArrayTo{ctx_suffix}({array_id}, {action}, {size}); // {array_name}")

            else:
                self.print(f"pairs_runtime->copyArrayTo{ctx_suffix}({array_id}, {action}); // {array_name}")

        if isinstance(ast_node, CopyContactProperty):
            prop_id = ast_node.contact_prop().id()
            prop_name = ast_node.contact_prop().name()
            action = Actions.c_keyword(ast_node.action())
            ctx_suffix = "Device" if ast_node.context() == Contexts.Device else "Host"
            size = self.generate_expression(ast_node.contact_prop().copy_size())
            self.print(f"pairs_runtime->copyContactPropertyTo{ctx_suffix}({prop_id}, {action}, {size}); // {prop_name}")

        if isinstance(ast_node, CopyProperty):
            prop_id = ast_node.prop().id()
            prop_name = ast_node.prop().name()
            action = Actions.c_keyword(ast_node.action())
            ctx_suffix = "Device" if ast_node.context() == Contexts.Device else "Host"
            size = self.generate_expression(ast_node.prop().copy_size())
            self.print(f"pairs_runtime->copyPropertyTo{ctx_suffix}({prop_id}, {action}, {size}); // {prop_name}")

        if isinstance(ast_node, CopyFeatureProperty):
            prop_id = ast_node.prop().id()
            prop_name = ast_node.prop().name()
            if ast_node.context() == Contexts.Device:
                assert ast_node.action()==Actions.ReadOnly, "Feature properties can only be read from device."
                self.print(f"pairs_runtime->copyFeaturePropertyToDevice({prop_id}); // {prop_name}")

        if isinstance(ast_node, CopyVar):
            var_name = ast_node.variable().name()
            ctx_suffix = "Device" if ast_node.context() == Contexts.Device else "Host"
            ref = self.generate_object_reference(ast_node.variable(), device=True)
            self.print(f"{ref}.copyTo{ctx_suffix}();")

        if isinstance(ast_node, For):
            iterator = self.generate_expression(ast_node.iterator)
            lower_range = self.generate_expression(ast_node.min)
            upper_range = self.generate_expression(ast_node.max)

            if self.target.is_openmp() and ast_node.is_kernel_candidate():
                self.print("#pragma omp parallel for")

            self.print(f"for(int {iterator} = {lower_range}; {iterator} < {upper_range}; {iterator}++) {{")
            self.loop_scope = True
            self.generate_statement(ast_node.block)
            self.loop_scope = False
            self.print("}")


        if isinstance(ast_node, Malloc):
            tkw = Types.c_keyword(self.sim, ast_node.array.type())
            size = self.generate_expression(ast_node.size)
            array_name = ast_node.array.name()

            if ast_node.decl:
                self.print(f"{tkw} *{array_name} = ({tkw} *) malloc({size});")
                if self.target.is_gpu() and ast_node.array.device_flag:
                    self.print(f"{tkw} *{array_name}_d = ({tkw} *) pairs::device_alloc({size});")
            else:
                self.print(f"{array_name} = ({tkw} *) malloc({size});")
                if self.target.is_gpu() and ast_node.array.device_flag:
                    self.print(f"{array_name}_d = ({tkw} *) pairs::device_alloc({size});")

        if isinstance(ast_node, KernelLaunch):
            range_start = self.generate_expression(ScalarOp.inline(ast_node.min))
            kernel = ast_node.kernel
            kernel_params = f"{range_start}"

            for param in kernel.parameters():
                kernel_params += f", {param.name()}"

            for var in kernel.read_only_variables():
                kernel_params += f", {var.name()}"

            for var in kernel.write_variables():
                kernel_params += f", {var.name()}"

            for it in kernel.iters():
                kernel_params += f", {it.name()}"

            for array in kernel.arrays():
                if array.is_static():
                    continue
                kernel_params += f", {array.name()}"

            for prop in kernel.properties():
                kernel_params += f", {prop.name()}"

            for contact_prop in kernel.contact_properties():
                kernel_params += f", {contact_prop.name()}"

            for feature_prop in kernel.feature_properties():
                if feature_prop.device_flag:
                    continue     
                kernel_params += f", {feature_prop.name()}"

            for array_access in kernel.array_accesses():
                kernel_params += f", {self.generate_expression(array_access)}"

            for scalar_op in kernel.scalar_ops():
                kernel_params += f", {self.generate_expression(scalar_op)}"

            threads_per_block = self.generate_expression(ast_node.threads_per_block)
            nblocks = self.generate_expression(ast_node.nblocks)
            self.print(f"if({nblocks} > 0 && {threads_per_block} > 0) {{")
            self.print.add_indent(4)
            self.print(f"{kernel.name}<<<{nblocks}, {threads_per_block}>>>({kernel_params});")
            self.print("pairs_runtime->sync();")
            self.print.add_indent(-4)
            self.print("}")

        if isinstance(ast_node, ModuleCall):
            module_params = ["pairs_runtime", "pobj"]

            module_params += [f"{param.name()}" for param in ast_node.module.parameters()]

            print_params = ", ".join(module_params)
            self.print(f"pairs::internal::{ast_node.module.name}({print_params});")

        if isinstance(ast_node, Print):
            args = ast_node.args
            exprs = [self.generate_expression(arg) for arg in args]
            toPrint = "PAIRS_DEBUG(\""
            for arg in args:
                if Types.is_real(arg.type()):
                    format = "%f "
                elif Types.is_integer(arg.type()):
                    format = "%d "
                else:
                    format = "%s "
                toPrint += format

            toPrint = toPrint + "\\n\", " + ", ".join(map(str, exprs)) + ");"
            self.print(toPrint)

        if isinstance(ast_node, PrintCode):
            toPrint = self.generate_expression(ast_node.arg)
            self.print(toPrint[1:-1])

        if isinstance(ast_node, Realloc):
            tkw = Types.c_keyword(self.sim, ast_node.array.type())
            size = self.generate_expression(ast_node.size)
            array_name = ast_node.array.name()
            ptr = self.generate_object_reference(ast_node)
            self.print(f"{ptr} = ({tkw} *) realloc({ptr}, {size});")

            if self.target.is_gpu() and ast_node.array.device_flag:
                d_ptr = self.generate_object_reference(ast_node, device=True)
                self.print(f"{d_ptr} = ({tkw} *) pairs::device_realloc({d_ptr}, {size});")

        if isinstance(ast_node, RegisterArray):
            a = ast_node.array()
            tkw = Types.c_keyword(self.sim, a.type())
            size = self.generate_expression(ast_node.size())

            if a.is_static():
                ptr_ref = self.generate_object_reference(a)
                d_ptr_ref = self.generate_object_reference(a, device=True)
                self.print(f"pairs_runtime->addStaticArray({a.id()}, \"{a.name()}\", {ptr_ref}, {d_ptr_ref}, {size});")

            else:
                ptr_addr = self.generate_object_address(a)
                d_ptr_addr = self.generate_object_address(a, device=True)
                self.print(f"pairs_runtime->addArray({a.id()}, \"{a.name()}\", {ptr_addr}, {d_ptr_addr}, {size});")

        if isinstance(ast_node, RegisterProperty):
            p = ast_node.property()
            ptr_addr = self.generate_object_address(p)
            d_ptr_addr = self.generate_object_address(p, device=True)
            tkw = Types.c_keyword(self.sim, p.type())
            ptype = Types.c_property_keyword(p.type())
            assert ptype != "Prop_Invalid", "Invalid property type!"

            playout = Layouts.c_keyword(p.layout())
            vol = 1 if p.is_volatile() else 0
            sizes = ", ".join([str(self.generate_expression(ScalarOp.inline(size))) for size in ast_node.sizes()])
            self.print(f"pairs_runtime->addProperty({p.id()}, \"{p.name()}\", {ptr_addr}, {d_ptr_addr}, {ptype}, {playout}, {vol}, {sizes});")

        if isinstance(ast_node, RegisterContactProperty):
            p = ast_node.property()
            ptr_addr = self.generate_object_address(p)
            d_ptr_addr = self.generate_object_address(p, device=True)
            tkw = Types.c_keyword(self.sim, p.type())
            ptype = Types.c_property_keyword(p.type())
            assert ptype != "Prop_Invalid", "Invalid property type!"

            playout = Layouts.c_keyword(p.layout())
            sizes = ", ".join([str(self.generate_expression(ScalarOp.inline(size))) for size in ast_node.sizes()])
            self.print(f"pairs_runtime->addContactProperty({p.id()}, \"{p.name()}\", {ptr_addr}, {d_ptr_addr}, {ptype}, {playout}, {sizes});")

        if isinstance(ast_node, RegisterFeatureProperty):
            fp = ast_node.feature_property()
            ptr = self.generate_object_reference(fp)
            ptr_addr = self.generate_object_address(fp)
            d_ptr_addr = self.generate_object_address(fp, device=True)
            array_size = fp.array_size()
            nkinds = fp.feature().nkinds()
            tkw = Types.c_keyword(self.sim, fp.type())
            fptype = Types.c_property_keyword(fp.type())
            assert fptype != "Prop_Invalid", "Invalid feature property type!"

            self.print(f"pairs_runtime->addFeatureProperty({fp.id()}, \"{fp.name()}\", {ptr_addr}, {d_ptr_addr}, {fptype}, {nkinds}, {array_size} * sizeof({tkw}));")

            for i in range(array_size):
                self.print(f"{ptr}[{i}] = {fp.data()[i]};")

            if self.target.is_gpu() and fp.device_flag:
                self.print(f"pairs_runtime->copyFeaturePropertyToDevice({fp.id()}); // {fp.name()}")

        if isinstance(ast_node, ReallocProperty):
            p = ast_node.property()
            ptr_addr = self.generate_object_address(p)
            d_ptr_addr = self.generate_object_address(p, device=True)
            sizes = ", ".join([str(self.generate_expression(ScalarOp.inline(size))) for size in ast_node.sizes()])
            self.print(f"pairs_runtime->reallocProperty({p.id()}, {ptr_addr}, {d_ptr_addr}, {sizes});")

        if isinstance(ast_node, ReallocArray):
            a = ast_node.array()
            size = self.generate_expression(ast_node.size())
            ptr_addr = self.generate_object_address(a)
            d_ptr_addr = self.generate_object_address(a, device=True)
            self.print(f"pairs_runtime->reallocArray({a.id()}, {ptr_addr}, {d_ptr_addr}, {size});")

        if isinstance(ast_node, DeclareVariable):
            var_name = ast_node.var.name()
            tkw = Types.c_keyword(self.sim, ast_node.var.type())
            prefix_decl = f"{tkw} " if ast_node.var.temporary() else ""

            if ast_node.var.is_scalar():
                var = self.generate_expression(ast_node.var)
                addr = self.generate_object_address(ast_node.var)
                init = self.generate_expression(ast_node.var.init_value())
                self.print(f"{prefix_decl}{var} = {init};")

                if ast_node.var.runtime_track():
                    self.print(f"pairs_runtime->trackVariable(\"{var_name}\", {addr});")

            else:
                for i in range(Types.number_of_elements(self.sim, ast_node.var.type())):
                    var = self.generate_expression(ast_node.var, index=i)
                    init = self.generate_expression(ast_node.var.init_value(), index=i)
                    self.print(f"{prefix_decl}{var} = {init};")

            if not self.kernel_context and self.target.is_gpu() and ast_node.var.device_flag:
                addr = self.generate_object_address(ast_node.var)
                ref = self.generate_object_reference(ast_node.var, device=True)
                self.print(f"{prefix_decl}{ref} = pairs_runtime->addDeviceVariable({addr});")

        if isinstance(ast_node, While):
            cond = self.generate_expression(ast_node.cond)
            self.print(f"while({cond}) {{")
            self.loop_scope = True
            self.generate_statement(ast_node.block)
            self.loop_scope = False
            self.print("}")

        if isinstance(ast_node, Return):
            expr = self.generate_expression(ast_node.expr)
            self.print(f"return {expr};")

    def generate_expression(self, ast_node, mem=False, index=None):
        if isinstance(ast_node, Array):
            return self.generate_object_reference(ast_node)

        if isinstance(ast_node, ArrayAccess):
            if mem or ast_node.inlined is True:
                array_name = self.generate_expression(ast_node.array)
                acc_index = self.generate_expression(ast_node.flat_index)
                return f"{array_name}[{acc_index}]"

            return f"{ast_node.name()}"

        if isinstance(ast_node, AtomicAdd):
            return f"{ast_node.name()}"

        if isinstance(ast_node, ScalarOp):
            if ast_node.inlined is True:
                lhs = self.generate_expression(ast_node.lhs, mem, index)
                rhs = self.generate_expression(ast_node.rhs, index=index)
                operator = ast_node.operator()
                return f"({operator.symbol()}({lhs}))" if operator.is_unary() else \
                       f"({lhs} {operator.symbol()} {rhs})"

            return f"{ast_node.name()}"

        if isinstance(ast_node, Call):
            extra_params = []

            if ast_node.name().startswith("pairs::"):
                extra_params += ["pairs_runtime"]

            params = ", ".join(extra_params + [str(self.generate_expression(p)) for p in ast_node.parameters()])
            return f"{ast_node.name()}({params})"

        if isinstance(ast_node, Cast):
            tkw = Types.c_keyword(self.sim, ast_node.cast_type)
            expr = self.generate_expression(ast_node.expr)
            return f"({tkw})({expr})"

        if isinstance(ast_node, ContactProperty):
            return self.generate_object_reference(ast_node)

        if isinstance(ast_node, Deref):
            var = self.generate_expression(ast_node.var)
            # Dereferences are ignored for write variables when full objects
            # are generated since they can be directly written into
            return var if (self.generate_full_object_names or ast_node.var.force_read) else f"(*{var})"

        if isinstance(ast_node, DeviceStaticRef):
            elem = self.generate_expression(ast_node.elem)
            return f"{elem}_d"

        if isinstance(ast_node, FeatureProperty):
            return self.generate_object_reference(ast_node)

        if isinstance(ast_node, HostRef):
            elem = self.generate_expression(ast_node.elem)
            return f"{elem}_h"

        if isinstance(ast_node, Iter):
            assert mem is False, "Iterator is not lvalue!"
            return f"{ast_node.name()}"

        if isinstance(ast_node, Lit):
            assert mem is False, "Literal is not lvalue!"
            if ast_node.type() == Types.String:
                return f"\"{ast_node.value}\""
            
            if ast_node.type() == Types.Boolean:
                if ast_node.value == True:
                    return "true"
                if ast_node.value == False:
                    return "false"

            if not ast_node.is_scalar():
                assert index is not None, "Index must be set for non-scalar literals."
                return ast_node.value[index]

            if isinstance(ast_node.value, float) and math.isinf(ast_node.value):
                if self.kernel_context:
                    return "CUDART_INF"
                else:
                    return f"std::numeric_limits<{self.real_type()}>::infinity()"

            return ast_node.value

        if isinstance(ast_node, MathFunction):
            assert mem is False, "Math function calls cannot be lvalue!"

            if ast_node.inlined is True:
                params = ", ".join([str(self.generate_expression(p)) for p in ast_node.parameters()])
                return f"{ast_node.function_name()}({params})"

            return f"{ast_node.name()}"

        if isinstance(ast_node, Property):
            return self.generate_object_reference(ast_node)

        if isinstance(ast_node, PropertyAccess):
            assert ast_node.is_scalar() or index is not None, \
                "Index must be set for non-scalar property access."
            prop_name = self.generate_expression(ast_node.prop)

            if mem or ast_node.inlined is True:
                index_expr = self.generate_expression(
                    ast_node.index if ast_node.is_scalar() else \
                    ast_node.vector_index(index))

                return f"{prop_name}[{index_expr}]"

            return f"{ast_node.name()}" + (f"_{index}" if not ast_node.is_scalar() else "")

        if isinstance(ast_node, ContactPropertyAccess):
            assert ast_node.is_scalar() or index is not None, \
                "Index must be set for non-scalar property access."
            prop_name = self.generate_expression(ast_node.contact_prop)

            if mem or ast_node.inlined is True:
                index_expr = self.generate_expression(
                    ast_node.index if ast_node.is_scalar() else \
                    ast_node.vector_index(index))

                return f"{prop_name}[{index_expr}]"

            return f"{ast_node.name()}" + (f"_{index}" if not ast_node.is_scalar() else "")

        if isinstance(ast_node, FeaturePropertyAccess):
            assert ast_node.is_scalar() or index is not None, \
                "Index must be set for non-scalar property access."
            feature_name = self.generate_expression(ast_node.feature_prop)

            if mem or ast_node.inlined is True:
                index_expr = self.generate_expression(
                    ast_node.index if ast_node.is_scalar() else \
                    ast_node.vector_index(index))

                return f"{feature_name}[{index_expr}]"

            return f"{ast_node.name()}" + (f"_{index}" if not ast_node.is_scalar() else "")

        if isinstance(ast_node, ParticleAttributeList):
            tid = CGen.temp_id
            list_ref = f"attr_list_{tid}"
            list_def = ", ".join([str(a.id()) for a in ast_node])
            self.print(f"const int {list_ref}[] = {{{list_def}}};")
            CGen.temp_id += 1
            return list_ref

        if isinstance(ast_node, Sizeof):
            assert mem is False, "Sizeof expression is not lvalue!"
            tkw = Types.c_keyword(self.sim, ast_node.data_type)
            return f"sizeof({tkw})"

        if isinstance(ast_node, Select):
            assert mem is False, "Select expression is not lvalue!"

            if ast_node.inlined is True:
                assert ast_node.is_scalar(), "Only scalar operations can be inlined!"
                cond = self.generate_expression(ast_node.cond, index=index)
                expr_if = self.generate_expression(ast_node.expr_if, index=index)
                expr_else = self.generate_expression(ast_node.expr_else, index=index)
                return f"(({cond}) ? ({expr_if}) : ({expr_else}))"

            if not ast_node.is_scalar():
                assert index is not None, "Index must be set for non-scalar reference."
                return f"{ast_node.name()}_{index}"

            return f"{ast_node.name()}"

        if isinstance(ast_node, Var):
            return self.generate_object_reference(ast_node, index=index)
        
        if isinstance(ast_node, Parameter):
            return ast_node.name()
        
        if isinstance(ast_node, VectorAccess):
            return self.generate_expression(ast_node.expr, mem, self.generate_expression(ast_node.index))

        if isinstance(ast_node, MatrixAccess):
            return self.generate_expression(ast_node.expr, mem, self.generate_expression(ast_node.index))

        if isinstance(ast_node, QuaternionAccess):
            return self.generate_expression(ast_node.expr, mem, self.generate_expression(ast_node.index))

        if isinstance(ast_node, Vector):
            assert index is not None, "Index must be set for vector."
            return f"{ast_node.name()}_{index}"

        if isinstance(ast_node, Matrix):
            assert index is not None, "Index must be set for matrix."
            return f"{ast_node.name()}_{index}"

        if isinstance(ast_node, Quaternion):
            assert index is not None, "Index must be set for quaternion."
            return f"{ast_node.name()}_{index}"

        if isinstance(ast_node, VectorOp):
            assert index is not None, "Index must be set for vector operation."
            return f"{ast_node.name()}_{index}"

        if isinstance(ast_node, MatrixOp):
            assert index is not None, "Index must be set for matrix operation."
            return f"{ast_node.name()}_{index}"

        if isinstance(ast_node, QuaternionOp):
            assert index is not None, "Index must be set for quaternion operation."
            return f"{ast_node.name()}_{index}"

        if isinstance(ast_node, ZeroVector):
            return "0.0"
