from pairs.ir.assign import Assign
from pairs.ir.atomic import AtomicAdd
from pairs.ir.arrays import Array, ArrayAccess, ArrayDecl, RegisterArray, ReallocArray
from pairs.ir.block import Block
from pairs.ir.branches import Branch
from pairs.ir.cast import Cast
from pairs.ir.contexts import Contexts
from pairs.ir.bin_op import BinOp, Decl, VectorAccess
from pairs.ir.device import ClearArrayFlag, ClearPropertyFlag, CopyArray, CopyProperty, CopyVar, SetArrayFlag, SetPropertyFlag, HostRef
from pairs.ir.functions import Call
from pairs.ir.kernel import Kernel, KernelLaunch
from pairs.ir.layouts import Layouts
from pairs.ir.lit import Lit
from pairs.ir.loops import For, Iter, ParticleFor, While
from pairs.ir.math import Ceil, Sqrt
from pairs.ir.memory import Malloc, Realloc
from pairs.ir.module import ModuleCall
from pairs.ir.features import RegisterFeature
from pairs.ir.particle_attributes import ParticleAttributeList
from pairs.ir.properties import Property, PropertyAccess, RegisterProperty, ReallocProperty
from pairs.ir.select import Select
from pairs.ir.sizeof import Sizeof
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.ir.variables import Var, VarDecl, Deref
from pairs.sim.timestep import Timestep
from pairs.code_gen.printer import Printer


class CGen:
    temp_id = 0

    def __init__(self, ref, debug=False):
        self.sim = None
        self.target = None
        self.print = None
        self.ref = ref
        self.debug = debug

    def assign_simulation(self, sim):
        self.sim = sim

    def assign_target(self, target):
        self.target = target

    def generate_program(self, ast_node):
        ext = ".cu" if self.target.is_gpu() else ".cpp"
        self.print = Printer(self.ref + ext)
        self.print.start()

        if self.target.is_gpu():
            self.print("#define PAIRS_TARGET_CUDA")

        self.print("#include <math.h>")
        self.print("#include <stdbool.h>")
        self.print("#include <stdio.h>")
        self.print("#include <stdlib.h>")
        self.print("//---")
        self.print("#include \"runtime/pairs.hpp\"")
        self.print("#include \"runtime/read_from_file.hpp\"")
        self.print("#include \"runtime/vtk.hpp\"")

        #if self.target.is_gpu():
        #    self.print("#include \"runtime/devices/cuda.hpp\"")
        #else:
        #    self.print("#include \"runtime/devices/dummy.hpp\"")

        self.print("")
        self.print("using namespace pairs;")
        self.print("")

        if self.target.is_gpu():
            for array in self.sim.arrays.statics():
                if array.device_flag:
                    t = array.type()
                    tkw = Types.c_keyword(t)
                    size = self.generate_expression(BinOp.inline(array.alloc_size()))
                    self.print(f"__constant__ {tkw} d_{array.name()}[{size}];")

            for feature_prop in self.sim.feature_properties():
                if feature_prop.device_flag:
                    t = feature_prop.type()
                    tkw = Types.c_keyword(t)
                    size = self.generate_expression(BinOp.inline(feature_prop.array_size()))
                    self.print(f"__constant__ {tkw} d_{feature_prop.name()}[{size}];")

        self.print("")

        for kernel in self.sim.kernels():
            self.generate_kernel(kernel)

        for module in self.sim.modules():
            self.generate_module(module)

        self.print.end()

    def generate_module(self, module):
        if module.name == 'main':
            ndims = module.sim.ndims()
            nprops = module.sim.properties.nprops()
            narrays = module.sim.arrays.narrays()
            self.print("int main(int argc, char **argv) {")
            self.print(f"    PairsSimulation *pairs = new PairsSimulation({nprops}, {narrays}, DimRanges);")
            self.generate_statement(module.block)
            self.print("    delete pairs;")
            self.print("    return 0;")
            self.print("}")

        else:
            module_params = "PairsSimulation *pairs"
            for var in module.read_only_variables():
                type_kw = Types.c_keyword(var.type())
                decl = f"{type_kw} {var.name()}"
                module_params += f", {decl}"

            for var in module.write_variables():
                type_kw = Types.c_keyword(var.type())
                decl = f"{type_kw} *{var.name()}"
                module_params += f", {decl}"

            for array in module.arrays():
                type_kw = Types.c_keyword(array.type())
                decl = f"{type_kw} *{array.name()}"
                module_params += f", {decl}"

                if array in module.host_references():
                    decl = f"{type_kw} *h_{array.name()}"
                    module_params += f", {decl}"

            for prop in module.properties():
                type_kw = Types.c_keyword(prop.type())
                decl = f"{type_kw} *{prop.name()}"
                module_params += f", {decl}"

                if prop in module.host_references():
                    decl = f"{type_kw} *h_{prop.name()}"
                    module_params += f", {decl}"

            self.print(f"void {module.name}({module_params}) {{")

            if self.debug:
                self.print.add_indent(4)
                self.print(f"PAIRS_DEBUG(\"{module.name}\\n\");")
                self.print.add_indent(-4)

            self.print.add_indent(4)
            for t in module.temporaries:
                self.generate_statement(VarDecl(module.sim, t))
            self.print.add_indent(-4)

            self.generate_statement(module.block)
            self.print("}")

    def generate_kernel(self, kernel):
        kernel_params = "int range_start"
        for var in kernel.read_only_variables():
            type_kw = Types.c_keyword(var.type())
            decl = f"{type_kw} {var.name()}"
            kernel_params += f", {decl}"

        for var in kernel.write_variables():
            type_kw = Types.c_keyword(var.type())
            decl = f"{type_kw} *{var.name()}"
            kernel_params += f", {decl}"

        for array in kernel.arrays():
            type_kw = Types.c_keyword(array.type())
            decl = f"{type_kw} *{array.name()}"
            kernel_params += f", {decl}"

        for prop in kernel.properties():
            type_kw = Types.c_keyword(prop.type())
            decl = f"{type_kw} *{prop.name()}"
            kernel_params += f", {decl}"

        for array_access in kernel.array_accesses():
            type_kw = Types.c_keyword(array_access.type())
            decl = f"{type_kw} a{array_access.id()}"
            kernel_params += f", {decl}"

        for bin_op in kernel.bin_ops():
            type_kw = Types.c_keyword(bin_op.type())
            decl = f"{type_kw} e{bin_op.id()}"
            kernel_params += f", {decl}"

        self.print(f"__global__ void {kernel.name}({kernel_params}) {{")
        self.print(f"    const int {kernel.iterator.name()} = blockIdx.x * blockDim.x + threadIdx.x + range_start;")
        self.print.add_indent(4)
        self.generate_statement(kernel.block)
        self.print.add_indent(-4)
        self.print("}")

    def generate_statement(self, ast_node):
        if isinstance(ast_node, ArrayDecl):
            t = ast_node.array.type()
            tkw = Types.c_keyword(t)
            size = self.generate_expression(BinOp.inline(ast_node.array.alloc_size()))
            if ast_node.array.init_value is not None:
                v_str = str(ast_node.array.init_value)
                if t == Types.Int64:
                    v_str += "LL"
                if t == Types.UInt64:
                    v_str += "ULL"

                init_string = v_str + (f", {v_str}" * (size - 1))
                self.print(f"{tkw} {ast_node.array.name()}[{size}] = {{{init_string}}};")
            else:
                self.print(f"{tkw} {ast_node.array.name()}[{size}];")

        if isinstance(ast_node, Assign):
            for assign_dest, assign_src in ast_node.assignments:
                dest = self.generate_expression(assign_dest, mem=True)
                src = self.generate_expression(assign_src)
                self.print(f"{dest} = {src};")

        if isinstance(ast_node, Block):
            self.print.add_indent(4)
            for stmt in ast_node.statements():
                self.generate_statement(stmt)
            self.print.add_indent(-4)

        # TODO: Why there are Decls for other types?
        if isinstance(ast_node, Decl):
            if isinstance(ast_node.elem, BinOp):
                bin_op = ast_node.elem
                if bin_op.inlined is False:
                    if bin_op.is_vector_kind():
                        for i in bin_op.indexes():
                            lhs = self.generate_expression(bin_op.lhs, bin_op.mem, index=i)
                            rhs = self.generate_expression(bin_op.rhs, index=i)
                            operator = bin_op.operator()
                            self.print(f"const double e{bin_op.id()}_{i} = {lhs} {operator.symbol()} {rhs};")
                    else:
                        lhs = self.generate_expression(bin_op.lhs, bin_op.mem)
                        rhs = self.generate_expression(bin_op.rhs)
                        operator = bin_op.operator()
                        tkw = Types.c_keyword(bin_op.type())
                        self.print(f"const {tkw} e{bin_op.id()} = {lhs} {operator.symbol()} {rhs};")

            if isinstance(ast_node.elem, ArrayAccess):
                array_access = ast_node.elem
                array_name = self.generate_expression(array_access.array)
                tkw = Types.c_keyword(array_access.type())
                acc_index = self.generate_expression(array_access.flat_index)
                acc_ref = f"a{array_access.id()}"
                self.print(f"const {tkw} {acc_ref} = {array_name}[{acc_index}];")

            if isinstance(ast_node.elem, FeatureAccess):
                feature_access = ast_node.elem
                feature_name = self.generate_expression(feature_access.feature.name())
                tkw = Types.c_keyword(feature_access.type())
                acc_index = self.generate_expression(feature_access.index)
                acc_ref = f"f{array_access.id()}"
                self.print(f"const {tkw} {acc_ref} = {feature_name}[{acc_index}];")

            if isinstance(ast_node.elem, PropertyAccess):
                prop_access = ast_node.elem
                prop_name = self.generate_expression(prop_access.prop)
                acc_ref = f"p{prop_access.id()}"

                if prop_access.is_vector_kind():
                    for i in prop_access.indexes():
                        i_expr = self.generate_expression(prop_access.get_index_expression(i))
                        self.print(f"const double {acc_ref}_{i} = {prop_name}[{i_expr}];")
                else:
                    tkw = Types.c_keyword(prop_access.type())
                    index_g = self.generate_expression(prop_access.index)
                    self.print(f"const {tkw} {acc_ref} = {prop_name}[{index_g}];")

            if isinstance(ast_node.elem, AtomicAdd):
                atomic_add = ast_node.elem
                elem = self.generate_expression(atomic_add.elem)
                value = self.generate_expression(atomic_add.value)
                tkw = Types.c_keyword(atomic_add.type())
                acc_ref = f"atm_add{atomic_add.id()}"
                prefix = "" if ast_node.elem.device_flag else "host_"

                if atomic_add.check_for_resize():
                    resize = self.generate_expression(atomic_add.resize)
                    capacity = self.generate_expression(atomic_add.capacity)
                    self.print(f"const {tkw} {acc_ref} = pairs::{prefix}atomic_add_resize_check(&({elem}), {value}, &({resize}), {capacity});")
                else:
                    self.print(f"const {tkw} {acc_ref} = pairs::{prefix}atomic_add(&({elem}), {value});")

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
            array_id = ast_node.array.id()
            array_name = ast_node.array.name()

            if ast_node.context() == Contexts.Device:
                self.print(f"pairs->copyArrayToDevice({array_id}); // {array_name}")
            else:
                self.print(f"pairs->copyArrayToHost({array_id}); // {array_name}")

        if isinstance(ast_node, CopyProperty):
            prop_id = ast_node.prop.id()
            prop_name = ast_node.prop.name()

            if ast_node.context() == Contexts.Device:
                self.print(f"pairs->copyPropertyToDevice({prop_id}); // {prop_name}")
            else:
                self.print(f"pairs->copyPropertyToHost({prop_id}); // {prop_name}")

        if isinstance(ast_node, CopyVar):
            var_name = ast_node.variable.name()

            if ast_node.context() == Contexts.Device:
                self.print(f"rv_{var_name}.copyToDevice();")
            else:
                self.print(f"rv_{var_name}.copyToHost();")

        if isinstance(ast_node, ClearArrayFlag):
            array_id = ast_node.array.id()
            array_name = ast_node.array.name()

            if ast_node.context() == Contexts.Device:
                self.print(f"pairs->clearArrayDeviceFlag({array_id}); // {array_name}")
            else:
                self.print(f"pairs->clearArrayHostFlag({array_id}); // {array_name}")

        if isinstance(ast_node, ClearPropertyFlag):
            prop_id = ast_node.prop.id()
            prop_name = ast_node.prop.name()

            if ast_node.context() == Contexts.Device:
                self.print(f"pairs->clearPropertyDeviceFlag({prop_id}); // {prop_name}")
            else:
                self.print(f"pairs->clearPropertyHostFlag({prop_id}); // {prop_name}")

        if isinstance(ast_node, SetArrayFlag):
            array_id = ast_node.array.id()
            array_name = ast_node.array.name()

            if ast_node.context() == Contexts.Device:
                self.print(f"pairs->setArrayDeviceFlag({array_id}); // {array_name}")
            else:
                self.print(f"pairs->setArrayHostFlag({array_id}); // {array_name}")

        if isinstance(ast_node, SetPropertyFlag):
            prop_id = ast_node.prop.id()
            prop_name = ast_node.prop.name()

            if ast_node.context() == Contexts.Device:
                self.print(f"pairs->setPropertyDeviceFlag({prop_id}); // {prop_name}")
            else:
                self.print(f"pairs->setPropertyHostFlag({prop_id}); // {prop_name}")

        if isinstance(ast_node, For):
            iterator = self.generate_expression(ast_node.iterator)
            lower_range = self.generate_expression(ast_node.min)
            upper_range = self.generate_expression(ast_node.max)
            self.print(f"for(int {iterator} = {lower_range}; {iterator} < {upper_range}; {iterator}++) {{")
            self.generate_statement(ast_node.block)
            self.print("}")


        if isinstance(ast_node, Malloc):
            tkw = Types.c_keyword(ast_node.array.type())
            size = self.generate_expression(ast_node.size)
            array_name = ast_node.array.name()

            if ast_node.decl:
                self.print(f"{tkw} *{array_name} = ({tkw} *) malloc({size});")
                if self.target.is_gpu() and ast_node.array.device_flag:
                    self.print(f"{tkw} *d_{array_name} = ({tkw} *) pairs::device_alloc({size});")
            else:
                self.print(f"{array_name} = ({tkw} *) malloc({size});")
                if self.target.is_gpu() and ast_node.array.device_flag:
                    self.print(f"d_{array_name} = ({tkw} *) pairs::device_alloc({size});")

        if isinstance(ast_node, KernelLaunch):
            range_start = self.generate_expression(BinOp.inline(ast_node.min))
            kernel = ast_node.kernel
            kernel_params = f"{range_start}"

            for var in kernel.read_only_variables():
                kernel_params += f", {var.name()}"

            for var in kernel.write_variables():
                kernel_params += f", {var.name()}"

            for array in kernel.arrays():
                kernel_params += f", {array.name()}"

            for prop in kernel.properties():
                kernel_params += f", {prop.name()}"

            for array_access in kernel.array_accesses():
                kernel_params += f", {self.generate_expression(array_access)}"

            for bin_op in kernel.bin_ops():
                kernel_params += f", {self.generate_expression(bin_op)}"

            threads_per_block = self.generate_expression(ast_node.threads_per_block)
            nblocks = self.generate_expression(ast_node.nblocks)
            self.print(f"if({nblocks} > 0 && {threads_per_block} > 0) {{")
            self.print.add_indent(4)
            self.print(f"{kernel.name}<<<{nblocks}, {threads_per_block}>>>({kernel_params});")
            self.print("pairs->sync();")
            self.print.add_indent(-4)
            self.print("}")

        if isinstance(ast_node, ModuleCall):
            module = ast_node.module
            module_params = "pairs"
            device_cond = module.run_on_device and self.target.is_gpu()

            for var in module.read_only_variables():
                decl = var.name()
                module_params += f", {decl}"

            for var in module.write_variables():
                decl = f"rv_{var.name()}.getDevicePointer()" if device_cond and var.device_flag else f"&{var.name()}"
                module_params += f", {decl}"

            for array in module.arrays():
                decl = f"d_{array.name()}" if device_cond else array.name()
                module_params += decl if len(module_params) <= 0 else f", {decl}"
                if array in module.host_references():
                    decl = array.name()
                    module_params += f", {decl}"

            for prop in module.properties():
                decl = f"d_{prop.name()}" if device_cond else prop.name()
                module_params += f", {decl}"
                if prop in module.host_references():
                    decl = prop.name()
                    module_params += f", {decl}"

            self.print(f"{module.name}({module_params});")

        if isinstance(ast_node, Print):
            self.print(f"PAIRS_DEBUG(\"{ast_node.string}\\n\");")

        if isinstance(ast_node, Realloc):
            tkw = Types.c_keyword(ast_node.array.type())
            size = self.generate_expression(ast_node.size)
            array_name = ast_node.array.name()
            self.print(f"{array_name} = ({tkw} *) realloc({array_name}, {size});")
            if self.target.is_gpu() and ast_node.array.device_flag:
                self.print(f"d_{array_name} = ({tkw} *) pairs::device_realloc(d_{array_name}, {size});")

        if isinstance(ast_node, RegisterArray):
            a = ast_node.array()
            ptr = a.name()
            d_ptr = f"d_{ptr}" if self.target.is_gpu() and a.device_flag else "nullptr"
            tkw = Types.c_keyword(a.type())
            size = self.generate_expression(ast_node.size())

            if a.is_static():
                self.print(f"pairs->addStaticArray({a.id()}, \"{a.name()}\", {ptr}, {d_ptr}, {size});") 

            else:
                if self.target.is_gpu() and a.device_flag:
                    self.print(f"{tkw} *{ptr}, *{d_ptr};")
                    d_ptr = f"&{d_ptr}"
                else:
                    self.print(f"{tkw} *{ptr};")

                self.print(f"pairs->addArray({a.id()}, \"{a.name()}\", &{ptr}, {d_ptr}, {size});")

        if isinstance(ast_node, RegisterProperty):
            p = ast_node.property()
            ptr = p.name()
            d_ptr = f"d_{ptr}" if self.target.is_gpu() and p.device_flag else "nullptr"
            tkw = Types.c_keyword(p.type())
            ptype = "Prop_Integer"  if p.type() == Types.Int32 else \
                    "Prop_Float"    if p.type() == Types.Double else \
                    "Prop_Vector"   if p.type() == Types.Vector else \
                    "Prop_Invalid"

            assert ptype != "Prop_Invalid", "Invalid property type!"

            playout = "AoS" if p.layout() == Layouts.AoS else \
                      "SoA" if p.layout() == Layouts.SoA else \
                      "Invalid"

            sizes = ", ".join([str(self.generate_expression(BinOp.inline(size))) for size in ast_node.sizes()])

            if self.target.is_gpu() and p.device_flag:
                self.print(f"{tkw} *{ptr}, *{d_ptr};")
                d_ptr = f"&{d_ptr}"
            else:
                self.print(f"{tkw} *{ptr};")

            self.print(f"pairs->addProperty({p.id()}, \"{p.name()}\", &{ptr}, {d_ptr}, {ptype}, {playout}, {sizes});")

        if isinstance(ast_node, RegisterFeatureProperty):
            fp = ast_node.feature_property()
            ptr = fp.name()
            d_ptr = f"d_{ptr}" if self.target.is_gpu() and p.device_flag else "nullptr"
            array_size = fp.array_size()
            nkinds = fp.feature().nkinds()
            self.print(f"{tkw} {ptr}[{array_size}];")
            self.print(f"pairs->addFeatureProperty({fp.id()}, \"{fp.name()}\", &{ptr}, {d_ptr}, {nkinds}, {array_size});")


        if isinstance(ast_node, Timestep):
            self.generate_statement(ast_node.block)

        if isinstance(ast_node, ReallocProperty):
            p = ast_node.property()
            ptr = p.name()
            d_ptr_addr = f"&d_{ptr}" if self.target.is_gpu() and p.device_flag else "nullptr"
            sizes = ", ".join([str(self.generate_expression(BinOp.inline(size))) for size in ast_node.sizes()])
            self.print(f"pairs->reallocProperty({p.id()}, &{ptr}, {d_ptr_addr}, {sizes});")
            #self.print(f"pairs->reallocProperty({p.id()}, (void **) &{ptr}, (void **) &d_{ptr}, {sizes});")

        if isinstance(ast_node, ReallocArray):
            a = ast_node.array()
            size = self.generate_expression(ast_node.size())
            ptr = a.name()
            d_ptr_addr = f"&d_{ptr}" if self.target.is_gpu() and a.device_flag else "nullptr"
            self.print(f"pairs->reallocArray({a.id()}, &{ptr}, {d_ptr_addr}, {size});")
            #self.print(f"pairs->reallocArray({a.id()}, (void **) &{ptr}, (void **) &d_{ptr}, {size});")

        if isinstance(ast_node, VarDecl):
            tkw = Types.c_keyword(ast_node.var.type())
            self.print(f"{tkw} {ast_node.var.name()} = {ast_node.var.init_value()};")

            if self.target.is_gpu() and ast_node.var.device_flag:
                self.print(f"RuntimeVar<{tkw}> rv_{ast_node.var.name()} = pairs->addDeviceVariable(&({ast_node.var.name()}));")
                #self.print(f"{tkw} *d_{ast_node.var.name()} = pairs->addDeviceVariable(&({ast_node.var.name()}));")

        if isinstance(ast_node, While):
            cond = self.generate_expression(ast_node.cond)
            self.print(f"while({cond}) {{")
            self.generate_statement(ast_node.block)
            self.print("}")

    def generate_expression(self, ast_node, mem=False, index=None):
        if isinstance(ast_node, Array):
            return ast_node.name()

        if isinstance(ast_node, ArrayAccess):
            if mem or ast_node.inlined is True:
                array_name = self.generate_expression(ast_node.array)
                acc_index = self.generate_expression(ast_node.flat_index)
                return f"{array_name}[{acc_index}]"

            return f"a{ast_node.id()}"

        if isinstance(ast_node, AtomicAdd):
            return f"atm_add{ast_node.id()}"

        if isinstance(ast_node, BinOp):
            if ast_node.inlined is True:
                assert ast_node.type() != Types.Vector, "Vector operations cannot be inlined!"
                lhs = self.generate_expression(ast_node.lhs, mem, index)
                rhs = self.generate_expression(ast_node.rhs, index=index)
                operator = ast_node.operator()
                return f"({lhs} {operator.symbol()} {rhs})"

            if ast_node.is_vector_kind():
                assert index is not None, "Index must be set for vector reference!"
                return f"e{ast_node.id()}[{index}]" if ast_node.mem else f"e{ast_node.id()}_{index}"

            return f"e{ast_node.id()}"

        if isinstance(ast_node, Call):
            extra_params = []

            if ast_node.name().startswith("pairs::"):
                extra_params += ["pairs"]

            if ast_node.name() == "pairs->initDomain":
                extra_params += ["&argc", "&argv"]

            params = ", ".join(extra_params + [str(self.generate_expression(p)) for p in ast_node.parameters()])
            return f"{ast_node.name()}({params})"

        if isinstance(ast_node, Cast):
            tkw = Types.c_keyword(ast_node.cast_type)
            expr = self.generate_expression(ast_node.expr)
            return f"({tkw})({expr})"

        if isinstance(ast_node, Ceil):
            assert mem is False, "Ceil call is not lvalue!"
            expr = self.generate_expression(ast_node.expr)
            return f"ceil({expr})"

        if isinstance(ast_node, Deref):
            var = self.generate_expression(ast_node.var)
            return f"(*{var})"

        if isinstance(ast_node, HostRef):
            elem = self.generate_expression(ast_node.elem)
            return f"h_{elem}"

        if isinstance(ast_node, Iter):
            assert mem is False, "Iterator is not lvalue!"
            return f"i{ast_node.id()}"

        if isinstance(ast_node, Lit):
            assert mem is False, "Literal is not lvalue!"
            if ast_node.type() == Types.String:
                return f"\"{ast_node.value}\""

            return ast_node.value

        if isinstance(ast_node, Property):
            return ast_node.name()

        if isinstance(ast_node, PropertyAccess):
            assert not ast_node.is_vector_kind() or index is not None, "Index must be set for vector property access!"
            prop_name = self.generate_expression(ast_node.prop)

            if mem or ast_node.inlined is True:
                index_expr = ast_node.index if not ast_node.is_vector_kind() else ast_node.get_index_expression(index)
                index_g = self.generate_expression(index_expr)
                return f"{prop_name}[{index_g}]"

            return f"p{ast_node.id()}" + (f"_{index}" if ast_node.is_vector_kind() else "")

        if isinstance(ast_node, FeatureAccess):
            feature_name = self.generate_expression(ast_node.feature.name())
            if mem or ast_node.inlined is True:
                index = self.generate_expression(ast_node.index)
                return f"{feature_name}[{index}]"

            return f"f{ast_node.id()}"

        if isinstance(ast_node, ParticleAttributeList):
            tid = CGen.temp_id
            list_ref = f"attr_list_{tid}"
            list_def = ", ".join(str([a.id() for p in ast_node]))
            self.print(f"const int {list_ref}[] = {{{list_def}}};")
            CGen.temp_id += 1
            return list_ref

        if isinstance(ast_node, Sizeof):
            assert mem is False, "Sizeof expression is not lvalue!"
            tkw = Types.c_keyword(ast_node.data_type)
            return f"sizeof({tkw})"

        if isinstance(ast_node, Sqrt):
            assert mem is False, "Square root call is not lvalue!"
            expr = self.generate_expression(ast_node.expr)
            return f"sqrt({expr})"

        if isinstance(ast_node, Select):
            assert mem is False, "Select expression is not lvalue!"
            cond = self.generate_expression(ast_node.cond)
            expr_if = self.generate_expression(ast_node.expr_if)
            expr_else = self.generate_expression(ast_node.expr_else)
            return f"({cond}) ? ({expr_if}) : ({expr_else})"

        if isinstance(ast_node, Var):
            return ast_node.name()

        if isinstance(ast_node, VectorAccess):
            return self.generate_expression(ast_node.expr, mem, self.generate_expression(ast_node.index))
