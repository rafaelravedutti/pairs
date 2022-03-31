from pairs.ir.assign import Assign
from pairs.ir.arrays import Array, ArrayAccess, ArrayDecl
from pairs.ir.block import Block
from pairs.ir.branches import Branch
from pairs.ir.cast import Cast
from pairs.ir.bin_op import BinOp, Decl, VectorAccess
from pairs.ir.device import CopyToDevice, CopyToHost
from pairs.ir.functions import Call
from pairs.ir.kernel import Kernel, KernelLaunch
from pairs.ir.layouts import Layouts
from pairs.ir.lit import Lit
from pairs.ir.loops import For, Iter, ParticleFor, While
from pairs.ir.math import Ceil, Sqrt
from pairs.ir.memory import Malloc, Realloc
from pairs.ir.module import ModuleCall
from pairs.ir.properties import Property, PropertyAccess, PropertyList, RegisterProperty, UpdateProperty
from pairs.ir.select import Select
from pairs.ir.sizeof import Sizeof
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.ir.variables import Var, VarDecl, Deref
from pairs.sim.timestep import Timestep
from pairs.code_gen.printer import Printer


class CGen:
    temp_id = 0

    def __init__(self, output, target, debug=False):
        self.sim = None
        self.target = None
        self.debug = debug
        self.print = Printer(output)

    def assign_simulation(self, sim):
        self.sim = sim

    def assign_target(self, target):
        self.target = target

    def generate_program(self, ast_node):
        self.print.start()
        self.print("#include <math.h>")
        self.print("#include <stdbool.h>")
        self.print("#include <stdio.h>")
        self.print("#include <stdlib.h>")
        self.print("//---")
        self.print("#include \"runtime/pairs.hpp\"")
        self.print("#include \"runtime/read_from_file.hpp\"")
        self.print("#include \"runtime/vtk.hpp\"")
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

        self.print("")

        for kernel in self.sim.kernels():
            self.generate_kernel(kernel)

        for module in self.sim.modules():
            self.generate_module(module)

        self.print.end()

    def generate_module(self, module):
        if module.name == 'main':
            self.print("int main() {")
            self.print("    PairsSim *ps = new PairsSim();")
            self.generate_statement(module.block)
            self.print("    return 0;")
            self.print("}")

        else:
            module_params = ""
            for var in module.read_only_variables():
                type_kw = Types.c_keyword(var.type())
                decl = f"{type_kw} {var.name()}"
                module_params += decl if len(module_params) <= 0 else f", {decl}"

            for var in module.write_variables():
                type_kw = Types.c_keyword(var.type())
                decl = f"{type_kw} *{var.name()}"
                module_params += decl if len(module_params) <= 0 else f", {decl}"

            for array in module.arrays():
                type_kw = Types.c_keyword(array.type())
                decl = f"{type_kw} *{array.name()}"
                module_params += decl if len(module_params) <= 0 else f", {decl}"

            for prop in module.properties():
                type_kw = Types.c_keyword(prop.type())
                decl = f"{type_kw} *{prop.name()}"
                module_params += decl if len(module_params) <= 0 else f", {decl}"

            self.print(f"void {module.name}({module_params}) {{")

            if self.debug:
                self.print.add_indent(4)
                self.generate_statement(Print(self.sim, module.name + " --- enter"))
                self.print.add_indent(-4)

            self.generate_statement(module.block)

            if self.debug:
                self.print.add_indent(4)
                self.generate_statement(Print(self.sim, module.name + " --- exit"))
                self.print.add_indent(-4)

            self.print("}")

    def generate_kernel(self, kernel):
        kernel_params = ""
        for var in kernel.read_only_variables():
            type_kw = Types.c_keyword(var.type())
            decl = f"{type_kw} {var.name()}"
            kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

        for var in kernel.write_variables():
            type_kw = Types.c_keyword(var.type())
            decl = f"{type_kw} *{var.name()}"
            kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

        for array in kernel.arrays():
            type_kw = Types.c_keyword(array.type())
            decl = f"{type_kw} *{array.name()}"
            kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

        for prop in kernel.properties():
            type_kw = Types.c_keyword(prop.type())
            decl = f"{type_kw} *{prop.name()}"
            kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

        for bin_op in kernel.bin_ops():
            type_kw = Types.c_keyword(bin_op.type())
            decl = f"{type_kw} e{bin_op.id()}"
            kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

        self.print(f"__global__ void {kernel.name}({kernel_params}) {{")
        self.print(f"    const int {kernel.iterator.name()} = blockIdx.x * blockDim.x + threadIdx.x;")
        self.generate_statement(kernel.block)
        self.print("}")

    def generate_statement(self, ast_node, bypass_checking=False):
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
            if isinstance(ast_node.elem, BinOp) and (bypass_checking or ast_node.used):
                bin_op = ast_node.elem

                if bin_op.inlined is False and bin_op.generated is False:
                    if bin_op.is_vector_kind():
                        for i in bin_op.indexes():
                            lhs = self.generate_expression(bin_op.lhs, bin_op.mem, index=i)
                            rhs = self.generate_expression(bin_op.rhs, index=i)
                            self.print(f"const double e{bin_op.id()}_{i} = {lhs} {bin_op.operator()} {rhs};")
                    else:
                        lhs = self.generate_expression(bin_op.lhs, bin_op.mem)
                        rhs = self.generate_expression(bin_op.rhs)
                        tkw = Types.c_keyword(bin_op.type())
                        self.print(f"const {tkw} e{bin_op.id()} = {lhs} {bin_op.operator()} {rhs};")

                ast_node.elem.generated = True

            if isinstance(ast_node.elem, PropertyAccess) and (bypass_checking or ast_node.used):
                prop_access = ast_node.elem
                prop_name = prop_access.prop.name()
                acc_ref = f"p{prop_access.id()}"

                if prop_access.is_vector_kind():
                    for i in prop_access.indexes():
                        i_expr = self.generate_expression(prop_access.get_index_expression(i))
                        self.print(f"const double {acc_ref}_{i} = {prop_name}[{i_expr}];")
                else:
                    tkw = Types.c_keyword(prop_access.type())
                    index_g = self.generate_expression(prop_access.index)
                    self.print(f"const {tkw} {acc_ref} = {prop_name}[{index_g}];")

                ast_node.elem.generated = True

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

        if isinstance(ast_node, CopyToDevice):
            array_name = ast_node.prop.name()
            self.print(f"pairs::copy_to_device({array_name}, d_{array_name})")

        if isinstance(ast_node, CopyToHost):
            array_name = ast_node.prop.name()
            self.print(f"pairs::copy_to_host(d_{array_name}, {array_name})")

        if isinstance(ast_node, For):
            iterator = self.generate_expression(ast_node.iterator)
            lower_range = None
            upper_range = None

            if isinstance(ast_node, ParticleFor):
                n = self.sim.nlocal if ast_node.local_only else self.sim.nlocal + self.sim.pbc.npbc
                lower_range = 0
                upper_range = self.generate_expression(n)

            else:
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
            kernel = ast_node.kernel
            kernel_params = ""
            for var in kernel.read_only_variables():
                decl = var.name()
                kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

            for var in kernel.write_variables():
                decl = f"&{var.name()}"
                kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

            for array in kernel.arrays():
                decl = f"d_{array.name()}"
                kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

            for prop in kernel.properties():
                decl = f"d_{prop.name()}"
                kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

            for bin_op in kernel.bin_ops():
                decl = self.generate_expression(bin_op)
                kernel_params += decl if len(kernel_params) <= 0 else f", {decl}"

            elems = ast_node.max - ast_node.min
            blocks = self.generate_expression((elems + ast_node.threads_per_block - 1) / ast_node.threads_per_block)
            self.print(f"{kernel.name}<<<{blocks}, {ast_node.threads_per_block}>>>({kernel_params});")

        if isinstance(ast_node, ModuleCall):
            module = ast_node.module
            module_params = ""
            device_cond = module.run_on_device and self.target.is_gpu()

            for var in module.read_only_variables():
                decl = var.name()
                module_params += decl if len(module_params) <= 0 else f", {decl}"

            for var in module.write_variables():
                decl = f"&{var.name()}"
                module_params += decl if len(module_params) <= 0 else f", {decl}"

            for array in module.arrays():
                decl = f"d_{array.name()}" if device_cond else array.name()
                module_params += decl if len(module_params) <= 0 else f", {decl}"

            for prop in module.properties():
                decl = f"d_{prop.name()}" if device_cond else prop.name()
                module_params += decl if len(module_params) <= 0 else f", {decl}"

            self.print(f"{module.name}({module_params});")

        if isinstance(ast_node, Print):
            self.print(f"fprintf(stdout, \"{ast_node.string}\\n\");")
            self.print(f"fflush(stdout);")

        if isinstance(ast_node, Realloc):
            tkw = Types.c_keyword(ast_node.array.type())
            size = self.generate_expression(ast_node.size)
            array_name = ast_node.array.name()
            self.print(f"{array_name} = ({tkw} *) realloc({array_name}, {size});")
            if self.target.is_gpu() and ast_node.array.device_flag:
                self.print(f"d_{array_name} = ({tkw} *) pairs::device_realloc(d_{array_name}, {size});")

        if isinstance(ast_node, RegisterProperty):
            p = ast_node.property()
            ptype = "Prop_Integer"  if p.type() == Types.Int32 else \
                    "Prop_Float"    if p.type() == Types.Double else \
                    "Prop_Vector"   if p.type() == Types.Vector else \
                    "Prop_Invalid"

            assert ptype != "Prop_Invalid", "Invalid property type!"

            playout = "AoS" if p.layout() == Layouts.AoS else \
                      "SoA" if p.layout() == Layouts.SoA else \
                      "Invalid"

            if p.type() != Types.Vector or p.layout() == Layouts.Invalid:
                self.print(f"ps->addProperty(Property({p.id()}, \"{p.name()}\", {p.name()}, {ptype}));")
            else:
                sizes = ", ".join([str(self.generate_expression(size)) for size in ast_node.sizes()])
                self.print(f"ps->addProperty(Property({p.id()}, \"{p.name()}\", {p.name()}, {ptype}, {playout}, {sizes}));")

        if isinstance(ast_node, Timestep):
            self.generate_statement(ast_node.block)

        if isinstance(ast_node, UpdateProperty):
            p = ast_node.property()
            if p.type() != Types.Vector or p.layout() == Layouts.Invalid:
                self.print(f"ps->updateProperty({p.id()}, {p.name()});")
            else:
                sizes = ", ".join([str(self.generate_expression(size)) for size in ast_node.sizes()])
                self.print(f"ps->updateProperty({p.id()}, {p.name()}, {sizes});")

        if isinstance(ast_node, VarDecl):
            tkw = Types.c_keyword(ast_node.var.type())
            self.print(f"{tkw} {ast_node.var.name()} = {ast_node.var.init_value()};")

        if isinstance(ast_node, While):
            cond = self.generate_expression(ast_node.cond)
            self.print(f"while({cond}) {{")
            self.generate_statement(ast_node.block)
            self.print("}")

    def generate_expression(self, ast_node, mem=False, index=None):
        if isinstance(ast_node, Array):
            return ast_node.name()

        if isinstance(ast_node, ArrayAccess):
            array_name = ast_node.array.name()
            acc_index = self.generate_expression(ast_node.index)
            if mem or ast_node.inlined is True:
                return f"{array_name}[{acc_index}]"

            acc_ref = f"a{ast_node.id()}"
            if not ast_node.generated:
                tkw = Types.c_keyword(ast_node.type())
                self.print(f"const {tkw} {acc_ref} = {array_name}[{acc_index}];")
                ast_node.generated = True

            return acc_ref

        if isinstance(ast_node, BinOp):
            lhs = self.generate_expression(ast_node.lhs, mem, index)
            rhs = self.generate_expression(ast_node.rhs, index=index)

            if ast_node.inlined is True:
                assert ast_node.type() != Types.Vector, "Vector operations cannot be inlined!"
                return f"({lhs} {ast_node.operator()} {rhs})"

            # Some expressions can be defined on-the-fly during transformations, hence they do not have
            # a declaration statement in the tree, so we generate them right before use
            if not ast_node.generated:
                self.generate_statement(ast_node.declaration(), bypass_checking=True)

            if ast_node.is_vector_kind():
                assert index is not None, "Index must be set for vector reference!"
                return f"e{ast_node.id()}[{index}]" if ast_node.mem else f"e{ast_node.id()}_{index}"

            return f"e{ast_node.id()}"

        if isinstance(ast_node, Call):
            params = ", ".join(["ps"] + [str(self.generate_expression(p)) for p in ast_node.parameters()])
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
            prop_name = ast_node.prop.name()

            if mem or ast_node.inlined is True:
                index_expr = ast_node.index if not ast_node.is_vector_kind() else ast_node.get_index_expression(index)
                index_g = self.generate_expression(index_expr)
                return f"{prop_name}[{index_g}]"

            if not ast_node.generated:
                self.generate_statement(ast_node.declaration(), bypass_checking=True) 

            acc_ref = f"p{ast_node.id()}"
            if ast_node.is_vector_kind():
                acc_ref += f"_{index}"

            return acc_ref

        if isinstance(ast_node, PropertyList):
            tid = CGen.temp_id
            list_ref = f"prop_list_{tid}"
            list_def = ", ".join(str(p.id()) for p in ast_node)
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
