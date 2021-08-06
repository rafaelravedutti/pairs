from ir.assign import Assign
from ir.arrays import Array, ArrayAccess, ArrayDecl
from ir.block import Block
from ir.branches import Branch
from ir.cast import Cast
from ir.bin_op import BinOp, BinOpDef
from ir.data_types import Type_Int, Type_Float, Type_String, Type_Vector
from ir.functions import Call
from ir.layouts import Layout_AoS, Layout_SoA, Layout_Invalid
from ir.lit import Lit
from ir.loops import For, Iter, ParticleFor, While
from ir.math import Ceil, Sqrt
from ir.memory import Malloc, Realloc
from ir.properties import Property, PropertyList, RegisterProperty
from ir.select import Select
from ir.sizeof import Sizeof
from ir.utils import Print
from ir.variables import Var, VarDecl
from sim.timestep import Timestep
from sim.vtk import VTKWrite
from code_gen.printer import Printer


class CGen:
    temp_id = 0

    def type2keyword(type_):
        return (
            'double' if type_ == Type_Float or type_ == Type_Vector
            else 'int' if type_ == Type_Int
            else 'bool'
        )

    def __init__(self, output):
        self.sim = None
        self.print = Printer(output)

    def assign_simulation(self, sim):
        self.sim = sim

    def generate_program(self, ast_node):
        self.print.start()
        self.print("#include <math.h>")
        self.print("#include <stdbool.h>")
        self.print("#include <stdio.h>")
        self.print("#include <stdlib.h>")
        self.print("//---")
        self.print("#include \"runtime/pairs.hpp\"")
        self.print("#include \"runtime/read_from_file.hpp\"")
        self.print("")
        self.print("using namespace pairs;")
        self.print("")
        self.print("int main() {")
        self.print("    PairsSim *ps = new PairsSim();")
        self.generate_statement(ast_node)
        self.print("}")
        self.print.end()

    def generate_statement(self, ast_node):
        if isinstance(ast_node, ArrayDecl):
            tkw = CGen.type2keyword(ast_node.array.type())
            size = self.generate_expression(BinOp.inline(ast_node.array.alloc_size()))
            self.print(f"{tkw} {ast_node.array.name()}[{size}];")

        if isinstance(ast_node, Assign):
            for assign_dest, assign_src in ast_node.assignments:
                dest = self.generate_expression(assign_dest, mem=True)
                src = self.generate_expression(assign_src)
                self.print(f"{dest} = {src};")

        if isinstance(ast_node, Block):
            self.print.add_ind(4)

            for stmt in ast_node.statements():
                self.generate_statement(stmt)

            self.print.add_ind(-4)

        if isinstance(ast_node, BinOpDef):
            bin_op = ast_node.bin_op

            if not isinstance(bin_op, BinOp) or not ast_node.used:
                return None

            if bin_op.inlined is False and bin_op.operator() != '[]' and bin_op.generated is False:
                if bin_op.kind() == BinOp.Kind_Scalar:
                    lhs = self.generate_expression(bin_op.lhs, bin_op.mem)
                    rhs = self.generate_expression(bin_op.rhs)
                    tkw = CGen.type2keyword(bin_op.type())
                    self.print(f"const {tkw} e{bin_op.id()} = {lhs} {bin_op.operator()} {rhs};")

                elif bin_op.kind() == BinOp.Kind_Vector:
                    for i in bin_op.vector_indexes:
                        lhs = self.generate_expression(bin_op.lhs, bin_op.mem, index=i)
                        rhs = self.generate_expression(bin_op.rhs, index=i)
                        self.print(f"const double e{bin_op.id()}_{i} = {lhs} {bin_op.operator()} {rhs};")

                else:
                    raise Exception("Invalid BinOp kind!")

                bin_op.generated = True

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
            self.print("{call};")

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
            tkw = CGen.type2keyword(ast_node.array.type())
            size = self.generate_expression(ast_node.size)
            array_name = ast_node.array.name()

            if ast_node.decl:
                self.print(f"{tkw} *{array_name} = ({tkw} *) malloc({size});")
            else:
                self.print(f"{array_name} = ({tkw} *) malloc({size});")

        if isinstance(ast_node, Print):
            self.print(f"fprintf(stdout, \"{ast_node.string}\\n\");")
            self.print(f"fflush(stdout);")

        if isinstance(ast_node, Realloc):
            tkw = CGen.type2keyword(ast_node.array.type())
            size = self.generate_expression(ast_node.size)
            array_name = ast_node.array.name()
            self.print(f"{array_name} = ({tkw} *) realloc({array_name}, {size});")

        if isinstance(ast_node, RegisterProperty):
            p = ast_node.property()
            ptype = "Prop_Integer"  if p.type() == Type_Int else \
                    "Prop_Float"    if p.type() == Type_Float else \
                    "Prop_Vector"   if p.type() == Type_Vector else \
                    "Prop_Invalid"

            assert ptype != "Prop_Invalid", "Invalid property type!"

            playout = "AoS" if p.layout() == Layout_AoS else \
                      "SoA" if p.layout() == Layout_SoA else \
                      "Invalid"

            if p.type() != Type_Vector or p.layout() == Layout_Invalid:
                self.print(f"ps->addProperty(Property({p.id()}, \"{p.name()}\", {p.name()}, {ptype}));")
            else:
                sizes = ", ".join([str(self.generate_expression(size)) for size in ast_node.sizes()])
                self.print(f"ps->addProperty(Property({p.id()}, \"{p.name()}\", {p.name()}, {ptype}, {playout}, {sizes}));")

        if isinstance(ast_node, Timestep):
            self.generate_statement(ast_node.block)

        if isinstance(ast_node, VarDecl):
            tkw = CGen.type2keyword(ast_node.var.type())
            self.print(f"{tkw} {ast_node.var.name()} = {ast_node.var.init_value()};")

        if isinstance(ast_node, VTKWrite):
            nlocal = self.generate_expression(self.sim.nlocal)
            npbc = self.generate_expression(self.sim.pbc.npbc)
            nall = self.generate_expression(self.sim.nlocal + self.sim.pbc.npbc)
            timestep = self.generate_expression(ast_node.timestep)
            self.generate_vtk_writing(ast_node.vtk_id * 2, f"{ast_node.filename}_local", 0, nlocal, nlocal, timestep)
            self.generate_vtk_writing(ast_node.vtk_id * 2 + 1, f"{ast_node.filename}_pbc", nlocal, nall, npbc, timestep)

        if isinstance(ast_node, While):
            cond = self.generate_expression(ast_node.cond)
            self.print(f"while({cond}) {{")
            self.generate_statement(ast_node.block)
            self.print("}")

    def generate_expression(self, ast_node, mem=False, index=None):
        if isinstance(ast_node, Array):
            return ast_node.name()

        if isinstance(ast_node, ArrayAccess):
            index = self.generate_expression(ast_node.index)
            array_name = ast_node.array.name()

            if mem:
                return f"{array_name}[{index}]"

            acc_ref = f"a{ast_node.id()}"
            if ast_node.generated is False:
                tkw = CGen.type2keyword(ast_node.type())
                self.print(f"const {tkw} {acc_ref} = {array_name}[{index}];")
                ast_node.generated = True

            return acc_ref

        if isinstance(ast_node, BinOp):
            if isinstance(ast_node.lhs, BinOp) and ast_node.lhs.kind() == BinOp.Kind_Vector and ast_node.operator() == '[]':
                return self.generate_expression(ast_node.lhs, ast_node.mem, self.generate_expression(ast_node.rhs))

            lhs = self.generate_expression(ast_node.lhs, mem, index)
            rhs = self.generate_expression(ast_node.rhs, index=index)

            if ast_node.operator() == '[]':
                idx = self.generate_expression(ast_node.mapped_vector_index(index)) if ast_node.is_vector_property_access() else rhs
                return f"{lhs}[{idx}]" if ast_node.mem else f"{lhs}_{idx}"

            if ast_node.inlined is True:
                assert ast_node.type() != Type_Vector, "Vector operations cannot be inlined!"
                return f"({lhs} {ast_node.operator()} {rhs})"

            # Some expressions can be defined on-the-fly during transformations, hence they do not have
            # a definition statement in the tree, so we generate them right before use
            if not ast_node.generated:
                self.generate_statement(ast_node.definition())

            if ast_node.kind() == BinOp.Kind_Vector:
                assert index is not None, "Index must be set for vector reference!"
                return f"e{ast_node.id()}[{index}]" if ast_node.mem else f"e{ast_node.id()}_{index}"

            return f"e{ast_node.id()}"

        if isinstance(ast_node, Call):
            params = ", ".join(["ps"] + [str(self.generate_expression(p)) for p in ast_node.parameters()])
            return f"{ast_node.name()}({params})"

        if isinstance(ast_node, Cast):
            tkw = CGen.type2keyword(ast_node.cast_type)
            expr = self.generate_expression(ast_node.expr)
            return f"({tkw})({expr})"

        if isinstance(ast_node, Ceil):
            assert mem is False, "Ceil call is not lvalue!"
            expr = self.generate_expression(ast_node.expr)
            return f"ceil({expr})"

        if isinstance(ast_node, Iter):
            assert mem is False, "Iterator is not lvalue!"
            return f"i{ast_node.id()}"

        if isinstance(ast_node, Lit):
            assert mem is False, "Literal is not lvalue!"

            if ast_node.type() == Type_String:
                return f"\"{ast_node.value}\""

            return ast_node.value

        if isinstance(ast_node, Property):
            return ast_node.name()

        if isinstance(ast_node, PropertyList):
            tid = CGen.temp_id
            list_ref = f"prop_list_{tid}"
            list_def = ", ".join(str(p.id()) for p in ast_node)
            self.print(f"const int {list_ref}[] = {{{list_def}}};")
            CGen.temp_id += 1
            return list_ref

        if isinstance(ast_node, Sizeof):
            assert mem is False, "Sizeof expression is not lvalue!"
            tkw = CGen.type2keyword(ast_node.data_type)
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

    def generate_vtk_writing(self, id, filename, start, end, n, timestep):
        # TODO: Do this in a more elegant way, without hard coded stuff
        header = "# vtk DataFile Version 2.0\n" \
                 "Particle data\n" \
                 "ASCII\n" \
                 "DATASET UNSTRUCTURED_GRID\n"

        filename_var = f"filename{id}"
        filehandle_var = f"vtk{id}"
        self.print(f"char {filename_var}[128];")
        self.print(f"snprintf({filename_var}, sizeof {filename_var}, \"{filename}_%d.vtk\", {timestep});")
        self.print(f"FILE *{filehandle_var} = fopen({filename_var}, \"w\");")
        for line in header.split('\n'):
            if len(line) > 0:
                self.print(f"fwrite(\"{line}\\n\", 1, {len(line) + 1}, {filehandle_var});")

        # Write positions
        self.print(f"fprintf({filehandle_var}, \"POINTS %d double\\n\", {n});")
        self.print(f"for(int i = {start}; i < {end}; i++) {{")
        self.print.add_ind(4)
        self.print(f"fprintf({filehandle_var}, \"%.4f %.4f %.4f\\n\", position[i * 3], position[i * 3 + 1], position[i * 3 + 2]);")
        self.print.add_ind(-4)
        self.print("}")
        self.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        # Write cells
        self.print(f"fprintf({filehandle_var}, \"CELLS %d %d\\n\", {n}, {n} * 2);")
        self.print(f"for(int i = {start}; i < {end}; i++) {{")
        self.print.add_ind(4)
        self.print(f"fprintf({filehandle_var}, \"1 %d\\n\", i - {start});")
        self.print.add_ind(-4)
        self.print("}")
        self.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        # Write cell types
        self.print(f"fprintf({filehandle_var}, \"CELL_TYPES %d\\n\", {n});")
        self.print(f"for(int i = {start}; i < {end}; i++) {{")
        self.print.add_ind(4)
        self.print(f"fwrite(\"1\\n\", 1, 2, {filehandle_var});")
        self.print.add_ind(-4)
        self.print("}")
        self.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        # Write masses
        self.print(f"fprintf({filehandle_var}, \"POINT_DATA %d\\n\", {n});")
        self.print(f"fprintf({filehandle_var}, \"SCALARS mass double\\n\");")
        self.print(f"fprintf({filehandle_var}, \"LOOKUP_TABLE default\\n\");")
        self.print(f"for(int i = {start}; i < {end}; i++) {{")
        self.print.add_ind(4)
        #self.print(f"fprintf({filehandle_var}, \"%4.f\\n\", mass[i]);")
        self.print(f"fprintf({filehandle_var}, \"1.0\\n\");")
        self.print.add_ind(-4)
        self.print("}")
        self.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")
        self.print(f"fclose({filehandle_var});")
