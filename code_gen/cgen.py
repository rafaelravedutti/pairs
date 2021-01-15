from ast.assign import Assign
from ast.arrays import ArrayAccess, ArrayDecl
from ast.block import Block
from ast.branches import Branch
from ast.cast import Cast
from ast.bin_op import BinOp, BinOpDef
from ast.data_types import Type_Int, Type_Float, Type_Vector
from ast.lit import Lit
from ast.loops import For, Iter, ParticleFor, While
from ast.math import Sqrt
from ast.memory import Malloc, Realloc
from ast.properties import Property
from ast.select import Select
from ast.sizeof import Sizeof
from ast.utils import Print
from ast.variables import Var, VarDecl
from sim.timestep import Timestep
from sim.vtk import VTKWrite
from code_gen.printer import printer


class CGen:
    def type2keyword(type_):
        return (
            'double' if type_ == Type_Float or type_ == Type_Vector
            else 'int' if type_ == Type_Int
            else 'bool'
        )

    def generate_program(sim, ast_node):
        printer.print("#include <stdio.h>")
        printer.print("#include <stdlib.h>")
        printer.print("#include <stdbool.h>")
        printer.print("")
        printer.print("int main() {")
        CGen.generate_statement(sim, ast_node)
        printer.print("}")

    def generate_statement(sim, ast_node):
        if isinstance(ast_node, ArrayDecl):
            tkw = CGen.type2keyword(ast_node.array.type())
            size = CGen.generate_expression(sim, BinOp.inline(ast_node.array.alloc_size()))
            printer.print(f"{tkw} {ast_node.array.name()}[{size}];")

        if isinstance(ast_node, Assign):
            for assign_dest, assign_src in ast_node.assignments:
                dest = CGen.generate_expression(sim, assign_dest, mem=True)
                src = CGen.generate_expression(sim, assign_src)
                printer.print(f"{dest} = {src};")

        if isinstance(ast_node, Block):
            printer.add_ind(4)

            for stmt in ast_node.statements():
                CGen.generate_statement(sim, stmt)

            printer.add_ind(-4)

        if isinstance(ast_node, BinOpDef):
            bin_op = ast_node.bin_op

            if not isinstance(bin_op, BinOp):
                return None

            if bin_op.inlined is False and bin_op.operator() != '[]' and bin_op.generated is False:
                if bin_op.kind() == BinOp.Kind_Scalar:
                    lhs = CGen.generate_expression(sim, bin_op.lhs, bin_op.mem)
                    rhs = CGen.generate_expression(sim, bin_op.rhs)
                    tkw = CGen.type2keyword(bin_op.type())
                    printer.print(f"const {tkw} e{bin_op.id()} = {lhs} {bin_op.operator()} {rhs};")

                elif bin_op.kind() == BinOp.Kind_Vector:
                    for i in bin_op.vector_indexes():
                        lhs = CGen.generate_expression(sim, bin_op.lhs, bin_op.mem, index=i)
                        rhs = CGen.generate_expression(sim, bin_op.rhs, index=i)
                        printer.print(f"const double e{bin_op.id()}_{i} = {lhs} {bin_op.operator()} {rhs};")

                else:
                    raise Exception("Invalid BinOp kind!")

                bin_op.generated = True

        if isinstance(ast_node, Branch):
            cond = CGen.generate_expression(sim, ast_node.cond)
            printer.print(f"if({cond}) {{")
            CGen.generate_statement(sim, ast_node.block_if)

            if ast_node.block_else is not None:
                printer.print("} else {")
                CGen.generate_statement(sim, ast_node.block_else)

            printer.print("}") 

        if isinstance(ast_node, For):
            iterator = CGen.generate_expression(sim, ast_node.iterator)
            lower_range = None
            upper_range = None

            if isinstance(ast_node, ParticleFor):
                n = sim.nlocal if ast_node.local_only else sim.nlocal + sim.pbc.npbc
                lower_range = 0
                upper_range = CGen.generate_expression(sim, n)

            else:
                lower_range = CGen.generate_expression(sim, ast_node.min)
                upper_range = CGen.generate_expression(sim, ast_node.max)

            printer.print(f"for(int {iterator} = {lower_range}; {iterator} < {upper_range}; {iterator}++) {{")
            CGen.generate_statement(sim, ast_node.block)
            printer.print("}")


        if isinstance(ast_node, Malloc):
            tkw = CGen.type2keyword(ast_node.array.type())
            size = CGen.generate_expression(sim, ast_node.size)
            array_name = ast_node.array.name()

            if ast_node.decl:
                printer.print(f"{tkw} *{array_name} = ({tkw} *) malloc({size});")
            else:
                printer.print(f"{array_name} = ({tkw} *) malloc({size});")

        if isinstance(ast_node, Print):
            printer.print(f"fprintf(stdout, \"{ast_node.string}\\n\");")
            printer.print(f"fflush(stdout);")

        if isinstance(ast_node, Realloc):
            tkw = CGen.type2keyword(ast_node.array.type())
            size = CGen.generate_expression(sim, ast_node.size)
            array_name = ast_node.array.name()
            printer.print(f"{array_name} = ({tkw} *) realloc({array_name}, {size});")

        if isinstance(ast_node, Timestep):
            CGen.generate_statement(sim, ast_node.block)

        if isinstance(ast_node, VarDecl):
            tkw = CGen.type2keyword(ast_node.var.type())
            printer.print(f"{tkw} {ast_node.var.name()} = {ast_node.var.init_value()};")

        if isinstance(ast_node, VTKWrite):
            nlocal = CGen.generate_expression(sim, sim.nlocal)
            npbc = CGen.generate_expression(sim, sim.pbc.npbc)
            nall = CGen.generate_expression(sim, sim.nlocal + sim.pbc.npbc)
            timestep = CGen.generate_expression(sim, ast_node.timestep)
            CGen.generate_vtk_writing(ast_node.vtk_id * 2, f"{ast_node.filename}_local", 0, nlocal, nlocal, timestep)
            CGen.generate_vtk_writing(ast_node.vtk_id * 2 + 1, f"{ast_node.filename}_pbc", nlocal, nall, npbc, timestep)

        if isinstance(ast_node, While):
            cond = CGen.generate_expression(sim, ast_node.cond)
            printer.print(f"while({cond}) {{")
            CGen.generate_statement(sim, ast_node.block)
            printer.print("}")

    def generate_expression(sim, ast_node, mem=False, index=None):
        if isinstance(ast_node, ArrayAccess):
            index = CGen.generate_expression(sim, ast_node.index)
            array_name = ast_node.array.name()

            if mem:
                return f"{array_name}[{index}]"

            acc_ref = f"a{ast_node.id()}"
            if ast_node.generated is False:
                tkw = CGen.type2keyword(ast_node.type())
                printer.print(f"const {tkw} {acc_ref} = {array_name}[{index}];")
                ast_node.generated = True

            return acc_ref

        if isinstance(ast_node, BinOp):
            if isinstance(ast_node.lhs, BinOp) and ast_node.lhs.kind() == BinOp.Kind_Vector and ast_node.operator() == '[]':
                return CGen.generate_expression(sim, ast_node.lhs, ast_node.mem, CGen.generate_expression(sim, ast_node.rhs))

            lhs = CGen.generate_expression(sim, ast_node.lhs, mem, index)
            rhs = CGen.generate_expression(sim, ast_node.rhs, index=index)

            if ast_node.operator() == '[]':
                idx = CGen.generate_expression(sim, ast_node.mapped_vector_index(index)) if ast_node.is_vector_property_access() else rhs
                return f"{lhs}[{idx}]" if ast_node.mem else f"{lhs}_{idx}"

            if ast_node.inlined is True:
                assert ast_node.type() != Type_Vector, "Vector operations cannot be inlined!"
                return f"({lhs} {ast_node.operator()} {rhs})"

            # Some expressions can be defined on-the-fly during transformations, hence they do not have
            # a definition statement in the tree, so we generate them right before use
            if not ast_node.generated:
                CGen.generate_statement(sim, ast_node.definition())

            if ast_node.kind() == BinOp.Kind_Vector:
                assert index is not None, "Index must be set for vector reference!"
                return f"e{ast_node.id()}[{index}]" if ast_node.mem else f"e{ast_node.id()}_{index}"

            return f"e{ast_node.id()}"

        if isinstance(ast_node, Cast):
            tkw = CGen.type2keyword(ast_node.cast_type)
            expr = CGen.generate_expression(sim, ast_node.expr)
            return f"({tkw})({expr})"

        if isinstance(ast_node, Iter):
            assert mem is False, "Iterator is not lvalue!"
            return f"i{ast_node.id()}"

        if isinstance(ast_node, Lit):
            assert mem is False, "Literal is not lvalue!"
            return ast_node.value

        if isinstance(ast_node, Property):
            return ast_node.name()

        if isinstance(ast_node, Sizeof):
            assert mem is False, "Sizeof expression is not lvalue!"
            tkw = CGen.type2keyword(ast_node.data_type)
            return f"sizeof({tkw})"

        if isinstance(ast_node, Sqrt):
            assert mem is False, "Square root call is not lvalue!"
            expr = CGen.generate_expression(sim, ast_node.expr)
            return f"sqrt({expr})"

        if isinstance(ast_node, Select):
            assert mem is False, "Select expression is not lvalue!"
            cond = CGen.generate_expression(sim, ast_node.cond)
            expr_if = CGen.generate_expression(sim, ast_node.expr_if)
            expr_else = CGen.generate_expression(sim, ast_node.expr_else)
            return f"({cond}) ? ({expr_if}) : ({expr_else})"

        if isinstance(ast_node, Var):
            return ast_node.name()

    def generate_vtk_writing(id, filename, start, end, n, timestep):
        # TODO: Do this in a more elegant way, without hard coded stuff
        header = "# vtk DataFile Version 2.0\n" \
                 "Particle data\n" \
                 "ASCII\n" \
                 "DATASET UNSTRUCTURED_GRID\n"

        filename_var = f"filename{id}"
        filehandle_var = f"vtk{id}"
        printer.print(f"char {filename_var}[128];")
        printer.print(f"snprintf({filename_var}, sizeof {filename_var}, \"{filename}_%d.vtk\", {timestep});")
        printer.print(f"FILE *{filehandle_var} = fopen({filename_var}, \"w\");")
        for line in header.split('\n'):
            if len(line) > 0:
                printer.print(f"fwrite(\"{line}\\n\", 1, {len(line) + 1}, {filehandle_var});")

        # Write positions
        printer.print(f"fprintf({filehandle_var}, \"POINTS %d double\\n\", {n});")
        printer.print(f"for(int i = {start}; i < {end}; i++) {{")
        printer.add_ind(4)
        printer.print(f"fprintf({filehandle_var}, \"%.4f %.4f %.4f\\n\", position[i * 3], position[i * 3 + 1], position[i * 3 + 2]);")
        printer.add_ind(-4)
        printer.print("}")
        printer.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        # Write cells
        printer.print(f"fprintf({filehandle_var}, \"CELLS %d %d\\n\", {n}, {n} * 2);")
        printer.print(f"for(int i = {start}; i < {end}; i++) {{")
        printer.add_ind(4)
        printer.print(f"fprintf({filehandle_var}, \"1 %d\\n\", i - {start});")
        printer.add_ind(-4)
        printer.print("}")
        printer.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        # Write cell types
        printer.print(f"fprintf({filehandle_var}, \"CELL_TYPES %d\\n\", {n});")
        printer.print(f"for(int i = {start}; i < {end}; i++) {{")
        printer.add_ind(4)
        printer.print(f"fwrite(\"1\\n\", 1, 2, {filehandle_var});")
        printer.add_ind(-4)
        printer.print("}")
        printer.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        # Write masses
        printer.print(f"fprintf({filehandle_var}, \"POINT_DATA %d\\n\", {n});")
        printer.print(f"fprintf({filehandle_var}, \"SCALARS mass double\\n\");")
        printer.print(f"fprintf({filehandle_var}, \"LOOKUP_TABLE default\\n\");")
        printer.print(f"for(int i = {start}; i < {end}; i++) {{")
        printer.add_ind(4)
        #printer.print(f"fprintf({filehandle_var}, \"%4.f\\n\", mass[i]);")
        printer.print(f"fprintf({filehandle_var}, \"1.0\\n\");")
        printer.add_ind(-4)
        printer.print("}")
        printer.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        printer.print(f"fclose({filehandle_var});")
