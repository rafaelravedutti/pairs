from ast.data_types import Type_Int, Type_Float, Type_Vector
from code_gen.printer import printer


class CGen:
    def type2keyword(type_):
        return (
            'double' if type_ == Type_Float or type_ == Type_Vector
            else 'int' if type_ == Type_Int
            else 'bool'
        )

    def generate_program_preamble():
        printer.print("#include <stdio.h>")
        printer.print("#include <stdlib.h>")
        printer.print("#include <stdbool.h>")
        printer.print("")
        printer.print("int main() {")

    def generate_program_epilogue():
        printer.print("}")

    def generate_block_preamble():
        printer.add_ind(4)

    def generate_block_epilogue():
        printer.add_ind(-4)

    def generate_cast(ctype, expr):
        tkw = CGen.type2keyword(ctype)
        return f"({tkw})({expr})"

    def generate_if(cond):
        printer.print(f"if({cond}) {{")

    def generate_else():
        printer.print("} else {")

    def generate_endif():
        printer.print("}")

    def generate_assignment(dest, src):
        printer.print(f"{dest} = {src};")

    def generate_array_decl(array, a_type, size):
        tkw = CGen.type2keyword(a_type)
        printer.print(f"{tkw} {array}[{size}];")

    def generate_array_access_ref(acc_id, array, index, mem=False):
        if mem:
            return f"{array}[{index}]"

        return f"a{acc_id}"

    def generate_array_access(acc_id, acc_type, array, index):
        ref = CGen.generate_array_access_ref(acc_id, array, index)
        tkw = CGen.type2keyword(acc_type)
        acc = f"const {tkw} {ref} = {array}[{index}];"
        printer.print(acc)

    def generate_malloc(array, a_type, size, decl):
        tkw = CGen.type2keyword(a_type)
        if decl:
            printer.print(f"{tkw} *{array} = ({tkw} *) malloc({size});")
        else:
            printer.print(f"{array} = ({tkw} *) malloc({size});")

    def generate_realloc(array, a_type, size):
        tkw = CGen.type2keyword(a_type)
        printer.print(f"{array} = ({tkw} *) realloc({array}, {size});")

    def generate_sizeof(data_type):
        tkw = CGen.type2keyword(data_type)
        return f"sizeof({tkw})"

    def generate_for_preamble(iter_id, rmin, rmax):
        printer.print(f"for(int {iter_id} = {rmin}; {iter_id} < {rmax}; {iter_id}++) {{")

    def generate_for_epilogue():
        printer.print("}")

    def generate_while_preamble(cond):
        printer.print(f"while({cond}) {{")

    def generate_while_epilogue():
        printer.print("}")

    def generate_expr_ref(expr_id):
        return f"e{expr_id}"

    def generate_expr(expr_id, expr_type, lhs, rhs, op):
        ref = CGen.generate_expr_ref(expr_id)
        tkw = CGen.type2keyword(expr_type)
        printer.print(f"const {tkw} {ref} = {lhs} {op} {rhs};")

    def generate_expr_access(lhs, rhs, mem):
        return f"{lhs}[{rhs}]" if mem else f"{lhs}_{rhs}"

    def generate_vec_expr_ref(expr_id, index, mem):
        return f"e{expr_id}[{index}]" if mem else f"e{expr_id}_{index}"

    def generate_vec_expr(expr_id, index, lhs, rhs, op, mem):
        ref = CGen.generate_vec_expr_ref(expr_id, index, mem)
        printer.print(f"const double {ref} = {lhs} {op} {rhs};")

    def generate_inline_expr(lhs, rhs, op):
        return f"({lhs} {op} {rhs})"

    def generate_var_decl(v_name, v_type, v_value):
        tkw = CGen.type2keyword(v_type)
        printer.print(f"{tkw} {v_name} = {v_value};")

    def generate_sqrt(expr):
        return f"sqrt({expr})"

    def generate_select(cond, expr_if, expr_else):
        return f"({cond}) ? ({expr_if}) : ({expr_else})"

    def generate_print(string):
        printer.print(f"fprintf(stdout, \"{string}\\n\"); fflush(stdout);")

    def generate_vtk_writing(id, filename, start, n, timestep):
        # TODO: Do this in a more elegant way, without hard coded stuff
        header = "# vtk DataFile Version 2.0\n" \
                 "Particle data\n" \
                 "ASCII\n" \
                 "DATASET UNSTRUCTURED_GRID\n"

        filename_var = f"filename{id}"
        filehandle_var = f"vtk{id}"
        printer.print(f"char {filename_var}[128];")
        printer.print(f"snprintf({filename_var}, sizeof {filename_var}, \"{filename}_%d.vtk\", {timestep.generate()});")
        printer.print(f"FILE *{filehandle_var} = fopen({filename_var}, \"w\");")
        for line in header.split('\n'):
            if len(line) > 0:
                printer.print(f"fwrite(\"{line}\\n\", 1, {len(line) + 1}, {filehandle_var});")

        # Write positions
        printer.print(f"fprintf({filehandle_var}, \"POINTS %d double\\n\", {n.generate()});")
        CGen.generate_for_preamble("i", start, n.generate())
        printer.add_ind(4)
        printer.print(f"fprintf({filehandle_var}, \"%.4f %.4f %.4f\\n\", position[i * 3], position[i * 3 + 1], position[i * 3 + 2]);")
        printer.add_ind(-4)
        CGen.generate_for_epilogue()
        printer.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        # Write cells
        printer.print(f"fprintf({filehandle_var}, \"CELLS %d %d\\n\", {n.generate()}, {n.generate()} * 2);")
        CGen.generate_for_preamble("i", start, n.generate())
        printer.add_ind(4)
        printer.print(f"fprintf({filehandle_var}, \"1 %d\\n\", i);")
        printer.add_ind(-4)
        CGen.generate_for_epilogue()
        printer.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        # Write cell types
        printer.print(f"fprintf({filehandle_var}, \"CELL TYPES %d\\n\", {n.generate()});")
        CGen.generate_for_preamble("i", start, n.generate())
        printer.add_ind(4)
        printer.print(f"fwrite(\"1\\n\", 1, 2, {filehandle_var});")
        printer.add_ind(-4)
        CGen.generate_for_epilogue()
        printer.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        # Write masses
        printer.print(f"fprintf({filehandle_var}, \"POINT_DATA %d\\n\", {n.generate()});")
        printer.print(f"fprintf({filehandle_var}, \"SCALARS mass double\\n\");")
        printer.print(f"fprintf({filehandle_var}, \"LOOKUP_TABLE default\\n\");")
        CGen.generate_for_preamble("i", start, n.generate())
        printer.add_ind(4)
        printer.print(f"fprintf({filehandle_var}, \"%4.f\\n\", mass[i]);")
        printer.add_ind(-4)
        CGen.generate_for_epilogue()
        printer.print(f"fwrite(\"\\n\\n\", 1, 2, {filehandle_var});")

        printer.print(f"fclose({filehandle_var});")
