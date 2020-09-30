from ast.data_types import Type_Int, Type_Float
from code_gen.printer import printer


class CGen:
    def generate_program_preamble():
        printer.print("int main() {")

    def generate_program_epilogue():
        printer.print("}")

    def generate_block_preamble():
        printer.add_ind(4)

    def generate_block_epilogue():
        printer.add_ind(-4)

    def generate_cast(ctype, expr):
        t = ('double' if ctype == Type_Float
             else 'int' if ctype == Type_Int else 'bool')

        return f"({t})({expr})"

    def generate_if(cond):
        printer.print(f"if({cond}) {{")

    def generate_else():
        printer.print("} else {")

    def generate_endif():
        printer.print("}")

    def generate_assignment(dest, src):
        printer.print(f"{dest} = {src};")

    def generate_array_decl(array, a_type, sizes):
        t = ('double' if a_type == Type_Float
             else 'int' if a_type == Type_Int else 'bool')

        gen_str = f"{t} {array}"
        for s in sizes:
            gen_str += f"[{s}]"

        printer.print(gen_str)

    def generate_array_access(array, index):
        return f"{array}[{index}]"

    def generate_realloc(array, size):
        printer.print(f"{array} = realloc({size});")

    def generate_for_preamble(iter_id, rmin, rmax):
        printer.print(
            f"for(int {iter_id} = {rmin}; {iter_id} < {rmax}; {iter_id}++) {{")

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
        t = ('double' if expr_type == Type_Float
             else 'int' if expr_type == Type_Int else 'bool')

        printer.print(f"const {t} {ref} = {lhs} {op} {rhs};")

    def generate_expr_access(lhs, rhs, mem):
        return f"{lhs}[{rhs}]" if mem else f"{lhs}_{rhs}"

    def generate_vec_expr_ref(expr_id, index, mem):
        return (f"e{expr_id}[{index}]" if mem else f"e{expr_id}_{index}")

    def generate_vec_expr(expr_id, index, lhs, rhs, op, mem):
        ref = CGen.generate_vec_expr_ref(expr_id, index, mem)
        printer.print(f"const double {ref} = {lhs} {op} {rhs};")

    def generate_inline_expr(lhs, rhs, op):
        return f"{lhs} {op} {rhs}"