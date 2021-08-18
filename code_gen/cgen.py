from ir.assign import Assign
from ir.arrays import Array, ArrayAccess, ArrayDecl
from ir.block import Block
from ir.branches import Branch
from ir.cast import Cast
from ir.bin_op import BinOp, Decl, VectorAccess
from ir.data_types import Type_Int, Type_Float, Type_String, Type_Vector
from ir.functions import Call
from ir.layouts import Layout_AoS, Layout_SoA, Layout_Invalid
from ir.lit import Lit
from ir.loops import For, Iter, ParticleFor, While
from ir.math import Ceil, Sqrt
from ir.memory import Malloc, Realloc
from ir.properties import Property, PropertyAccess, PropertyList, RegisterProperty, UpdateProperty
from ir.select import Select
from ir.sizeof import Sizeof
from ir.utils import Print
from ir.variables import Var, VarDecl
from sim.timestep import Timestep
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
        self.print("#include \"runtime/vtk.hpp\"")
        self.print("")
        self.print("using namespace pairs;")
        self.print("")
        self.print("int main() {")
        self.print("    PairsSim *ps = new PairsSim();")
        self.generate_statement(ast_node)
        self.print("}")
        self.print.end()

    def generate_statement(self, ast_node, bypass_checking=False):
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

        # TODO: Why there are Decls for other types?
        if isinstance(ast_node, Decl):
            if isinstance(ast_node.elem, BinOp):
                bin_op = ast_node.elem
                if not bypass_checking and (not isinstance(bin_op, BinOp) or not ast_node.used):
                    return None

                if bin_op.inlined is False and bin_op.generated is False:
                    if bin_op.is_vector_kind():
                        for i in bin_op.indexes():
                            lhs = self.generate_expression(bin_op.lhs, bin_op.mem, index=i)
                            rhs = self.generate_expression(bin_op.rhs, index=i)
                            self.print(f"const double e{bin_op.id()}_{i} = {lhs} {bin_op.operator()} {rhs};")

                    else:
                        lhs = self.generate_expression(bin_op.lhs, bin_op.mem)
                        rhs = self.generate_expression(bin_op.rhs)
                        tkw = CGen.type2keyword(bin_op.type())
                        self.print(f"const {tkw} e{bin_op.id()} = {lhs} {bin_op.operator()} {rhs};")

                ast_node.elem.generated = True

            if isinstance(ast_node.elem, PropertyAccess):
                prop_access = ast_node.elem
                prop_name = prop_access.prop.name()
                acc_ref = f"p{prop_access.id()}"

                if prop_access.is_vector_kind():
                    for i in prop_access.indexes():
                        i_expr = self.generate_expression(prop_access.get_index_expression(i))
                        self.print(f"const double {acc_ref}_{i} = {prop_name}[{i_expr}];")

                else:
                    tkw = CGen.type2keyword(prop_access.type())
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

        if isinstance(ast_node, UpdateProperty):
            p = ast_node.property()

            if p.type() != Type_Vector or p.layout() == Layout_Invalid:
                self.print(f"ps->updateProperty({p.id()}, {p.name()});")
            else:
                sizes = ", ".join([str(self.generate_expression(size)) for size in ast_node.sizes()])
                self.print(f"ps->updateProperty({p.id()}, {p.name()}, {sizes});")

        if isinstance(ast_node, VarDecl):
            tkw = CGen.type2keyword(ast_node.var.type())
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

            if mem:
                return f"{array_name}[{acc_index}]"

            acc_ref = f"a{ast_node.id()}"
            if not ast_node.generated:
                tkw = CGen.type2keyword(ast_node.type())
                self.print(f"const {tkw} {acc_ref} = {array_name}[{acc_index}];")
                ast_node.generated = True

            return acc_ref

        if isinstance(ast_node, BinOp):
            lhs = self.generate_expression(ast_node.lhs, mem, index)
            rhs = self.generate_expression(ast_node.rhs, index=index)

            if ast_node.inlined is True:
                assert ast_node.type() != Type_Vector, "Vector operations cannot be inlined!"
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

        if isinstance(ast_node, PropertyAccess):
            assert not ast_node.is_vector_kind() or index is not None, "Index must be set for vector property access!"
            prop_name = ast_node.prop.name()

            if mem:
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

        if isinstance(ast_node, VectorAccess):
            return self.generate_expression(ast_node.expr, mem, self.generate_expression(ast_node.index))
