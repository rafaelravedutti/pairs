import clang.cindex
from clang.cindex import CursorKind as kind


def print_tree(node, indent=0):
    spaces = ' ' * indent
    line = node.location.line
    column = node.location.column
    print(f"{spaces}{line}:{column}> {node.spelling} ({node.kind})")

    for c in node.get_children():
        print_tree(c, indent + 2)


def get_subtree(node, ref):
    splitted_ref = ref.split("::", 1)
    if len(splitted_ref) == 2:
        look_for, remaining = splitted_ref
    else:
        look_for = splitted_ref[0]
        remaining = None

    for c in node.get_children():
        cond_namespace = remaining is not None and \
                         (c.kind == kind.NAMESPACE or
                          c.kind == kind.CLASS_DECL) and \
                         c.spelling == look_for

        cond_func = remaining is None and \
                    (c.kind == kind.FUNCTION_TEMPLATE or \
                     c.kind == kind.CXX_METHOD or \
                     c.kind == kind.NAMESPACE or \
                     c.kind == kind.CLASS_DECL) and \
                    c.spelling == look_for

        if cond_namespace or cond_func:
            if remaining is None:
                print_tree(c)
            else:
                get_subtree(c, remaining)


walberla_path = "/home/rzlin/az16ahoq/repositories/walberla"
walberla_src = f"{walberla_path}/src"
walberla_build_src = f"{walberla_path}/build/src"
clang_include_path = "/software/anydsl/llvm_build/lib/clang/7.0.1/include"
mpi_include_path = "/software/openmpi/4.0.0-llvm/include"
mesapd_path = f"{walberla_src}/mesa_pd"
filename = "SpringDashpot.hpp"
index = clang.cindex.Index.create()
tu = index.parse(
    f"{mesapd_path}/kernel/{filename}",
    args = [
        "-Wall",
        f"-I{walberla_src}",
        f"-I{walberla_build_src}",
        f"-I{clang_include_path}",
        f"-I{mpi_include_path}"
    ]
)

diagnostics = tu.diagnostics
for i in range(0, len(diagnostics)):
    print(diagnostics[i])

print(f"Translation unit: {tu.spelling}")
# print_kernel(tu.cursor, "walberla::mesa_pd::kernel::SpringDashpot")
get_subtree(tu.cursor, "walberla::mesa_pd::kernel")
