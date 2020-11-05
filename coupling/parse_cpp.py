import clang.cindex


def find_typerefs(node):
    print(node)
    print(node.kind)

    for c in node.get_children():
        print(c)

    #for t in node.get_tokens():
    #    print(t.spelling, ' ... ', t.kind, ' ... ', t.cursor.kind)

    if node.kind.is_reference():
        ref_node = clang.cindex.Cursor_ref(node)
        line = node.location.line
        column = node.location.column
        print(f"Found {ref_node.spelling} [line={line}, col={column}]")

    # Recurse for children of this node
    for c in node.get_children():
        find_typerefs(c)

walberla_path = "/home/rzlin/az16ahoq/repositories/walberla"
walberla_src = f"{walberla_path}/src"
walberla_build_src = f"{walberla_path}/build/src"
clang_include_path = "/software/anydsl/llvm_build/lib/clang/7.0.1/include"
mpi_include_path = "/software/openmpi/4.0.0-llvm/include"
mesapd_path = f"{walberla_src}/mesa_pd"
filename = "SpringDashpot.hpp"
typename = "SpringDashpot"
index = clang.cindex.Index.create()
tu = index.parse(
    f"{mesapd_path}/kernel/{filename}",
    args=["-Wall", f"-I{walberla_src}", f"-I{walberla_build_src}", f"-I{clang_include_path}", f"-I{mpi_include_path}"])

diagnostics = tu.diagnostics
for i in range(0, len(diagnostics)):
    print(diagnostics[i])

print(f"Translation unit: {tu.spelling}")
find_typerefs(tu.cursor)
