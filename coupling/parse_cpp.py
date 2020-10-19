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

path = "/home/rzlin/az16ahoq/repositories/walberla/src/mesa_pd/kernel"
file = "SpringDashpot.h"
typename = "SpringDashpot"
index = clang.cindex.Index.create()
tu = index.parse(f"{path}/{file}")

diagnostics = tu.diagnostics
for i in range(0, len(diagnostics)):
    print(diagnostics[i])

print(f"Translation unit: {tu.spelling}")
find_typerefs(tu.cursor)
