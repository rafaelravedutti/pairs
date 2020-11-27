from coupling.parse_cpp import parse_walberla_file
from coupling.parse_cpp import get_class_method, print_tree

filename = "mesa_pd/kernel/SpringDashpot.hpp"
translation_unit = parse_walberla_file(filename)

# subtree = get_subtree(tu.cursor, "walberla::mesa_pd::kernel")
# print_tree(subtree)

kernel = get_class_method(
        translation_unit.cursor,
        "walberla::mesa_pd::kernel::SpringDashpot",
        "operator()")
print_tree(kernel)
