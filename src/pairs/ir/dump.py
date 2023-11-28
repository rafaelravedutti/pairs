class Dump:
    currentIndent = 0

    def __init__(self):
        pass

    def get_method(method_name):
        method = getattr(Dump, method_name, None)
        return method if callable(method) else None

    def dump(ast_node):
        if ast_node is None:
            ast_node = self.ast

        method = Dump.get_method(f"dump_{type(ast_node).__name__}")
        if method is not None:
            return method(ast_node)

        method_unknown = Dump.get_method("dump_Unknown")
        if method_unknown is not None:
            return method_unknown(ast_node)

    def dump_ScalarOp(ast_node):
        print(' ' * Dump.currentIndent, end="")
        print(f"ScalarOp<{ast_node.op.symbol()}")

        Dump.currentIndent += 2
        print(' ' * Dump.currentIndent, end="")
        Dump.dump(ast_node.lhs)
        print(' ' * Dump.currentIndent, end="")
        Dump.dump(ast_node.rhs)

        Dump.currentIndent -= 2
        print(' ' * Dump.currentIndent, end="")
        print(">")

    def dump_Unknown(ast_node):
        print(ast_node, end="")
