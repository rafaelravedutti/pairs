from ast.arrays import Array
from ast.bin_op import BinOp, BinOpDef
from ast.lit import Lit
from ast.loops import Iter
from ast.properties import Property
from ast.variables import Var
from ast.visitor import Visitor
from graphviz import Digraph


class ASTGraph:
    def __init__(self, ast_node, filename, ref="AST", max_depth=0):
        self.graph = Digraph(ref, filename=filename, node_attr={'color': 'lightblue2', 'style': 'filled'})
        self.graph.attr(size='6,6')
        self.visitor = Visitor(ast_node, max_depth=max_depth)

    def generate_and_view(self):
        def generate_edges_for_node(ast_node, graph, generated):
            node_id = id(ast_node)
            if not isinstance(ast_node, BinOpDef) and node_id not in generated:
                node_ref = f"n{id(ast_node)}"
                generated.append(node_id)
                graph.node(node_ref, label=ASTGraph.get_node_label(ast_node))
                for child in ast_node.children():
                    if not isinstance(child, BinOpDef):
                        child_ref = f"n{id(child)}"
                        graph.node(child_ref, label=ASTGraph.get_node_label(child))
                        graph.edge(node_ref, child_ref)

        generated = []
        for node in self.visitor:
            generate_edges_for_node(node, self.graph, generated)

        self.graph.view()

    def get_node_label(ast_node):
        if isinstance(ast_node, (Array, Property, Var)):
            return ast_node.name()

        if isinstance(ast_node, BinOp):
            return ast_node.operator()

        if isinstance(ast_node, Iter):
            return f"Iter({ast_node.id()})"

        if isinstance(ast_node, Lit):
            return str(ast_node.value)

        return type(ast_node).__name__
