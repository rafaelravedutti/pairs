from graphviz import Digraph
from pairs.ir.arrays import Array
from pairs.ir.bin_op import BinOp, Decl
from pairs.ir.features import Feature, FeatureProperty
from pairs.ir.lit import Lit
from pairs.ir.loops import Iter
from pairs.ir.properties import Property, ContactProperty
from pairs.ir.variables import Var
from pairs.ir.visitor import Visitor


class ASTGraph:
    last_reference = 0

    def __init__(self, ast_node, filename, ref="AST", max_depth=0):
        self.graph = Digraph(ref, filename=filename, node_attr={'color': 'lightblue2', 'style': 'filled'})
        self.graph.attr(size='6,6')
        self.ast = ast_node

    def new_unique_reference():
        ASTGraph.last_reference += 1
        return f"unique_ref{ASTGraph.last_reference}"

    def generate_edges_for_node(ast_node, graph, generated):
        node_ref = ASTGraph.get_node_reference(ast_node)
        if not isinstance(ast_node, Decl) and node_ref not in generated:
            generated.append(node_ref)
            graph.node(node_ref, label=ASTGraph.get_node_label(ast_node))

            for child in ast_node.children():
                if not isinstance(child, Decl):
                    child_ref = ASTGraph.generate_edges_for_node(child, graph, generated)
                    graph.edge(node_ref, child_ref)

        return node_ref

    def render(self):
        ASTGraph.generate_edges_for_node(self.ast, self.graph, [])
        self.graph.render()

    def view(self):
        self.graph.view()

    def get_node_reference(ast_node):
        if isinstance(ast_node, (Array, Property, Var, ContactProperty, FeatureProperty, Iter)):
            return ASTGraph.new_unique_reference()

        return f"n{id(ast_node)}"

    def get_node_label(ast_node):
        if isinstance(ast_node, (Array, Property, Var, ContactProperty, FeatureProperty)):
            return ast_node.name()

        if isinstance(ast_node, BinOp):
            return ast_node.operator().symbol()

        if isinstance(ast_node, Iter):
            return f"Iter({ast_node.id()})"

        if isinstance(ast_node, Lit):
            return str(ast_node.value)

        return type(ast_node).__name__
