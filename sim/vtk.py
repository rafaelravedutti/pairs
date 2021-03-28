from ir.lit import as_lit_ast
from ir.ast_node import ASTNode


class VTKWrite(ASTNode):
    vtk_id = 0

    def __init__(self, sim, filename, timestep):
        super().__init__(sim)
        self.vtk_id = VTKWrite.vtk_id
        self.filename = filename
        self.timestep = as_lit_ast(sim, timestep)
        VTKWrite.vtk_id += 1
