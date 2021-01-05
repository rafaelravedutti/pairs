from ast.lit import as_lit_ast

class VTKWrite:
    vtk_id = 0

    def __init__(self, sim, filename, timestep):
        self.sim = sim
        self.vtk_id = VTKWrite.vtk_id
        self.filename = filename
        self.timestep = as_lit_ast(sim, timestep)
        VTKWrite.vtk_id += 1

    def children(self):
        return []

    def transform(self, fn):
        return fn(self)
