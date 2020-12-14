from ast.lit import as_lit_ast

class VTKWrite:
    vtk_id = 0

    def __init__(self, sim, filename, timestep):
        self.sim = sim
        self.filename = filename
        self.timestep = as_lit_ast(sim, timestep)

    def children(self):
        return []

    def generate(self):
        self.sim.code_gen.generate_vtk_writing(
            VTKWrite.vtk_id * 2, f"{self.filename}_local",
            as_lit_ast(self.sim, 0), self.sim.nlocal, self.timestep)

        self.sim.code_gen.generate_vtk_writing(
            VTKWrite.vtk_id * 2 + 1, f"{self.filename}_pbc",
            self.sim.nlocal, self.sim.pbc.npbc, self.timestep)

        VTKWrite.vtk_id += 1

    def transform(self, fn):
        return fn(self)
