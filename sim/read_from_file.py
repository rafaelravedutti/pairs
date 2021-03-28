from ir.data_types import Type_Int, Type_Float, Type_Vector
from ir.loops import For
from sim.grid import Grid

class ReadFromFile():
    def __init__(self, sim, filename, props):
        self.sim = sim
        self.filename = filename
        self.props = props
        self.grid = None

    def lower(self):
        ndims = None
        nlocal = self.sim.nlocal
        line_number = 0

        self.sim.clear_block()
        with open(self.filename, "r") as fp:
            for line in fp:
                current_data = line.split(',')
                if line_number == 0:
                    assert len(current_data) % 2 == 0, "Number of dimensions in header must be even!"
                    ndims = int(len(current_data) / 2)
                    config = [[float(current_data[d * 2]), float(current_data[d * 2 + 1])] for d in range(0, ndims)]
                    self.grid = Grid(self.sim, config)
                else:
                    i = 0
                    for p in self.props:
                        if p.type() == Type_Vector:
                            for d in range(0, ndims):
                                p[nlocal][d].set(float(current_data[i + d]))

                            i += ndims

                        else:
                            if p.type() == Type_Int:
                                p[nlocal].set(int(current_data[i]))
                            elif p.type() == Type_Float:
                                p[nlocal].set(float(current_data[i]))
                            else:
                                raise Exception(f"Invalid property type at line {line_number}!")

                            i += 1

                    nlocal.set(nlocal + 1)

                line_number += 1

        return self.sim.block
