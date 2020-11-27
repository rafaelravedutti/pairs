class Grid:
    def __init__(self, sim, config):
        self.sim = sim
        self.ndims = len(config)
        self.config = config

    def range(self, dim):
        return self.config[dim]

    def min(self, dim):
        return self.config[dim][0]

    def max(self, dim):
        return self.config[dim][1]

    def xmin(self, dim):
        return self.min(0)

    def xmax(self, dim):
        return self.max(0)

    def ymin(self, dim):
        return self.min(1)

    def ymax(self, dim):
        return self.max(1)

    def zmin(self, dim):
        return self.min(2)

    def zmax(self, dim):
        return self.max(2)

    def length(self, dim):
        return self.max(dim) - self.min(dim)

    def get_ncells_for_spacing(self, spacing):
        return [int(self.max(d) - self.min(d) / spacing)
                for d in range(0, self.ndims)]


class Grid2D(Grid):
    def __init__(self, sim, xmin, xmax, ymin, ymax):
        config = [[xmin, xmax], [ymin, ymax]]
        super().__init__(sim, config)


class Grid3D(Grid):
    def __init__(self, sim, xmin, xmax, ymin, ymax, zmin, zmax):
        config = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        super().__init__(sim, config)
