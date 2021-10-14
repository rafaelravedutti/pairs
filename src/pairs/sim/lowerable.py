class Lowerable:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        raise Exception("Error: lower() method must be implemented for Lowerable inherited classes!")
