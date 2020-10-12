class Scope:
    def __init__(self, block):
        self.block = block

    def __lt__(self, other):
        return self.block.level < other.block.level

    def __le__(self, other):
        return self.block.level <= other.block.level

    def __gt__(self, other):
        return self.block.level > other.block.level

    def __ge__(self, other):
        return self.block.level >= other.block.level

    def level(self):
        return self.block.level
