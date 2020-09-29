class Printer:
    def __init__(self):
        self.indent = 0

    def add_ind(self, offset):
        self.indent += offset

    def print(self, text):
        print(self.indent * ' ' + text)


printer = Printer()
