class Printer:
    def __init__(self, output):
        self.output = output
        self.stream = None
        self.indent = 0

    def add_indent(self, offset):
        self.indent += offset

    def start(self):
        self.stream = open(self.output, 'w')

    def end(self):
        self.stream.close()
        self.stream = None

    def __call__(self, text):
        assert self.stream is not None, "Invalid stream!"
        self.stream.write(self.indent * ' ' + text + '\n')
