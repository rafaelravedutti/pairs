class Layouts:
    Invalid = -1
    AoS = 0
    SoA = 1

    def c_keyword(layout):
        return "AoS" if layout == Layouts.AoS else \
               "SoA" if layout == Layouts.SoA else \
               "Invalid"
