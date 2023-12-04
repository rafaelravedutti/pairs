class DomainPartitioners:
    Invalid = -1
    Regular = 0
    RegularXY = 1
    BoxList = 2

    def c_keyword(layout):
        return "Regular"    if layout == DomainPartitioners.Regular else \
               "RegularXY"  if layout == DomainPartitioners.RegularXY else \
               "BoxList"    if layout == DomainPartitioners.BoxList else \
               "Invalid"
