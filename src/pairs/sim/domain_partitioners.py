class DomainPartitioners:
    Invalid = -1
    Regular = 0
    RegularXY = 1
    BlockForest = 2

    def c_keyword(layout):
        return "RegularPartitioning"        if layout == DomainPartitioners.Regular else \
               "RegularXYPartitioning"      if layout == DomainPartitioners.RegularXY else \
               "BlockForestPartitioning"    if layout == DomainPartitioners.BlockForest else \
               "Invalid"
