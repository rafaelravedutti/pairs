class Timers:
    Invalid = -1
    All = 0
    Communication = 1
    DeviceTransfers = 2
    Offset = 3

    def name(timer):
        return "all"            if timer == Timers.All else             \
               "mpi"            if timer == Timers.Communication else   \
               "transfers"      if timer == Timers.DeviceTransfers else \
               "invalid"
