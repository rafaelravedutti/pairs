class Timers:
    Invalid = -1
    Communication = 0
    DeviceTransfers = 1
    Offset = 2

    def name(timer):
        return "MARKERS::MPI"                if timer == Timers.Communication else   \
               "MARKERS::DEVICE_TRANSFERS"   if timer == Timers.DeviceTransfers else \
               "INVALID"
