class Contexts:
    Invalid = -1
    Host = 0
    Device = 1

    def as_string(t):
        return (
            'host' if t == Types.Host
            else 'device' if t == Types.Device
            else '<invalid context>'
        )

    def is_host(ctx):
        return ctx == Contexts.Host

    def is_target(ctx):
        return ctx == Contexts.Device
