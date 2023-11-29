class Actions:
    Invalid = -1
    NoAction = 0
    ReadAfterWrite = 1
    WriteAfterRead = 2
    ReadOnly = 3
    WriteOnly = 4
    Ignore = 5

    def update_rule(action, new_op):
        if action == Actions.NoAction:
            return Actions.ReadOnly if new_op == 'r' else Actions.WriteOnly

        if action == Actions.ReadOnly and new_op == 'w':
            return Actions.WriteAfterRead

        if action == Actions.WriteOnly and new_op == 'r':
            return Actions.ReadAfterWrite

        return action

    def c_keyword(action):
        return "NoAction"       if action == Actions.NoAction else       \
               "ReadAfterWrite" if action == Actions.ReadAfterWrite else \
               "WriteAfterRead" if action == Actions.WriteAfterRead else \
               "ReadOnly"       if action == Actions.ReadOnly else       \
               "WriteOnly"      if action == Actions.WriteOnly else      \
               "Ignore"         if action == Actions.Ignore else         \
               "Invalid"
