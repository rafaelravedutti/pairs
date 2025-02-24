class LoadBalancingAlgorithms:
    Morton = 0
    Hilbert = 1
    Diffusive = 3
    Metis = 2

    def c_keyword(algorithm):
        return "Hilbert"        if algorithm == LoadBalancingAlgorithms.Hilbert else \
               "Morton"         if algorithm == LoadBalancingAlgorithms.Morton else \
               "Diffusive"      if algorithm == LoadBalancingAlgorithms.Diffusive else \
               "Metis"          if algorithm == LoadBalancingAlgorithms.Metis else \
               "Invalid"
    