
def sharpe(returns,rf=0):
    import numpy as np
    excess = returns - rf
    return np.mean(excess)/np.std(excess,ddof=1)

# add other metrics...
