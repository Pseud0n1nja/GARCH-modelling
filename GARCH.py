import pandas as pd

#import numpy as np
#import statsmodels.formula.api as sm
#import matplotlib.pyplot as plt
#import datetime as dt
from arch import arch_model



prices = pd.read_csv("stock_data.csv")
prices.info()

returns  = prices["NIFTY"].pct_change().dropna()

#volatilty clustering
returns.plot()

#mean is zero for HFT in ts
model  = arch_model(returns,vol ="GARCH", p=1, o=0, q=1, mean = "Zero", dist = "Normal")
model  = arch_model(returns,vol ="GARCH", p=1, o=0, q=1, dist = "Normal")
#help(arch_model)
results = model.fit()

print(results.summary())

results.plot()

para  = results.params

gamma = 1- (para[1]+para[2])
lvol = (para[1]/gamma)**0.5
print(gamma, lvol)

se = results.std_err
print(se)

tstats = results.tvalues
print(tstats)

ts = para/se 

cv = results.conditional_volatility

