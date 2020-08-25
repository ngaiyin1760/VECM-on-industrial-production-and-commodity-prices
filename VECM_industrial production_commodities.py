import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from itertools import permutations
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import VECM


#Commodities data==============================================================
#load data from excel file downloaded from IEA
commodities = pd.read_excel('YOUR OWN DIRECTORY/CMOHistoricalDataMonthly.xlsx', 'Monthly Prices')
commodities = commodities[3:]
commodities.columns = commodities.iloc[0,:]
commodities = commodities.drop(commodities.index[[0,1,2]])
commodities = commodities.set_index(commodities.iloc[:,0], drop=True).drop(commodities.columns[0], axis=1)

#Convert dataframe's index to monthly time series
month_series_commodities = pd.Series(pd.date_range("1960-01-01", periods=commodities.shape[0], freq="M"))
commodities = commodities.set_index(month_series_commodities)

#Select specific commodities price
commodities_energy_metal = commodities[['Crude oil, Brent', 'Coal, South African', 'Natural gas, US',
                                        'Aluminum', 'Iron ore, cfr spot', 'Copper', 'Platinum', 'Silver']]


#Industrial Production data====================================================
indust_prod = pd.read_excel('YOUR OWN DIRECTORY/INDPRO.xls')
indust_prod = indust_prod[10:]
indust_prod = indust_prod.set_index(indust_prod.iloc[:,0], drop=True).drop(indust_prod.columns[0], axis=1)
indust_prod = indust_prod.rename(columns={'Unnamed: 1': 'US Industrial Production Index'})

#Convert dataframe's index to month time series
month_series_ip = pd.Series(pd.date_range("1919-01-01", periods=indust_prod.shape[0], freq="M"))
indust_prod = indust_prod.set_index(month_series_ip)


#Combine the dataset===========================================================
#Select the period beginning from 1990 January
commodities_energy_metal = commodities_energy_metal.loc['1990-1-1':]
commodities_energy_metal = commodities_energy_metal.astype(float)

indust_prod = indust_prod.loc['1990-1-1':]
indust_prod = indust_prod.astype(float)

df = pd.concat([indust_prod, commodities_energy_metal], axis=1, sort=False)
df.columns = ['Prod', 'Oil', 'Coal', 'N.Gas', 'Alum', 'Iron', 'Copp', 'Plat', 'Silv']

#plot the data and correlation matrix==========================================
plt.figure(figsize=(40,32))
df.plot()
df[['Prod', 'Oil', 'Coal', 'N.Gas', 'Iron', 'Silv']].plot()

plt.figure(figsize=(10,8))
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)


#Test for stationary===========================================================
# plots the autocorrelation plots for each commodties price at 75 lags
for i in df:
    plot_acf(df[i], lags = 75)
    pyplot.title('ACF for %s' % i) 
    pyplot.show()

#construct time series with first difference
df_diff = df.diff().dropna()

# plots the autocorrelation plots for the difference in each commodities price from the
# price the previous month at 75 lags
for i in df_diff:
    plot_acf(df_diff[i], lags = 75)
    pyplot.title('ACF for %s' % i) 
    pyplot.show()

# performs the Augmented Dickey-Fuller Test for all our variables of interest without
# a constant, with a constant, and with a constant and linear trend
for i in df_diff:
    for j in ['nc', 'c', 'ct']:
        result = adfuller(df_diff[i], regression = j)
        print('ADF Statistic with %s for %s: %f' % (j, i, result[0]))
        print('p-value: %f' % result[1])


#Granger Causality Test===========================================================
# creates a list of tuples containing the permutations of length 2
df_perms = list(permutations(df, 2))

# loops through the list of stock permutations
#for i in range(len(df_perms)):
for i in range(8):
    temp_list = list(df_perms[i])
    temp_df = df[temp_list]
    print('Does a lag of ' + temp_list[1] + ' predict ' + temp_list[0])
    print(grangercausalitytests(temp_df, maxlag = 3, addconst = True, verbose = True))
    print('')
    print('')

df_selected = df[['Prod', 'Oil', 'Coal', 'Alum', 'Iron', 'Copp', 'Plat', 'Silv']]

#Johansen Cointegration test===================================================
def johansen_trace(y, p):
        N, l = y.shape
        joh_trace = coint_johansen(y, 0, p)
        r = 0
        for i in range(l):
            if joh_trace.lr1[i] > joh_trace.cvt[i, 1]:
                r = i + 1
        joh_trace.r = r

        return joh_trace

# loops through 1 to 6 lags of months
for i in range(1, 7): 
    # tests for cointegration at i lags
    joh_trace = johansen_trace(df_selected, i)
    print('Using the Trace Test, there are', joh_trace.r, '''cointegrating vectors at 
    %s lags between the series''' % i)
    print()


#Vector error corretion model (VECM)===========================================
# estimates the VECM on the closing prices with 6 lags, 1 cointegrating relationship, and
# a constant within the cointegration relationship
model_vecm = VECM(endog = df_selected, k_ar_diff = 6, coint_rank = 1, deterministic = 'aic')
model_vecm_fit = model_vecm.fit()
model_vecm_fit.summary()


#Impulse response function=====================================================
irf = model_vecm_fit.irf(24)
irf.plot(orth = False)


#Dynamic forecasting===========================================================
model_vecm_fit.plot_forecast(12, n_last_obs=60)
