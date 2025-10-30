import pandas as pd
import statsmodels.api as sm

# Load data
data = pd.read_csv('Data_CW1.csv')

# Create income categories
data['MedInc'] = ((data['Annual income'] > 20) & (data['Annual income'] <= 70)).astype(int)
data['HigInc'] = (data['Annual income'] > 70).astype(int)

# Create car ownership dummies
data['1CarLowInc'] = ((data['Cars'] == 1) & (data['Annual income'] <= 20)).astype(int)
data['1CarMedInc'] = ((data['Cars'] == 1) & (data['MedInc'] == 1)).astype(int)
data['1CarHigInc'] = ((data['Cars'] == 1) & (data['HigInc'] == 1)).astype(int)
data['2+Cars'] = (data['Cars'] >= 2).astype(int)

# Prepare regression data
X = data[['MedInc', 'HigInc', '1CarLowInc', '1CarMedInc', '1CarHigInc', '2+Cars', 'HH Size']]
X = sm.add_constant(X)  # Add intercept
y = data['NoTrips']

# Run regression
#model = sm.OLS(y, X).fit()
model = sm.OLS(y, X).fit(cov_type='HC3')  # White's robust standard errors

################# Diagnostic tests ###########
import numpy as np
from statsmodels.stats.diagnostic import het_white, het_breuschpagan, linear_harvey_collier
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# 1. Heteroskedasticity tests
bp_test = het_breuschpagan(model.resid, model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
bp_results = dict(zip(labels, bp_test))

white_test = het_white(model.resid, model.model.exog)
white_results = dict(zip(['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value'], white_test))


# 2. Normality of residuals (Jarque-Bera)
jb_test = jarque_bera(model.resid)
jb_results = {'JB Statistic': jb_test[0], 'p-value': jb_test[1], 'Skew': jb_test[2], 'Kurtosis': jb_test[3]}

# 3. Autocorrelation test (Breusch-Godfrey)
bg_test = acorr_breusch_godfrey(model, nlags=1)
bg_results = {'LM Statistic': bg_test[0], 'p-value': bg_test[1], 'F-statistic': bg_test[2], 'F p-value': bg_test[3]}

# 4. Multicollinearity (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns[1:]  # Exclude constant
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]

# 5. Outlier detection (standardized residuals)
std_resid = model.resid / np.sqrt(model.mse_resid)
outliers = np.sum(np.abs(std_resid) > 2.5)
extreme_outliers = np.sum(np.abs(std_resid) > 3.0)





##########Display#################

# Save results to file and print clean summary
with open('regression_results.txt', 'w') as f:
    f.write(str(model.summary()))
    f.write('\n\n')
    f.write('DIAGNOSTIC TESTS\n')
    f.write('='*60 + '\n\n')
    
    f.write('1. HETEROSKEDASTICITY TESTS\n')
    f.write('-'*30 + '\n')
    f.write('Breusch-Pagan Test:\n')
    for key, value in bp_results.items():
        f.write(f'  {key:<20}: {value:.3f}\n')
    f.write('\nWhite Test:\n')
    for key, value in white_results.items():
        f.write(f'  {key:<20}: {value:.3f}\n')
    

    f.write('\n2. AUTOCORRELATION TEST (Breusch-Godfrey)\n')
    f.write('-'*30 + '\n')
    for key, value in bg_results.items():
        f.write(f'  {key:<20}: {value:.3f}\n')

    f.write('\n3. MULTICOLLINEARITY (VIF)\n')
    f.write('-'*30 + '\n')
    for _, row in vif_data.iterrows():
        f.write(f'  {row["Variable"]:<20}: {row["VIF"]:.3f}\n')

    f.write('\n4. OUTLIER ANALYSIS\n')
    f.write('-'*30 + '\n')
    f.write(f'  Observations with |std_resid| > 2.5: {outliers}\n')
    f.write(f'  Observations with |std_resid| > 3.0: {extreme_outliers}\n')
    f.write(f'  Percentage outliers (>2.5): {100*outliers/len(model.resid):.2f}%\n')

# Print clean summary
print("Trip Generation Model Results")
print("="*50)
print(f"R-squared: {model.rsquared:.3f}")
print(f"Adj. R-squared: {model.rsquared_adj:.3f}")
print(f"F-statistic: {model.fvalue:.1f}")
print(f"Prob (F-statistic): {model.f_pvalue:.2e}")
print(f"No. Observations: {int(model.nobs)}")
print("\nCoefficients:")
print("-"*70)
print(f"{'Variable':<12} {'Coef':>10} {'Std Err':>10} {'t':>10} {'P>|t|':>10}")
print("-"*70)
for i, var in enumerate(model.params.index):
    coef = model.params[i]
    std_err = model.bse[i]
    t_stat = model.tvalues[i]
    p_val = model.pvalues[i]
    print(f"{var:<12} {coef:>10.3f} {std_err:>10.3f} {t_stat:>10.3f} {p_val:>10.3f}")

print("\nDIAGNOSTIC TESTS SUMMARY:")
print("="*50)

print("\n1. Heteroskedasticity (should have p > 0.05):")
print(f"   Breusch-Pagan p-value: {bp_results['LM-Test p-value']:.3f}")
print(f"   White Test p-value: {white_results['LM-Test p-value']:.3f}")

print("\n2. Autocorrelation (should have p > 0.05):")
print(f"   Breusch-Godfrey p-value: {bg_results['p-value']:.3f}")

print("\n3. Multicollinearity (VIF should be < 10):")
for _, row in vif_data.iterrows():
    print(f"   {row['Variable']}: {row['VIF']:.2f}")

print(f"\n5. Outliers:")
print(f"   {outliers} observations with |standardized residual| > 2.5")
print(f"   {extreme_outliers} observations with |standardized residual| > 3.0")

print(f"\nFull diagnostic results saved to: regression_results.txt")
