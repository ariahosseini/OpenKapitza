import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, cross_val_predict, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# Read input files
df_atom_energy_vasp = pd.read_excel(io='atom_energy_vasp.xlsx', sheet_name='Sheet1', header=0, skiprows=1,
                     dtype={'Atomic number': int, 'Symbol': str, 'PAW potential name': str,
                            'Total energy of an isolated atom (eV/atom)': float})
df_descriptor_dataset = pd.read_excel(io='descriptor_dataset.xlsx', sheet_name='descriptor dataset', header=0,
                                      skiprows=0, index_col=0,
                                      dtype={'Material': str, 'Formula': str, 'Cv (J/gK)': float,
                                             'Melting point (K)': float, 'Density (g/cm3)': float,
                                             'Volume (10-29 m3/f.u.)': float, 'R1': int, 'R2': int,
                                             'Mass (u)': float, 'AC1x': int, 'AC2x': int, 'AC2y': int,
                                             'ENc': float, 'ENa': float, 'IPc': float, 'IPa': float,
                                             'Eb(eV/f.u.)': float})
df_training_dataset = pd.read_excel(io='training_dataset_itr_prediction.xlsx', sheet_name='training data', header=0,
                                    skiprows=0, index_col=0,
                                    dtype={'Interface': str, 'Film': str, 'substrate': str, 'interlayer1': str,
                                           'interlayer2': str, 'interlayer': int, 'T': float, 'fthick': float,
                                           'fheatcap': float, 'fmelt': float, 'fdensity': float, 'funit': float,
                                           'fR1': int, 'fR2': int, 'fAC1x': int, 'fAC1y': int, 'fAC2x': int,
                                           'fAC2y': int, 'fENc': float, 'fENa': float, 'fIPc': float, 'fIPa': float,
                                           'fEb': float, 'fmass': float, 'sheatcap': float, 'smelt': float,
                                           'sdensity': float, 'sunit': float, 'sR1': int, 'sR2': int, 'sAC1x': int,
                                           'sAC1y': int, 'sAC2x': int, 'sAC2y': int, 'sENc': float, 'sENa': float,
                                           'sIPc': float, 'sIPa': float, 'sEb': float, 'smass': float, 'itr': float})

df_descriptor_dataset_corr_pearson = df_descriptor_dataset.corr(method='pearson')
df_training_dataset_corr_pearson = df_training_dataset.corr(method='pearson')
# descriptor_dataset_pearson = df_descriptor_dataset_corr_pearson.unstack().abs().sort_values(kind="quicksort")
# training_dataset_pearson = df_training_dataset_corr_pearson.unstack().abs().sort_values(kind="quicksort")

itr = df_training_dataset['itr']
features = [_ for _ in df_training_dataset.select_dtypes(include=[float]).columns if _ != 'itr']

x = df_training_dataset[features]

#
reg = linear_model.LinearRegression()
reg.fit(x.values, y)
print(reg.coef_)
print(reg.intercept_)
y_pred = reg.predict(x)
y_mse = mean_squared_error(y, y_pred)
print(reg.predict([[100.0, 80.0, 0.111941, 1337.0, 19.3, 1.695, 2.54, 2.54, 9.225, 9.225, -3.814051, 196.97, 0.275127,
                    1687.0, 2.33, 2.003, 1.9, 1.9, 8.151, 8.151, -4.622464, 28.09]]))
print(y.values[0])
print(y_mse)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
y_cv = cross_validate(reg, x, y, cv=kfold)  # bad practice
y_pred_cv = cross_val_predict(reg, x, y, cv=kfold)
r2_reg = r2_score(y, y_pred_cv)
y_cv_mse = mean_squared_error(y, y_pred_cv)

sel = SelectKBest(f_regression, k=3)
x_new = sel.fit_transform(x, y)

def identify_columns(x_new, nrows=np.shape(x.values)[1]):
    columns = x.columns
    xvalues = x.values
    dist = np.linalg.norm(xvalues[:nrows, :, None] - x_new[:nrows, None, :], axis=0)
    return columns[np.argmin(dist, axis=0)].values

print(f'Selected features {identify_columns(x_new)}')

y_best = cross_val_predict(reg, x_new, y, cv=kfold)
r2_best= r2_score(y, y_best)
y_mse_best = mean_squared_error(y, y_best)

scaler = StandardScaler()
scaler.fit(x)
z = scaler.transform(x)

cv_results = []
coeffs = []
alphas = np.logspace(-2, 2, 71)

for alpha in alphas:

    ridge = linear_model.Ridge(alpha=alpha, max_iter=10000)
    ridge.fit(z, y)
    scores = cross_validate(ridge, z, y, cv=kfold, scoring="neg_mean_squared_error")
    cv_results.append([alpha, -np.mean(scores["test_score"])]+ list(ridge.coef_))


cv_results = pd.DataFrame(cv_results, columns=["alpha", "score"] + features)

best_alpha = cv_results["alpha"][cv_results["score"].idxmin()]

ridge_best = linear_model.Ridge(alpha=best_alpha, max_iter=10000)
ridge_best.fit(z, y)
means_ = scaler.mean_
stds_ = scaler.scale_

real_coeff = ridge_best.coef_/ stds_  #  convert back to un-normalized inputs
real_interp = ridge_best.intercept_ - means_.dot(real_coeff)  #  convert back to un-normalized inputs

equation = ["%.2e %s" % (v, f) for v, f in zip(real_coeff, features)]
print("itc = %.1f + %s" % (real_interp, " + ".join(equation)) )

y_ridge = cross_val_predict(ridge_best, z, y, cv=kfold)
r2_ridge= r2_score(y, y_ridge)
y_mse_ridge = mean_squared_error(y, y_ridge)




cv_results_lasso = []
coeffs_lasso = []
alphas = np.logspace(-2, 2, 71)

for alpha in alphas:

    lasso = linear_model.Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(z, y)
    scores_lasso = cross_validate(lasso, z, y, cv=kfold, scoring="neg_mean_squared_error")
    cv_results_lasso.append([alpha, -np.mean(scores_lasso["test_score"])]+ list(lasso.coef_))


cv_results_lasso = pd.DataFrame(cv_results_lasso, columns=["alpha", "score"] + features)
best_alpha_lasso = cv_results_lasso["alpha"][cv_results_lasso["score"].idxmin()]





lasso_best = linear_model.Lasso(alpha=best_alpha_lasso, max_iter=10000)
y_lesso = cross_val_predict(lasso_best, z, y, cv=kfold)
r2_lesso= r2_score(y, y_lesso)
y_mse_lesso = mean_squared_error(y, y_lesso)
lasso_best.fit(z, y)

real_coeff_lasso = lasso_best.coef_/ stds_  #  convert back to un-normalized inputs
real_interp_lasso = lasso_best.intercept_ - means_.dot(real_coeff_lasso)  #  convert back to un-normalized inputs

equation_lasso = ["%.2e %s" % (v, f) for v, f in zip(real_coeff_lasso, features) if abs(v) > 1e-4]
print("itc = %.1f + %s" % (real_interp_lasso, " + ".join(equation_lasso)) )



pls = PLSRegression(n_components=3)
pls.fit(x, y)
y_pls = cross_val_predict(pls, x, y, cv=kfold)
r2_pls= r2_score(y, y_pls)
y_mse_pls = mean_squared_error(y, y_pls)

exec(open("fig.py").read())