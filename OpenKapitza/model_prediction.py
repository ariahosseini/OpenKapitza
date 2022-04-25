import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, cross_val_predict, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

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

itr = df_training_dataset['itr']
features = [_ for _ in df_training_dataset.select_dtypes(include=[float]).columns if _ != 'itr']
x = df_training_dataset[features]
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()
scaler.fit(x)
norm_x = scaler.transform(x)
means_ = scaler.mean_
stds_ = scaler.scale_
cv = []
alphas = np.logspace(-2, 2, 101)

for alpha in alphas:

    lasso = linear_model.Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(norm_x, itr)
    scores = cross_validate(lasso, norm_x, itr, cv=kfold, scoring="neg_mean_squared_error")
    cv.append([alpha, -np.mean(scores["test_score"])]+ list(lasso.coef_))

cv = pd.DataFrame(cv, columns=["alpha", "score"] + features)
opt_alpha = cv["alpha"][cv["score"].idxmin()]

reg = linear_model.Lasso(alpha=opt_alpha, max_iter=10000)
reg.fit(norm_x, itr)
itr_lasso = cross_val_predict(reg, norm_x, itr, cv=kfold)
r2_lasso = r2_score(itr, itr_lasso)
mse_lasso = mean_squared_error(itr, itr_lasso)

coeff = reg.coef_/ stds_  # Convert back to un-normalized inputs
interp = reg.intercept_ - means_.dot(coeff)  # Convert back to un-normalized inputs
# eq = ["%.2e %s" % (v, f) for v, f in zip(coeff, features) if abs(v) > 1e-4]
# print("itc = %.1f + %s" % (interp, " + ".join(eq)) )

# Partial Least Squares Regression

pls = PLSRegression(n_components=3)
pls.fit(x, itr)
itr_pls = cross_val_predict(pls, x, itr, cv=kfold)
r2_pls= r2_score(itr, itr_pls)
mse_pls = mean_squared_error(itr, itr_pls)

# Principal Component Regression

pca = PCA()
pca.fit(norm_x)
z_pca = pca.transform(norm_x)
print(pca.explained_variance_)
reg_pca = linear_model.LinearRegression()
itr_pca = cross_val_predict(reg_pca, z_pca[:, : 3], itr, cv=kfold)
r2_pca= r2_score(itr, itr_pca)
mse_pca = mean_squared_error(itr, itr_pca)

exec(open("fig.py").read())


