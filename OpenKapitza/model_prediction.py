import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, cross_val_predict, KFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF, ConstantKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

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
                                           'interlayer2': str, 'interlayer': float, 'T': float, 'fthick': float,
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
    cv.append([alpha, -np.mean(scores["test_score"])] + list(lasso.coef_))

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


# Kernel Ridge Regression


krr = KernelRidge()
krr.fit(norm_x, itr)
itr_krr = cross_val_predict(krr, norm_x, itr, cv=kfold)
r2_krr = r2_score(itr, itr_krr)
mse_krr = mean_squared_error(itr, itr_krr)


# Partial Least Squares Regression

pls = PLSRegression(n_components=5)
pls.fit(x, itr)
itr_pls = cross_val_predict(pls, x, itr, cv=kfold)
r2_pls= r2_score(itr, itr_pls)
mse_pls = mean_squared_error(itr, itr_pls)

# Principal Component Regression

pca = PCA()
pca.fit(norm_x)
z_pca = pca.transform(norm_x)
# print(pca.explained_variance_)
reg_pca = linear_model.LinearRegression()

# r2_pca_list = []
#
# for i in range(3, np.shape(z_pca)[1]):
#
#     itr_pca = cross_val_predict(reg_pca, z_pca[:, :int(i)], itr, cv=kfold)
#     r2_pca= r2_score(itr, itr_pca)
#     mse_pca = mean_squared_error(itr, itr_pca)
#     r2_pca_list.append(r2_pca)

itr_pca = cross_val_predict(reg_pca, z_pca[:, :6], itr, cv=kfold)
r2_pca= r2_score(itr, itr_pca)
mse_pca = mean_squared_error(itr, itr_pca)

# Stochastic Gradient Descent

reg_sgd = linear_model.SGDRegressor()
reg_sgd.fit(norm_x, itr)
itr_sgd = cross_val_predict(reg_sgd, norm_x, itr, cv=kfold)
r2_sgd = r2_score(itr, itr_sgd)
mse_sgd = mean_squared_error(itr, itr_sgd)

# Elastic Net

# elnet = linear_model.ElasticNet()
# grid = dict()
# grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
# grid['l1_ratio'] = np.arange(0, 1, 0.01)
# grid_elnet = GridSearchCV(elnet, grid, cv=kfold, n_jobs=-1) # scoring='neg_mean_absolute_error'
# grid_elnet.fit(norm_x, itr)
# print(" Results from Grid Search ")
# print("\n The best estimator across ALL searched params:\n", grid_elnet.best_estimator_)
# print("\n The best score across ALL searched params:\n", grid_elnet.best_score_)
# print("\n The best parameters across ALL searched params:\n", grid_elnet.best_params_)
# exit()

reg_elnet = linear_model.ElasticNet(alpha=0.01, l1_ratio=0.01)
reg_elnet.fit(norm_x, itr)
itr_elnet = cross_val_predict(reg_elnet, norm_x, itr, cv=kfold)
r2_elnet = r2_score(itr, itr_elnet)
mse_elnet = mean_squared_error(itr, itr_elnet)


# Support Vector Machines

# SVR = SVR(kernel="rbf", gamma=0.1)
#
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf']}
#
# grid_SVR = GridSearchCV(SVR, param_grid, cv=kfold, refit=True, verbose=3)
#
#
# grid_SVR.fit(norm_x, itr)
# print(" Results from Grid Search ")
# print("\n The best estimator across ALL searched params:\n", grid_SVR.best_estimator_)
# print("\n The best score across ALL searched params:\n", grid_SVR.best_score_)
# print("\n The best parameters across ALL searched params:\n", grid_SVR.best_params_)
# exit()


svr = SVR(kernel="rbf", C=1000, gamma=0.1, epsilon=0.1)
svr.fit(norm_x, itr)

itr_svr = cross_val_predict(svr, norm_x, itr, cv=kfold)
r2_svr = r2_score(itr, itr_svr)
mse_svr = mean_squared_error(itr, itr_svr)

# Gaussian Regression Processes

kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 10.0), nu=2.5) + \
         WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e3)) + ConstantKernel()


gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
gpr.fit(norm_x, itr)

itr_gpr = cross_val_predict(gpr, norm_x, itr, cv=kfold)
r2_gpr = r2_score(itr, itr_gpr)
mse_gpr = mean_squared_error(itr, itr_gpr)

# Gradient Boosting Regression

# GBR = GradientBoostingRegressor(random_state=0)
#
# parameters = {'learning_rate': [0.01,0.02,0.03,0.04],
#                   'subsample'    : [0.9,0.5,0.2,0.1],
#                   'n_estimators' : [100,500,1000,1500],
#                   'max_depth'    : [3,4,6,8,10]
#               }
# grid_GBR = GridSearchCV(estimator=GBR, param_grid = parameters, cv = kfold, n_jobs=-1)
# grid_GBR.fit(norm_x, itr)
# print(" Results from Grid Search ")
# print("\n The best estimator across ALL searched params:\n", grid_GBR.best_estimator_)
# print("\n The best score across ALL searched params:\n", grid_GBR.best_score_)
# print("\n The best parameters across ALL searched params:\n", grid_GBR.best_params_)
# exit()

# gbr = GradientBoostingRegressor(random_state=0, max_depth=3, n_estimators=100, subsample=1.0)
gbr = GradientBoostingRegressor(random_state=0, max_depth=6, n_estimators=500, subsample=0.5, learning_rate=0.01)
gbr.fit(norm_x, itr)

itr_gbr = cross_val_predict(gbr, norm_x, itr, cv=kfold)
r2_gbr = r2_score(itr, itr_gbr)
mse_gbr = mean_squared_error(itr, itr_gbr)

exec(open("fig.py").read())

