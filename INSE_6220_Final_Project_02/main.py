import pandas as pd
import numpy as np
from pca import pca
import matplotlib.pyplot as plt
from pycaret.classification import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import seaborn as sns

# load data
df = pd.read_csv(r'C:\Users\kofi4\PycharmProjects\INSE_6220_Final_Project_02\diabetes.csv')
print(df.head(n=25))
print(df.shape)
X = df.drop(columns=['Outcome'])

# 1.0 Descriptive Stats
X.describe().transpose()
Y = df['Outcome']
Xstd = StandardScaler().fit_transform(X)
df_Xstd = pd.DataFrame(Xstd)
print(df_Xstd)
Xcols = X.columns
df_Xstd.columns = Xcols
df_Xstd.describe().transpose()

# Bar Chart of Outcome
Y.value_counts().plot(kind='bar', rot=0)
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

# Box Plot
ax = plt.figure()
ax = sns.boxplot(data=df_Xstd, orient="v", palette="Set2")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

# Pair Plot
sns.pairplot(df_Xstd)
plt.show()
plt.close()

# Covariance Matrix
dfc = df_Xstd - df_Xstd.mean()
ax = sns.heatmap(dfc.cov(), cmap='RdYlGn_r', linewidths=0.5, annot=True, cbar=False, square=True)
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False, labeltop=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.show()
plt.close()

# 2.0 Principal Component Analysis (PCA)
model = pca()
output = model.fit_transform(X)
model2 = pca(n_components=8)
output2 = model2.fit_transform(X)
eigenvectors = pd.DataFrame(model2.results['loadings']).T
eigenvalues = pd.DataFrame(model.results['explained_var'])
print(eigenvectors)
print(eigenvalues)
eigenvectors.to_csv('eigenvectors.csv')
eigenvalues.to_csv('eigenvalues.csv')
pc_data = output['PC']
A = output['loadings'].T
print(pc_data)
print(A)
print(output['variance_ratio'])

model.plot()
plt.show()
plt.close()

model.biplot()
plt.show()
plt.close()

model.scatter()
plt.show()
plt.close()


# PCA Method 2
# pca = PCA()
# pca.fit(df_Xstd)
# Z = pca.fit_transform(df_Xstd)
# idx_Negative = np.where(Y == 0)
# idx_Positive = np.where(Y == 1)
#
# plt.figure(figsize=(15,10))
# plt.scatter(Z[idx_Negative, 0], Z[idx_Negative, 1], color='green', label='Negative Diabetes Diagnosis')
# plt.scatter(Z[idx_Positive, 0], Z[idx_Positive, 1], color='red', label='Positive Diabetes Diagnosis')
# plt.legend()
# plt.xlabel('Z1')
# plt.ylabel('Z2')
# plt.show()
# plt.close()
#
# # PCA - Step 2: Calculate the Covariance Matrix
# A = pca.components_.T
#
# print(pd.DataFrame(A))
# Eigen Values
# plt.figure(figsize=(8,4))
# plt.scatter(A[:, 0], A[:, 1], color='red')
# plt.xlabel('A1')
# plt.ylabel('A2')
# variables = Xcols
# for label, x, y, in zip(variables, A[:, 0], A[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(-20, 20))
#     textcoords = 'offset points',
#     ha = 'right',
#     va = 'bottom'
#     bbox = dict(boxstyle='round', pad=0.5, fc='yellow', alpha=0.5)
#     arrowprops = dict(arrowstyle='->', connectionstyle='arc3, rad=0')
#
# plt.show()
#
# Lambda = pca.explained_variance_
#
# A1 = A[:, 0]
# A2 = A[:, 1]
# Z1 = Z[:, 0]
# Z2 = Z[:, 1]
#
# x = np.arange(len(Lambda)) + 1
# plt.figure(figsize=(15,10))
# plt.plot(x, Lambda/sum(Lambda), 'ro-', lw=3)
# plt.xlabel('Number of components')
# plt.ylabel('Explained variance')
# plt.show()
#
# ell = pca.explained_variance_ratio_
# ind = np.arange(len(ell))
# plt.figure(figsize=(15,10))
# plt.bar(ind, ell, align='center', alpha=0.5)
# plt.plot(np.cumsum(ell))
# plt.show()
#
# plt.figure(figsize=(15,10))
# plt.xlabel('Z1')
# plt.ylabel('Z2')
#
# for i in range(len(A1)):
#     plt.arrow(0,0, A1[i]*max(Z1), A2[i]*max(Z2), color='k', width=0.0005, head_width=0.0025)
#     plt.text(A1[i]*max(Z1)*1.2, A2[i]*max(Z2)*1.2, Xcols[i], color='k')
# plt.scatter(Z[idx_Negative, 0], Z[idx_Negative, 1], color='r', label='Negative')
# plt.scatter(Z[idx_Positive, 0], Z[idx_Positive, 1], color='g', label='Positive')
# plt.show()

# 3.0 Machine Learning
# 3.1 Before PCA
train_data = df.sample(frac=0.7,random_state=786).reset_index(drop=True)
predict_data = df.drop(train_data.index).reset_index(drop=True)

print(f"Data for Modeling: {train_data.shape}")
print(f"Data for Predictions: {predict_data.shape}")

bin_clf = setup(data=train_data, target='Outcome', session_id=123)

compare_models()
gen_compare = pull()
gen_compare.to_csv('General_Model_Comparison.csv')
print(gen_compare)
# 3.2 After PCA
pc_data.insert(2, 'Outcome', df['Outcome'], True)
print(pc_data)
pc_train_data = pc_data.sample(frac=0.7,random_state=786).reset_index(drop=True)
pc_predict_data = pc_data.drop(pc_train_data.index).reset_index(drop=True)
pc_bin_clf = setup(data=pc_train_data, target='Outcome', session_id=123)
compare_models()
PCA_gen_compare = pull()
PCA_gen_compare.to_csv('PCA_Model_Comparison.csv')

# 3.3 Create Model
# LDA
lda = create_model('lda')
tuned_lda = tune_model(lda)
Tuned_LDA_Results = pull()
Tuned_LDA_Results.to_csv('Tuned_LDA_Results.csv')
print(tuned_lda)

# LR
lr = create_model('lr')
tuned_lr = tune_model(lr)
print(tuned_lr)

# ridge
ridge = create_model('ridge')
tuned_ridge = tune_model(ridge)
print(tuned_ridge)

# 3.4 Plots
# LDA
plot_model(tuned_lda, plot='auc', save=True)
os.rename('AUC.png', 'LDA_AUC.png')
plot_model(tuned_lda, plot='confusion_matrix', save=True)
os.rename('Confusion Matrix.png', 'LDA_Confusion_Matrix.png')
plot_model(tuned_lda, plot='boundary', save=True)
os.rename('Decision Boundary.png', 'LDA_Decision_Boundary.png')
plot_model(tuned_lda, plot='feature', save=True)
os.rename('Feature Importance.png', 'LDA_Feature_Importance.png')

# LR
plot_model(tuned_lr, plot='auc', save=True)
os.rename('AUC.png', 'LR_AUC.png')
plot_model(tuned_lr, plot='confusion_matrix', save=True)
os.rename('Confusion Matrix.png', 'LR_Confusion_Matrix.png')
plot_model(tuned_lr, plot='boundary', save=True)
os.rename('Decision Boundary.png', 'LR_Decision_Boundary.png')
plot_model(tuned_lr, plot='feature', save=True)
os.rename('Feature Importance.png', 'LR_Feature_Importance.png')

# Ridge
plot_model(tuned_ridge, plot='confusion_matrix', save=True)
os.rename('Confusion Matrix.png', 'Ridge_Confusion_Matrix.png')
plot_model(tuned_ridge, plot='boundary', save=True)
os.rename('Decision Boundary.png', 'Ridge_Decision_Boundary.png')
plot_model(tuned_ridge, plot='feature', save=True)
os.rename('Feature Importance.png', 'Ridge_Feature_Importance.png')


