import pandas as pd
from sklearn.datasets import load_digits

# loading the dataset:
dataset = load_digits()
print(dataset.keys())

# shape of data:
print(dataset.data.shape)

# Reshape the matrix:
print(dataset.data[0].reshape(8, 8))

# plotting the graph:
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(dataset.data[9].reshape(8,8))
plt.show()

# targeting the unique value using numpy:
import numpy as np
np_data = np.unique(dataset.target)
print(np_data)

# Converting into dataframe:
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df)
print(df.describe())

# dependent variables:
X = df
# Independent variables:
y = dataset.target

# StandardScaler classifier:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

# Training and testing the dataset:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

# Logistic Regression:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# Principal Component Analysis (PCA):
from sklearn.decomposition import PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X)
print(X_pca.shape) # it will check it some columns are not useful then it will remove it

# Variance of each columns by using the explained_variance_ratio:
# it will give each and every columns variance if some columns does not have it will remove by the PCA:
print(pca.explained_variance_ratio_)

# new update columns after the PCA method applied:
print(pca.n_components_)

# Again we will train the model after the new updated columns:
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)
model_pca = LogisticRegression(max_iter=1000)
model_pca.fit(X_train_pca, y_train)
print(model_pca.score(X_test_pca, y_test))

# Again PCA:
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X_pca.shape)
print(X_pca)
# again process:
print(pca.explained_variance_ratio_)
# again we have to check the accuracy with train the model again
# we have to choose what accuracy we want 
