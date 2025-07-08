import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# Step 1: Generate synthetic gene expression data
np.random.seed(42)
genes = ['gene' + str(i) for i in range(1, 101)]
samples = ['wt' + str(i) for i in range(1, 6)] + ['ko' + str(i) for i in range(1, 6)]

data_matrix = pd.DataFrame(index=genes, columns=samples)
for i in genes:
    wt_values = np.random.poisson(lam=np.random.randint(10, 1000), size=5)
    ko_values = np.random.poisson(lam=np.random.randint(10, 1000), size=5)
    data_matrix.loc[i] = np.concatenate([wt_values, ko_values])
data_matrix = data_matrix.astype(float)

scaler=StandardScaler()
scaled_data=scaler.fit_transform(data_matrix.T)
pca = PCA()
pca_data = pca.fit_transform(scaled_data)
pca_var_per = np.round(pca.explained_variance_ratio_ * 100, 1)

# PCA Plot
plt.figure(figsize=(6, 4))
for i, sample in enumerate(data_matrix.columns):
    plt.text(pca_data[i, 0], pca_data[i, 1], sample)
plt.xlabel(f'PC1 - {pca_var_per[0]}%')
plt.ylabel(f'PC2 - {pca_var_per[1]}%')
plt.title("PCA Graph")
plt.grid(True)
plt.tight_layout()
plt.show()

mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
mds_data = mds.fit_transform(scaled_data)

# MDS Plot (Euclidean)
plt.figure(figsize=(6, 4))
for i, sample in enumerate(data_matrix.columns):
    plt.text(mds_data[i, 0], mds_data[i, 1], sample)
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title("MDS Plot using Euclidean distance")
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 3: MDS with average log2 fold change distance
log2_matrix = np.log2(data_matrix)

# Custom distance matrix: avg(abs(log2(FC)))
n = log2_matrix.shape[1]
log2_dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        log2_dist_matrix[i, j] = np.mean(np.abs(log2_matrix.iloc[:, i] - log2_matrix.iloc[:, j]))

# Classical MDS with log2 FC distance
mds_logfc = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_logfc_data = mds_logfc.fit_transform(log2_dist_matrix)

# MDS Plot (log2 FC)
plt.figure(figsize=(6, 4))
for i, sample in enumerate(data_matrix.columns):
    plt.text(mds_logfc_data[i, 0], mds_logfc_data[i, 1], sample)
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title("MDS Plot using avg(logFC) as the distance")
plt.grid(True)
plt.tight_layout()
plt.show()