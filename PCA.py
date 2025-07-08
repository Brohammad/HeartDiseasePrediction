import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Generate dummy gene expression data
genes = ['gene' + str(i) for i in range(1, 101)]
wt = ['wt' + str(i) for i in range(1, 6)]
ko = ['ko' + str(i) for i in range(1, 6)]
data = pd.DataFrame(columns=[*wt, *ko], index=genes)

for gene in data.index:
    data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
    data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)

# Transpose and scale
scaled_data = preprocessing.scale(data.T)

# PCA
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# Explained variance
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PCA' + str(i) for i in range(1, len(per_var)+1)]

# Scree plot
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Variance Explained')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# PCA scatter plot
pca_df = pd.DataFrame(data=pca_data, columns=labels, index=data.columns)

plt.figure(figsize=(8,6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'])
plt.title('Principal Component Analysis')
plt.xlabel(f'PC1 - {per_var[0]}%')
plt.ylabel(f'PC2 - {per_var[1]}%')

# Annotate sample names
for sample in pca_df.index:
    x = pca_df.loc[sample, 'PCA1']
    y = pca_df.loc[sample, 'PCA2']
    plt.annotate(sample, (x, y))

plt.grid(True)
plt.show()

# Calculate loading scores for PCA1
# If you did NOT transpose:
#loading_scores = pd.Series(pca.components_[0], index=data.columns)

# If you DID transpose:
loading_scores = pd.Series(pca.components_[0], index=data.index)

# Get absolute values for contribution magnitude
abs_loading_scores = loading_scores.abs()

# Sort by descending absolute loading scores
sorted_loading_scores = abs_loading_scores.sort_values(ascending=False)

# Display top 10 genes contributing to PCA1
top_10_genes = sorted_loading_scores.head(10)
print("Top 10 Genes Contributing to PCA1 (with loading scores):\n")
for gene in top_10_genes.index:
    print(f"{gene}: {loading_scores[gene]}")
