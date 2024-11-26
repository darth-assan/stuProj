print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nProcessing sheet_03_test.py")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
debugmode = True

def calculate_covariance_matrix(data):
    n_samples = data.shape[0]  # for each sensor/title
    mean_centered_data = data - np.mean(data, axis=0)
    covariance_matrix = np.dot(mean_centered_data.T, mean_centered_data) / (n_samples - 1)
    return covariance_matrix

def eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors

def select_k_for_beta(eigenvalues, beta):
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues)  # array of cumulative sorted sum in assending order 
    k = np.argmax(cumulative_variance / total_variance > beta) + 1
    return k

def project_data(data, eigenvectors, k):
    mean_centered_data = data - np.mean(data, axis=0) # centered the data to see corelation visually, but can see point of relevance
    top_eigenvectors = eigenvectors[:, :k]
    projected_data = np.dot(mean_centered_data, top_eigenvectors)  # the data is wrapped w.r.t. eigenvectors (selectively), new space where data is more deviated
    return np.array(projected_data)

def pca_with_single_beta(data, beta):
    if isinstance(data, pd.DataFrame): data = data.values  # Convert DataFrame to numpy array
    #print("\ndata:\n",data)
    if debugmode: print("\ndata shape:\n",data.shape)
    cov_matrix = calculate_covariance_matrix(data)
    #print("\ncov_matrix:\n",cov_matrix)
    if debugmode: print("\ncov_matrix shape:\n",cov_matrix.shape)  # this matrix is symmetrical
    eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)
    if debugmode: print("\neigenvectors and eigenvalues shapes:\n",eigenvectors.shape, eigenvalues.shape)
    if debugmode: print("\neach eigenvectors and eigenvalues shapes:\n",eigenvectors.shape, eigenvalues.shape)
    k = select_k_for_beta(eigenvalues, beta)  # The number of components selected. = (len(output[0]))
    projected_data = project_data(data, eigenvectors, k)  # The data projected onto the top principal components.
    explained_variance_ratio = np.sum(eigenvalues[:k]) / np.sum(eigenvalues)  # explained_variance_ratio: The ratio of variance explained by the selected components.
    return projected_data, explained_variance_ratio



# Read the CSV file
df = pd.read_csv(r'..\..\data\oversampling\oversampling\synthetic_data-HU.csv')
#print("titles:", df.columns)
df_pca_S, evr_S = pca_with_single_beta(df, 0.998)  # bigger explained_variance_ratio ->portion of data captured by PCA
df_pca_M, evr_M = pca_with_single_beta(df, 0.895)
df_pca_L, evr_L = pca_with_single_beta(df, 0.879)  # smaller explained_variance_ratio ->portion of data captured by PCA
output = df_pca_S

# Plot each column

#for i in range(len(output[0])):
#    plt.scatter(output[i], label=i, s=0.5)
#print(len(df_pca_S))

for j in range(len(output[0])):
    plt.figure(j)
    for i in range(len(output[0])):
        plt.scatter(output[:,j], output[:,i], s=0.5, label=f'X={i+1}')
    plt.xlabel(f'df_pca_S[:,{j+1}]')
    plt.ylabel('df_pca_S[:,X]')
    plt.legend()
    plt.title(f'PCA of feature {j+1} vs X')
    plt.show()
