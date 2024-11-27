debugmode = False
if debugmode: print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nProcessing sheet_03_test.py")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from loguru import logger
import os

# Configure Loguru
logger.remove()
logger.add(sys.stdout, level="INFO", filter=lambda record: record["level"].name == "INFO")


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
    return projected_data


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
    

def prepare_dataset(dir):
    dftest_2103 = pd.DataFrame()
    dftrain_2103 = pd.DataFrame()
    for file_name in os.listdir(dir):  # List all files in the directory
        file_path = os.path.join(dir, file_name)
        if file_name.lower().endswith(".csv"):  # Ensure the file is a CSV
            if "test" in file_name.lower():  # Check if the file is a "test" file
                dftest_2103 = pd.concat([dftest_2103, pd.read_csv(file_path)], ignore_index=True)
            elif "train" in file_name.lower():  # Check if the file is a "train" file
                dftrain_2103 = pd.concat([dftrain_2103, pd.read_csv(file_path)], ignore_index=True)
    dftest_2103['time'] = range(0, len(dftest_2103) + 0)
    dftrain_2103['time'] = range(0, len(dftrain_2103) + 0)
    return dftest_2103, dftrain_2103

    """
    # Concatinating the hai files
    df1 = pd.read_csv(r'..\..\data\original\hai-21.03\test1.csv')
    df2 = pd.read_csv(r'..\..\data\original\hai-21.03\test2.csv')
    df3 = pd.read_csv(r'..\..\data\original\hai-21.03\test3.csv')
    df4 = pd.read_csv(r'..\..\data\original\hai-21.03\test4.csv')
    df5 = pd.read_csv(r'..\..\data\original\hai-21.03\test5.csv')
    df6 = pd.read_csv(r'..\..\data\original\hai-21.03\train1.csv')
    df7 = pd.read_csv(r'..\..\data\original\hai-21.03\train2.csv')
    df8 = pd.read_csv(r'..\..\data\original\hai-21.03\train3.csv')
    hai_2103_test = pd.concat([df1, df2, df3, df4, df5])
    hai_2103_test['time'] = range(1, len(hai_2103_test) + 1) # switch time to a range
    hai_2103_train = pd.concat([df6, df7, df8])
    hai_2103_train['time'] = range(1, len(hai_2103_train) + 1) # switch time to a range
    """

def main_func(dir_in, dir_out, beta):
    # main
    #dir = "..\\..\\data\\original\\hai-21.03"
    dftest_2103, dftrain_2103 = prepare_dataset(dir_in)

    temp_pca, temp_evr = pca_with_single_beta(dftest_2103, beta)
    np.savetxt(f'{dir_out}\hai_21.03_test_{beta}.csv', temp_pca, delimiter=",")
    temp_pca, temp_evr = pca_with_single_beta(dftrain_2103, beta)
    np.savetxt(f'{dir_in}\hai_21.03_train_{beta}.csv', temp_pca, delimiter=",")

    """
    betas = [0.998, 0.895, 0.879]
    for beta in betas:
        temp_pca, temp_evr = pca_with_single_beta(dftest_2103, beta)
        np.savetxt(f'hai_21.03_test_{beta}.csv', temp_pca, delimiter=",")
        temp_pca, temp_evr = pca_with_single_beta(dftrain_2103, beta)
        np.savetxt(f'hai_21.03_train_{beta}.csv', temp_pca, delimiter=",")
    """









"""
# EXAMPLE OUTPUT PLOT
df = pd.read_csv('hai_21.03_test_0.998.csv', header=None)
output = np.array(df)
for j in range(len(output[0])):
    plt.figure(j)
    for i in range(len(output[0])):
        plt.scatter(output[:,j], output[:,i], s=0.5, label=f'X={i+1}')
        #plt.scatter(output[:,4], output[:,0], s=0.5, label=f'X={i+1}')
    plt.xlabel(f'df_pca_S[:,{j+1}]')
    plt.ylabel('df_pca_S[:,X]')
    plt.legend()
    plt.title(f'PCA of feature {j+1} vs X')
    plt.show()
"""
