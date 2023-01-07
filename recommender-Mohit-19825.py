import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_array
from scipy import linalg
from scipy.sparse.linalg import svds
import time
from tqdm import tqdm
import tracemalloc

np.random.seed(42)

ds_path = "../data/ml-latest-small/ratings.csv"

data = pd.read_csv(ds_path, usecols = ['userId','movieId','rating'], dtype = {'userId':int, 'movieId':int, 'rating':float})
data = data.subtract([1,1,0], axis='columns')  
mat = coo_array((data['rating'], (data['userId'], data['movieId'])))

"""## Singular Value Decomposition"""

def plot_error_SVD(range_latent, mat):
    errs = []
    times = []
    mem = []
    U,S,Vh = svds(mat, k = mat.shape[0] - 1)
    tmp = S.argsort()[::-1]
    for i in tqdm(range(*range_latent)):
        tracemalloc.stop()
        tracemalloc.start()
        start = time.time()
        t = tmp[:i]
        u = U[:,t]
        vh = Vh[t,:]
        s = S[t]
        s = np.diag(s)
        vh = s @ vh
        err = np.square(u @ vh - mat).sum() / (mat.size)
        _, first_peak = tracemalloc.get_traced_memory()
        peak = first_peak/(1024*1024)
        mem.append(peak)
        errs.append(err)
        end = time.time()
        time_elapsed = end - start
        times.append(time_elapsed)

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(*range_latent), errs)
    plt.ylabel('error')
    plt.xlabel('latent factors')
    plt.title('mean square error vs latent factors')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(*range_latent), times)
    plt.ylabel('time elapsed in secs')
    plt.xlabel('latent factors')
    plt.title('time elapsed vs latent factors')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(*range_latent), mem)
    plt.ylabel('memory used in MB')
    plt.xlabel('latent factors')
    plt.title('memory used vs latent factors')
    plt.grid(True)
    plt.show()

    return

plot_error_SVD((10, 500, 30), mat)

"""## CUR decomposition"""

def sample_k(mat, k):    
    tmp = np.square(mat)
    energy_per_row = np.sum(tmp, axis=1)
    energy_per_column = np.sum(tmp, axis=0)
    
    row_proba = energy_per_row / sum(energy_per_row)
    col_proba = energy_per_column / sum(energy_per_column)

    tmp_row = np.random.choice(list(range(mat.shape[0])), k, p = row_proba)
    tmp_col = np.random.choice(list(range(mat.shape[1])), k, p = col_proba)
    
    norm_const_per_row =  1 / ((row_proba * k)**0.5 + 1e-8)
    norm_const_per_col =  1 / ((col_proba * k)**0.5 + 1e-8)
    
    tmp_mat = mat * norm_const_per_col.reshape(1,mat.shape[1])
    tmp_mat = tmp_mat * norm_const_per_row.reshape(mat.shape[0], 1)
    
    tmp_mat = tmp_mat.tocsr()[tmp_row,:].tocsc()[:,tmp_col]

    return tmp_mat.tocoo()


def plot_error_CUR(range_latent, mat):
    errs = []
    times = []
    mem = []
    for i in tqdm(range(*range_latent)):
        tracemalloc.stop()
        tracemalloc.start()
        start = time.time()
        mat = sample_k(mat, i)
        u,s,vh = svds(mat, k = mat.shape[0] - 1)
        t = np.argsort(s)[::-1]
        u = u[:,t]
        vh = vh[t,:]
        s = s[t]
        s = np.diag(s)
        vh = s @ vh
        err = np.square(u @ vh - mat).sum() / (mat.size)
        _, first_peak = tracemalloc.get_traced_memory()
        peak = first_peak/(1024*1024)
        mem.append(peak)
        errs.append(err)
        end = time.time()
        time_elapsed = end - start
        times.append(time_elapsed)
    
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(*range_latent), errs)
    plt.ylabel('error')
    plt.xlabel('latent factors')
    plt.title('mean square error vs latent factors')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(*range_latent), times)
    plt.ylabel('time elapsed in secs')
    plt.xlabel('latent factors')
    plt.title('time elapsed vs latent factors')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(*range_latent), mem)
    plt.ylabel('memory used in MB')
    plt.xlabel('latent factors')
    plt.title('memory used vs latent factors')
    plt.grid(True)
    plt.show()
    
    return

plot_error_CUR((10, 500, 30), mat)

"""## PQ Decomposition"""

def get_matrix(df, df_full):

    num_users = np.array(df_full['userId'].unique()).shape[0]
    movie_to_ind = {}
    unique_movies = np.array(df_full['movieId'].unique())
    i=0
    for movie in unique_movies:
        movie_to_ind[movie] = i
        i+=1
        
    num_movies = np.array(df_full['movieId'].unique()).shape[0]
    mat = np.zeros((num_users, num_movies))

    for id in df.index:
        uid = df['userId'][id]
        mid = movie_to_ind[df['movieId'][id]]
        mat[uid][mid] = df['rating'][id]
        
    return mat

def train_test_split(df, test_ratio):

    df_test = df.sample(frac = test_ratio)
    df_train = df.drop(df_test.index)
    train_mat = get_matrix(df_train, df)
    test_mat = get_matrix(df_test, df)
    
    return train_mat, test_mat

def get_gradient_P(PU, QI, RUI, lamda, M, N):
    tmp = -2 * (RUI - PU.dot(QI.T)) / (M * N)
    QI = tmp * QI
    gradient = QI + 2 * lamda * PU
    return gradient

def get_gradient_Q(PU, QI, RUI, lamda, M, N):
    tmp = -2 * (RUI - PU.dot(QI.T)) / (M * N)
    PU = tmp * PU
    gradient = PU + 2 * lamda * QI
    return gradient

def get_loss(P, Q, mat, train, lamda):
    mat_hat = P @ Q.T
    diff = np.subtract(mat, mat_hat)
    diff_sq = np.square(diff)
    mse = np.mean(diff_sq)
    if train:
        loss = mse + lamda*(np.sum(np.square(P)) + np.sum(np.square(Q)))
    else:
        loss = mse  
    return loss


def gradient_descent(mat, epochs, lamda, lr, R):

    M = mat.shape[0]
    N = mat.shape[1]
    
    P = np.random.rand(M, R)
    Q = np.random.rand(N, R)
    
    losses = []
    
    for epoch in range(epochs):
        loss = get_loss(P, Q, mat, True, lamda)
        for u in range(M):
            for i in range(N):
                QI = Q[i]
                PU = P[u]
                RUI = mat[u][i]
                if RUI > 0:
                    del_PU = get_gradient_P(PU, QI, RUI, lamda, M, N)
                    del_QI = get_gradient_Q(PU, QI, RUI, lamda, M, N)
                    P[u] = PU  - lr * del_PU
                    Q[i] = QI - lr * del_QI
        losses.append((epoch, loss))
    losses= np.array(losses)

    return P, Q, losses

train_mat, test_mat = train_test_split(data, 0.2)
P, Q, losses = gradient_descent(train_mat, 100, 0.01, 0.01, 20)
test_loss = get_loss(P, Q, test_mat, False, 0.01)

print(f'Test MSE loss: {test_loss}')

plt.figure(figsize=(5,3))
plt.plot(list(zip(*losses))[0], list(zip(*losses))[1])
plt.ylabel('train loss')
plt.xlabel('epoch')
plt.title('training loss vs epoch')
plt.grid(True)
plt.show()