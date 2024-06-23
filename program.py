from scipy.fft import dct
import numpy as np

# generate input like so: x ~ N(0, 𝚺), 𝚺 = DCT * delta * DCT(T), y = 𝐱(𝑇) * 𝛃 + ϵ, ϵ ~ N(0, 0.25)

d = 128
DCT = dct(np.eye(d), axis=0, norm='ortho')

delta = np.zeros((d, d))
for j in range(d):
    delta[j, j] = 3 * (0.9 ** j)

Sigma = DCT @ delta @ DCT.T
sigma_eps = 0.25
beta_tide = np.array([0.2178] * 20 + [0.0218] * 108)
beta = DCT @ beta_tide


def generate_random_x_cholesky(Sigma=Sigma, num_samples=1):
    L = np.linalg.cholesky(Sigma)
    dim = Sigma.shape[0]
    z = np.random.normal(size=(num_samples, dim))
    x = z @ L.T
    return x


def generate_scalar_normal(mean=0, variance=0.25):
    std_dev = np.sqrt(variance)
    scalar = np.random.normal(loc=mean, scale=std_dev)
    return scalar


data_size = 32
input_matrix = np.zeros((data_size, d))

sample_out = []

for i in range(data_size):
    vec = generate_random_x_cholesky()
    input_matrix[i, :] = vec
    y = vec[0].T @ beta + generate_scalar_normal()
    sample_out.append(y)

print("rank of X is " + str(np.linalg.matrix_rank(input_matrix)))
