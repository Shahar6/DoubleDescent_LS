from scipy.fft import dct
import numpy as np
import matplotlib.pyplot as plt

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


def generate_x(Sigma=Sigma, num_samples=1):
    L = np.linalg.cholesky(Sigma)
    dim = Sigma.shape[0]
    z = np.random.normal(size=(num_samples, dim))
    x = z @ L.T
    return x


def generate_epsilon(mean=0, variance=0.25):
    std_dev = np.sqrt(variance)
    scalar = np.random.normal(loc=mean, scale=std_dev)
    return scalar


data_size = 32
train_matrix = np.zeros((data_size, d))
sample_out = np.zeros((data_size, 1))

for i in range(data_size):
    vec = generate_x()
    train_matrix[i, :] = vec
    y = vec[0].T @ beta + generate_epsilon()
    sample_out[i, :] = y

print("rank of X is " + str(np.linalg.matrix_rank(train_matrix)))

test_size = 1000
test_matrix = np.zeros((test_size, d))

test_out = np.zeros((test_size, 1))

for i in range(test_size):
    vec = generate_x()
    test_matrix[i, :] = vec
    y = (vec[0].T @ beta) + generate_epsilon()
    test_out[i, :] = y

# create U
errors = []
params = []
for p in range(1, d + 1):
    U = np.zeros((d, p))
    for i in range(p):
        column = DCT[:, i]
        U[:, i] = column
    tetha = train_matrix @ U
    hasInv = np.linalg.det(tetha.T @ tetha) != 0
    if hasInv:
        alpha = np.linalg.inv(tetha.T @ tetha) @ tetha.T @ sample_out
    else:
        alpha = np.linalg.pinv(tetha.T @ tetha) @ tetha.T @ sample_out
    test_err = np.linalg.norm((alpha.T @ U.T @ test_matrix.T) - test_out) / test_size
    params.append(p)
    errors.append(test_err)

plt.plot(params, errors, marker='o', linestyle='-', color='b')

plt.title('Errors and Parameters Least Squares')
plt.xlabel('Parameters')
plt.ylabel('Error')
plt.show()
