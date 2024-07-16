from scipy.fft import dct
import numpy as np
import matplotlib.pyplot as plt

# if you read the code, note that I avoided for loops to calculate individual x's,
# instead I used matrix multiplications where I found it an option

d = 128
DCT = dct(np.eye(d), axis=0, norm='ortho')

delta = np.zeros((d, d))
for j in range(d):
    delta[j, j] = 3 * (0.9 ** j)

Sigma = DCT @ delta @ DCT.T
sigma_eps = 0.25
beta_tide = np.array([0.2178] * 20 + [0.0218] * 108)
beta = DCT @ beta_tide
gamma = [0.001, 0.1, 1]

def generate_x(Sigma=Sigma, num_samples=1):
    L = np.linalg.cholesky(Sigma)
    dim = Sigma.shape[0]
    z = np.random.normal(size=(num_samples, dim))
    x = z @ L.T
    return x


def generate_epsilon(mean=0, v=0.25):
    std_dev = np.sqrt(v)
    scalar = np.random.normal(loc=mean, scale=std_dev)
    return scalar


data_size = 32
train_matrix = np.zeros((data_size, d))
sample_out = np.zeros((data_size, 1))

for i in range(data_size):
    vec = generate_x()
    train_matrix[i, :] = vec
    y = vec @ beta + generate_epsilon()
    sample_out[i, :] = y

print("rank of X is " + str(np.linalg.matrix_rank(train_matrix)))

test_size = 1000
test_matrix = np.zeros((test_size, d))
test_out = np.zeros((test_size, 1))

for i in range(test_size):
    vec = generate_x()
    test_matrix[i, :] = vec
    eps = generate_epsilon()
    y = (vec @ beta) + eps
    test_out[i, :] = y
# create U
errors = []
errors_ridge_small = []
errors_ridge_medium = []
errors_ridge_big = []
params = [i for i in range(1, d + 1)]
for p in range(1, d + 1):
    U = np.zeros((d, p))
    for i in range(p):
        column = DCT[:, i]
        U[:, i] = column
    tetha = train_matrix @ U
    hasInv = np.linalg.det(tetha.T @ tetha) != 0
    eye_p = np.identity(p)
    tetha_norm = tetha.T @ tetha
    tetha_sample_out = tetha.T @ sample_out
    matrix_save = U.T @ test_matrix.T
    if hasInv:
        alpha = np.linalg.inv(tetha_norm) @ tetha_sample_out
    else:
        alpha = np.linalg.pinv(tetha_norm) @ tetha_sample_out
    alpha_ridge_small = np.linalg.inv(tetha_norm + eye_p * gamma[0]) @ tetha_sample_out
    alpha_ridge_medium = np.linalg.inv(tetha_norm + eye_p * gamma[1]) @ tetha_sample_out
    alpha_ridge_big = np.linalg.inv(tetha_norm + eye_p * gamma[2]) @ tetha_sample_out
    test_err = (np.linalg.norm((alpha.T @ matrix_save) - test_out.T) ** 2) / test_size
    small_err = (np.linalg.norm((alpha_ridge_small.T @ matrix_save) - test_out.T) ** 2) / test_size
    medium_err = (np.linalg.norm((alpha_ridge_medium.T @ matrix_save) - test_out.T) ** 2) / test_size
    big_err = (np.linalg.norm((alpha_ridge_big.T @ matrix_save) - test_out.T) ** 2) / test_size
    errors.append(test_err)
    errors_ridge_small.append(small_err)
    errors_ridge_medium.append(medium_err)
    errors_ridge_big.append(big_err)

plt.plot(params, errors, color='b', label='least squares')
plt.plot(params, errors_ridge_small, color='r', label='gamma=0.001')
plt.plot(params, errors_ridge_medium, color='g', label='gamma=0.1')
plt.plot(params, errors_ridge_big, color='m', label='gamma=1')
plt.ylim(0, 5)
plt.title('Errors dependent on Parameters in Least Squares')
plt.xlabel('Parameters')
plt.ylabel('Error')
plt.legend()
plt.show()

K = 500  # times to repeat the experiment
# create vector to store the errors
avg_errors = np.zeros(d)
avg_errors_small = np.zeros(d)
avg_errors_medium = np.zeros(d)
avg_errors_big = np.zeros(d)
#bias = np.zeros(d)
#variance = np.zeros(d)
Fs = np.zeros((K, test_size))

for p in range(1, d + 1):
    print(f'iter:{p}')
    Fx_avg = np.zeros((1, test_size))
    U = np.zeros((d, p))
    for i in range(p):
        column = DCT[:, i]
        U[:, i] = column
    for j in range(K):
        # generate training set
        train_matrix = np.zeros((data_size, d))
        sample_out = np.zeros((data_size, 1))
        for i in range(data_size):
            vec = generate_x()
            train_matrix[i, :] = vec
            y = vec @ beta + generate_epsilon()
            sample_out[i, :] = y
        tetha = train_matrix @ U
        hasInv = np.linalg.det(tetha.T @ tetha) != 0
        eye_p = np.identity(p)
        tetha_norm = tetha.T @ tetha
        tetha_sample_out = tetha.T @ sample_out
        matrix_save = U.T @ test_matrix.T
        if hasInv:
            alpha = np.linalg.inv(tetha_norm) @ tetha_sample_out
        else:
            alpha = np.linalg.pinv(tetha_norm) @ tetha_sample_out
        #curr_f = (alpha.T @ U.T @ test_matrix.T)
        #test_err = (np.linalg.norm(curr_f - test_out.T) ** 2) / test_size
        alpha_ridge_small = np.linalg.inv(tetha_norm + eye_p * gamma[0]) @ tetha_sample_out
        alpha_ridge_medium = np.linalg.inv(tetha_norm + eye_p * gamma[1]) @ tetha_sample_out
        alpha_ridge_big = np.linalg.inv(tetha_norm + eye_p * gamma[2]) @ tetha_sample_out
        test_err = (np.linalg.norm((alpha.T @ matrix_save) - test_out.T) ** 2) / test_size
        small_err = (np.linalg.norm((alpha_ridge_small.T @ matrix_save) - test_out.T) ** 2) / test_size
        medium_err = (np.linalg.norm((alpha_ridge_medium.T @ matrix_save) - test_out.T) ** 2) / test_size
        big_err = (np.linalg.norm((alpha_ridge_big.T @ matrix_save) - test_out.T) ** 2) / test_size
        avg_errors[p - 1] += (test_err / K)
        avg_errors_small[p - 1] += (small_err / K)
        avg_errors_medium[p - 1] += (medium_err / K)
        avg_errors_big[p - 1] += (big_err / K)
        #Fx_avg += (curr_f / K)
        #Fs[j, :] = curr_f
    #variance[p - 1] = np.sum(np.linalg.norm(Fs - Fx_avg, axis=1) ** 2) / (K * test_size)
    #Fx_avg = np.linalg.norm((Fx_avg - (test_matrix @ beta))) ** 2
    #bias[p - 1] = Fx_avg / test_size

plt.plot(params, avg_errors, color='b', label='Average Test Error')
plt.plot(params, avg_errors_small, color='r', label='gamma=0.001')
plt.plot(params, avg_errors_medium, color='g', label='gamma=0.1')
plt.plot(params, avg_errors_big, color='m', label='gamma=1')
#plt.plot(params, bias, color='c', ls='--', dashes=(5, 5), label='Bias Squared')
#plt.plot(params, variance, color='y', ls='--', dashes=(5, 5), label='Variance')
plt.title('Error, ridge regression error dependent on Parameters in Least Squares')
plt.xlabel('Parameters')
plt.ylim(0, 5)
plt.legend()
plt.show()
