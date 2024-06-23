from scipy.fft import dct
import numpy as np

# generate input like so: x ~ N(0, 𝚺), 𝚺 = DCT * delta * DCT(T), y = 𝐱(𝑇) * 𝛃 + ϵ, ϵ ~ N(0, 0.25)

C = dct(np.eye(d), axis=0, norm='ortho')
d = 128

delta = np.zeros((d, d))
for j in range(d):
    delta[j, j] = 3 * (0.9 ** j)

