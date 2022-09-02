import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import optimize
import scipy.special as sp
from functools import partial

c0 = 3e8
eps_t = 10
eps_l = 10
d = 0.016
n = 1


# Implicit function of the dispersion's equation
def f(beta, k):
    kce = np.sqrt((k ** 2) * eps_t - (beta ** 2))
    kcm = kce * np.sqrt(eps_l / eps_t)
    h_air = np.sqrt(beta ** 2 - k ** 2)
    X = (d / 2) * h_air
    Y_e = (d / 2) * kce
    Y_m = (d / 2) * kcm

    DTE = ((sp.kvp(n, X, 1)) / (X * sp.kn(n, X))) + ((sp.jvp(n, Y_e, 1)) / (Y_e * sp.jn(n, Y_e)))
    DTM = ((sp.kvp(n, X, 1)) / (X * sp.kn(n, X))) + (eps_l * (sp.jvp(n, Y_e, 1)) / (Y_e * sp.jn(n, Y_e)))
    DH = (n ** 2) * ((1 / X ** 2) + (1 / Y_e ** 2)) * ((1 / X ** 2) + (eps_l / Y_m ** 2))

    return [DTE, DTM, DH]


def dte(beta, k):
    return f(beta, k)[0]


def dtm(beta, k):
    return f(beta, k)[1]


def zero_deepfinder(f, a, b, xtol, Niter):
    U = np.linspace(a, b, Niter)  # evaluate function at 100 different points
    c = f(U)
    s = np.sign(c)
    res = []
    for i in range(Niter - 1):
        if s[i] + s[i + 1] == 0:  # oposite signs
            u = scipy.optimize.brentq(f, U[i], U[i + 1])
            z = f(u)
            if np.isnan(z) or (abs(z) > 1e-3 and abs(f(u - xtol / 2)) > abs(f(u - xtol))):
                continue
            # print('found zero at {}'.format(u))
            res.append(u)
    return res


Nk = 300  # Discretization
k_min = 10
k_max = 1000

k = np.linspace(k_min, k_max, Nk)

k = np.delete(k, -1, axis=0)  # The strict inequality implies to remove the last element

xtol = 0.0001
Beta_te = []
Beta_tm = []
#Beta_th=[]
for k0 in k:
    #Niter = int(1/2 * k0 * (np.sqrt(eps_t) - 1))
    Niter=100
    z_te = zero_deepfinder(partial(dte, k=k0), k0 * (10001 / 10000), k0 * np.sqrt(eps_t) * (9999 / 10000), xtol, Niter)
    z_tm = zero_deepfinder(partial(dtm, k=k0), k0 * (10001 / 10000), k0 * np.sqrt(eps_t) * (9999 / 10000), xtol, Niter)
    #z_th = zero_deepfinder(partial(dth, k=k0), k0 * (10001 / 10000), k0 * np.sqrt(eps_t) * (9999 / 10000), xtol, Niter)
    Beta_tm.append(z_tm)
    Beta_te.append(z_te)
    #Beta_th.append(z_th)



def X_dte(idx):
    x = []
    if len(Beta_te[-1]) < idx + 1:
        raise ValueError("this mode doesn't exist")
    else:
        for i in range(len(Beta_te)):
            if len(Beta_te[i]) <= idx:
                pass
            else:
                x.append(Beta_te[i][-idx - 1])
    return x


def Y_dte(idx):
    x = X_dte(idx)
    return k[Nk - 1 - len(x)::]


def X_dtm(idx):
    x = []
    if len(Beta_tm[-1]) < idx + 1:
        raise ValueError("this mode doesn't exist")
    else:
        for i in range(len(Beta_tm)):
            if len(Beta_tm[i]) <= idx:
                pass
            else:
                x.append(Beta_tm[i][-idx - 1])
    return x


def Y_dtm(idx):
    x = X_dtm(idx)
    return k[Nk - 1 - len(x)::]

u=9 # unit Giga
plt.figure()
plt.xlabel(r'$\beta$')
plt.grid(visible=True, linestyle='--')
plt.ylabel('$f$ [GHz]')
plt.ylim([0, k_max * (c0/(2*10**u*scipy.pi))])
plt.xlim([0, k_max * np.sqrt(eps_t)])
# -- limits domain
plt.axline((0, 0), slope=(c0/(2*10**u*scipy.pi)), c='red')
plt.axline((0, 0), slope=(c0/(2*10**u*scipy.pi)) / np.sqrt(eps_t), c='red')

plt.plot(X_dte(0), (c0/(2*10**u*scipy.pi))*Y_dte(0), label='TE0')
plt.plot(X_dte(1), (c0/(2*10**u*scipy.pi))*Y_dte(1), label='TE1')
plt.plot(X_dte(2), (c0/(2*10**u*scipy.pi))*Y_dte(2), label='TE2')
plt.plot(X_dte(3), (c0/(2*10**u*scipy.pi))*Y_dte(3), label='TE3')

plt.plot(X_dtm(0), (c0/(2*10**u*scipy.pi))*Y_dtm(0), label='TM0')
plt.plot(X_dtm(1), (c0/(2*10**u*scipy.pi))*Y_dtm(1), label='TM1')
plt.plot(X_dtm(2), (c0/(2*10**u*scipy.pi))*Y_dtm(2), label='TM2')
plt.plot(X_dtm(3), (c0/(2*10**u*scipy.pi))*Y_dtm(3), label='TM3')

plt.legend()
plt.savefig("graph.pdf", format="pdf", bbox_inches="tight")
plt.show()