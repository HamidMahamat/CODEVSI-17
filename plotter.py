import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from functools import partial

eps_t = 10
eps_l = 10
d = 0.016
n = 0


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


# Zero research with secants method
def zero(f, a, b, tol, Niter):
    j = 0
    if f(a) * f(b) >= 0:
        print("Secant method fails. The function must be strictly monotonous in the interval")
        return None
    else:
        an = a
        bn = b
        while abs(bn - an) > tol and j < Niter:
            j = j + 1
            mn = an - f(an) * (bn - an) / (f(bn) - f(an))
            f_mn = f(mn)
            if f(an) * f_mn < 0:
                an = an
                bn = mn
            elif f(bn) * f_mn < 0:
                an = mn
                bn = bn
            elif f_mn == 0:
                print("Found exact solution.")
                break
            else:
                print("Secant method fails.")
                return None
    return an - f(an) * (bn - an) / (f(bn) - f(an))


# Domain issues : points for which the bessel functions are defined and real
def isindomain(beta, k):
    return (beta > k) & (beta < np.sqrt(eps_t) * k)


Nb = 800  # Number of columns
Nk = 300  # Number of rows

# Maximum frequency in GHz
##f_max = 2
k_min = 230 #(2*z_min/(d*sqrt(eps_t-1)))*(1000001/1000000) where z_min is the second zero of J_n
k_max = 500

DATA = np.zeros((Nk, Nb))  # The first row will contain only zeros (if we put the matrix' shape [Nk,Nb]) that's why Nk-1

k = np.linspace(k_min, k_max, Nk)

for i in range(Nk):
    DATA[i] = np.linspace(k[i], np.sqrt(eps_t) * k[i], Nb)

DATA = np.delete(DATA, 0, axis=0)  # Delete the first row containing only zeros
DATA = np.delete(DATA, -1, axis=1)  # The strict inequality implies to remove the last column


# This function return the range of bessel function's zeros jn for a given k_0
def zeros_besselj(n, k):
    j = 1
    res = sp.jn_zeros(n, j)
    if k * (d / 2) * np.sqrt(eps_t - 1) < res:
        return []
    else:
        while k * (d / 2) * np.sqrt(eps_t - 1) > res[-1]:
            j = j + 1
            res = sp.jn_zeros(n, j)
        return res[0:-1]


# --- Test (To have an idea of the DTE_k(beta) curve's shape)
# plt.figure()
# z=zeros_besselj(0,3000)
# x1=np.sqrt((3000**2)*eps_t-((2/d)*z)**2)
# y1=f(x1, DATA[-50, 0])[0]
# plt.scatter(x1,y1, c='red', linewidths=2)
#     # plt.xlim([k_min, k_max * sqrt(eps_t)])
# plt.xlabel(r'$\beta$')
# plt.ylim([-100, 100])
# x = DATA[-50, 1:] # k_0=3000
# y = f(x, DATA[-50, 0])[0]
# plt.ylabel(r'$DTE_{k_0}(\beta)$')
# plt.plot(x, y)
# plt.axline((0,0),(1,0))
# plt.show()
# --- end of the test


# plt.figure()
# plt.xlim([0,100])
# plt.xlabel(r'$\beta$')
# plt.ylim([0,100])
# plt.ylabel('$\omega$')
# plt.axline((0,0), slope=1, c='red')
# plt.axline((0,0), slope=1/np.sqrt(eps_t), c='red')
# plt.show()


# first test
Beta_toplot = []
for k0 in DATA[:, 0]:
    kcible = []
    z = zeros_besselj(n, k0)
    x = np.sqrt((k0 ** 2) * eps_t - (4 * z**2/d**2))
    x = x[::-1]
    for i in range(len(x) - 1):
        kcible.append(
            zero(partial(dte, k=k0), x[i] + (x[i + 1] - x[i]) / 2000, x[i + 1] - (x[i + 1] - x[i]) / 2000, 0.01, 100))
    Beta_toplot.append(kcible)

plt.figure()
# plt.xlim([k_min, k_max * sqrt(eps_t)])
plt.xlabel(r'$\beta$')
y = DATA[:, 0]  # k_0=3000
x=[]
for i in range(len(Beta_toplot)):
    # if Beta_toplot[i] == [] :
    #     pass
    # else :
        x.append(Beta_toplot[i][-1])

plt.ylabel(r'$k$')
plt.plot(x, y)
#plt.plot(x1, y, c='green')

plt.axline((0, 0), slope=1, c='red')

plt.axline((0, 0), slope=1 / np.sqrt(eps_t), c='red')
plt.show()
