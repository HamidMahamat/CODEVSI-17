import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import optimize
import scipy.special as sp
from functools import partial

c0 = 299792458
eps_t = None#10
eps_l = None# 10
d = None #0.016
n = None #1


def mouse_move(event):
    x, y = event.xdata, event.ydata
    print(x, y)

def plot_unit_circle():
    angs = np.linspace(0, 2 * np.pi, 10**6)
    rs = np.zeros_like(angs) + 1
    xs = rs * np.cos(angs)
    ys = rs * np.sin(angs)
    plt.plot(xs, ys)



# Implicit function of the dispersion's equation
def f(beta, k):
    kce = np.sqrt((k ** 2) * eps_t - (beta ** 2))
    kcm = kce * np.sqrt(eps_l / eps_t)
    h_air = np.sqrt(beta ** 2 - k ** 2)
    X = (d / 2) * h_air
    Y_e = (d / 2) * kce
    #Y_m = (d / 2) * kcm

    DTE = ((sp.kvp(n, X, 1)) / (X * sp.kn(n, X))) + ((sp.jvp(n, Y_e, 1)) / (Y_e * sp.jn(n, Y_e)))
    DTM = ((sp.kvp(n, X, 1)) / (X * sp.kn(n, X))) + (eps_l * (sp.jvp(n, Y_e, 1)) / (Y_e * sp.jn(n, Y_e)))
    #DH = (n ** 2) * ((1 / X ** 2) + (1 / Y_e ** 2)) * ((1 / X ** 2) + (eps_l / Y_m ** 2))

    return [DTE, DTM]#, DH]


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


def X_dte(idx,Beta_te):
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


def Y_dte(idx,k,Nk,Beta_te):
    x = X_dte(idx,Beta_te)
    return k[Nk - 1 - len(x)::]


def X_dtm(idx,Beta_tm,):
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


def Y_dtm(idx,k,Nk,Beta_tm):
    x = X_dtm(idx,Beta_tm)
    return k[Nk - 1 - len(x)::]

def main(esp_t_p,esp_l_p,d_p,n_p,f_max,index, list_te_tm):
    global eps_l,eps_t,d,n
    eps_t = esp_t_p#10
    eps_l = esp_l_p# 10
    d = d_p #0.016
    n = n_p #1
    u=9 # unit Giga
    Nk = 300  # Discretization
    k_min = 10
    k_max = (2*scipy.pi*f_max*(10**u))/(299792458)

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



    u=9 # unit Giga
    plt.figure()
    plt.xlabel(r'$\beta$ [m$^{-1}$]')
    plt.grid(visible=True, which='both', linestyle='--')
    plt.ylabel('$f$ [GHz]')
    #plt.ylim([0, k_max * (c0/(2*10**u*scipy.pi))])
    plt.ylim([0, k_max*(299792458)/(2*scipy.pi*(10**u))])
    plt.xlim([0, k_max * np.sqrt(eps_t)])
    # -- limits domain
    plt.axline((0, 0), slope=(c0/(2*10**u*scipy.pi)), c='red')
    plt.axline((0, 0), slope=(c0/(2*10**u*scipy.pi)) / np.sqrt(eps_t), c='red')

    for i in list_te_tm:
        plt.plot(X_dte(int(i),Beta_te), (299792458/(2*scipy.pi*(10**u)))*Y_dte(int(i),k,Nk,Beta_te), label=f'TE$_{{{int(n)}}}$,$_{{{i}}}$')
        plt.plot(X_dtm(int(i),Beta_tm), (299792458/(2*scipy.pi*(10**u)))*Y_dtm(int(i),k,Nk,Beta_tm), label=f'TM$_{{{int(n)}}}$,$_{{{i}}}$')


    """plt.plot(X_dte(1,Beta_te), (c0/(2*10**u*scipy.pi))*Y_dte(1,k,Nk,Beta_te), label='TE1')
    plt.plot(X_dte(2,Beta_te), (c0/(2*10**u*scipy.pi))*Y_dte(2,k,Nk,Beta_te), label='TE2')
    plt.plot(X_dte(3,Beta_te), (c0/(2*10**u*scipy.pi))*Y_dte(3,k,Nk,Beta_te), label='TE3')

    plt.plot(X_dtm(0,Beta_tm), (c0/(2*10**u*scipy.pi))*Y_dtm(0,k,Nk,Beta_tm), label='TM0')
    plt.plot(X_dtm(1,Beta_tm), (c0/(2*10**u*scipy.pi))*Y_dtm(1,k,Nk,Beta_tm), label='TM1')
    plt.plot(X_dtm(2,Beta_tm), (c0/(2*10**u*scipy.pi))*Y_dtm(2,k,Nk,Beta_tm), label='TM2')
    plt.plot(X_dtm(3,Beta_tm), (c0/(2*10**u*scipy.pi))*Y_dtm(3,k,Nk,Beta_tm), label='TM3')"""

    plt.legend()
    plt.connect('motion_notify_event', mouse_move)
    plot_unit_circle()
    plt.savefig(".images/graph"+str(index)+".png", format="png", bbox_inches="tight")
    #plt.show()
