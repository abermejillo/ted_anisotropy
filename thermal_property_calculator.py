"""
This script contains functions for the calculation of the specific heat and thermal conductivity.

by Álvaro
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve

# constants
k_b = 1.380649*10**(-23) # J / K
h_bar = 1.054571*10**(-34) # J s
N_A = 6.022*10**(23)
gamma = 4.5/2*1.76*10**(11)*h_bar # J / T
mu_0 = 4*np.pi*10**(-7) # T m / A
mu_b = 9.27*10**(-24)

def c_einstein(T, T_einstein):
    '''
    Computes the specific heat in a temperature range according to the Einstein model.
    :param T: np.array
        Temperature points
    :param T_einstein: float
        Einstein temperature
    :return: np.array
        Array containing the specific heat per number of atoms in international units [J/K]
    '''
    theta = T_einstein / T
    c_ei = 3*k_b*theta**2 * np.exp(theta) / (np.exp(theta)-1)**2
    return c_ei


def integrand_debye(q):
    """
    Integrand to compute the specific heat according to Debye's model.
    :param q: float
        Crystal momentum
    :return: float
        Value of the integrand
    """
    integrand = q**3 * np.exp(q) / (np.exp(q) - 1)**2
    return integrand


def c_debye(T, T_debye):
    """
    Computes the specific heat in a temperature range according to the Debye model.
    :param T: np.array
        Temperature points
    :param T_debye: float
        Debye temperature
    :return: np.array
        Array containing the specific heat per number of atoms in international units [J/K]
    """
    T = np.array(T)
    constant = 6 * k_b * (T / T_debye) ** 2
    if T.size == 1:  # some condition to deal with T being a float or an array. The function works with both.
        integral = integrate.quad(integrand_debye, 0, T_debye/T)[0]
        return constant*integral
    else:
        integral = []
        for t in T:
            integral.append(integrate.quad(integrand_debye, 0, T_debye/t)[0])
        return constant*integral


def c_comb(T, T_debye, T_einstein, x, n):
    """
    Computes the specific heat in a temperature range as a combination of the Debye and Einstein models.
    :param T: np.array
        Temperature points
    :param T_debye: float
        Debye temperature
    :param T_einstein: float
        Einstein temperature
    :param x: float
        Fraction of Einstein participation to the specific heat
    :param n: int
        Number of atoms per molecule
    :return: np.array
        Array containing the specific heat per mol in international units [J/molK]
    """
    c_db = c_debye(T, T_debye) * n * N_A
    c_ei = c_einstein(T, T_einstein) * n * N_A
    c_cb = (1 - x) * c_db + x * c_ei
    return c_cb


def integrand_Npho(q, T, T_db, v):
    """
    Integrand for the computeation of the number of magnons.
    :param q: float
        Crystal momentum
    :param T: float
        Temperature
    :param T_N: float
        Neel temperature
    :param H_E: float
        Exchange effective magnetic field
    :param H_A: float
        Anisotropy effective magnetic field
    :return: float
        Integrand value
    """
    kdb = k_b*T_db/(h_bar*v)
    factor = kdb**2/(2*np.pi)
    nq = 1 / (np.exp(q*T_db/T) - 1)

    return factor*q*nq

def n_phonon(T, T_db, v):
    """
    Computes the number of magnons.
    :param T: np.array
        Temperature points
    :param T_N: float
        Neel temperature
    :param H_E: float
        Exchange effective magnetic field
    :param H_A: float
        Anisotropy effective magnetic field
    :return: np.array
        Number of magnons at each temperature point
    """
    integral = []
    for i, t in enumerate(T):
        print('\rN_mag -> Temperature step:', t, end='')
        integral.append(integrate.quad(integrand_Npho, 0, 1, args=(t, T_db, v))[0])
    return integral


def k(T, T_debye, T_einstein, x, n, v, tau_ph):
    """
    Computes the thermal conductivity in a temperature range from the specific heat calculated as a Debye Einstein
    mixture.
    :param T: np.array
        Temperature points
    :param T_debye: Float
        Debye temperature
    :param T_einstein: Float
        Einstein temperature
    :param x: float
        Fraction of the Einstein participation to the specific heat
    :param n: int
        Number of atoms per molecule
    :param v: float
        Average phonon velocity
    :param tau_ph: float
        Average phonon lifetime
    :return: np.array
        Array containing the thermal conductivity in international units [W/mK]
    """
    c = c_comb(T, T_debye, T_einstein, x, n)
    kappa = c*v**2*tau_ph
    return kappa


def k_elastic(T, T_debye, T_einstein, x, n, v, tau_ph, thickness):
    """
    Computes the thermal conductivity in a temperature range from the specific heat calculated as a Debye Einstein
    mixture.
    :param T: np.array
        Temperature points
    :param T_debye: Float
        Debye temperature
    :param T_einstein: Float
        Einstein temperature
    :param x: float
        Fraction of the Einstein participation to the specific heat
    :param n: int
        Number of atoms per molecule
    :param v: float
        Average phonon velocity
    :param tau_ph: float
        Average phonon lifetime
    :return: np.array
        Array containing the thermal conductivity in international units [W/mK]
    """
    k_db = k_b * T_debye / (h_bar * v)
    N_db = k_db ** 2 / (4 * np.pi)
    c = c_comb(T, T_debye, T_einstein, x, n)
    kappa = c*v**2*tau_ph*N_db/(thickness*5*N_A)
    return kappa

def integrand_ku(q, T, T_db, v, a, tau_ph, gamma_E, m):
    k_db = k_b * T_db / (h_bar * v)
    w_q = 2*v/a * np.sin(np.pi*q/2)
    w_db = 2*v/a

    T_U = gamma_E**2 * k_b *T / (m*v**2*w_db)
    tau_k = (1/tau_ph + T_U*w_q**2)**-1
    print((T_U*w_q**2)**-1, tau_k, w_q)

    f_BE = np.exp(q) / (np.exp(q) - 1)**2
    f_BE = np.exp(h_bar * w_q / (k_b * T)) / (np.exp(h_bar * w_q / (k_b * T)) - 1) ** 2

    integrand = q*w_q**2*tau_k*f_BE
    return integrand


def k_umklapp(T, T_debye, v, a, tau_ph,gamma_E, m, thickness):
    k_db = k_b * T_debye / (h_bar * v)
    factor = k_db**2*h_bar**2/(4*np.pi) * 3*v/(k_b*T**2)
    integral = []
    for i, t in enumerate(T):
        integral.append(integrate.quad(integrand_ku, 0, 1, args=(t, T_debye, v, a, tau_ph, gamma_E, m))[0])

    kappa = factor*integral/thickness
    return kappa

def c_debye_fit(t, T_debye, n=3):
    """
    Function to perform fits with the Debye model. Change the value of n here.
    :param t: float
        Temperature
    :param T_debye: float
        Debye temperature
    :param n: int
        Number of atoms per molecule
    :return: float
        Specific heat at temperature t per mol in international units [J/molK]
    """
    c_db = c_debye(t, T_debye) * n * N_A
    return c_db


def c_fit(t, T_debye, T_einstein, x, n=3):
    """
    Function to perform fits with a Debye Einstein mixture model. Change the value of n here.
    :param t: float
        Temperature
    :param T_debye: float
        Debye temperature
    :param T_einstein: float
        Einstein temperature
    :param x: float
        Fraction of the Einstein participation to the specific heat
    :param n: int
        Number of atoms per molecule
    :return: float
        Specific heat at temperature t per mol in international units [J/molK]
    """
    c_db = c_debye(t, T_debye) * n * N_A
    c_ei = c_einstein(t, T_einstein) * n * N_A
    c_cb = (1 - x) * c_db + x * c_ei
    return c_cb

# Here start the magnetic contributions

def langevin(x, T):
    """
    Implicit equation for the computation of the temperature dependence of the order parameter.
    :param x: float
        Order parameter
    :param T: float
        Temperature
    :return:
        Value of the implicit equation
    """
    return np.tanh(x/T) - x

def BJ(x, J):
    return (2*J + 1)/(2*J)*(np.tanh((2*J + 1)*x/(2*J)))**(-1) - 1/(2*J)*(np.tanh(x/(2*J)))**(-1)

def brillouin_func(x, T, J, g, n, TN):
    meff = g * mu_b * (J * (J + 2)) ** 0.5
    C = mu_0 * n * meff ** 2 / (3 * k_b)
    nw = TN / C
    brillouin = BJ(x, J)
    return brillouin - T*(J+1)*x/(3*J*C*nw)

def t_dep(T, T_N):
    """
    Temperature dependence of the order parameter (spontaneous magnetization)
    :param T: np.array or float
        Temperature points
    :param T_N: float
        Neel temperature
    :return:
        Value of the order parameter at temperatures T
    """
    T = np.array(T)
    if T.size == 1: # some condition to deal with T being a float or an array. The function works with both.
        return fsolve(langevin, [0, 1], (T/T_N))[1]
    else:
        m = []
        for t in T:
            if t>=T_N:
                m.append(0)
            else:
                m.append(fsolve(langevin, [0, 1], (t/T_N))[1])
        return np.array(m)

def t_dep_BJ(T, T_N, J):
    T = np.array(T)
    g, n = 1, 1
    if T.size == 1:  # some condition to deal with T being a float or an array. The function works with both.
        if T>=T_N:
            return 0
        else:
            return BJ(fsolve(brillouin_func, [0.01, T_N/T**0.5], (T, J, g, n, T_N))[1],J)
    else:
        m = []
        for t in T:
            if t >= T_N:
                m.append(0)
            else:
                m.append(BJ(fsolve(brillouin_func, [0.01, T_N/t**0.5], (t, J, g, n, T_N))[1],J))
        return np.array(m)

def p_theta(theta1, theta2):
    """
    Integrand parameter for the Ising contribution
    :param theta1: float
        Angle 1
    :param theta2: flaot
        Angle 2
    :return:
        Value of the parameter
    """
    return np.cos(theta1) + np.cos(theta2) + np.cos(theta1 + theta2)

def integrand_ising(theta1, theta2, K):
    """
    Inegrand for the computation of the Ising specific heat
    :param theta1: float
        Angle 1
    :param theta2: float
        Angle 2
    :param K: float
        Normalized temperature
    :return: float
        Value of the integrand
    """
    num1 = 6*np.sinh(4*K)*np.sinh(2*K)-4*np.cosh(4*K)*(2*p_theta(theta1, theta2)-3*np.cosh(2*K))
    den1 = np.cosh(2*K)**3 + 1 - np.sinh(2*K)**2*p_theta(theta1, theta2)
    num2 = np.sinh(4*K)**2*(2*p_theta(theta1, theta2) - 3*np.cosh(2*K))**2
    den2 = (np.cosh(2*K)**3 + 1 - np.sinh(2*K)**2*p_theta(theta1, theta2))**2

    return num1/den1 - num2/den2

def integrand_ising_TN(theta1, theta2):
    return  24 / (3 - p_theta(theta1, theta2)) - 8/3

def c_ising(T, T_N, n):
    """
    Computes the Ising contribution to the specific heat.
    :param T: np.array
        Temperature points
    :param T_N: float
        Neel temperature
    :return: np.array
        Specific heat per mol in international units [J/molK]
    """
    J = T_N * k_b * np.log(2 + np.sqrt(3)) / 2
    K = J/(k_b*T)
    factor = K**2*k_b*n*N_A/(16*np.pi**2)
    integral = []
    for i,t in enumerate(T):
        print('\rc_ising -> Temperature step:', t, end='')
        if t==T_N:
            integral.append(integrate.dblquad(integrand_ising_TN, 0, 2 * np.pi, 0, 2 * np.pi)[0])
        else:
            integral.append(integrate.dblquad(integrand_ising, 0, 2*np.pi, 0, 2*np.pi, args=[K[i]])[0])
    print('')
    return factor*integral

def integrand_cmag(q, T, T_N, H_E, H_A, J):
    """
    Integrand for the magnon contribution to the specific heat.
    :param q: float
        Crystal momentum
    :param T: float
        Temperature
    :param T_N: float
        Neel temperature
    :param H_E: float
        Exchange effective magnetic field
    :param H_A: float
        Anisotropy effective magnetic field
    :return: float
        Value of the integrand
    """
    etha = H_A / H_E
    wq = gamma*mu_0*H_E*t_dep_BJ(T, T_N, J)/h_bar * np.sqrt(np.sin(q*np.pi/2)**2 + etha**2 + 2*etha)
    if h_bar*wq/(k_b*T) == 0:
        return 0
    else:
        nq = np.exp(h_bar*wq/(k_b*T)) / (np.exp(h_bar*wq/(k_b*T)) - 1)**2
        integrand = q * wq**2 * nq
        return integrand

def c_magnon(T, T_N, H_E, H_A, J):
    """
    Magnon contribution to the specific heat.
    :param T: np.array
        Temperature points
    :param T_N: float
        Neel temperature
    :param a: float
        Lattice parameter
    :param H_E: float
        Exchange effective magnetic field
    :param H_A:
        Anisotropy effective magnetic field
    :return: np.array
        Specific heat per mol in international units [J/molK]
    """
    factor = 2*h_bar**2/(k_b*T**2)*N_A
    integral = []
    for i,t in enumerate(T):
        print('\rc_magnon -> Temperature step:', t, end='')
        integral.append(integrate.quad(integrand_cmag,0,1,args=(t, T_N, H_E, H_A, J))[0])
    print('')
    return factor*np.array(integral)

def integrand_Nmag(q,T, T_N, a, H_E, H_A, J):
    """
    Integrand for the computeation of the number of magnons.
    :param q: float
        Crystal momentum
    :param T: float
        Temperature
    :param T_N: float
        Neel temperature
    :param H_E: float
        Exchange effective magnetic field
    :param H_A: float
        Anisotropy effective magnetic field
    :return: float
        Integrand value
    """
    km = np.sqrt(np.pi) * 2 / a
    factor = km**2/(2*np.pi)
    etha = H_A / H_E
    wq = gamma * mu_0 * H_E * t_dep_BJ(T, T_N, J) / h_bar * np.sqrt(np.sin(q * np.pi / 2) ** 2 + etha ** 2 + 2 * etha)
    if (h_bar * wq / (k_b * T) == 0) or (T == 114):
        nq = 0
    else:
        nq = 1 / (np.exp(h_bar * wq / (k_b * T)) - 1)

    return factor*q*nq

def n_magnon(T, T_N, a, H_E, H_A, J):
    """
    Computes the number of magnons.
    :param T: np.array
        Temperature points
    :param T_N: float
        Neel temperature
    :param H_E: float
        Exchange effective magnetic field
    :param H_A: float
        Anisotropy effective magnetic field
    :return: np.array
        Number of magnons at each temperature point
    """
    integral = []
    for i, t in enumerate(T):
        print('\rN_mag -> Temperature step:', t, end='')
        integral.append(integrate.quad(integrand_Nmag, 0, 1, args=(t, T_N, a, H_E, H_A, J))[0])
    return integral

def integrand_kmag(q, T, T_N, H_E, H_A, J):
    """
    Integrand for magnon contribution to thermal conductivity.
    :param q: float
        Crystal momentum
    :param T: float
        Temperature
    :param T_N: float
        Neel temperature
    :param H_E: float
        Exchange effective magnetic field
    :param H_A: float
        Anisotropy effective magnetic field
    :return: float
        Integrand value
    """
    etha = H_A / H_E
    wq = gamma*mu_0*H_E*t_dep_BJ(T, T_N, J)/h_bar * np.sqrt(np.sin(q*np.pi/2)** 2 + etha**2 + 2*etha)
    if (h_bar * wq / (k_b * T) == 0):
        nq = 0
    else:
        nq = np.exp(h_bar*wq / (k_b*T)) / (np.exp(h_bar*wq / (k_b*T)) - 1)**2
    vel = np.sin(q*np.pi)**2 / (np.sin(q*np.pi/2)**2 + etha**2 + 2*etha)

    return q * wq**2 * nq * vel

def k_magnon(T, T_N, H_E, H_A, J, tau_mag):
    """
    Computes the manon contribution to the thermal conductivity.
    :param T: np.array
        Temperature points
    :param T_N: float
        Neel temperature
    :param H_E: float
        Exchange effective magnetic field
    :param H_A: float
        Anisotropy effective magnetic field
    :param tau_mag: float
        Magnon average lifetime
    :return: np.array
        Magnon contribution to the thermal conductivity
    """
    factor = (gamma*mu_0*H_E*t_dep_BJ(T, T_N, J)/8)**2 * np.pi/(k_b*T**2) * tau_mag
    integral = []
    for i, t in enumerate(T):
        print('\rK_mag -> Temperature step:', t, end='')
        integral.append(integrate.quad(integrand_kmag, 0, 1, args=(t, T_N, H_E, H_A, J))[0])
    print('')
    return factor * np.array(integral)

def k_mag(T, T_N, a, H_E, H_A, J, v, tau):
    c_mag = c_magnon(T, T_N, a, H_E, H_A, J)
    km = np.sqrt(np.pi) * 2 / a
    N = km ** 2 / (4 * np.pi)
    c_mag = c_mag*N/N_A
    k = c_mag * v**2 * tau
    return k


