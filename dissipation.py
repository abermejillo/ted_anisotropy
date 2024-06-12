"""
This script contains functions for the calculation of the dissipation from the Lifschitz landau model (Xiang).

by Álvaro
"""
import numpy as np
import scipy.special as sp
from scipy.optimize import fsolve

integral_alpha = np.loadtxt('data/integral_alpha.csv', delimiter=',')
T_integral_alpha = np.loadtxt('data/T_integral_alpha.csv', delimiter=',')

def tension_thick(integral_alpha, thickness, E, nu):
    return -E*thickness/(1-nu)*integral_alpha

def system_ab(x, D, a, F):
    alpha, beta = x[0], x[1]
    return [alpha * sp.jv(1, alpha) / sp.jv(0, alpha) + beta * sp.iv(1, beta) / sp.iv(0, beta),
            beta ** 2 - alpha ** 2 - a**2*F/D]

def alpha_beta(D, a, F):
    F = np.array(F)
    if F.size == 1:
        params = fsolve(system_ab, [3, 4], maxfev=10000, args=(D, a, f))
        return params
    else:
        alpha = []
        beta = []
        for f in F:
            params = fsolve(system_ab, [2.5, 20], maxfev=10000, args=(D, a, f))
            alpha.append(params[0])
            beta.append(params[1])
        return np.array(alpha), np.array(beta)

def omega_tension_pl(thickness, radius, nu, E, rho, tension):
    D = E * thickness ** 3 / (12 * (1 - nu**2))
    alpha, beta = alpha_beta(D, radius, tension)
    omega = np.sqrt(D / (rho * thickness)) / (2 * radius ** 2) * np.sqrt(
        (alpha ** 2 + beta ** 2) ** 2 - (radius ** 2 * tension / D) ** 2)
    return omega

###########################################################################

def dissipation_tau_r(r, mu, nu, E, rho, T, c_v , kappa, alpha, w0):
    Xi = r/mu * np.sqrt(np.pi*w0*rho*c_v/(2*kappa))
    factor1 = (1 + nu) / (1 - 2*nu) * E * alpha ** 2 * T / (rho * c_v)
    if max(Xi)<10:
        factor2 = 6/Xi**2 - 6/Xi**3*(np.sinh(Xi)+np.sin(Xi))/(np.cosh(Xi) + np.cos(Xi))
    else: # approximation of the sinh and cosh if Xi values are too high.
        factor2 = 6 / Xi**2 - 6 / Xi ** 3
    Q = factor1*factor2
    return Q

def dissipation_tau_z(thickness, nu, E, rho, T, c_v , kappa, alpha, w0):
    Xi = thickness * np.sqrt(w0*rho*c_v/(2*kappa))
    factor1 = (1 + nu) / (1 - 2*nu) * E * alpha ** 2 * T / (rho * c_v)
    if max(Xi)<10:
        factor2 = 6/Xi**2 - 6/Xi**3*(np.sinh(Xi)+np.sin(Xi))/(np.cosh(Xi) + np.cos(Xi))
    else: # approximation of the sinh and cosh if Xi values are too high.
        factor2 = 6 / Xi**2 - 6 / Xi ** 3
    Q = factor1*factor2
    return Q

def dissipation_kz_kr(h, r, nu, E, rho, T, c_v , kappa, alpha_T, w0, N_MAX):
    Q = []
    x0 = sp.jn_zeros(0, N_MAX)
    alpha, beta = 3.19622, 3.19622  # This implies zeroth order and no pretension: alpha=beta
    if w0[0]==-1:
        print('We changed the frequency')
        D = E * h ** 3 / (12 * (1 - 2 * nu))
        sqrt1 = np.sqrt(D / (rho * h))
        w0 = [(alpha**2 + beta**2)/(2*r**2)*sqrt1]*len(T)
    for t, temp in enumerate(T):
        sum = []
        for x0_n in x0:
            l_n = (1/2 * ( (x0_n/r)**2 + np.sqrt((x0_n/r)**4 + (w0[t]*rho*c_v[t]/kappa[t])**2)))**0.5
            m_n = (1/2 * (-(x0_n/r)**2 + np.sqrt((x0_n/r)**4 + (w0[t]*rho*c_v[t]/kappa[t])**2)))**0.5

            c1_n = np.cosh(l_n*h/2) * np.cos(m_n*h/2)
            c2_n = np.sinh(l_n*h/2) * np.sin(m_n*h/2)
            c3_n = np.cosh(l_n*h/2) * np.sin(m_n*h/2)
            c4_n = np.sinh(l_n*h/2) * np.cos(m_n*h/2)

            d1_n = (l_n ** 3 - 3 * l_n * m_n ** 2) * (l_n * h / (l_n ** 2 + m_n ** 2) * (c1_n ** 2 + c2_n ** 2) +
                                4 * l_n * m_n / (l_n ** 2 + m_n ** 2) ** 2 * (c2_n * c4_n - c1_n * c3_n) -
                                2 * (l_n ** 2 - m_n ** 2) / (l_n ** 2 + m_n ** 2) ** 2 * (c1_n * c4_n + c2_n * c3_n))
            d2_n = (3 * l_n ** 2 * m_n - m_n ** 3) * (-m_n * h / (l_n ** 2 + m_n ** 2) * (c1_n ** 2 + c2_n ** 2) +
                                4 * l_n * m_n / (l_n ** 2 + m_n ** 2) ** 2 * (c1_n * c4_n + c2_n * c3_n) +
                                2 * (l_n ** 2 - m_n ** 2) / (l_n ** 2 + m_n ** 2) ** 2 * (c2_n * c4_n - c1_n * c3_n))

            integral = (alpha ** 2 + beta ** 2) * x0_n ** 3 * sp.jv(0, alpha) * sp.jv(1, x0_n) / \
                                                                    ((alpha ** 2 - x0_n ** 2) * (beta ** 2 + x0_n ** 2))

            numerator_n = (d1_n + d2_n - (l_n**4-m_n**4)*(c1_n**2+c2_n**2)*h**3/12) * integral**2
            denominator_n = (l_n**2 + m_n**2)**3 * (c1_n**2+c2_n**2) * sp.jv(1,x0_n)**2
            sum_n = numerator_n / denominator_n
            sum.append(sum_n)

        factor_DE = -4 * w0[t] * np.pi ** 2 * alpha_T[t] ** 2 * E ** 2 * temp  / \
                                                                    (kappa[t] * (1 - 2 * nu) * (1 - nu) * r ** 2)
        sum = np.array(sum).sum()
        DELTA_E = factor_DE * sum

        factor_EM = 0.5 * np.pi * h * rho * w0[t] ** 2 * r ** 2
        int_DE = (2 - 4 * beta * sp.iv(1, beta) / ((alpha ** 2 + beta ** 2) * sp.iv(0, beta)) - sp.iv(1,beta) ** 2 / \
                        sp.iv(0, beta) ** 2) * sp.jv(0, alpha) ** 2 - 4 * alpha * sp.jv(0, alpha) * sp.jv(1, alpha) / \
                        (alpha ** 2 + beta ** 2) + sp.jv(1, alpha) ** 2

        E_MAX = factor_EM * int_DE

        Q.append(1 / (2*np.pi) * DELTA_E / E_MAX)

    return np.array(Q)

def dissipation_kz_kr_anis(h, r, nu, E, rho, T, c_v , kappa_z, kappa_r, alpha_T, w0, N_MAX):
    Q = []
    x0 = sp.jn_zeros(0, N_MAX)
    alpha, beta = 3.19622, 3.19622  # This implies zeroth order and no pretension: alpha=beta
    if w0[0]==-1:
        print('We changed the frequency')
        D = E * h ** 3 / (12 * (1 - 2 * nu))
        sqrt1 = np.sqrt(D / (rho * h))
        w0 = [(alpha**2 + beta**2)/(2*r**2)*sqrt1]*len(T)
    for t, temp in enumerate(T):
        sum = []
        for x0_n in x0:
            l_n = (1/2 * ( (kappa_r[t]/kappa_z[t])*(x0_n/r)**2 + np.sqrt((kappa_r[t]/kappa_z[t])**2*(x0_n/r)**4 + (w0[t]*rho*c_v[t]/kappa_z[t])**2)))**0.5
            m_n = (1/2 * ( -(kappa_r[t]/kappa_z[t])*(x0_n/r)**2 + np.sqrt((kappa_r[t]/kappa_z[t])**2*(x0_n/r)**4 + (w0[t]*rho*c_v[t]/kappa_z[t])**2)))**0.5

            c1_n = np.cosh(l_n*h/2) * np.cos(m_n*h/2)
            c2_n = np.sinh(l_n*h/2) * np.sin(m_n*h/2)
            c3_n = np.cosh(l_n*h/2) * np.sin(m_n*h/2)
            c4_n = np.sinh(l_n*h/2) * np.cos(m_n*h/2)

            d1_n = (l_n ** 3 - 3 * l_n * m_n ** 2) * (l_n * h / (l_n ** 2 + m_n ** 2) * (c1_n ** 2 + c2_n ** 2) +
                                4 * l_n * m_n / (l_n ** 2 + m_n ** 2) ** 2 * (c2_n * c4_n - c1_n * c3_n) -
                                2 * (l_n ** 2 - m_n ** 2) / (l_n ** 2 + m_n ** 2) ** 2 * (c1_n * c4_n + c2_n * c3_n))
            d2_n = (3 * l_n ** 2 * m_n - m_n ** 3) * (-m_n * h / (l_n ** 2 + m_n ** 2) * (c1_n ** 2 + c2_n ** 2) +
                                4 * l_n * m_n / (l_n ** 2 + m_n ** 2) ** 2 * (c1_n * c4_n + c2_n * c3_n) +
                                2 * (l_n ** 2 - m_n ** 2) / (l_n ** 2 + m_n ** 2) ** 2 * (c2_n * c4_n - c1_n * c3_n))

            integral = (alpha ** 2 + beta ** 2) * x0_n ** 3 * sp.jv(0, alpha) * sp.jv(1, x0_n) / \
                                                                    ((alpha ** 2 - x0_n ** 2) * (beta ** 2 + x0_n ** 2))

            numerator_n = (d1_n + d2_n - (l_n**4-m_n**4)*(c1_n**2+c2_n**2)*h**3/12) * integral**2
            denominator_n = (l_n**2 + m_n**2)**3 * (c1_n**2+c2_n**2) * sp.jv(1,x0_n)**2
            sum_n = numerator_n / denominator_n
            sum.append(sum_n)

        factor_DE = -4 * w0[t] * np.pi ** 2 * alpha_T[t] ** 2 * E ** 2 * temp  / \
                                                                    (kappa_z[t] * (1 - 2 * nu) * (1 - nu) * r ** 2)
        sum = np.array(sum).sum()
        DELTA_E = factor_DE * sum

        factor_EM = 0.5 * np.pi * h * rho * w0[t] ** 2 * r ** 2
        int_DE = (2 - 4 * beta * sp.iv(1, beta) / ((alpha ** 2 + beta ** 2) * sp.iv(0, beta)) - sp.iv(1,beta) ** 2 / \
                        sp.iv(0, beta) ** 2) * sp.jv(0, alpha) ** 2 - 4 * alpha * sp.jv(0, alpha) * sp.jv(1, alpha) / \
                        (alpha ** 2 + beta ** 2) + sp.jv(1, alpha) ** 2

        E_MAX = factor_EM * int_DE

        Q.append(1 / (2*np.pi) * DELTA_E / E_MAX)

    return np.array(Q)

def dissipation_tau_z_tension(thickness, radius, nu, E, rho, T, c_v , kappa, alpha, w0, tension):
    Xi = thickness * np.sqrt(w0*rho*c_v/(2*kappa))
    factor1 = (1 + nu) / (1 - nu) * E * alpha ** 2 * T / (rho * c_v)
    if max(Xi)<10:
        factor2 = 6/Xi**2 - 6/Xi**3*(np.sinh(Xi)+np.sin(Xi))/(np.cosh(Xi) + np.cos(Xi))
    else: # approximation of the sinh and cosh if Xi values are too high.
        factor2 = 6 / Xi**2 - 6 / Xi ** 3
    D = E * thickness ** 3 / (12 * (1 - 2 * nu))
    alpha, beta = alpha_beta(D, thickness, radius, tension)
    omega = np.sqrt(D / (rho * thickness)) / (2 * radius ** 2) * np.sqrt(
        (alpha ** 2 + beta ** 2) ** 2 - (radius ** 2 * tension / D) ** 2)
    integral_den = 1/radius**2 * (0.5*(alpha**4*(sp.jv(0, alpha)**2+sp.jv(1, alpha)) + beta**4*(sp.iv(0, beta)**2 - sp.iv(1, beta))) + 2*alpha**2*beta**2 * (beta*sp.iv(1, beta)*sp.jv(0,alpha) + alpha*sp.jv(1,alpha)*sp.iv(0,beta)) / (alpha**2+beta**2))
    integral_num = 0.5 * (sp.jv(0, alpha)**2 * (alpha**2 - beta**2 + beta*sp.iv(1,beta)/sp.iv(0,beta)**2 * (2*(beta**2 - alpha**2)*sp.iv(0,beta)/(alpha**2+beta**2) + beta*sp.iv(1,beta))) + (2*alpha*(beta**2 - alpha**2)*sp.jv(0, alpha)*sp.jv(1, alpha))/(alpha**2+beta**2) + alpha**2*sp.jv(1, alpha)**2)
    F = tension/D * integral_num/integral_den
    factor3 = 1/(1+F)
    Q = factor1*factor2*factor3
    return Q, omega
    
###########################################################
# Calculations of dissipation with tension contributions

def dissipation_tau_z_tension_w(thickness, radius, nu, E, rho, T, c_v , kappa, alpha_T, tension):
    D = E * thickness ** 3 / (12 * (1 - 2 * nu))
    alpha, beta = alpha_beta(D, thickness, radius, tension)
    w0 = np.sqrt(D / (rho * thickness)) / (2 * radius ** 2) * np.sqrt(
        (alpha ** 2 + beta ** 2) ** 2 - (radius ** 2 * tension / D) ** 2)

    Xi = thickness * np.sqrt(w0*rho*c_v/(2*kappa))
    factor1 = (1 + nu) / (1 - nu) * E * alpha_T ** 2 * T / (rho * c_v)
    if max(Xi)<10:
        factor2 = 6/Xi**2 - 6/Xi**3*(np.sinh(Xi)+np.sin(Xi))/(np.cosh(Xi) + np.cos(Xi))
    else: # approximation of the sinh and cosh if Xi values are too high.
        factor2 = 6 / Xi**2 - 6 / Xi ** 3
    integral_den = 1/radius**2 * (0.5*(alpha**4*(sp.jv(0, alpha)**2+sp.jv(1, alpha)) + beta**4*(sp.iv(0, beta)**2 - sp.iv(1, beta))) + 2*alpha**2*beta**2 * (beta*sp.iv(1, beta)*sp.jv(0,alpha) + alpha*sp.jv(1,alpha)*sp.iv(0,beta)) / (alpha**2+beta**2))
    integral_num = 0.5 * (sp.jv(0, alpha)**2 * (alpha**2 - beta**2 + beta*sp.iv(1,beta)/sp.iv(0,beta)**2 * (2*(beta**2 - alpha**2)*sp.iv(0,beta)/(alpha**2+beta**2) + beta*sp.iv(1,beta))) + (2*alpha*(beta**2 - alpha**2)*sp.jv(0, alpha)*sp.jv(1, alpha))/(alpha**2+beta**2) + alpha**2*sp.jv(1, alpha)**2)
    F = tension/D * integral_num/integral_den
    factor3 = 1/(1+F)
    Q = factor1*factor2*factor3
    return np.array(Q), np.array(w0)

def dissipation_tau_z_tension(h, radius, nu, E, rho, T, c_v , kappa, alpha_T, pretension):
    tension = tension_thick(integral_alpha, h, E, nu) + pretension
    tension = np.interp(T, T_integral_alpha, tension) # tension is computed with a specific spacing, here we correct for that

    D = E * h ** 3 / (12 * (1 - nu**2))
    alpha, beta = alpha_beta(D, radius, tension)
    w0 = omega_tension_pl(h, radius, nu, E, rho, tension)

    Xi = h * np.sqrt(w0*rho*c_v/(2*kappa))
    factor1 = (1 + nu) / (1 - 2*nu) * E * alpha_T ** 2 * T / (rho * c_v)

    if max(Xi)<10:
        factor2 = 6/Xi**2 - 6/Xi**3*(np.sinh(Xi)+np.sin(Xi))/(np.cosh(Xi) + np.cos(Xi))
    else: # approximation of the sinh and cosh if Xi values are too high.
        factor2 = 6 / Xi**2 - 6 / Xi ** 3

    B = 1/(sp.iv(0,beta)/sp.jv(0,alpha) )
    integral_den = (1/radius**2) * (0.5*(alpha**4*(sp.jv(0, alpha)**2+sp.jv(1, alpha)**2) + B**2*beta**4*(sp.iv(0, beta)**2 - sp.iv(1, beta)**2)) + 2*B*alpha**2*beta**2 * (beta*sp.iv(1, beta)*sp.jv(0,alpha) + alpha*sp.jv(1,alpha)*sp.iv(0,beta)) / (alpha**2+beta**2))
    integral_num = 0.5 * (sp.jv(0, alpha)**2 * (alpha**2 - beta**2 + beta*sp.iv(1,beta)/sp.iv(0,beta)**2 * (2*(beta**2 - alpha**2)*sp.iv(0,beta)/(alpha**2+beta**2) + beta*sp.iv(1,beta))) + (2*alpha*(beta**2 - alpha**2)*sp.jv(0, alpha)*sp.jv(1, alpha))/(alpha**2+beta**2) + alpha**2*sp.jv(1, alpha)**2)

    gamma_tension = tension / D * integral_num / integral_den

    factor3 = 1 / (1 + gamma_tension)

    Q = factor1*factor2*factor3

    return np.array(Q), np.array(w0)

def dissipation_z_tension(h, radius, nu, E, rho, T, c_v , kappa, alpha_T, pretension):
    
    tension = tension_thick(integral_alpha, h, E, nu) + pretension
    tension = np.interp(T, T_integral_alpha, tension) # tension is computed with a specific spacing, here we correct for that

    D = E * h ** 3 / (12 * (1 - nu**2))
    alpha, beta = alpha_beta(D, radius, tension)
    w0 = omega_tension_pl(h, radius, nu, E, rho, tension)

    Xi = h * np.sqrt(w0*rho*c_v/(2*kappa))
    factor1 = E**2 * alpha_T**2 * T * h**2 / (6 * rho * c_v * (1 - 2*nu) * (1 - nu))

    if max(Xi)<10:
        factor2 = 6/Xi**2 - 6/Xi**3*(np.sinh(Xi)+np.sin(Xi))/(np.cosh(Xi) + np.cos(Xi))
    else: # approximation of the sinh and cosh if Xi values are too high.
        factor2 = 6 / Xi**2 - 6 / Xi** 3

    B = 1/(sp.iv(0,beta)/sp.jv(0,alpha) )
    integral = 0.5*(alpha**4*(sp.jv(0, alpha)**2+sp.jv(1, alpha)**2) + B**2*beta**4*(sp.iv(0, beta)**2 - sp.iv(1, beta)**2)) + 2*B*alpha**2*beta**2 * (beta*sp.iv(1, beta)*sp.jv(0,alpha) + alpha*sp.jv(1,alpha)*sp.iv(0,beta)) / (alpha**2+beta**2)

    factor_EM = rho * w0** 2 * radius ** 4
    int_DE = (2 - 4 * beta * sp.iv(1, beta) / ((alpha ** 2 + beta ** 2) * sp.iv(0, beta)) - sp.iv(1,beta) ** 2 / \
                        sp.iv(0, beta) ** 2) * sp.jv(0, alpha) ** 2 - 4 * alpha * sp.jv(0, alpha) * sp.jv(1, alpha) / \
                        (alpha ** 2 + beta ** 2) + sp.jv(1, alpha)** 2

    DELTA_E = factor1 * factor2 * integral
    E_MAX = factor_EM * int_DE

    Q = DELTA_E / E_MAX

    return np.array(Q), np.array(w0)

def dissipation_kz_kr_tension_anis(h, r, nu, E, rho, T, c_v , kappa_z, kappa_r, alpha_T, pretension, N_MAX):
    Q = []
    x0 = sp.jn_zeros(0, N_MAX)
    tension = tension_thick(integral_alpha, h, E, nu) + pretension
    tension = np.interp(T, T_integral_alpha, tension) # tension is computed with a specific spacing, here we correct for that
    D = E * h ** 3 / (12 * (1 - nu**2))
    alpha, beta = alpha_beta(D, r, tension)
    w0 = omega_tension_pl(h, r, nu, E, rho, tension)

    for t, temp in enumerate(T):
        sum = []
        for x0_n in x0:
            l_n = (1/2 * ( (kappa_r[t]/kappa_z[t])*(x0_n/r)**2 + np.sqrt((kappa_r[t]/kappa_z[t])**2*(x0_n/r)**4 + (w0[t]*rho*c_v[t]/kappa_z[t])**2)))**0.5
            m_n = (1/2 * ( -(kappa_r[t]/kappa_z[t])*(x0_n/r)**2 + np.sqrt((kappa_r[t]/kappa_z[t])**2*(x0_n/r)**4 + (w0[t]*rho*c_v[t]/kappa_z[t])**2)))**0.5

            c1_n = np.cosh(l_n*h/2) * np.cos(m_n*h/2)
            c2_n = np.sinh(l_n*h/2) * np.sin(m_n*h/2)
            c3_n = np.cosh(l_n*h/2) * np.sin(m_n*h/2)
            c4_n = np.sinh(l_n*h/2) * np.cos(m_n*h/2)

            d1_n = (l_n ** 3 - 3 * l_n * m_n ** 2) * (l_n * h / (l_n ** 2 + m_n ** 2) * (c1_n ** 2 + c2_n ** 2) +
                                4 * l_n * m_n / (l_n ** 2 + m_n ** 2) ** 2 * (c2_n * c4_n - c1_n * c3_n) -
                                2 * (l_n ** 2 - m_n ** 2) / (l_n ** 2 + m_n ** 2) ** 2 * (c1_n * c4_n + c2_n * c3_n))
            d2_n = (3 * l_n ** 2 * m_n - m_n ** 3) * (-m_n * h / (l_n ** 2 + m_n ** 2) * (c1_n ** 2 + c2_n ** 2) +
                                4 * l_n * m_n / (l_n ** 2 + m_n ** 2) ** 2 * (c1_n * c4_n + c2_n * c3_n) +
                                2 * (l_n ** 2 - m_n ** 2) / (l_n ** 2 + m_n ** 2) ** 2 * (c2_n * c4_n - c1_n * c3_n))

            integral = (alpha[t] ** 2 + beta[t] ** 2) * x0_n ** 3 * sp.jv(0, alpha[t]) * sp.jv(1, x0_n) / \
                                                                    ((alpha[t] ** 2 - x0_n ** 2) * (beta[t] ** 2 + x0_n ** 2))

            numerator_n = (d1_n + d2_n - (l_n**4-m_n**4)*(c1_n**2+c2_n**2)*h**3/12) * integral**2
            denominator_n = (l_n**2 + m_n**2)**3 * (c1_n**2+c2_n**2) * sp.jv(1,x0_n)**2
            sum_n = numerator_n / denominator_n
            sum.append(sum_n)

        factor_DE = -4 * w0[t] * np.pi ** 2 * alpha_T[t] ** 2 * E ** 2 * temp  / \
                                                                    (kappa_z[t] * (1 - 2 * nu) * (1 - nu) * r ** 2)
        sum = np.array(sum).sum()
        DELTA_E = factor_DE * sum

        factor_EM = 0.5 * np.pi * h * rho * w0[t] ** 2 * r ** 2
        int_DE = (2 - 4 * beta[t] * sp.iv(1, beta[t]) / ((alpha[t] ** 2 + beta[t] ** 2) * sp.iv(0, beta[t])) - sp.iv(1,beta[t]) ** 2 / \
                        sp.iv(0, beta[t]) ** 2) * sp.jv(0, alpha[t]) ** 2 - 4 * alpha[t] * sp.jv(0, alpha[t]) * sp.jv(1, alpha[t]) / \
                        (alpha[t] ** 2 + beta[t] ** 2) + sp.jv(1, alpha[t]) ** 2

        E_MAX = factor_EM * int_DE

        Q.append(1 / (2*np.pi) * DELTA_E / E_MAX)

    return np.array(Q), np.array(w0)


def dissipation_r_curry(h, r, nu, E, rho, T, c_v , kappa, alpha_T, pretension):
    tension = tension_thick(integral_alpha, h, E, nu) + pretension
    tension = np.interp(T, T_integral_alpha, tension) # tension is computed with a specific spacing, here we correct for that
    omega = omega_tension_pl(h, r, nu, E, rho, tension)

    tau = r**2*rho*c_v / (sp.jn_zeros(0,1)**2 * kappa )
    mu = 5
    c4 = 1.8519
    c5 = 2*E*alpha_T**2*T/(c_v*rho*(1-2*nu))
    Q = 2*np.pi*(mu**2*omega*tau)*c4*c5 / ((mu**2 * omega*tau)**2*(2 + c5) + np.pi**2*c4)

    return Q, omega