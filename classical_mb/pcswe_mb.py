import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, argmin, argmax, pi, mean, exp, sqrt, zeros, ones, nan
import scipy
import matplotlib.pyplot as plt
from scipy.special import roots_legendre, eval_legendre
from numpy.polynomial import chebyshev



class PCSWE_mb():

    def __init__(self):
        self.debug = False

        # geometry
        self.A = 0.72
        self.H = 7.12 
        self.L = 8e3

        # tunable
        self.r = 0.24
        self.h0 = 0.0025

        self.bnd_LO_hop = nan
        self.bnd_FO_hop = nan

        # morphodynamics
        self.p = 0.4 # porosity
        self.c_d = 0.0025
        self.lmbda = 6.8e-6
        self.d50 = 0.13e-3

        # universal constants
        self.g = 9.81
        self.sigma = 1.4e-4
        self.rho_w = 1025
        self.rho_s = 2650

        self.set_derivative_vars()
        
    def set_derivative_vars(self):
        self.epsilon = self.A / self.H
        self.eta = self.sigma * self.L / sqrt(self.g * self.H)
        self.U = self.epsilon * self.sigma * self.L
        self.kappa = self.g * self.H / (self.sigma * self.L) ** 2

        self.s = self.rho_s / self.rho_w

        self.delta = 0.04 * self.c_d**(3/2) * self.A * (self.sigma * self.L)**4 / \
                    (self.g**2 * (self.s-1)**2 * self.d50 * self.H**6 * (1-self.p))

    def h_fx(self, x): return x
    def h_fx_dx(self, x): return 1
    def h_fx_dxx(self, x): return 0
    def h_fx_dxxx(self, x): return 0
    
    def y0_fx_dx(self, x_x, y0_x):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        h_x, h_x_dx, h_x_dxx = self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x)

        # derivatives of dzeta (using momentum equation)
        dz0c_x_dx = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0c_x - u0s_x)
        dz0s_x_dx = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0s_x + u0c_x)
        u0c_x_dx = (-dz0s_x + u0c_x * h_x_dx)  / (1 - h_x)
        u0s_x_dx = ( dz0c_x + u0s_x * h_x_dx)  / (1 - h_x)

   
        return np.array([dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx])

    def y0_fx_dx_hop(self, x_x, y0_x):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        h_x, h_x_dx, h_x_dxx = self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x)

        # derivatives of dzeta (using momentum equation)
        dz0c_x_dx = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0c_x - u0s_x)
        dz0s_x_dx = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0s_x + u0c_x)
        u0c_x_dx =  ( dz0s_x_dx - u0c_x * h_x_dxx)  / (2*h_x_dx)
        u0s_x_dx =  (-dz0c_x_dx - u0s_x * h_x_dxx)  / (2*h_x_dx)

        return np.array([dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx])

    def y0_fx_dxx_hop(self, x_x, y0_x, y0_x_dx):
        # this function only works at x = 1, since it uses hopital's rule to approximate the u0_x_dxx
        # to get the actual correct value, we should use np.where and have two cases for u0_x_dxx

        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        h_x, h_x_dx, h_x_dxx, h_x_dxxx = self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x), self.h_fx_dxxx(x_x)

        dz0c_x_dxx = 1 / self.kappa * (- self.r / (1 - h_x + self.h0)**2 * h_x_dx * u0c_x - self.r / (1 - h_x + self.h0) * u0c_x_dx - u0s_x_dx)
        dz0s_x_dxx = 1 / self.kappa * (- self.r / (1 - h_x + self.h0)**2 * h_x_dx * u0s_x - self.r / (1 - h_x + self.h0) * u0s_x_dx + u0c_x_dx)
        u0c_x_dxx = 1/3 * (- h_x_dxx / h_x_dx**2 * ( dz0s_x_dx - u0c_x * h_x_dxx) + 1 / h_x_dx * ( dz0s_x_dxx - u0c_x_dx * h_x_dxx - u0c_x * h_x_dxxx))
        u0s_x_dxx = 1/3 * (- h_x_dxx / h_x_dx**2 * (-dz0c_x_dx - u0s_x * h_x_dxx) + 1 / h_x_dx * (-dz0c_x_dxx - u0s_x_dx * h_x_dxx - u0s_x * h_x_dxxx))
        
        return np.array([dz0c_x_dxx, dz0s_x_dxx, u0c_x_dxx, u0s_x_dxx])

    def y1_fx_dx(self, x_x, y1_x, y0_x, y0_x_dx):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x = y1_x
        h_x, h_x_dx, h_x_dxx= self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x)

        # derivatives of dzeta (using momentum equation)
        dz1r_x_dx = (1 / (1 - h_x + self.h0) * (- self.r * u1r_x - 0.5 * (  dz0c_x *  u0s_x - dz0s_x *  u0c_x)
            - 0.5 * (  dz0s_x * dz0s_x_dx + dz0c_x * dz0c_x_dx) * self.kappa) - 0.5 * (u0c_x * u0c_x_dx + u0s_x * u0s_x_dx)            ) / self.kappa
        dz1c_x_dx = (1 / (1 - h_x + self.h0) * (- self.r * u1c_x - 0.5 * (  dz0c_x *  u0s_x + dz0s_x *  u0c_x)
            - 0.5 * (- dz0s_x * dz0s_x_dx + dz0c_x * dz0c_x_dx) * self.kappa) - 0.5 * (u0c_x * u0c_x_dx - u0s_x * u0s_x_dx) - 2 * u1s_x) / self.kappa
        dz1s_x_dx = (1 / (1 - h_x + self.h0) * (- self.r * u1s_x + 0.5 * (  dz0c_x *  u0c_x - dz0s_x *  u0s_x)
            - 0.5 * (  dz0c_x * dz0s_x_dx + dz0s_x * dz0c_x_dx) * self.kappa) - 0.5 * (u0c_x * u0s_x_dx + u0s_x * u0c_x_dx) + 2 * u1c_x) / self.kappa

        # derivatives of first order u at x = 1 (using l'hopital)
        u1r_x_dx = 1 / (1 - h_x) * (h_x_dx * u1r_x              - 1 / 2 * (dz0c_x * u0c_x_dx + dz0c_x_dx * u0c_x + dz0s_x * u0s_x_dx + dz0s_x_dx * u0s_x))
        u1c_x_dx = 1 / (1 - h_x) * (h_x_dx * u1c_x - 2 * dz1s_x - 1 / 2 * (dz0c_x * u0c_x_dx + dz0c_x_dx * u0c_x - dz0s_x * u0s_x_dx - dz0s_x_dx * u0s_x))
        u1s_x_dx = 1 / (1 - h_x) * (h_x_dx * u1s_x + 2 * dz1c_x - 1 / 2 * (dz0s_x * u0c_x_dx + dz0s_x_dx * u0c_x + dz0c_x * u0s_x_dx + dz0c_x_dx * u0s_x))

        return np.array([dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx]) 
    
    def y1_fx_dx_hop(self, x_x, y1_x, y0_x, y0_x_dx):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x = y1_x
        h_x, h_x_dx, h_x_dxx= self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x)

        # derivatives of dzeta (using momentum equation)
        dz1r_x_dx = (1 / (1 - h_x + self.h0) * (- self.r * u1r_x - 0.5 * (  dz0c_x *  u0s_x - dz0s_x *  u0c_x)
            - 0.5 * (  dz0s_x * dz0s_x_dx + dz0c_x * dz0c_x_dx) * self.kappa) - 0.5 * (u0c_x * u0c_x_dx + u0s_x * u0s_x_dx)            ) / self.kappa
        dz1c_x_dx = (1 / (1 - h_x + self.h0) * (- self.r * u1c_x - 0.5 * (  dz0c_x *  u0s_x + dz0s_x *  u0c_x)
            - 0.5 * (- dz0s_x * dz0s_x_dx + dz0c_x * dz0c_x_dx) * self.kappa) - 0.5 * (u0c_x * u0c_x_dx - u0s_x * u0s_x_dx) - 2 * u1s_x) / self.kappa
        dz1s_x_dx = (1 / (1 - h_x + self.h0) * (- self.r * u1s_x + 0.5 * (  dz0c_x *  u0c_x - dz0s_x *  u0s_x)
            - 0.5 * (  dz0c_x * dz0s_x_dx + dz0s_x * dz0c_x_dx) * self.kappa) - 0.5 * (u0c_x * u0s_x_dx + u0s_x * u0c_x_dx) + 2 * u1c_x) / self.kappa

        # derivatives of first order u at x = 1 (using l'hopital)
        dz0c_x_dxx, dz0s_x_dxx, u0c_x_dxx, u0s_x_dxx = self.y0_fx_dxx_hop(x_x, y0_x, y0_x_dx)

        # second derivatives of products of leading order components:
        dz_u_cc_x_dxx = (dz0c_x * u0c_x_dxx + 2 * dz0c_x_dx * u0c_x_dx + dz0c_x_dxx * u0c_x)
        dz_u_ss_x_dxx = (dz0s_x * u0s_x_dxx + 2 * dz0s_x_dx * u0s_x_dx + dz0s_x_dxx * u0s_x)
        dz_u_cs_x_dxx = (dz0s_x * u0c_x_dxx + 2 * dz0s_x_dx * u0c_x_dx + dz0s_x_dxx * u0c_x)
        dz_u_sc_x_dxx = (dz0c_x * u0s_x_dxx + 2 * dz0c_x_dx * u0s_x_dx + dz0c_x_dxx * u0s_x)

        # l'hopital
        u1r_x_dx = (             - u1r_x * h_x_dxx + (dz_u_cc_x_dxx + dz_u_ss_x_dxx) / 2) / (2*h_x_dx)
        u1c_x_dx = ( 2*dz1s_x_dx - u1c_x * h_x_dxx + (dz_u_cc_x_dxx - dz_u_ss_x_dxx) / 2) / (2*h_x_dx)
        u1s_x_dx = (-2*dz1c_x_dx - u1s_x * h_x_dxx + (dz_u_cs_x_dxx + dz_u_sc_x_dxx) / 2) / (2*h_x_dx)

        return np.array([dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx]) 
   
    def y_r(self, dz_r):
        dz0c_r, dz0s_r, dz1r_r, dz1c_r, dz1s_r = dz_r
        h_r_dx = self.h_fx_dx(1)

        u0c_r = dz0s_r / h_r_dx
        u0s_r = -dz0c_r / h_r_dx

        dz0c_r_dx, dz0s_r_dx, u0c_r_dx, u0s_r_dx = self.y0_fx_dx_hop(1, [dz0c_r, dz0s_r, u0c_r, u0s_r])

        u1r_r = 1 / h_r_dx * (1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r + dz0s_r * u0s_r_dx + dz0s_r_dx * u0s_r))
        u1c_r = 1 / h_r_dx * (1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r - dz0s_r * u0s_r_dx - dz0s_r_dx * u0s_r) + 2 * dz1s_r)
        u1s_r = 1 / h_r_dx * (1 / 2 * (dz0c_r * u0s_r_dx + dz0c_r_dx * u0s_r + dz0s_r * u0c_r_dx + dz0s_r_dx * u0c_r) - 2 * dz1c_r)

        return np.array([dz0c_r, dz0s_r, u0c_r, u0s_r, dz1r_r, dz1c_r, dz1s_r, u1r_r, u1c_r, u1s_r])

    def deriv(self, x_x, y_x):
        x_x, y_x = np.asarray(x_x), np.asarray(y_x)
        y0_x, y1_x = np.split(y_x, [4], axis=0)

        y0_x_dx = np.empty_like(y0_x, dtype=y0_x.dtype)
        mask_hop = abs(1 - x_x) < self.bnd_LO_hop
        y0_x_dx[:, ~mask_hop] = self.y0_fx_dx    (x_x[~mask_hop], y0_x[:, ~mask_hop])
        y0_x_dx[:,  mask_hop] = self.y0_fx_dx_hop(x_x[ mask_hop], y0_x[:,  mask_hop])

        y1_x_dx = np.empty_like(y1_x, dtype=y1_x.dtype)
        mask_hop = abs(1 - x_x) < self.bnd_FO_hop
        y1_x_dx[:, ~mask_hop] = self.y1_fx_dx    (x_x[~mask_hop], y1_x[:, ~mask_hop], y0_x[:, ~mask_hop], y0_x_dx[:, ~mask_hop])
        y1_x_dx[:,  mask_hop] = self.y1_fx_dx_hop(x_x[ mask_hop], y1_x[:,  mask_hop], y0_x[:,  mask_hop], y0_x_dx[:,  mask_hop])

        y_x_dx = np.concatenate((y0_x_dx, y1_x_dx), axis=0)

        return y_x_dx

    def deriv_h(self, x_x, y_x):
        x_x, y_x = np.asarray(x_x), np.asarray(y_x)
        y0_x, y1_x, yh_x = np.split(y_x, [4, 10], axis=0)


        y0_x_dx = np.empty_like(y0_x, dtype=y0_x.dtype)
        mask_hop = abs(1 - x_x) < self.bnd_LO_hop
        y0_x_dx[:, ~mask_hop] = self.y0_fx_dx    (x_x[~mask_hop], y0_x[:, ~mask_hop])
        y0_x_dx[:,  mask_hop] = self.y0_fx_dx_hop(x_x[ mask_hop], y0_x[:,  mask_hop])

        y1_x_dx = np.empty_like(y1_x, dtype=y1_x.dtype)
        mask_hop = abs(1 - x_x) < self.bnd_FO_hop
        y1_x_dx[:, ~mask_hop] = self.y1_fx_dx    (x_x[~mask_hop], y1_x[:, ~mask_hop], y0_x[:, ~mask_hop], y0_x_dx[:, ~mask_hop])
        y1_x_dx[:,  mask_hop] = self.y1_fx_dx_hop(x_x[ mask_hop], y1_x[:,  mask_hop], y0_x[:,  mask_hop], y0_x_dx[:,  mask_hop])

        # y_x_dx = np.concatenate((y0_x_dx, y1_x_dx), axis=0)

        dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x, h_x, h_x_dx = y_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx = y1_x_dx

        factor = 0.04 * self.c_d**(3/2) / (self.g * (self.s-1))**2 / self.d50 
        U5 = (self.A * self.sigma * self.L / self.H)**5
        q_x_dx = U5 * factor / 2 / pi / self.L * self.epsilon*(5*pi*u0c_x**4*u1c_x_dx/2 + 15*pi*u0c_x**4*u1r_x_dx/4 + 5*pi*u0c_x**3*u0s_x*u1s_x_dx + 10*pi*u0c_x**3*u1c_x*u0c_x_dx + 15*pi*u0c_x**3*u1r_x*u0c_x_dx + 5*pi*u0c_x**3*u1s_x*u0s_x_dx + 15*pi*u0c_x**2*u0s_x**2*u1r_x_dx/2 + 15*pi*u0c_x**2*u0s_x*u1r_x*u0s_x_dx + 15*pi*u0c_x**2*u0s_x*u1s_x*u0c_x_dx + 5*pi*u0c_x*u0s_x**3*u1s_x_dx + 15*pi*u0c_x*u0s_x**2*u1r_x*u0c_x_dx + 15*pi*u0c_x*u0s_x**2*u1s_x*u0s_x_dx - 5*pi*u0s_x**4*u1c_x_dx/2 + 15*pi*u0s_x**4*u1r_x_dx/4 - 10*pi*u0s_x**3*u1c_x*u0s_x_dx + 15*pi*u0s_x**3*u1r_x*u0s_x_dx + 5*pi*u0s_x**3*u1s_x*u0c_x_dx)
        h_x_dxx = q_x_dx / self.lmbda * self.L**2 / self.H / (1-self.p)

        return np.array([dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx, dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx, h_x_dx, h_x_dxx])
    
    def ivp(self, dz_r, dense_output=False):
        
        y_r = self.y_r(dz_r)

        x_range = [1, 0]
        sol = scipy.integrate.solve_ivp(
            self.deriv,
            x_range,
            y_r,
            rtol=1e-9,# atol=1e-7, 
            max_step=5e-2,  
            method='DOP853',
            dense_output=dense_output
        )
        
        return sol

    def solve_shooting(self):
        
        def mismatch(u_r):
            sol = self.ivp(u_r)

            dz0c_l, dz0s_l, u0c_l, u0s_l, dz1r_l, dz1c_l, dz1s_l, u1r_l, u1c_l, u1s_l = sol.y[:, -1]
            return [dz0c_l - 1, dz0s_l, dz1r_l, dz1c_l, dz1s_l]


        dz_guess = 0.001 * np.ones((5))
        res = scipy.optimize.root(mismatch, dz_guess)
        print(res)

        if res.status == 1:
            sol = self.ivp(res.x, dense_output=True)
            self.y = sol
            return res, sol
    
    def visualize_sol(self, start=0):

        try:
            x, y = self.y.t, self.y.y # solve_ivp (shooting method)
        except:
            x, y = self.y.x, self.y.y # solve_bvp

        y = y[:, x > start]
        x = x[x > start]

        fig, axs = plt.subplots(6, 6, figsize=(30, 30))
        labels=[r"$\zeta^0_{c1}$", r"$\zeta^0_{s1}$", r"$u^0_{c1}$", r"$u^0_{s1}$", r"$\zeta^1_{r}$", r"$\zeta^1_{c2}$", r"$\zeta^1_{s2}$", r"$u^1_{r}$", r"$u^1_{c2}$", r"$u^1_{s2}$"]

        y_x_dx = np.array([self.deriv(x_, y_) for x_, y_ in zip(x, y.T)]).T
        y0_x_dxx = np.array([self.y0_fx_dxx_hop(x_, y_[:4], y_dx_[:4]) for x_, y_, y_dx_ in zip(x, y.T, y_x_dx.T)]).T

        bnd_FO_hop = self.bnd_FO_hop 

        self.bnd_FO_hop = 1000
        y_x_dx_hop = np.array([self.deriv(x_, y_) for x_, y_ in zip(x, y.T)]).T
        self.bnd_FO_hop = bnd_FO_hop



        for i in range(4):
            axs[0, i].set_title(labels[i])
            axs[0, i].plot(x, y[i],'o-')
        for i in range(4):
            axs[1, i].plot(x, y_x_dx[i], 'o-')
            axs[1, i].plot(x, np.gradient(y[i], x, edge_order=2),'o-')
            axs[1, i].plot(x[1:], np.diff(y[i]) / np.diff(x),'o-')
            axs[1, i].legend(["analytical", "np.gradient (2nd order)", "np.diff (1st order)"])
        for i in range(4):
            axs[2, i].set_title(labels[i])
            axs[2, i].plot(x, y0_x_dxx[i],'o-')
            axs[2, i].plot(x, np.gradient(np.gradient(y[i], x, edge_order=2), x, edge_order=2),'o-')
            axs[2, i].legend(["analytical (only accurate at x = 1)", "np.gradient (2nd order)"])

        for i in range(6):
            axs[3, i].set_title(labels[4+i])
            axs[3, i].plot(x, y[4+i],'o-')
        for i in range(6):
            axs[4, i].plot(x, np.gradient(y[4+i], x, edge_order=2),'o-')
            axs[4, i].plot(x[1:], (np.diff(y[4+i]) / np.diff(x)),'o-')
            axs[4, i].plot(x, y_x_dx[4+i], 'o-')
        for i in range(3):
            axs[4, 3+i].plot(x, y_x_dx_hop[7+i], 'o-')
            axs[4, 3+i].legend(["analytical", "np.gradient (2nd order)", "np.diff (1st order)", "l'hopital"])

        for i in range(6):
            axs[5, i].plot(x, np.gradient(np.gradient(y[4+i], x, edge_order=2), x, edge_order=2),'o-')

        plt.show()

    def bc(self, y_left, y_right):
        dz0c_l, dz0s_l, u0c_l, u0s_l, dz1r_l, dz1c_l, dz1s_l, u1r_l, u1c_l, u1s_l = y_left
        dz0c_r, dz0s_r, u0c_r, u0s_r, dz1r_r, dz1c_r, dz1s_r, u1r_r, u1c_r, u1s_r = y_right

        h_r_dx = self.h_fx_dx(1)
        dz0c_r_dx, dz0s_r_dx, u0c_r_dx, u0s_r_dx = self.y0_fx_dx_hop(1, [dz0c_r, dz0s_r, u0c_r, u0s_r])

        return [
            dz0c_l - 1,
            dz0s_l,
            u0c_r - ( dz0s_r / h_r_dx),
            u0s_r - (-dz0c_r / h_r_dx),
            dz1r_l, 
            dz1s_l, 
            dz1c_l,
            u1r_r - 1 / h_r_dx * (1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r + dz0s_r * u0s_r_dx + dz0s_r_dx * u0s_r)),
            u1c_r - 1 / h_r_dx * (1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r - dz0s_r * u0s_r_dx - dz0s_r_dx * u0s_r) + 2 * dz1s_r),
            u1s_r - 1 / h_r_dx * (1 / 2 * (dz0c_r * u0s_r_dx + dz0c_r_dx * u0s_r + dz0s_r * u0c_r_dx + dz0s_r_dx * u0c_r) - 2 * dz1c_r)
        ]

    def bc_h(self, y_left, y_right):
        dz0c_l, dz0s_l, u0c_l, u0s_l, dz1r_l, dz1c_l, dz1s_l, u1r_l, u1c_l, u1s_l, h_l, h_l_dx = y_left
        dz0c_r, dz0s_r, u0c_r, u0s_r, dz1r_r, dz1c_r, dz1s_r, u1r_r, u1c_r, u1s_r, h_r, h_r_dx = y_right

        h_r_dx = self.h_fx_dx(1)
        dz0c_r_dx, dz0s_r_dx, u0c_r_dx, u0s_r_dx = self.y0_fx_dx_hop(1, [dz0c_r, dz0s_r, u0c_r, u0s_r])

        return [
            dz0c_l - 1,
            dz0s_l,
            u0c_r - ( dz0s_r / h_r_dx),
            u0s_r - (-dz0c_r / h_r_dx),
            dz1r_l, 
            dz1s_l, 
            dz1c_l,
            u1r_r - 1 / h_r_dx * (1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r + dz0s_r * u0s_r_dx + dz0s_r_dx * u0s_r)),
            u1c_r - 1 / h_r_dx * (1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r - dz0s_r * u0s_r_dx - dz0s_r_dx * u0s_r) + 2 * dz1s_r),
            u1s_r - 1 / h_r_dx * (1 / 2 * (dz0c_r * u0s_r_dx + dz0c_r_dx * u0s_r + dz0s_r * u0c_r_dx + dz0s_r_dx * u0c_r) - 2 * dz1c_r),
            h_l,
            h_r - self.max_h
        ]

    def solve_bvp(self):
        x_x = linspace(0, 1, 2000)
        y_guess = 0.1 * np.ones((10, len(x_x)))

        # sol = scipy.integrate.solve_bvp(self.deriv, self.bc, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        sol = scipy.integrate.solve_bvp(self.deriv, self.bc, x_x, y_guess, tol=1e-7, max_nodes=10000, verbose=2)
        self.y = sol
        if sol.status or self.debug:
            print(sol)
            raise SystemError 

    def solve_h(self):
        x_x = linspace(0, 1, 1000)
        y_guess = 0.1 * np.ones((12, len(x_x)))
        y_guess[-2, :] = self.h_fx(x_x)
        y_guess[-1, :] = self.h_fx_dx(x_x)

        self.max_h = self.h_fx(1)
        
        sol = scipy.integrate.solve_bvp(self.deriv_h, self.bc_h, x_x, y_guess, max_nodes=10000, verbose=2)
        self.y = sol
        if sol.status or self.debug:
            print(sol)
            raise SystemError
        
    def visualize_sol_h(self):
        fig, axs = plt.subplots(2, 6, figsize=(30, 10))
        labels=[r"$\zeta^0_{c1}$", r"$\zeta^0_{s1}$", r"$u^0_{c1}$", r"$u^0_{s1}$", r"$\zeta^1_{r}$", r"$\zeta^1_{c2}$", r"$\zeta^1_{s2}$", r"$u^1_{r}$", r"$u^1_{c2}$", r"$u^1_{s2}$", r"$h$", r"$h_x$"]
        for i in range(4):
            axs[0, i].set_title(labels[i])
            axs[0, i].plot(self.y.x, self.y.y[i])
        for i in range(6):
            axs[1, i].set_title(labels[4 + i])
            axs[1, i].plot(self.y.x, self.y.y[4 + i], 'k')
        for i in range(2):
            axs[0, 4+i].set_title(labels[10 + i])
            axs[0, 4+i].plot(self.y.x, self.y.y[10 + i], 'brown')
        plt.show()
        

if __name__ == "__main__":
    pcswe = PCSWE_mb()
    pcswe.h0 = 0.1
    pcswe.r = 0.45

    pcswe.bnd_FO_hop = 1e-14
    pcswe.bnd_LO_hop = 1e-11
    pcswe.set_derivative_vars()

    res, sol = pcswe.solve_shooting()
    pcswe.visualize_sol()

    pcswe.solve_bvp()
    pcswe.visualize_sol(0.995)
