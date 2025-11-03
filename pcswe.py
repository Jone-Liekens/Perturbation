import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt, zeros, ones, nan
import scipy
import matplotlib.pyplot as plt

class PerturbationCSWE():

    def __init__(self):
        self.debug = False


  

        # geometry
        self.A = 0.72
        self.H = 7.12 
        self.L = 8e3

        # tunable
        self.r = 0.24
        self.h0 = 0.0025
        self.small_number = nan
        self.bc = self.bc_moving_boundary



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

    def deriv_LO(self, x_x, y0_x):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        h_x, h_x_dx, h_x_dxx = self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x)

        # derivatives of dzeta (using momentum equation)
        dz0c_x_dx = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0c_x - u0s_x)
        dz0s_x_dx = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0s_x + u0c_x)

        # derivatives of u (using continuity equation)
        with np.errstate(divide='ignore', invalid='ignore'):
            u0c_x_dx = (-dz0s_x + u0c_x * h_x_dx)  / (1 - h_x)
            u0s_x_dx = ( dz0c_x + u0s_x * h_x_dx)  / (1 - h_x)

        # derivatives of u at x = 1 (using l'hopital)
        u0c_x_dx[-1] =  ( dz0s_x_dx[-1] - u0c_x[-1] * h_x_dxx)  / (2*h_x_dx)
        u0s_x_dx[-1] =  (-dz0c_x_dx[-1] - u0s_x[-1] * h_x_dxx)  / (2*h_x_dx)

        return np.array([dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx])
    
    def deriv_FO(self, x_x, y1_x, y0_x, y0_x_dx):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x = y1_x
        h_x, h_x_dx, h_x_dxx, h_x_dxxx = self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x), self.h_fx_dxxx(x_x)

        # derivatives of dzeta (using momentum equation)
        dz1r_x_dx = (1 / (1 - h_x + self.h0) * (- self.r * u1r_x - 0.5 * (  dz0c_x *  u0s_x - dz0s_x *  u0c_x)
            - 0.5 * (  dz0s_x * dz0s_x_dx + dz0c_x * dz0c_x_dx) * self.kappa) - 0.5 * (u0c_x * u0c_x_dx + u0s_x * u0s_x_dx)            ) / self.kappa
        dz1c_x_dx = (1 / (1 - h_x + self.h0) * (- self.r * u1c_x - 0.5 * (  dz0c_x *  u0s_x + dz0s_x *  u0c_x)
            - 0.5 * (- dz0s_x * dz0s_x_dx + dz0c_x * dz0c_x_dx) * self.kappa) - 0.5 * (u0c_x * u0c_x_dx - u0s_x * u0s_x_dx) - 2 * u1s_x) / self.kappa
        dz1s_x_dx = (1 / (1 - h_x + self.h0) * (- self.r * u1s_x + 0.5 * (  dz0c_x *  u0c_x - dz0s_x *  u0s_x)
            - 0.5 * (  dz0c_x * dz0s_x_dx + dz0s_x * dz0c_x_dx) * self.kappa) - 0.5 * (u0c_x * u0s_x_dx + u0s_x * u0c_x_dx) + 2 * u1c_x) / self.kappa

        # derivatives of u (using continuity equation)
        u1r_x_dx = 1 / (1 - h_x) * (h_x_dx * u1r_x              - 1 / 2 * (dz0c_x * u0c_x_dx + dz0c_x_dx * u0c_x + dz0s_x * u0s_x_dx + dz0s_x_dx * u0s_x))
        u1c_x_dx = 1 / (1 - h_x) * (h_x_dx * u1c_x - 2 * dz1s_x - 1 / 2 * (dz0c_x * u0c_x_dx + dz0c_x_dx * u0c_x - dz0s_x * u0s_x_dx - dz0s_x_dx * u0s_x))
        u1s_x_dx = 1 / (1 - h_x) * (h_x_dx * u1s_x + 2 * dz1c_x - 1 / 2 * (dz0s_x * u0c_x_dx + dz0s_x_dx * u0c_x + dz0c_x * u0s_x_dx + dz0c_x_dx * u0s_x))

        # derivatives of first order u at x = 1 (using l'hopital)
        u1r_x_dx[-1], u1c_x_dx[-1], u1s_x_dx[-1] = self.lhopital_FO(x_x[-1], y1_x[:, -1], y0_x[:, -1], y0_x_dx[:, -1], (dz1r_x_dx[-1], dz1c_x_dx[-1], dz1s_x_dx[-1]))


        self.product_dxx(x_x, y0_x, y0_x_dx)

        return np.array([dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx])

    def lhopital_FO(self, x_r, y1_r, y0_r, y0_r_dx, dz1_r_dx):
        dz0c_r, dz0s_r, u0c_r, u0s_r = y0_r
        dz0c_r_dx, dz0s_r_dx, u0c_r_dx, u0s_r_dx = y0_r_dx
        dz1r_r, dz1c_r, dz1s_r, u1r_r, u1c_r, u1s_r = y1_r
        dz1r_r_dx, dz1c_r_dx, dz1s_r_dx = dz1_r_dx
        h_r, h_r_dx, h_r_dxx, h_r_dxxx = self.h_fx(x_r), self.h_fx_dx(x_r), self.h_fx_dxx(x_r), self.h_fx_dxxx(x_r)

        # second derivatives of leading order components:
        dz0c_r_dxx = 1 / self.kappa * (- self.r / (1 - h_r + self.h0)**2 * h_r_dx * u0c_r - self.r / (1 - h_r + self.h0) * u0c_r_dx - u0s_r_dx)
        dz0s_r_dxx = 1 / self.kappa * (- self.r / (1 - h_r + self.h0)**2 * h_r_dx * u0s_r - self.r / (1 - h_r + self.h0) * u0s_r_dx + u0c_r_dx)
        u0c_r_dxx = 0.5 * (- h_r_dxx / h_r_dx**2 * ( dz0s_r_dx - u0c_r * h_r_dxx) + 1 / h_r_dx * ( dz0s_r_dxx - u0c_r_dx * h_r_dxx - u0c_r * h_r_dxxx))
        u0s_r_dxx = 0.5 * (- h_r_dxx / h_r_dx**2 * (-dz0c_r_dx - u0s_r * h_r_dxx) + 1 / h_r_dx * (-dz0c_r_dxx - u0s_r_dx * h_r_dxx - u0s_r * h_r_dxxx))

        # second derivatives of products of leading order components:
        dz_u_cc_r_dxx = (dz0c_r * u0c_r_dxx + 2 * dz0c_r_dx * u0c_r_dx + dz0c_r_dxx * u0c_r)
        dz_u_ss_r_dxx = (dz0s_r * u0s_r_dxx + 2 * dz0s_r_dx * u0s_r_dx + dz0s_r_dxx * u0s_r)
        dz_u_cs_r_dxx = (dz0s_r * u0c_r_dxx + 2 * dz0s_r_dx * u0c_r_dx + dz0s_r_dxx * u0c_r)
        dz_u_sc_r_dxx = (dz0c_r * u0s_r_dxx + 2 * dz0c_r_dx * u0s_r_dx + dz0c_r_dxx * u0s_r)

        # l'hopital
        u1r_r_dx = (             - u1r_r * h_r_dxx + (dz_u_cc_r_dxx + dz_u_ss_r_dxx) / 2) / (2*h_r_dx)
        u1c_r_dx = ( 2*dz1s_r_dx - u1c_r * h_r_dxx + (dz_u_cc_r_dxx - dz_u_ss_r_dxx) / 2) / (2*h_r_dx)
        u1s_r_dx = (-2*dz1c_r_dx - u1s_r * h_r_dxx + (dz_u_cs_r_dxx + dz_u_sc_r_dxx) / 2) / (2*h_r_dx)

        return np.array([u1r_r_dx, u1c_r_dx, u1s_r_dx])
    
    def deriv_interpolate(self, x_x, y_x, x_start, n_points, degree, mu, debug=False):
        # return derivatives at x = x_start by making a polynomial interpolation

        vandermonde = lambda x_vec, x_start, degree: (x_vec - x_start)[:, np.newaxis]  ** np.arange(0, degree+1)[np.newaxis, :]
        
        A = vandermonde(x_x[-n_points:], x_start, degree)
        p, res, A_rank, A_svd = np.linalg.lstsq(A, y_x[-n_points:])

        if debug:
            continuous_x = np.linspace(x_x[-n_points], x_x[-1], 1000)
            plt.plot(x_x[-n_points:], y_x[-n_points:], 'o')
            plt.plot(continuous_x, vandermonde(continuous_x, x_start, degree) @ p)
            plt.show()

        # take mu'th derivative, at point x_point, of the polynomial with coefficients p for (x-x_start)
        deriv = lambda x_point, x_start, p, mu: (x_point - x_start)**np.arange(0, len(p)) @ np.linalg.matrix_power(np.diag(np.arange(1, len(p)-1+1), k=1), mu) @ p
        return deriv(x_start, x_start, p, mu)

    def product_dxx(self, x_x, y0_x, y0_x_dx):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        h_x, h_x_dx, h_x_dxx, h_x_dxxx = self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x), self.h_fx_dxxx(x_x)

        print("Estimating second derivative of products")

        if True: # analytical
            x_r = x_x[-1]
            dz0c_r, dz0s_r, u0c_r, u0s_r = y0_x[:, -1]
            dz0c_r_dx, dz0s_r_dx, u0c_r_dx, u0s_r_dx = y0_x_dx[:, -1]
            h_r, h_r_dx, h_r_dxx, h_r_dxxx = self.h_fx(x_r), self.h_fx_dx(x_r), self.h_fx_dxx(x_r), self.h_fx_dxxx(x_r)

            # second derivatives of leading order components:
            dz0c_r_dxx = 1 / self.kappa * (- self.r / (1 - h_r + self.h0)**2 * h_r_dx * u0c_r - self.r / (1 - h_r + self.h0) * u0c_r_dx - u0s_r_dx)
            dz0s_r_dxx = 1 / self.kappa * (- self.r / (1 - h_r + self.h0)**2 * h_r_dx * u0s_r - self.r / (1 - h_r + self.h0) * u0s_r_dx + u0c_r_dx)
            u0c_r_dxx = 0.5 * (- h_r_dxx / h_r_dx**2 * ( dz0s_r_dx - u0c_r * h_r_dxx) + 1 / h_r_dx * ( dz0s_r_dxx - u0c_r_dx * h_r_dxx - u0c_r * h_r_dxxx))
            u0s_r_dxx = 0.5 * (- h_r_dxx / h_r_dx**2 * (-dz0c_r_dx - u0s_r * h_r_dxx) + 1 / h_r_dx * (-dz0c_r_dxx - u0s_r_dx * h_r_dxx - u0s_r * h_r_dxxx))

            # second derivatives of products of leading order components:
            dz_u_cc_r_dxx = (dz0c_r * u0c_r_dxx + 2 * dz0c_r_dx * u0c_r_dx + dz0c_r_dxx * u0c_r)
            dz_u_ss_r_dxx = (dz0s_r * u0s_r_dxx + 2 * dz0s_r_dx * u0s_r_dx + dz0s_r_dxx * u0s_r)
            dz_u_cs_r_dxx = (dz0s_r * u0c_r_dxx + 2 * dz0s_r_dx * u0c_r_dx + dz0s_r_dxx * u0c_r)
            dz_u_sc_r_dxx = (dz0c_r * u0s_r_dxx + 2 * dz0c_r_dx * u0s_r_dx + dz0c_r_dxx * u0s_r)

            print(dz_u_cc_r_dxx )

        if True:

            n_points = 3
            degree = 2

            dz_u_cc_r_dxx = self.deriv_interpolate(x_x, dz0c_x * u0c_x, x_x[-1], n_points, degree, 2)
            print(dz_u_cc_r_dxx)
            dz_u_cc_r_dxx = self.deriv_interpolate(x_x, dz0c_x_dx * u0c_x + dz0c_x * u0c_x_dx, x_x[-1], n_points, degree, 1)
            print(dz_u_cc_r_dxx)

            n_points = 20
            degree = 4

            dz_u_cc_r_dxx = self.deriv_interpolate(x_x, dz0c_x * u0c_x, x_x[-1], n_points, degree, 2)
            print(dz_u_cc_r_dxx)
            dz_u_cc_r_dxx = self.deriv_interpolate(x_x, dz0c_x_dx * u0c_x + dz0c_x * u0c_x_dx, x_x[-1], n_points, degree, 1)
            print(dz_u_cc_r_dxx)

        
        a = input("continue?")
            
    def deriv(self, x_x, y_x):
        y0_x, y1_x = np.split(y_x, [4], axis=0)

        y0_x_dx = self.deriv_LO(x_x, y0_x)
        y1_x_dx = self.deriv_FO(x_x, y1_x, y0_x, y0_x_dx)

        y_x_dx = np.concatenate((y0_x_dx, y1_x_dx), axis=0)

        return y_x_dx

    def bc_moving_boundary(self, y_left, y_right):
        dz0c_l, dz0s_l, u0c_l, u0s_l, dz1r_l, dz1c_l, dz1s_l, u1r_l, u1c_l, u1s_l = y_left
        dz0c_r, dz0s_r, u0c_r, u0s_r, dz1r_r, dz1c_r, dz1s_r, u1r_r, u1c_r, u1s_r = y_right

        h_r, h_r_dx, h_r_dxx = self.h_fx(1), self.h_fx_dx(1), self.h_fx_dxx(1)

        dz0c_r_dx = 1 / self.kappa * ( - self.r / (1 - h_r + self.h0) * u0c_r - u0s_r)
        dz0s_r_dx = 1 / self.kappa * ( - self.r / (1 - h_r + self.h0) * u0s_r + u0c_r)

        u0c_r_dx =  ( dz0s_r_dx - u0c_r * h_r_dxx)  / (2*h_r_dx)
        u0s_r_dx =  (-dz0c_r_dx - u0s_r * h_r_dxx)  / (2*h_r_dx)

        return [
            dz0c_l - 1,
            dz0s_l,
            dz0s_r - h_r_dx * u0c_r,
            dz0c_r + h_r_dx * u0s_r,
            dz1r_l, 
            dz1s_l, 
            dz1c_l,
            h_r_dx * u1r_r - 1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r + dz0s_r * u0s_r_dx + dz0s_r_dx * u0s_r),
            h_r_dx * u1c_r - 1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r - dz0s_r * u0s_r_dx - dz0s_r_dx * u0s_r) - 2 * dz1s_r,
            h_r_dx * u1s_r - 1 / 2 * (dz0c_r * u0s_r_dx + dz0c_r_dx * u0s_r + dz0s_r * u0c_r_dx + dz0s_r_dx * u0c_r) + 2 * dz1c_r
        ]
    
    def bc_moving_boundary_LO(self, y0_left, y0_right):
        dz0c_l, dz0s_l, u0c_l, u0s_l = y0_left
        dz0c_r, dz0s_r, u0c_r, u0s_r = y0_right

        h_r, h_r_dx, h_r_dxx = self.h_fx(1), self.h_fx_dx(1), self.h_fx_dxx(1)

        return [
            dz0c_l - 1,
            dz0s_l,
            dz0s_r - h_r_dx * u0c_r,
            dz0c_r + h_r_dx * u0s_r,
        ]
    
    def bc_moving_boundary_FO(self, y1_left, y1_right, y0_left, y0_right):
        dz0c_l, dz0s_l, u0c_l, u0s_l = y0_left
        dz0c_r, dz0s_r, u0c_r, u0s_r = y0_right
        dz1r_l, dz1c_l, dz1s_l, u1r_l, u1c_l, u1s_l = y1_left
        dz1r_r, dz1c_r, dz1s_r, u1r_r, u1c_r, u1s_r = y1_right

        h_r, h_r_dx, h_r_dxx = self.h_fx(1), self.h_fx_dx(1), self.h_fx_dxx(1)

        dz0c_r_dx = 1 / self.kappa * ( - self.r / (1 - h_r + self.h0) * u0c_r - u0s_r)
        dz0s_r_dx = 1 / self.kappa * ( - self.r / (1 - h_r + self.h0) * u0s_r + u0c_r)

        u0c_r_dx =  ( dz0s_r_dx - u0c_r * h_r_dxx)  / (2*h_r_dx)
        u0s_r_dx =  (-dz0c_r_dx - u0s_r * h_r_dxx)  / (2*h_r_dx)

        return [
            dz1r_l, 
            dz1s_l, 
            dz1c_l,
            h_r_dx * u1r_r - 1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r + dz0s_r * u0s_r_dx + dz0s_r_dx * u0s_r),
            h_r_dx * u1c_r - 1 / 2 * (dz0c_r * u0c_r_dx + dz0c_r_dx * u0c_r - dz0s_r * u0s_r_dx - dz0s_r_dx * u0s_r) - 2 * dz1s_r,
            h_r_dx * u1s_r - 1 / 2 * (dz0c_r * u0s_r_dx + dz0c_r_dx * u0s_r + dz0s_r * u0c_r_dx + dz0s_r_dx * u0c_r) + 2 * dz1c_r
        ]
    
    def bc_combined(self, y_l, y_r):
        y0_l, y1_l = np.split(y_l, [4], axis=0)
        y0_r, y1_r = np.split(y_r, [4], axis=0)

        bc0 = self.deriv_LO(y0_l, y0_r)
        bc1 = self.deriv_FO(y1_l, y1_r, y0_l, y0_r)

        bc = np.concatenate((bc0, bc1), axis=0)
        return bc

    def solve(self):
        self.x = linspace(0, 1, 2000)

        # if you want to have a different initial mesh
        # dx = 0.01
        # transf1 = lambda x: (np.log(x + dx) - np.log(dx)) / (np.log(1 + dx) - np.log(dx))
        # self.x = transf1(self.x)
        # plt.plot(self.x[::50], ones(self.x.shape)[::50], 'o')
        # plt.show()

        # initial guess
        y_guess = 0.1 * np.ones((10, len(self.x)))

        sol = scipy.integrate.solve_bvp(self.deriv, self.bc, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)

        if sol.status or self.debug:
            print(sol)
            raise SystemError
        
        self.y = sol

    def solve_LO(self):
        self.x = linspace(0, 1, 2000)
        y_guess = 0.1 * np.ones((4, len(self.x)))
        sol = scipy.integrate.solve_bvp(self.deriv_LO, self.bc_moving_boundary_LO, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)

        if sol.status or self.debug:
            print(sol)
            raise SystemError
        
        self.y0 = sol

    def visualize_components(self):
        # assume self.solve() has been called

        fig, axs = plt.subplots(2, 6, figsize=(30, 10))
        labels=[r"$\zeta^0_{c1}$", r"$\zeta^0_{s1}$", r"$u^0_{c1}$", r"$u^0_{s1}$", r"$\zeta^1_{r}$", r"$\zeta^1_{c2}$", r"$\zeta^1_{s2}$", r"$u^1_{r}$", r"$u^1_{c2}$", r"$u^1_{s2}$"]
        for i in range(4):
            axs[0, i].set_title(labels[i])
            axs[0, i].plot(self.y.x, self.y.y[i])
        for i in range(6):
            axs[1, i].set_title(labels[4 + i])
            axs[1, i].plot(self.y.x, self.y.y[4 + i])

        plt.show()

    def visualize_LO(self):
        fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        st = np.argmin(abs(self.y0.x - 0.9))
        for nu in range(3):
            for i in range(4):
                axs[nu, i].plot(self.y0.x[st:], self.y0.sol(self.y0.x[st:], nu=nu)[i])
        plt.show()





if __name__ == "__main__":
    pcswe = PerturbationCSWE()
    pcswe.H = 7.12
    pcswe.A = 0.72
    pcswe.L = 8e3

    pcswe.r = 0.45
    pcswe.h0 = 0.1

    pcswe.tol = 1e-8
    pcswe.small_number = 0

    # pcswe.solve()
    # pcswe.visualize_components()

    pcswe.solve_LO()
    pcswe.product_dxx(pcswe.y0.x, pcswe.y0.y, pcswe.deriv_LO(pcswe.y0.x, pcswe.y0.y))
    pcswe.visualize_LO()

   


