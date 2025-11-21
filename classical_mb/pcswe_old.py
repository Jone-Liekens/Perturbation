import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt, zeros, ones, nan
import scipy
import matplotlib.pyplot as plt
from scipy.special import roots_legendre, eval_legendre
from numpy.polynomial import chebyshev


# 
class PerturbationCSWESolution(): pass

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

    def generate_solution(self):
        # assume we have ran self.solve()
        t = linspace(0, 2*pi, 1000)
        x = self.y.x

        # x_mesh = x[:, np.newaxis] * ones(t.shape)[np.newaxis, :]
        # t_mesh = np.tile(t, (len(x), 1))


        dz0_xt = self.y.y[0][:, np.newaxis] * cos(t)[np.newaxis, :] + \
            self.y.y[1][:, np.newaxis] * sin(t)[np.newaxis, :]

        dz1_xt = self.y.y[4][:, np.newaxis] * ones(t.shape)[np.newaxis, :] + \
            self.y.y[5][:, np.newaxis] * cos(2*t)[np.newaxis, :] + \
            self.y.y[6][:, np.newaxis] * sin(2*t)[np.newaxis, :]


        u0_xt = self.y.y[2][:, np.newaxis] * cos(t)[np.newaxis, :] + \
            self.y.y[3][:, np.newaxis] * sin(t)[np.newaxis, :]

        u1_xt = self.y.y[7][:, np.newaxis] * ones(t.shape)[np.newaxis, :] + \
            self.y.y[8][:, np.newaxis] * cos(2*t)[np.newaxis, :] + \
            self.y.y[9][:, np.newaxis] * sin(2*t)[np.newaxis, :]
        
       
        dz_xt = dz0_xt + self.epsilon * dz1_xt
        u_xt = u0_xt + self.epsilon * u1_xt

        sol = PerturbationCSWESolution
        sol.x, sol.t, sol.dz_xt, sol.u_xt = x, t, dz_xt, u_xt
        return sol
    
    def h_fx(self, x): return x

    def h_fx_dx(self, x): return 1
    
    def h_fx_dxx(self, x): return 0

    def h_fx_dxxx(self, x): return 0

    def deriv_LO(self, x_x, y0_x):
        # x_x, y0_x = np.array(x_x), np.array(y0_x)
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

        print(u0c_x_dx, dz0s_x_dx, u0c_x)
        print("here", h_x_dx)

        if isinstance(x_x, np.ndarray):
            
            u0c_x_dx[-1] =  ( dz0s_x_dx[-1] - u0c_x[-1] * h_x_dxx)  / (2*h_x_dx[-1])
            u0s_x_dx[-1] =  (-dz0c_x_dx[-1] - u0s_x[-1] * h_x_dxx)  / (2*h_x_dx[-1])

            # u0c_x_dx = np.nan_to_num(u0c_x_dx, nan=0.0, posinf=0.0, neginf=0.0)

        elif abs(x_x - 1) < 1e-14:
            u0c_x_dx =  ( dz0s_x_dx - u0c_x * h_x_dxx)  / (2*h_x_dx)
            u0s_x_dx =  (-dz0c_x_dx - u0s_x * h_x_dxx)  / (2*h_x_dx)

        


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
        u1r_x_dx = 1 / (1 - h_x + self.small_number) * (h_x_dx * u1r_x              - 1 / 2 * (dz0c_x * u0c_x_dx + dz0c_x_dx * u0c_x + dz0s_x * u0s_x_dx + dz0s_x_dx * u0s_x))
        u1c_x_dx = 1 / (1 - h_x + self.small_number) * (h_x_dx * u1c_x - 2 * dz1s_x - 1 / 2 * (dz0c_x * u0c_x_dx + dz0c_x_dx * u0c_x - dz0s_x * u0s_x_dx - dz0s_x_dx * u0s_x))
        u1s_x_dx = 1 / (1 - h_x + self.small_number) * (h_x_dx * u1s_x + 2 * dz1c_x - 1 / 2 * (dz0s_x * u0c_x_dx + dz0s_x_dx * u0c_x + dz0c_x * u0s_x_dx + dz0c_x_dx * u0s_x))

        # derivatives of first order u at x = 1 (using l'hopital)
        # u1r_x_dx[-1], u1c_x_dx[-1], u1s_x_dx[-1] = self.lhopital_FO(x_x[-1], y1_x[:, -1], y0_x[:, -1], y0_x_dx[:, -1], (dz1r_x_dx[-1], dz1c_x_dx[-1], dz1s_x_dx[-1]))
        # self.product_dxx(x_x, y0_x, y0_x_dx)

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

    def bc_wall(self, y_left, y_right):
        dz0c_l, dz0s_l, u0c_l, u0s_l, dz1r_l, dz1c_l, dz1s_l, u1r_l, u1c_l, u1s_l = y_left
        dz0c_r, dz0s_r, u0c_r, u0s_r, dz1r_r, dz1c_r, dz1s_r, u1r_r, u1c_r, u1s_r = y_right

        return [
            dz0c_l - 1,
            dz0s_l,
            u0c_r,
            u0s_r,
            dz1r_l, 
            dz1s_l, 
            dz1c_l,
            u1r_r,
            u1c_r,
            u1s_r
        ]

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

        # sol = scipy.integrate.solve_bvp(self.deriv, self.bc, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        sol = scipy.integrate.solve_bvp(self.deriv, self.bc, self.x, y_guess, tol=self.tol, max_nodes=20000)
        self.y = sol
        if sol.status or self.debug:
            print(sol)
            raise SystemError

    def solve_LO(self):
        self.x = linspace(0, 1, 2000)
        
        # if you want to have a different initial mesh
        # dx = 0.003
        # transf1 = lambda x: (np.log(x + dx) - np.log(dx)) / (np.log(1 + dx) - np.log(dx))
        # self.x = transf1(self.x)
        y_guess = 0.1 * np.ones((4, len(self.x)))

        # sol = scipy.integrate.solve_bvp(self.deriv_LO, self.bc_moving_boundary_LO, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        sol = solve_bvp_s45(self.deriv_LO, self.bc_moving_boundary_LO, self.x, y_guess, s=3, tol=self.tol, max_nodes=20000, verbose=2)
        # sol = solve_bvp_s2(self.deriv_LO, self.bc_moving_boundary_LO, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        # sol = solve_bvp_orig(self.deriv_LO, self.bc_moving_boundary_LO, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        try:
            self.y0 = sol.sol
        except:
            self.y0 = sol
        return

        if sol.status or self.debug:
            print(sol)
            raise SystemError
        
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

        x = linspace(0, 1, 10000)
        fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        st = np.argmin(abs(x - 0.9))
        for nu in range(3):
            for i in range(4):
                axs[nu, i].plot(x[st:], self.y0.sol(x[st:], nu=nu)[i])
        plt.show()

    def transport(self):
        # assume solve has already been executed, and we have all 10 components (4 LO 6 FO) calculated
        # compute local transport

        sol = self.generate_solution()
        x = sol.x 
        u_xt = sol.u_xt

        try:
            # if solve has been run
            dz0_c, dz0_s, u0_c, u0_s, dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = self.y.y
            h_x = self.h_fx(x)
            h_x_dx = self.h_fx_dx(x)
        except:
            # if solve with h has been run
            dz0_c, dz0_s, u0_c, u0_s, dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s, h, h_x = self.y.y
            h_x = h(x)
            h_x_dx = h_x(x)


        d_dz0_c = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0_c - u0_s)
        d_dz0_s = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0_s + u0_c)

        d_u0_c = (-dz0_s + u0_c * h_x_dx)  / (1 - h_x + self.small_number)
        d_u0_s = ( dz0_c + u0_s * h_x_dx)  / (1 - h_x + self.small_number) 

        d_u1_r = 1 / (1 - h_x + self.small_number) * (h_x_dx * u1_r             - 1 / 2 * (dz0_c * d_u0_c + d_dz0_c * u0_c + dz0_s * d_u0_s + d_dz0_s * u0_s))
        d_u1_c = 1 / (1 - h_x + self.small_number) * (h_x_dx * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * d_u0_c + d_dz0_c * u0_c - dz0_s * d_u0_s - d_dz0_s * u0_s))
        d_u1_s = 1 / (1 - h_x + self.small_number) * (h_x_dx * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * d_u0_c + d_dz0_s * u0_c + dz0_c * d_u0_s + d_dz0_c * u0_s))


        # Engelund Hansen formula
        factor = 0.04 * self.c_d**(3/2) / (self.g * (self.s-1))**2 / self.d50 
        U5 = (self.A * self.sigma * self.L / self.H)**5

        print(factor)
        print(U5)



        # just u^5 dimensionless
        u5_xt = u_xt**5
        # instantaneous transport
        q_xt = factor * U5 * u_xt**5


        # time integration of instantaneous transport
        # -------------------------------------------
        u5_x = self.epsilon*((5*pi*d_u1_c/2 + 15*pi*d_u1_r/4)*u0_c**4 + (5*pi*u0_s*d_u1_s + 5*pi*u1_s*d_u0_s)*u0_c**3 + (15*pi*u0_s**2*d_u1_r/2 + 15*pi*u0_s*u1_r*d_u0_s)*u0_c**2 + (5*pi*u0_s**3*d_u1_s + 15*pi*u0_s**2*u1_r*d_u0_c + 15*pi*u0_s**2*u1_s*d_u0_s)*u0_c + (10*pi*u0_c**3*u1_c + 15*pi*u0_c**3*u1_r + 15*pi*u0_c**2*u0_s*u1_s + 5*pi*u0_s**3*u1_s)*d_u0_c - 5*pi*u0_s**4*d_u1_c/2 + 15*pi*u0_s**4*d_u1_r/4 - 10*pi*u0_s**3*u1_c*d_u0_s + 15*pi*u0_s**3*u1_r*d_u0_s)
        # analytically
        q_x = U5 * factor / 2 / pi * self.epsilon*((5*pi*u1_c/2 + 15*pi*u1_r/4)*u0_c**4 + 5*pi*u0_c**3*u0_s*u1_s + 15*pi*u0_c**2*u0_s**2*u1_r/2 + 5*pi*u0_c*u0_s**3*u1_s - 5*pi*u0_s**4*u1_c/2 + 15*pi*u0_s**4*u1_r/4)
        # trapezoidal rule
        q_x_trap = np.mean(q_xt, 1)
        # # Gauss_Legendre
        # roots, weights = roots_legendre(100)
        # q_xt_interp = np.array([np.interp(roots*2*pi, linspace(0, 2*pi, q_xt.shape[1]), q_xt[x_i, :]) for x_i in range(len(x))]) # time is equidistant
        # q_x_gl = np.sum(q_xt_interp * weights[np.newaxis, :], 1) * 2 * pi / self.sigma


        # spatial derivative of time integration of instantaneous transport
        # -----------------------------------------------------------------

        print(U5 * factor / 2 / pi / self.L * self.epsilon)
        
        u5_x_dx = self.epsilon*((5*pi*d_u1_c/2 + 15*pi*d_u1_r/4)*u0_c**4 + (5*pi*u0_s*d_u1_s + 5*pi*u1_s*d_u0_s)*u0_c**3 + (15*pi*u0_s**2*d_u1_r/2 + 15*pi*u0_s*u1_r*d_u0_s)*u0_c**2 + (5*pi*u0_s**3*d_u1_s + 15*pi*u0_s**2*u1_r*d_u0_c + 15*pi*u0_s**2*u1_s*d_u0_s)*u0_c + (10*pi*u0_c**3*u1_c + 15*pi*u0_c**3*u1_r + 15*pi*u0_c**2*u0_s*u1_s + 5*pi*u0_s**3*u1_s)*d_u0_c - 5*pi*u0_s**4*d_u1_c/2 + 15*pi*u0_s**4*d_u1_r/4 - 10*pi*u0_s**3*u1_c*d_u0_s + 15*pi*u0_s**3*u1_r*d_u0_s)
        q_x_dx = U5 * factor / 2 / pi / self.L * self.epsilon*((5*pi*d_u1_c/2 + 15*pi*d_u1_r/4)*u0_c**4 + (5*pi*u0_s*d_u1_s + 5*pi*u1_s*d_u0_s)*u0_c**3 + (15*pi*u0_s**2*d_u1_r/2 + 15*pi*u0_s*u1_r*d_u0_s)*u0_c**2 + (5*pi*u0_s**3*d_u1_s + 15*pi*u0_s**2*u1_r*d_u0_c + 15*pi*u0_s**2*u1_s*d_u0_s)*u0_c + (10*pi*u0_c**3*u1_c + 15*pi*u0_c**3*u1_r + 15*pi*u0_c**2*u0_s*u1_s + 5*pi*u0_s**3*u1_s)*d_u0_c - 5*pi*u0_s**4*d_u1_c/2 + 15*pi*u0_s**4*d_u1_r/4 - 10*pi*u0_s**3*u1_c*d_u0_s + 15*pi*u0_s**3*u1_r*d_u0_s)
        q_x_dx_num = np.gradient(q_x, x) / self.L
        u5_x = self.epsilon*((5*pi*u1_c/2 + 15*pi*u1_r/4)*u0_c**4 + 5*pi*u0_c**3*u0_s*u1_s + 15*pi*u0_c**2*u0_s**2*u1_r/2 + 5*pi*u0_c*u0_s**3*u1_s - 5*pi*u0_s**4*u1_c/2 + 15*pi*u0_s**4*u1_r/4)
        
        # plt.plot(x, u5_x_dx)
        # plt.show()
        # plt.plot(x, u5_x)
        # plt.show()
        

        # bedload transport
        # -----------------
        b_x = (1 - self.p) * self.lmbda * self.H / self.L * h_x_dx * np.ones(x.shape)

        # spatial derivative of bedload transport?
        # cannot compute this analytically without d^2h/dx^2  
        # numerical approx:
        b_x_dx = np.gradient(b_x, x) / self.L


        # effect on the time derivative of the bed

        print(self.H / self.sigma / (1-self.p))
        h_x_dt = (-q_x_dx + b_x_dx) / self.H / self.sigma / (1 - self.p) # (dimensionless)

        return x, u5_xt, q_xt, q_x, q_x_trap, q_x_dx, q_x_dx_num, b_x, b_x_dx, h_x_dt



    def dh_dx(self):
                # assume solve has already been executed, and we have all 10 components (4 LO 6 FO) calculated
        # compute local transport

        sol = self.generate_solution()
        x = sol.x 
        u_xt = sol.u_xt

        try:
            # if solve has been run
            dz0_c, dz0_s, u0_c, u0_s, dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = self.y.y
            h_x = self.h_fx(x)
            h_x_dx = self.h_fx_dx(x)
        except:
            # if solve with h has been run
            dz0_c, dz0_s, u0_c, u0_s, dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s, h, h_x = self.y.y
            h_x = h(x)
            h_x_dx = h_x(x)


        d_dz0_c = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0_c - u0_s)
        d_dz0_s = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0_s + u0_c)

        d_u0_c = (-dz0_s + u0_c * h_x_dx)  / (1 - h_x + self.small_number)
        d_u0_s = ( dz0_c + u0_s * h_x_dx)  / (1 - h_x + self.small_number) 

        d_u1_r = 1 / (1 - h_x + self.small_number) * (h_x_dx * u1_r             - 1 / 2 * (dz0_c * d_u0_c + d_dz0_c * u0_c + dz0_s * d_u0_s + d_dz0_s * u0_s))
        d_u1_c = 1 / (1 - h_x + self.small_number) * (h_x_dx * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * d_u0_c + d_dz0_c * u0_c - dz0_s * d_u0_s - d_dz0_s * u0_s))
        d_u1_s = 1 / (1 - h_x + self.small_number) * (h_x_dx * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * d_u0_c + d_dz0_s * u0_c + dz0_c * d_u0_s + d_dz0_c * u0_s))



        # Engelund Hansen formula
        factor = 0.04 * self.c_d**(3/2) / (self.g * (self.s-1))**2 / self.d50 * (self.A * self.sigma * self.L / self.H)**5


        # spatial derivative of time integration of instantaneous transport
        # -----------------------------------------------------------------
        q_x_dx =  factor / self.sigma / self.L * self.epsilon*((5*pi*d_u1_c/2 + 15*pi*d_u1_r/4)*u0_c**4 + (5*pi*u0_s*d_u1_s + 5*pi*u1_s*d_u0_s)*u0_c**3 + (15*pi*u0_s**2*d_u1_r/2 + 15*pi*u0_s*u1_r*d_u0_s)*u0_c**2 + (5*pi*u0_s**3*d_u1_s + 15*pi*u0_s**2*u1_r*d_u0_c + 15*pi*u0_s**2*u1_s*d_u0_s)*u0_c + (10*pi*u0_c**3*u1_c + 15*pi*u0_c**3*u1_r + 15*pi*u0_c**2*u0_s*u1_s + 5*pi*u0_s**3*u1_s)*d_u0_c - 5*pi*u0_s**4*d_u1_c/2 + 15*pi*u0_s**4*d_u1_r/4 - 10*pi*u0_s**3*u1_c*d_u0_s + 15*pi*u0_s**3*u1_r*d_u0_s)
         

        # bedload transport
        # -----------------
        b_x = (1 - self.p) * self.lmbda / self.L * self.H * h_x_dx * np.ones(x.shape)

        # spatial derivative of bedload transport?
        # cannot compute this analytically without d^2h/dx^2  
        # numerical approx:
        b_x_dx = np.gradient(b_x, x) / self.L  


        # effect on the time derivative of the bed
        h_x_dt = (-q_x_dx + b_x_dx) * self.sigma / self.H # (dimensionless)

        return h_x_dt


    def transport_step(self, timestep=1e-3):

        def cheby_approx(x, y):
            deg = 5
            x_mapped = 2 * x - 1
            coefs = chebyshev.chebfit(x_mapped, y, deg)
            p = chebyshev.Chebyshev(coefs, domain=[0, 1])
            return p, p.deriv()


        self.solve()
        
        x, u5_xt, q_xt, q_x, q_x_trap, q_x_dx, q_x_dx_num, b_x, b_x_dx, h_x_dt = self.transport()


        size = (np.max(h_x_dt) - np.min(h_x_dt))

        print(size)

        new_h = self.h_fx(x) + timestep * h_x_dt / size


        mx, mn = new_h[-1], new_h[0]

        new_h = 0.9 * new_h / (mx - mn) + (0 - mn)




        h_fx, h_fx_dx = cheby_approx(x, new_h)
        
        self.h_fx = h_fx
        self.h_fx_dx = h_fx_dx

        return
    
    def visualize_transport(self):
        x, u5_xt, q_xt, q_x, q_x_trap, q_x_dx, q_x_dx_num, b_x, b_x_dx, h_x_dt = self.transport()

        fig, axs = plt.subplots(2, 3, figsize=(9, 6))
        axs[0, 0].plot(x, self.h_fx(x), 'brown', linewidth=4)
        axs[0, 1].plot(x, q_x_trap, 'o')
        # axs[0, 1].plot(x, q_x_gl, 'x', ms=5)
        axs[0, 1].plot(x, q_x, 'k')
        axs[0, 2].plot(x, b_x)

        axs[1, 0].plot(x, h_x_dt, color='brown', linestyle='--')
        axs[1, 1].plot(x, q_x_dx)
        axs[1, 2].plot(x, b_x_dx)
        plt.show()

        return
  



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

   


