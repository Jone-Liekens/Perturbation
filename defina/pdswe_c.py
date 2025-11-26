import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt, zeros, ones, nan, real, imag
import scipy
import matplotlib.pyplot as plt
from scipy.special import roots_legendre, eval_legendre
from numpy.polynomial import chebyshev


class PDSWESolution(): pass

class PDSWE_C():

    def __init__(self):
        self.debug = False

        # geometry
        self.A = 0.72
        self.H = 7.12 
        self.L = 8e3

        # tunable
        self.r = 0.45
        self.h0 = 0.0025
        self.small_number = nan

        # defina
        self.a_r = 0.1
        self.dL = 0.1

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
        
    def g0_fx(self, x_x):
        h_x = self.h_fx(x_x)

        s1_x = scipy.special.erf(2 * (1 - h_x) / self.a_r)
        s2_x = exp(-4 * (1-h_x)**2 / self.a_r**2)
        eta0_x = 0.5 * (1 + s1_x)
        Y0_x  = eta0_x * (1 - h_x) + self.a_r / 4 / pi**0.5 * s2_x
        return np.array([s1_x, s2_x, eta0_x, Y0_x])
    
    def g0_fx_dx(self, x_x, g0_x):
        h_x, h_x_dx = self.h_fx(x_x), self.h_fx_dx(x_x)
        s1_x, s2_x, eta0_x, Y0_x = g0_x

        s1_x_dx = h_x_dx * (-4) / pi**0.5 / self.a_r * s2_x
        s2_x_dx = h_x_dx * 8 * (1-h_x) / self.a_r**2 * s2_x
        eta0_x_dx = 0.5 * s1_x_dx
        Y0_x_dx = 0.5 * (s1_x_dx * (1 - h_x) - (s1_x + 1) * h_x_dx) + self.a_r / 4 / pi**0.5 * s2_x_dx
        return np.array([s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx])

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

        sol = PDSWESolution
        sol.x, sol.t, sol.dz_xt, sol.u_xt = x, t, dz_xt, u_xt
        return sol
    
    def h_fx(self, x): return x

    def h_fx_dx(self, x): return 1

    def h_fx_dxx(self, x): return 0

    def y0_x_dx(self, x_x, y_x):
        h_x, h_x_dx = self.h_fx(x_x), self.h_fx_dx(x_x)

        g0_x = self.g0_fx(x_x)
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = self.g0_fx_dx(x_x, g0_x)
    
        dz_x, u_x = y_x
        dz_x_dx = 1 / self.kappa * (- self.r / Y0_x * u_x - 1j * u_x)
        u_x_dx = (-1j * eta0_x * dz_x - u_x * Y0_x_dx)  / Y0_x
        return [dz_x_dx, u_x_dx]

    def bc(self, y_l, y_r):
        dz_l, u_l = y_l
        dz_r, u_r = y_r
        return [dz_l - 1, u_r]
    
    def solve_coupled(self):
        x_x = np.linspace(0, 1 + self.dL, 1000)
        y_guess = 0.1 * np.ones((2, len(x_x))) + 0.1j * np.ones((2, len(x_x)))
        sol = scipy.integrate.solve_bvp(self.y0_x_dx, self.bc, x_x, y_guess, tol=1e-7, max_nodes=20000, verbose=2)
        self.y = sol

    def visualize_coupled(self):
        x_x = self.y.x
        dz_x, u_x = self.y.y 

        dzc_x = real(dz_x)
        dzs_x = imag(dz_x)
        uc_x = real(u_x)
        us_x = imag(u_x)

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].plot(x_x, dzc_x)
        axs[1].plot(x_x, dzs_x)
        axs[2].plot(x_x, uc_x)
        axs[3].plot(x_x, us_x)
        plt.show()

    def dz_x(self, x_x, u_x, u_x_dx):

        g0_x = self.g0_fx(x_x)
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = self.g0_fx_dx(x_x, g0_x)

        dz_x = 1j * (Y0_x_dx * u_x + Y0_x * u_x_dx) / eta0_x
        return dz_x

    def u_x_dx(self, x_x, dz_x, u_x):

        g0_x = self.g0_fx(x_x)
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = self.g0_fx_dx(x_x, g0_x)
    
        u_x_dx = (- eta0_x * 1j * dz_x - Y0_x_dx * u_x ) / Y0_x
        return u_x_dx

    def u_x_dxx(self, x_x, y_x):
        h_x, h_x_dx, h_x_dxx = self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x)

        # helper functions that often appear
        g0_x = self.g0_fx(x_x)
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = self.g0_fx_dx(x_x, g0_x)
    
        s1_x_dxx = -4 / pi**0.5 / self.a_r * (h_x_dxx * s2_x + h_x_dx * s2_x_dx)
        s2_x_dxx = 8 / self.a_r**2 * (h_x_dxx * (1-h_x) * s2_x - h_x_dx  * h_x_dx * s2_x + h_x_dx * (1-h_x) * s2_x_dx)
        Y0_x_dxx = 0.5 * (s1_x_dxx * (1 - h_x) - s1_x_dx * h_x_dx - s1_x_dx * h_x_dx - (s1_x + 1) * h_x_dxx) + self.a_r / 4 / pi**0.5 * s2_x_dxx

        u_x, u_x_dx = y_x
        A = (2 * Y0_x_dx * eta0_x - Y0_x * eta0_x_dx) / (Y0_x * eta0_x)
        B = (Y0_x_dxx * eta0_x - Y0_x_dx * eta0_x_dx + eta0_x**2 / self.kappa) / (Y0_x * eta0_x)
        C = - (eta0_x**2 * self.r / self.kappa / Y0_x) / (Y0_x * eta0_x)
        u_x_dxx = - A * u_x_dx - (B + C *1j) * u_x


        dz_x = 1j * (Y0_x_dx * u_x + Y0_x * u_x_dx) / eta0_x
        dz_x_dx = 1 / self.kappa * (- self.r / Y0_x * u_x - 1j * u_x)
        u_x_dxx = -1j * (
            eta0_x_dx * dz_x / Y0_x +
            eta0_x * dz_x_dx / Y0_x +
            - eta0_x * dz_x * Y0_x_dx / Y0_x**2
        ) - (
            Y0_x_dxx * u_x / Y0_x + 
            Y0_x_dx * u_x_dx / Y0_x +
            - Y0_x_dx * u_x * Y0_x_dx / Y0_x**2
        )

        return [u_x_dx, u_x_dxx]

    def ivp(self, u_start, dense_output=False):
        x_range = [0, 1 + self.dL]

        u_l = u_start[0] + 1j * u_start[1]
        dz_l = 1 + 0j

        u_l_dx = self.u_x_dx(0, dz_l, u_l)

        y0 = np.array([u_l, u_l_dx])
        sol = scipy.integrate.solve_ivp(self.u_x_dxx, x_range, y0, dense_output=dense_output, rtol=1e-7)

        return sol

    def solve_u_xx(self):

        def mismatch(s):
            # velocity should be zero at boundary
            sol = self.ivp(s)
            return [real(sol.y[0, -1]), imag(sol.y[0, -1])]


        s_guess = [0.1, 0.1]
        res = scipy.optimize.root(mismatch, s_guess, tol=1e-9)

        self.res = res
        print(res)

    def visualize_u_xx(self):
        
        sol = self.ivp(self.res.x, True)


        # x_x = np.linspace(0, 1, 1000)
        x_x = sol.t
        u_x, u_x_dx = sol.sol(x_x)

        

        g0_x = self.g0_fx(x_x)
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = self.g0_fx_dx(x_x, g0_x)

        dz_x = 1j * (Y0_x_dx * u_x + Y0_x * u_x_dx) / eta0_x

        print(dz_x)

        uc_x, us_x = real(u_x), imag(u_x)
        dzc_x, dzs_x = real(dz_x), imag(dz_x)

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].plot(x_x, dzc_x, 'o-')
        axs[1].plot(x_x, dzs_x, 'o-')
        axs[2].plot(x_x, uc_x, 'o-')
        axs[3].plot(x_x, us_x, 'o-')
        plt.show()

    def u_x_dxx_test(self, x_x, y_x):
        h_x, h_x_dx, h_x_dxx = self.h_fx(x_x), self.h_fx_dx(x_x), self.h_fx_dxx(x_x)

        # helper functions that often appear
        g0_x = self.g0_fx(x_x)
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = self.g0_fx_dx(x_x, g0_x)
    

        s1_x_dxx = -4 / pi**0.5 / self.a_r * (h_x_dxx * s2_x + h_x_dx * s2_x_dx)
        s2_x_dxx = 8 / self.a_r**2 * (h_x_dxx * (1-h_x) * s2_x - h_x_dx  * h_x_dx * s2_x + h_x_dx * (1-h_x) * s2_x_dx)
        Y0_x_dxx = 0.5 * (s1_x_dxx * (1 - h_x) - s1_x_dx * h_x_dx - s1_x_dx * h_x_dx - (s1_x + 1) * h_x_dxx) + self.a_r / 4 / pi**0.5 * s2_x_dxx

        u_x, u_x_dx = y_x

        dz_x = 1j * (Y0_x_dx * u_x + Y0_x * u_x_dx) / eta0_x
        dz_x_dx = 1 / self.kappa * (- self.r / Y0_x * u_x - 1j * u_x)
        # dz_x_dx2 = 1j * (Y0_x_dxx * u_x + Y0_x_dx * u_x_dx + Y0_x_dx + u_x_dx + Y0_x * u_x_dxx) / eta0_x - 1j / eta0_x**2 * eta0_x_dx * (Y0_x_dx * u_x + Y0_x * u_x_dx) 
        
        u_x_dxx = -1j * (
            eta0_x_dx * dz_x / Y0_x +
            eta0_x * dz_x_dx / Y0_x +
            - eta0_x * dz_x * Y0_x_dx / Y0_x**2
        ) - (
            Y0_x_dxx * u_x / Y0_x + 
            Y0_x_dx * u_x_dx / Y0_x +
            - Y0_x_dx * u_x * Y0_x_dx / Y0_x**2
        )

        A = (2 * Y0_x_dx * eta0_x - Y0_x * eta0_x_dx) / (Y0_x * eta0_x)
        B = (Y0_x_dxx * eta0_x - Y0_x_dx * eta0_x_dx + eta0_x**2 / self.kappa) / (Y0_x * eta0_x)
        C = - (eta0_x**2 * self.r / self.kappa / Y0_x) / (Y0_x * eta0_x)

        u_x_dxx2 = - A * u_x_dx - (B + C *1j) * u_x

        plt.plot(x_x, u_x_dxx, 'o')
        plt.plot(x_x, u_x_dxx2)
        plt.show()

        return [u_x_dx, u_x_dxx]



if __name__ == "__main__":
    pdswec = PDSWE_C()

    pdswec.a_r = 0.1
    pdswec.dL = 0
    pdswec.set_derivative_vars()

    # pdswec.solve_coupled()
    # pdswec.visualize_coupled()

    pdswec.solve_u_xx()
    pdswec.visualize_u_xx()