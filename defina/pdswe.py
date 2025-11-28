import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt, zeros, ones, nan
import scipy
import matplotlib.pyplot as plt


class PDSWE_sol(): pass

class PDSWE():

    def __init__(self):
        self.debug = False


  

        # geometry
        self.A = 0.72
        self.H = 7.12 
        self.L = 8e3

        # tunable
        self.r = 0.24
        self.small_number = nan

        # defina
        self.a_r = 0
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


    def signature(self):
        return r"DSWE $r = {}, a_r = {}, dL = {}, h(1) = {} (A = {}, H = {}, L = {})$".format(self.r, self.a_r, self.dL, self.h_fx(1), self.A, self.H, self.L)



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

        sol = PDSWE_sol()
        sol.x, sol.t, sol.dz_xt, sol.u_xt = x, t, dz_xt, u_xt
        return sol

    def h_fx(self, x): return x

    def h_fx_dx(self, x): return 1 
    
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

    def y0_fx_dx(self, x_x, y0_x, g0_x, g0_x_dx):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = g0_x_dx

        dz0c_x_dx = 1 / self.kappa * (- self.r / Y0_x * u0c_x - u0s_x)
        dz0s_x_dx = 1 / self.kappa * (- self.r / Y0_x * u0s_x + u0c_x)
        u0c_x_dx = (-eta0_x * dz0s_x - u0c_x * Y0_x_dx)  / Y0_x
        u0s_x_dx = ( eta0_x * dz0c_x - u0s_x * Y0_x_dx)  / Y0_x
        return np.array([dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx])
    
    def g1_fx(self, x_x, y0_x, y0_x_dx, g0_x, g0_x_dx):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = g0_x_dx

        eta1c_x = dz0c_x * 2 / pi**0.5 / self.a_r * s2_x
        eta1s_x = dz0s_x * 2 / pi**0.5 / self.a_r * s2_x
        Y1c_x = dz0c_x * eta0_x
        Y1s_x = dz0s_x * eta0_x
        return np.array([eta1c_x, eta1s_x, Y1c_x, Y1s_x])
    
    def g1_fx_dx(self, x_x, y0_x, y0_x_dx, g0_x, g0_x_dx):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = g0_x_dx

        eta1c_x_dx = 2 / pi**0.5 / self.a_r * (dz0c_x_dx * s2_x + dz0c_x * s2_x_dx)
        eta1s_x_dx = 2 / pi**0.5 / self.a_r * (dz0s_x_dx * s2_x + dz0s_x * s2_x_dx)
        Y1c_x_dx = dz0c_x_dx * eta0_x + dz0c_x * 0.5 * s1_x_dx
        Y1s_x_dx = dz0s_x_dx * eta0_x + dz0s_x * 0.5 * s1_x_dx
        return np.array([eta1c_x_dx, eta1s_x_dx, Y1c_x_dx, Y1s_x_dx])
     
    def y1_fx_dx(self, x_x, y0_x, y0_x_dx, y1_x, g0_x, g0_x_dx, g1_x, g1_x_dx):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        s1_x, s2_x, eta0_x, Y0_x = g0_x
        s1_x_dx, s2_x_dx, eta0_x_dx, Y0_x_dx = g0_x_dx


        dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x = y1_x
        eta1c_x, eta1s_x, Y1c_x, Y1s_x = g1_x
        eta1c_x_dx, eta1s_x_dx, Y1c_x_dx, Y1s_x_dx = g1_x_dx


        dz1r_x_dx = (
            (- self.r * u1r_x - 0.5 * (  Y1c_x *  u0s_x - Y1s_x *  u0c_x) - 0.5 * ( Y1c_x * dz0c_x_dx + Y1s_x * dz0s_x_dx) * self.kappa) / Y0_x - 0.5 * (u0c_x * u0c_x_dx + u0s_x * u0s_x_dx)            ) / self.kappa 
        dz1c_x_dx = (
            (- self.r * u1c_x - 0.5 * (  Y1c_x *  u0s_x + Y1s_x *  u0c_x) - 0.5 * ( Y1c_x * dz0c_x_dx - Y1s_x * dz0s_x_dx) * self.kappa) / Y0_x - 0.5 * (u0c_x * u0c_x_dx - u0s_x * u0s_x_dx) - 2 * u1s_x) / self.kappa
        dz1s_x_dx = (
            (- self.r * u1s_x - 0.5 * ( -Y1c_x *  u0c_x + Y1s_x *  u0s_x) - 0.5 * ( Y1c_x * dz0s_x_dx + Y1s_x * dz0c_x_dx) * self.kappa) / Y0_x - 0.5 * (u0c_x * u0s_x_dx + u0s_x * u0c_x_dx) + 2 * u1c_x) / self.kappa
        
        u1r_x_dx = -1 / Y0_x * (
            + 0.5 * ( eta1c_x * dz0s_x - eta1s_x * dz0c_x)
            + 0.5 * ( Y1c_x_dx * u0c_x + Y1s_x_dx * u0s_x)
            + 0.5 * ( Y1c_x * u0c_x_dx + Y1s_x * u0s_x_dx)
            + Y0_x_dx * u1r_x
        )

        u1c_x_dx = -1 / Y0_x * (
            + 2 * eta0_x * dz1s_x
            + 0.5 * ( eta1c_x * dz0s_x + eta1s_x * dz0c_x)
            + 0.5 * ( Y1c_x_dx * u0c_x - Y1s_x_dx * u0s_x)
            + 0.5 * ( Y1c_x * u0c_x_dx - Y1s_x * u0s_x_dx)
            + Y0_x_dx * u1c_x
        )

        u1s_x_dx = -1 / Y0_x * (
            - 2 * eta0_x * dz1c_x
            + 0.5 * (-eta1c_x * dz0c_x + eta1s_x * dz0s_x)
            + 0.5 * ( Y1c_x_dx * u0s_x + Y1s_x_dx * u0c_x)
            + 0.5 * ( Y1c_x * u0s_x_dx + Y1s_x * u0c_x_dx)
            + Y0_x_dx * u1s_x
        )

        return np.array([dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx])

    def bc(self, y_l, y_r):
        dz0c_l, dz0s_l, u0c_l, u0s_l, dz1r_l, dz1c_l, dz1s_l, u1r_l, u1c_l, u1s_l = y_l
        dz0c_r, dz0s_r, u0c_r, u0s_r, dz1r_r, dz1c_r, dz1s_r, u1r_r, u1c_r, u1s_r = y_r

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
 
    def bc_LO(self, y_l, y_r):
        dz0c_l, dz0s_l, u0c_l, u0s_l = y_l
        dz0c_r, dz0s_r, u0c_r, u0s_r = y_r

        return [
            dz0c_l - 1,
            dz0s_l,
            u0c_r,
            u0s_r
        ]
    
    def deriv_LO(self, x_x, y_x):
        g0_x = self.g0_fx(x_x)
        g0_x_dx = self.g0_fx_dx(x_x, g0_x)
        y0_x_dx = self.y0_fx_dx(x_x, y_x, g0_x, g0_x_dx)

        return y0_x_dx

    def deriv(self, x_x, y_x):
        y0_x, y1_x = np.split(y_x, [4], axis=0)

        g0_x = self.g0_fx(x_x)
        g0_x_dx = self.g0_fx_dx(x_x, g0_x)
        y0_x_dx = self.y0_fx_dx(x_x, y0_x, g0_x, g0_x_dx)

        g1_x = self.g1_fx(x_x, y0_x, y0_x_dx, g0_x, g0_x_dx)
        g1_x_dx = self.g1_fx_dx(x_x, y0_x, y0_x_dx, g0_x, g0_x_dx)
        y1_x_dx = self.y1_fx_dx(x_x, y0_x, y0_x_dx, y1_x, g0_x, g0_x_dx, g1_x, g1_x_dx)


        return np.concatenate((y0_x_dx, y1_x_dx), axis=0)

    def deriv_h(self, x_x, y_x):

        dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x, h_x, h_x_dx = y_x

        y_x_dx = self.deriv(x_x, y_x[:10])
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx, dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx = y_x_dx

        factor = 0.04 * self.c_d**(3/2) / (self.g * (self.s-1))**2 / self.d50 
        U5 = (self.A * self.sigma * self.L / self.H)**5


        q_x_dx = U5 * factor / 2 / pi / self.L * self.epsilon*(5*pi*u0c_x**4*u1c_x_dx/2 + 15*pi*u0c_x**4*u1r_x_dx/4 + 5*pi*u0c_x**3*u0s_x*u1s_x_dx + 10*pi*u0c_x**3*u1c_x*u0c_x_dx + 15*pi*u0c_x**3*u1r_x*u0c_x_dx + 5*pi*u0c_x**3*u1s_x*u0s_x_dx + 15*pi*u0c_x**2*u0s_x**2*u1r_x_dx/2 + 15*pi*u0c_x**2*u0s_x*u1r_x*u0s_x_dx + 15*pi*u0c_x**2*u0s_x*u1s_x*u0c_x_dx + 5*pi*u0c_x*u0s_x**3*u1s_x_dx + 15*pi*u0c_x*u0s_x**2*u1r_x*u0c_x_dx + 15*pi*u0c_x*u0s_x**2*u1s_x*u0s_x_dx - 5*pi*u0s_x**4*u1c_x_dx/2 + 15*pi*u0s_x**4*u1r_x_dx/4 - 10*pi*u0s_x**3*u1c_x*u0s_x_dx + 15*pi*u0s_x**3*u1r_x*u0s_x_dx + 5*pi*u0s_x**3*u1s_x*u0c_x_dx)
        # b_x = (1 - self.p) * self.lmbda * self.H / self.L * h_x_dx * np.ones(x_x.shape)
        # b_x_dx = np.gradient(b_x, x_x) / self.L
        # h_x_dt = (-q_x_dx + b_x_dx) / self.H / self.sigma / (1 - self.p) # (dimensionless)
        h_x_dxx = q_x_dx / self.lmbda * self.L**2 / self.H / (1-self.p)

        return np.array([dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx, dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx, h_x_dx, h_x_dxx])
    
    def bc_h(self, y_l, y_r):
        dz0c_l, dz0s_l, u0c_l, u0s_l, dz1r_l, dz1c_l, dz1s_l, u1r_l, u1c_l, u1s_l, h_l, h_l_dx = y_l
        dz0c_r, dz0s_r, u0c_r, u0s_r, dz1r_r, dz1c_r, dz1s_r, u1r_r, u1c_r, u1s_r, h_r, h_r_dx = y_r

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
            u1s_r,
            h_l,
            h_r - 0.9
        ]

    def solve(self):
        n = 1000
        self.x = linspace(0, 1 + self.dL, n)

        # initial guess
        y_guess = 0.1 * np.ones((10, len(self.x)))
        
        sol = scipy.integrate.solve_bvp(self.deriv, self.bc, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        self.y = sol
        if sol.status or self.debug:
            print(sol)
            raise SystemError
        
    def solve_LO(self):
        n = 1000
        self.x = linspace(0, 1 + self.dL, n)

        # initial guess
        y_guess = 0.1 * np.ones((4, len(self.x)))
        
        sol = scipy.integrate.solve_bvp(self.deriv_LO, self.bc_LO, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        self.y = sol
        if sol.status or self.debug:
            print(sol)
            raise SystemError        
    
    def solve_h(self):
        n = 1000
        self.x = linspace(0, 1 + self.dL, n)

        # initial guess
        y_guess = 0.1 * np.ones((12, len(self.x)))
        y_guess[-2, :] = np.linspace(0, 0.9, n)
        y_guess[-1, :] = 0.9
        

        # sol = scipy.integrate.solve_bvp(self.deriv, self.bc, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        sol = scipy.integrate.solve_bvp(self.deriv_h, self.bc_h, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
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

    def visualize_components(self, bnd=0, axs=None):
        # assume self.solve() has been called

        if axs is None:
            fig, axs = plt.subplots(2, 6, figsize=(30, 10))
            # fig, axs = plt.subplots(3, 6, figsize=(30, 15))

        labels=[r"$\zeta^0_{c1}$", r"$\zeta^0_{s1}$", r"$u^0_{c1}$", r"$u^0_{s1}$", r"$\zeta^1_{r}$", r"$\zeta^1_{c2}$", r"$\zeta^1_{s2}$", r"$u^1_{r}$", r"$u^1_{c2}$", r"$u^1_{s2}$"]
        st = np.argmin(abs(self.y.x - bnd))
        for i in range(4):
            axs[0, i].set_title(labels[i])
            axs[0, i].plot(self.y.x[st:], self.y.y[i, st:])
        for i in range(6):
            axs[1, i].set_title(labels[4 + i])
            axs[1, i].plot(self.y.x[st:], self.y.y[4 + i, st:])

        if axs is None:
            plt.show()

    def visualize_amplitudes(self, bnd=None, axs=None):
        # assume self.solve() has been called

        cmplx = np.array([
            self.y.y[0] + 1j * self.y.y[1],
            self.y.y[5] + 1j * self.y.y[6],
            self.y.y[4],


            self.y.y[2] + 1j * self.y.y[3],
            self.y.y[8] + 1j * self.y.y[9],
            self.y.y[7]
        ])

        ampl = np.abs(cmplx)
        phase = np.angle(cmplx)
        # phase = np.unwrap(phase, axis=1)
        # phase = np.unwrap(np.unwrap(phase, axis=1), axis=0)
        if axs is None:
            fig, axs = plt.subplots(2, 6, figsize=(30, 10))
            # fig, axs = plt.subplots(3, 6, figsize=(30, 15))

        bnd = bnd if bnd is not None and bnd > 1e-5 else self.y.x[1]
        st = np.argmin(abs(self.y.x - bnd))


        labels = [r"$\zeta^0_1$",  r"$\zeta^1_2$", r"$\zeta^1_{res,0}$", r"$u^0_1$",  r"$u^1_2$",  r"$u^1_{res,0}$"]

        for j in range(6):
            axs[0, j].set_title(labels[j])
            axs[0, j].plot(self.y.x[st:], ampl[j, st:])
            axs[1, j].plot(self.y.x[st:], phase[j, st:])
            # axs[2, j].plot(self.y.x[st:], np.gradient(phase[j, st:], self.y.x[st:], edge_order=2))

        

        

        if axs is None:
            plt.show()

    def visualize_defina_vars(self, bnd=0, axs=None):

        st = np.argmin(abs(self.y.x - bnd))
        x_x, y_x = self.y.x[st:], self.y.y[:, st:]
        x = x_x

        

        dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x = y_x
        h_x, h_x_dx = self.h_fx(x_x), self.h_fx_dx(x_x)

        # helper functions that often appear
        s1_x = scipy.special.erf(2 * (1 - h_x) / self.a_r)
        s2_x = exp(-4 * (1-h_x)**2 / self.a_r**2)
        s1_x_dx = h_x_dx * (-4) / pi**0.5 / self.a_r * s2_x
        s2_x_dx = h_x_dx * 8 * (1-h_x) / self.a_r**2 * s2_x

        # leading order
        eta0_x = 0.5 * (1 + s1_x)
        Y0_x  = eta0_x * (1 - h_x) + self.a_r / 4 / pi**0.5 * s2_x
        Y0_x_dx = 0.5 * s1_x_dx * (1 - h_x) + self.a_r / 4 / pi**0.5 * s2_x_dx

        dz0c_x_dx = 1 / self.kappa * (- self.r / Y0_x * u0c_x - u0s_x)
        dz0s_x_dx = 1 / self.kappa * (- self.r / Y0_x * u0s_x + u0c_x)
        u0c_x_dx = (-eta0_x * dz0s_x - u0c_x * Y0_x_dx)  / Y0_x
        u0s_x_dx = ( eta0_x * dz0c_x - u0s_x * Y0_x_dx)  / Y0_x

        # first order
        eta1c_x = dz0c_x * 2 / pi**0.5 / self.a_r * s2_x
        eta1s_x = dz0s_x * 2 / pi**0.5 / self.a_r * s2_x

        Y1c_x = dz0c_x * eta0_x
        Y1s_x = dz0s_x * eta0_x
        Y1c_x_dx = dz0c_x_dx * eta0_x + dz0c_x * 0.5 * s1_x_dx
        Y1s_x_dx = dz0s_x_dx * eta0_x + dz0s_x * 0.5 * s1_x_dx

        etaf = 2 / pi**0.5 / self.a_r * s2_x
        Yf = eta0_x # kind of useless



        if axs is None:
            fig, axs = plt.subplots(2, 6, figsize=(30, 10))

        

        eta0_x, Y0_x, eta1c_x, eta1s_x, dz0c_x, dz0s_x

        axs[0, 0].plot(x, eta0_x)
        axs[0, 0].set_title(r"$\eta_0$")

        axs[1, 0].plot(x, Y0_x)
        axs[1, 0].set_title(r"$Y_0$")

        axs[0, 1].plot(x, eta1c_x)
        axs[0, 1].set_title(r"$\eta_{1c}$")

        axs[0, 2].plot(x, eta1c_x)
        axs[0, 2].set_title(r"$\eta_{1s}$")

        axs[1, 1].plot(x, Y1c_x)
        axs[1, 1].set_title(r"$Y_{1c}$")

        axs[1, 2].plot(x, Y1s_x)
        axs[1, 2].set_title(r"$Y_{1s}$")

        eta_cmplx = eta1c_x + 1j * eta1s_x
        axs[0, 4].plot(x, np.abs(eta_cmplx))
        axs[0, 5].plot(x, np.angle(eta_cmplx))


        Y_cmplx = Y1c_x + 1j * Y1s_x
        axs[1, 4].plot(x, np.abs(Y_cmplx))
        axs[1, 5].plot(x, np.angle(Y_cmplx))

        

        if axs is None:
            plt.show()




        return eta0_x, Y0_x, eta1c_x, eta1s_x, dz0c_x, dz0s_x, Y0_x_dx, 0.5 * s1_x_dx

    def visualize_LO_effects(self):

        x_x, y_x = self.y.x, self.y.y

        dz0c_x, dz0s_x, u0c_x, u0s_x = y_x
        h_x, h_x_dx = self.h_fx(x_x), self.h_fx_dx(x_x)

        # helper functions that often appear
        s1_x = scipy.special.erf(2 * (1 - h_x) / self.a_r)
        s2_x = exp(-4 * (1-h_x)**2 / self.a_r**2)
        s1_x_dx = h_x_dx * (-4) / pi**0.5 / self.a_r * s2_x
        s2_x_dx = h_x_dx * 8 * (1-h_x) / self.a_r**2 * s2_x

        # leading order
        eta0_x = 0.5 * (1 + s1_x)
        Y0_x  = eta0_x * (1 - h_x) + self.a_r / 4 / pi**0.5 * s2_x
        Y0_x_dx = 0.5 * (s1_x_dx * (1 - h_x) - (s1_x + 1) * h_x_dx) + self.a_r / 4 / pi**0.5 * s2_x_dx

        dz0c_x_dx = 1 / self.kappa * (- self.r / Y0_x * u0c_x - u0s_x)
        dz0s_x_dx = 1 / self.kappa * (- self.r / Y0_x * u0s_x + u0c_x)
        u0c_x_dx = (-eta0_x * dz0s_x - u0c_x * Y0_x_dx)  / Y0_x
        u0s_x_dx = ( eta0_x * dz0c_x - u0s_x * Y0_x_dx)  / Y0_x

        # first order
        eta1c_x = dz0c_x * 2 / pi**0.5 / self.a_r * s2_x
        eta1s_x = dz0s_x * 2 / pi**0.5 / self.a_r * s2_x

        Y1c_x = dz0c_x * eta0_x
        Y1s_x = dz0s_x * eta0_x
        Y1c_x_dx = dz0c_x_dx * eta0_x + dz0c_x * 0.5 * s1_x_dx
        Y1s_x_dx = dz0s_x_dx * eta0_x + dz0s_x * 0.5 * s1_x_dx
        
        # dz1r_x_dx = (
        #     (- self.r * u1r_x - 0.5 * (  Y1c_x *  u0s_x - Y1s_x *  u0c_x) - 0.5 * ( Y1c_x * dz0c_x_dx + Y1s_x * dz0s_x_dx) * self.kappa) / Y0_x - 0.5 * (u0c_x * u0c_x_dx + u0s_x * u0s_x_dx)            ) / self.kappa 
        # dz1c_x_dx = (
        #     (- self.r * u1c_x - 0.5 * (  Y1c_x *  u0s_x + Y1s_x *  u0c_x) - 0.5 * ( Y1c_x * dz0c_x_dx - Y1s_x * dz0s_x_dx) * self.kappa) / Y0_x - 0.5 * (u0c_x * u0c_x_dx - u0s_x * u0s_x_dx) - 2 * u1s_x) / self.kappa
        # dz1s_x_dx = (
        #     (- self.r * u1s_x - 0.5 * ( -Y1c_x *  u0c_x + Y1s_x *  u0s_x) - 0.5 * ( Y1c_x * dz0s_x_dx + Y1s_x * dz0c_x_dx) * self.kappa) / Y0_x - 0.5 * (u0c_x * u0s_x_dx + u0s_x * u0c_x_dx) + 2 * u1c_x) / self.kappa
        
        # u1r_x_dx = -1 / Y0_x * (
        #     + 0.5 * ( eta1c_x * dz0s_x - eta1s_x * dz0c_x)
        #     + 0.5 * ( Y1c_x_dx * u0c_x + Y1s_x_dx * u0s_x)
        #     + 0.5 * ( Y1c_x * u0c_x_dx + Y1s_x * u0s_x_dx)
        #     + Y0_x_dx * u1r_x
        # )

        # u1c_x_dx = -1 / Y0_x * (
        #     + 2 * eta0_x * dz1s_x
        #     + 0.5 * ( eta1c_x * dz0s_x + eta1s_x * dz0c_x)
        #     + 0.5 * ( Y1c_x_dx * u0c_x - Y1s_x_dx * u0s_x)
        #     + 0.5 * ( Y1c_x * u0c_x_dx - Y1s_x * u0s_x_dx)
        #     + Y0_x_dx * u1c_x
        # )

        # u1s_x_dx = -1 / Y0_x * (
        #     - 2 * eta0_x * dz1c_x
        #     + 0.5 * (-eta1c_x * dz0c_x + eta1s_x * dz0s_x)
        #     + 0.5 * ( Y1c_x_dx * u0s_x + Y1s_x_dx * u0c_x)
        #     + 0.5 * ( Y1c_x * u0s_x_dx + Y1s_x * u0c_x_dx)
        #     + Y0_x_dx * u1s_x
        # )



        return [
            # - self.r * u1r_x / Y0_x,
            - 0.5 * (  Y1c_x *  u0s_x - Y1s_x *  u0c_x) / Y0_x,
            - 0.5 * ( Y1c_x * dz0c_x_dx + Y1s_x * dz0s_x_dx) * self.kappa,
            - 0.5 * (u0c_x * u0c_x_dx + u0s_x * u0s_x_dx)
        ]
        





