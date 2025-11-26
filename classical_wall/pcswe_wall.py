import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt, zeros, ones, nan
import scipy
import matplotlib.pyplot as plt

from quartic_solver import * 

class PCSWE_wall_sol(): pass

class PCSWE_wall():

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

        # morphodynamics
        self.p = 0.4 # porosity
        self.c_d = 0.0025
        self.lmbda = 6.8e-6
        self.d50 = 0.13e-3
        self.phi = pi / 6 # 30 degrees 

        # universal constants
        self.g = 9.81
        self.sigma = 1.4e-4
        self.rho_w = 1025
        self.rho_s = 2650
        
        self.use_alpha = False
        
    def set_derivative_vars(self):
        self.epsilon = self.A / self.H
        self.eta = self.sigma * self.L / sqrt(self.g * self.H)
        self.U = self.epsilon * self.sigma * self.L
        self.kappa = self.g * self.H / (self.sigma * self.L) ** 2

        self.s = self.rho_s / self.rho_w

        self.delta = 0.04 * self.c_d**(3/2) * self.A * (self.sigma * self.L)**4 / \
                    (self.g**2 * (self.s-1)**2 * self.d50 * self.H**6 * (1-self.p))

        self.f = 0.04 * self.c_d**(3/2) / (self.g**2 * (self.s-1)**2 * self.d50)
        
        self.bA_ = self.f * self.U**5 / self.L / 2 / pi
        self.bA  = self.f * self.U**5 / self.L / 2 / pi / self.sigma / self.H / (1-self.p)

        self.bB_ = (1-self.p) * self.lmbda * self.H / self.L**2
        self.bB  = (1-self.p) * self.lmbda * self.H / self.L**2 / self.sigma/ self.H / (1-self.p)

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

        sol = PCSWE_wall_sol()
        sol.x, sol.t, sol.dz_xt, sol.u_xt = x, t, dz_xt, u_xt
        return sol
    
    def h_fx(self, x): return 0.9 * x

    def h_fx_dx(self, x): return 0.9

    def deriv_LO(self, x_x, y0_x):
        # x_x, y0_x = np.array(x_x), np.array(y0_x)
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        h_x, h_x_dx = self.h_fx(x_x), self.h_fx_dx(x_x)

        # derivatives of dzeta (using momentum equation)
        dz0c_x_dx = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0c_x - u0s_x)
        dz0s_x_dx = 1 / self.kappa * ( - self.r / (1 - h_x + self.h0) * u0s_x + u0c_x)

        # derivatives of u (using continuity equation)
        u0c_x_dx = (-dz0s_x + u0c_x * h_x_dx)  / (1 - h_x)
        u0s_x_dx = ( dz0c_x + u0s_x * h_x_dx)  / (1 - h_x)

        return np.array([dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx])
    
    def deriv_FO(self, x_x, y1_x, y0_x, y0_x_dx):
        dz0c_x, dz0s_x, u0c_x, u0s_x = y0_x
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x = y1_x
        h_x, h_x_dx = self.h_fx(x_x), self.h_fx_dx(x_x)

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

        return np.array([dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx])

    def deriv(self, x_x, y_x):
        y0_x, y1_x = np.split(y_x, [4], axis=0)

        y0_x_dx = self.deriv_LO(x_x, y0_x)
        y1_x_dx = self.deriv_FO(x_x, y1_x, y0_x, y0_x_dx)

        y_x_dx = np.concatenate((y0_x_dx, y1_x_dx), axis=0)

        return y_x_dx

    def deriv_h(self, x_x, y_x):

        dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x, h_x, h_x_dx = y_x
        y0_x, y1_x, yh_x = np.split(y_x, [4, 10], axis=0)

        y0_x_dx = self.deriv_LO(x_x, y0_x)
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx

        y1_x_dx = self.deriv_FO(x_x, y1_x, y0_x, y0_x_dx)
        dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx = y1_x_dx

        if not self.use_alpha:
            u5_x_dx = self.epsilon*(5*pi*u0c_x**4*u1c_x_dx/2 + 15*pi*u0c_x**4*u1r_x_dx/4 + 5*pi*u0c_x**3*u0s_x*u1s_x_dx + 10*pi*u0c_x**3*u1c_x*u0c_x_dx + 15*pi*u0c_x**3*u1r_x*u0c_x_dx + 5*pi*u0c_x**3*u1s_x*u0s_x_dx + 15*pi*u0c_x**2*u0s_x**2*u1r_x_dx/2 + 15*pi*u0c_x**2*u0s_x*u1r_x*u0s_x_dx + 15*pi*u0c_x**2*u0s_x*u1s_x*u0c_x_dx + 5*pi*u0c_x*u0s_x**3*u1s_x_dx + 15*pi*u0c_x*u0s_x**2*u1r_x*u0c_x_dx + 15*pi*u0c_x*u0s_x**2*u1s_x*u0s_x_dx - 5*pi*u0s_x**4*u1c_x_dx/2 + 15*pi*u0s_x**4*u1r_x_dx/4 - 10*pi*u0s_x**3*u1c_x*u0s_x_dx + 15*pi*u0s_x**3*u1r_x*u0s_x_dx + 5*pi*u0s_x**3*u1s_x*u0c_x_dx)
        elif self.use_alpha:
            

            t = np.linspace(0, 2*pi, 300)
            u0_xt = u0c_x[:,None] * cos(  t)[None,:] + u0s_x[:,None] * sin(  t)[None,:]
            u1_xt = u1c_x[:,None] * cos(2*t)[None,:] + u1s_x[:,None] * sin(2*t)[None,:] + u1r_x[:,None] * ones(t.shape)[None,:]
            u_xt = u0_xt + self.epsilon * u1_xt
  
            c2 = cos(atan(self.H/self.L * h_x_dx)) # ~ 1
            alpha_xt = tan(self.phi) / (c2[:, None] * (tan(self.phi) + self.H / self.L * h_x_dx[:, None] * np.sign(u_xt)))
     
            y = alpha_xt * u_xt**5

            # with alpha
            # u_x_trap = np.trapezoid(y, x=t) # sol.t goes from 0 to 2 pi
            u_x_simp = scipy.integrate.simpson(y, x=t)

            # without alpha
            # u5_x = self.epsilon*(5*pi*u0c_x**4*u1c_x/2 + 15*pi*u0c_x**4*u1r_x/4 + 5*pi*u0c_x**3*u0s_x*u1s_x + 15*pi*u0c_x**2*u0s_x**2*u1r_x/2 + 5*pi*u0c_x*u0s_x**3*u1s_x - 5*pi*u0s_x**4*u1c_x/2 + 15*pi*u0s_x**4*u1r_x/4)
            # u5_x_simp = scipy.integrate.simpson(y, x=t)
            

            u5_x_dx = np.gradient(u_x_simp, x_x, edge_order=2)

        qe_x_dx = self.bA *u5_x_dx 
        h_x_dxx = qe_x_dx / self.bB


        return np.array([dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx, dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx, h_x_dx, h_x_dxx])
    
    def bc(self, y_left, y_right):
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
            h_r - self.max_h
        ]
    
    def bc_LO(self, y_left, y_right):
        dz0c_l, dz0s_l, u0c_l, u0s_l = y_left
        dz0c_r, dz0s_r, u0c_r, u0s_r = y_right

        return [
            dz0c_l - 1,
            dz0s_l,
            u0c_r,
            u0s_r
        ]

    def solve(self):
        x_x = linspace(0, 1, 2000)

        # if you want to have a different initial mesh
        # dx = 0.01
        # transf1 = lambda x: (np.log(x + dx) - np.log(dx)) / (np.log(1 + dx) - np.log(dx))
        # self.x = transf1(self.x)
        # plt.plot(self.x[::50], ones(self.x.shape)[::50], 'o')
        # plt.show()

        # initial guess
        y_guess = 0.1 * np.ones((10, len(x_x)))

        # sol = scipy.integrate.solve_bvp(self.deriv, self.bc, self.x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        sol = scipy.integrate.solve_bvp(self.deriv, self.bc, x_x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        self.y = sol
        if sol.status or self.debug:
            print(sol)
            raise SystemError

    def solve_LO(self):
        x_x = linspace(0, 1, 1000)
        y_guess = 0.1 * np.ones((4, len(x_x)))

        sol = scipy.integrate.solve_bvp(self.deriv_LO, self.bc_LO, x_x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)

        self.y0 = sol
        
        if sol.status or self.debug:
            print(sol)
            raise SystemError
    
    def solve_h(self):
        x_x = linspace(0, 1, 1000)
        y_guess = 0.1 * np.ones((12, len(x_x)))
        y_guess[-2, :] = self.h_fx(x_x)
        y_guess[-1, :] = self.h_fx_dx(x_x)

        self.max_h = self.h_fx(1)
        
        sol = scipy.integrate.solve_bvp(self.deriv_h, self.bc_h, x_x, y_guess, tol=self.tol, max_nodes=20000, verbose=2)
        self.y = sol
        if sol.status or self.debug:
            print(sol)
            raise SystemError

    def visualize_sol(self):
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

    def visualize_sol_LO(self):

        x = linspace(0, 1, 10000)
        fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        st = np.argmin(abs(x - 0.9))
        for nu in range(3):
            for i in range(4):
                axs[nu, i].plot(x[st:], self.y0.sol(x[st:], nu=nu)[i])
        plt.show()

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

    def transport_alpha(self):
        sol = self.generate_solution()
        x_x = sol.x # same as self.y.x (luckily!)
        u_xt = sol.u_xt
        try:
            # if solve has been run
            y0_x, y1_x = np.split(self.y.y, [4], axis=0)
            dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x = self.y.y
            h_x, h_x_dx = self.h_fx(x_x), self.h_fx_dx(x_x)
        except:
            # if solve_h has been run
            y0_x, y1_x, yh_x = np.split(self.y.y, [4, 10], axis=0)
            dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x, h_x, h_x_dx = self.y.y
        y0_x_dx = self.deriv_LO(x_x, y0_x)
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        y1_x_dx = self.deriv_FO(x_x, y1_x, y0_x, y0_x_dx)
        dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx = y1_x_dx



        c1 = tan(self.phi)
        c2 = cos(atan(self.H/self.L * h_x_dx)) # ~ 1
        c3 = self.H / self.L * h_x_dx # very small
        k2 = c2 * c1 
        k3 = c2 * c3 # very small
        alpha_xt = alpha = c1 / (k2 + k3 * np.sign(u_xt))

        y = alpha_xt * u_xt**5

        # with alpha
        u_x_trap = np.trapezoid(y, x=sol.t) # sol.t goes from 0 to 2 pi
        u_x_simp = scipy.integrate.simpson(y, x=sol.t)

        # without alpha
        u5_x = self.epsilon*(5*pi*u0c_x**4*u1c_x/2 + 15*pi*u0c_x**4*u1r_x/4 + 5*pi*u0c_x**3*u0s_x*u1s_x + 15*pi*u0c_x**2*u0s_x**2*u1r_x/2 + 5*pi*u0c_x*u0s_x**3*u1s_x - 5*pi*u0s_x**4*u1c_x/2 + 15*pi*u0s_x**4*u1r_x/4)
        u5_x_simp = scipy.integrate.simpson(y, x=sol.t)
        plt.plot(x_x, u_x_trap, 'o')
        plt.plot(x_x, u_x_simp)
        plt.plot(x_x, u5_x)
        plt.show()

        plt.plot(x_x, np.gradient(u_x_simp, x_x, edge_order=2), 'o')
        plt.plot(x_x, np.gradient(u5_x, x_x, edge_order=2))
        plt.plot(x_x, np.gradient(u5_x_simp, x_x, edge_order=2))
        plt.show()






        sign_changes = u_xt * np.roll(u_xt, -1, axis=1) < 0
        # print(sign_changes.shape)
        zero_idxs = [np.asarray(changes).nonzero() for changes in sign_changes]


        result = np.zeros((len(x_x), 2))
        t = sol.t
        # print(t[0])
        for x_i in range(len(zero_idxs)):
            try:
                for zero_i in range(2):
                    t_i = zero_idxs[x_i][0][zero_i]
                    t_cross = t[t_i] - u_xt[x_i,t_i] * (t[t_i+1] - t[t_i]) / (u_xt[x_i,t_i+1] - u_xt[x_i,t_i])
                    print(x_i, t_i, t_cross)
            except:
                pass

        eps = self.epsilon

        a, b, c, d, e = eps * u1r_x, u0c_x, u0s_x, eps * u1c_x, eps * u1s_x

        A = a - b + d
        B = 2*c - 4*e
        C = 2*a - 6*d
        D = 2*c + 4*e
        E = a + b + d
       

        sols = multi_quartic(A, B, C, D, E)
        #  t = 2 atan(x) + 2k pi from these solutions
        # print(sols)
            
     

        raise SystemError

        # alternative calculation: find all zeros 
        print(x_x.shape)
        print(sol.t.shape)









        # X, T = np.meshgrid(x_x, sol.t)
        T, X = np.meshgrid(sol.t, x_x)









        print(X.shape)
        print(u_xt.shape)

        # Suppose X, Y are 2D meshgrids and F is f(X,Y)
        plt.figure()
        cs = plt.contour(X, T, u_xt, levels=[0], colors='k')  # level-set f=0
        plt.show()


        zero_curves = cs.allsegs
        print(zero_curves[0])
        for item in zero_curves[0]:
            print(item.shape)
    
            plt.plot(item[:, 0], item[:, 1])
            plt.show()


        sign_change = u_xt[:, :-1] * u_xt[:, 1:] < 0    # shape (nx, ny-1)


        rows, cols = np.where(sign_change)

        print(rows.shape, cols.shape)
        print(rows)

        result = np.zeros(len(x_x))




        sign_change = u_xt[:, :-1] * u_xt[:, 1:] < 0   # (nx, ny-1)
        x = np.asarray(sign_change).nonzero()
        print(x)


    
        results = zeros((len(x_x), 2))       # output array
        for i in range(len(x_x)):
            


            cols = np.where(sign_change[i])[0]   # indices j where crossing between y[j] and y[j+1]
            
            if len(cols) == 0:
                continue
            
            # Interpolate crossings (vectorized for this row)
            j = cols
            y0, y1 = y[j], y[j+1]
            f0, f1 = u_xt[i, j], u_xt[i,j+1]
            y_cross = y0 - f0 * (y1 - y0) / (f1 - f0)
            
            # Keep only the first 2 crossings
            results[i, :min(2, len(y_cross))] = y_cross[:2]


        # print(zero_curves.shape)
        # print(zero_curves)

        # for path in zero_curves:
            
        #     curve = path.vertices      # Nx2 array of (x, y) points along the zero line
        #     print(curve)
        #     # do something with curve






        


    def transport2(self):
        
        sol = self.generate_solution()
        x_x = sol.x # same as self.y.x (luckily!)
        u_xt = sol.u_xt

        try:
            # if solve has been run
            y0_x, y1_x = np.split(self.y.y, [4], axis=0)
            dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x = self.y.y
            h_x, h_x_dx = self.h_fx(x_x), self.h_fx_dx(x_x)
        except:
            # if solve_h has been run
            y0_x, y1_x, yh_x = np.split(self.y.y, [4, 10], axis=0)
            dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x, h_x, h_x_dx = self.y.y

        y0_x_dx = self.deriv_LO(x_x, y0_x)
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        y1_x_dx = self.deriv_FO(x_x, y1_x, y0_x, y0_x_dx)
        dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx = y1_x_dx


        u5_x_dx = 5*pi*self.epsilon*(2*u0c_x**4*u1c_x_dx + 3*u0c_x**4*u1r_x_dx + 4*u0c_x**3*u0s_x*u1s_x_dx + 8*u0c_x**3*u1c_x*u0c_x_dx + 12*u0c_x**3*u1r_x*u0c_x_dx + 4*u0c_x**3*u1s_x*u0s_x_dx + 6*u0c_x**2*u0s_x**2*u1r_x_dx + 12*u0c_x**2*u0s_x*u1r_x*u0s_x_dx + 12*u0c_x**2*u0s_x*u1s_x*u0c_x_dx + 4*u0c_x*u0s_x**3*u1s_x_dx + 12*u0c_x*u0s_x**2*u1r_x*u0c_x_dx + 12*u0c_x*u0s_x**2*u1s_x*u0s_x_dx - 2*u0s_x**4*u1c_x_dx + 3*u0s_x**4*u1r_x_dx - 8*u0s_x**3*u1c_x*u0s_x_dx + 12*u0s_x**3*u1r_x*u0s_x_dx + 4*u0s_x**3*u1s_x*u0c_x_dx)/4
        
        q_x_dx = self.bA_ * u5_x_dx
        
        fig, axs = plt.subplots(4, 5, figsize=(25, 20))
        axs[0, 0].plot(x_x, self.h_fx(x_x), 'brown', linewidth=4)
        axs[0, 0].set_title(r"$h(x)$")

        axs[0, 1].plot(x_x, u5_x_dx)
        axs[0, 1].set_title(r"$\int u^5$")

        axs[1, 4].set_title(r"$(\int q_T)_x$")
        axs[1, 4].plot(x_x, q_x_dx)
        plt.show()



           


        # axs[1, 2].plot(x_x, np.max(q_xt, axis=1))
        # axs[1, 2].set_title("max($q_T$)")
        # axs[1, 3].plot(x_x, q_x_trap, 'o', label="trapz rule")
        # # axs[0, 1].plot(x, q_x_gl, 'x', ms=5)
        # axs[1, 3].plot(x_x, q_x, 'k', label="analytical")
        # axs[1, 3].set_title(r"$\int q_T$")
        # axs[1, 4].plot(x_x, q_x_dx)
        # axs[1, 4].set_title(r"$(\int q_T)_x$")
        # axs[2, 4].plot(x_x, b_x)
        # axs[2, 4].set_title(r"$\int (1-p)\lambda h_x$")
        # axs[2, 0].plot(x_x, h_x_dt, color='brown', linestyle='--')
        # axs[2, 0].set_title(r"$h_t$")
        # axs[2, 2].plot(x, b_x_dx)
        # axs[2, 2].set_title(r"$(\int (1-p)\lambda h_x)_x$")
        # plt.show()


    
        return


    def transport(self):
        # assume solve has already been executed, and we have all 10 components (4 LO 6 FO) calculated
        # compute local transport

        sol = self.generate_solution()
        x_x = sol.x # same as self.y.x (luckily!)
        u_xt = sol.u_xt

        try:
            # if solve has been run
            y0_x, y1_x = np.split(self.y.y, [4], axis=0)
            dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x = self.y.y
            h_x, h_x_dx = self.h_fx(x_x), self.h_fx_dx(x_x)
        except:
            # if solve_h has been run
            y0_x, y1_x, yh_x = np.split(self.y.y, [4, 10], axis=0)
            dz0c_x, dz0s_x, u0c_x, u0s_x, dz1r_x, dz1c_x, dz1s_x, u1r_x, u1c_x, u1s_x, h_x, h_x_dx = self.y.y

        y0_x_dx = self.deriv_LO(x_x, y0_x)
        dz0c_x_dx, dz0s_x_dx, u0c_x_dx, u0s_x_dx = y0_x_dx
        y1_x_dx = self.deriv_FO(x_x, y1_x, y0_x, y0_x_dx)
        dz1r_x_dx, dz1c_x_dx, dz1s_x_dx, u1r_x_dx, u1c_x_dx, u1s_x_dx = y1_x_dx

    

        # Engelund Hansen formula
        factor = 0.04 * self.c_d**(3/2) / (self.g * (self.s-1))**2 / self.d50 
        U5 = (self.A * self.sigma * self.L / self.H)**5

        # just u^5 dimensionless
        u5_xt = u_xt**5

        # instantaneous transport
        q_xt = factor * U5 * u_xt**5


        
        # time integration of instantaneous transport
        # -------------------------------------------
        u5_x = self.epsilon*(5*pi*u0c_x**4*u1c_x/2 + 15*pi*u0c_x**4*u1r_x/4 + 5*pi*u0c_x**3*u0s_x*u1s_x + 15*pi*u0c_x**2*u0s_x**2*u1r_x/2 + 5*pi*u0c_x*u0s_x**3*u1s_x - 5*pi*u0s_x**4*u1c_x/2 + 15*pi*u0s_x**4*u1r_x/4)

        # analytically
        q_x = U5 * factor / 2 / pi * u5_x
        
        # trapezoidal rule
        q_x_trap = np.mean(q_xt, 1)
        # # Gauss_Legendre
        # roots, weights = roots_legendre(100)
        # q_xt_interp = np.array([np.interp(roots*2*pi, linspace(0, 2*pi, q_xt.shape[1]), q_xt[x_i, :]) for x_i in range(len(x))]) # time is equidistant
        # q_x_gl = np.sum(q_xt_interp * weights[np.newaxis, :], 1) * 2 * pi / self.sigma


        # spatial derivative of time integration of instantaneous transport
        # -----------------------------------------------------------------
        u5_x_dx = 5*pi*self.epsilon*(2*u0c_x**4*u1c_x_dx + 3*u0c_x**4*u1r_x_dx + 4*u0c_x**3*u0s_x*u1s_x_dx + 8*u0c_x**3*u1c_x*u0c_x_dx + 12*u0c_x**3*u1r_x*u0c_x_dx + 4*u0c_x**3*u1s_x*u0s_x_dx + 6*u0c_x**2*u0s_x**2*u1r_x_dx + 12*u0c_x**2*u0s_x*u1r_x*u0s_x_dx + 12*u0c_x**2*u0s_x*u1s_x*u0c_x_dx + 4*u0c_x*u0s_x**3*u1s_x_dx + 12*u0c_x*u0s_x**2*u1r_x*u0c_x_dx + 12*u0c_x*u0s_x**2*u1s_x*u0s_x_dx - 2*u0s_x**4*u1c_x_dx + 3*u0s_x**4*u1r_x_dx - 8*u0s_x**3*u1c_x*u0s_x_dx + 12*u0s_x**3*u1r_x*u0s_x_dx + 4*u0s_x**3*u1s_x*u0c_x_dx)/4
        q_x_dx = U5 * factor / 2 / pi / self.L * u5_x_dx
        q_x_dx_num = np.gradient(q_x, x_x) / self.L


        # bedload transport
        # -----------------
        b_x = (1 - self.p) * self.lmbda * self.H / self.L * h_x_dx * np.ones(x_x.shape)

        # spatial derivative of bedload transport?
        # cannot compute this analytically without d^2h/dx^2  
        # numerical approx:
        b_x_dx = np.gradient(b_x, x_x) / self.L


        # effect on the time derivative of the bed

        print(self.H / self.sigma / (1-self.p))
        h_x_dt = (-q_x_dx + b_x_dx) / self.H / self.sigma / (1 - self.p) # (dimensionless)

        return x_x, u5_xt, u5_x, u5_x_dx, q_xt, q_x, q_x_trap, q_x_dx, q_x_dx_num, b_x, b_x_dx, h_x_dt

    def visualize_transport(self):
        x, u5_xt, u5_x, u5_x_dx, q_xt, q_x, q_x_trap, q_x_dx, q_x_dx_num, b_x, b_x_dx, h_x_dt = self.transport()

        fig, axs = plt.subplots(4, 5, figsize=(25, 20))
        axs[0, 0].plot(x, self.h_fx(x), 'brown', linewidth=4)
        axs[0, 0].set_title(r"$h(x)$")


        axs[0, 1].plot(x, u5_x_dx)
        axs[0, 1].set_title(r"$\int u^5$")



        axs[1, 0].plot(x, np.max(u5_xt, axis=1))
        axs[1, 0].set_title("max($u^5$)")

        axs[1, 1].plot(x, u5_x)
        axs[1, 1].set_title(r"$\int u^5$")

        axs[1, 2].plot(x, np.max(q_xt, axis=1))
        axs[1, 2].set_title("max($q_T$)")





        axs[1, 3].plot(x, q_x_trap, 'o', label="trapz rule")
        # axs[0, 1].plot(x, q_x_gl, 'x', ms=5)
        axs[1, 3].plot(x, q_x, 'k', label="analytical")
        axs[1, 3].set_title(r"$\int q_T$")

        axs[1, 4].plot(x, q_x_dx)
        axs[1, 4].set_title(r"$(\int q_T)_x$")



        axs[2, 4].plot(x, b_x)
        axs[2, 4].set_title(r"$\int (1-p)\lambda h_x$")


        axs[2, 0].plot(x, h_x_dt, color='brown', linestyle='--')
        axs[2, 0].set_title(r"$h_t$")


        

        axs[2, 2].plot(x, b_x_dx)
        axs[2, 2].set_title(r"$(\int (1-p)\lambda h_x)_x$")
        plt.show()

        return
  
"""
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
"""



if __name__ == "__main__":
    pcswe = PCSWE_wall()
    pcswe.H = 7.12
    pcswe.A = 0.72
    pcswe.L = 8e3

    pcswe.r = 0.45
    pcswe.h0 = 0.1

    pcswe.tol = 1e-5

    pcswe.set_derivative_vars()

    pcswe.solve_LO()
    pcswe.visualize_sol_LO()
    
    pcswe.solve()
    pcswe.visualize_sol()

    pcswe.solve_h()
    pcswe.visualize_sol_h()


   


