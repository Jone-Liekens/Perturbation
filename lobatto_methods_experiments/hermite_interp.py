
import numpy as np
import matplotlib.pyplot as plt
import scipy

np.set_printoptions(
    linewidth=150,   # allow long lines before wrapping
    precision=3,     # show only two decimals
    floatmode='maxprec',  # optional: consistent formatting
    # suppress=True    # avoid scientific notation for small numbers
)


def hermite_interp_1d(x, y, yd):
    x_hermite = np.repeat(x, 2)
    f_hermite = np.repeat(y, 2)
    n = len(x_hermite)

    # compute the coefficients
    f_table = np.empty((n, n), dtype=np.float64)
    f_table[0, :] = f_hermite
    f_table[1, ::2] = yd
    f_table[1, 1:-1:2] = (f_table[0, 2::2] - f_table[0, :-2:2]) / (x[1:] - x[:-1])
    for i in range(2, n):
        f_table[i, :-i] = (f_table[i-1, 1:n-i+1] - f_table[i-1, :n-i]) / (x_hermite[i:] - x_hermite[:-i])

    herm_coeffs = f_table[:, 0]

    # convert the hermite coeffs to a 'normal' polynomial
    coeffs = np.zeros((n, n))
    coeffs[0, 0] = 1
    for i in range(1, n):
        coeffs[i, :i] = - x_hermite[i-1] * coeffs[i-1, :i]
        coeffs[i, 1:i+1] += coeffs[i-1, :i]

    poly_coeffs = herm_coeffs @ coeffs
    return np.poly1d(poly_coeffs[::-1])



def hermite_interp(x, y, yd):
    # x = (s, m)
    # y = (n, s, m)
    # yd = (n, s, m)

    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        yd = np.expand_dims(yd, axis=1)
    if y.ndim == 2:
        y = np.expand_dims(y, axis=0)
        yd = np.expand_dims(yd, axis=0)

    x_boundaries = np.array([xs[0] for xs in x.T] + [x[-1][-1]])
    x = x - x[0]

    s2 = 2*x.shape[0]
    m = x.shape[1] # s, m
    n = y.shape[0] # n, s, m


    print("interpolation_dims", s2, m, n)


    y = y.transpose([1, 2, 0]) # move n-dimension to the back
    yd = yd.transpose([1, 2, 0]) 

    x_hermite = np.repeat(x, 2, axis=0) # 2s, m
    f_hermite = np.repeat(y, 2, axis=0) # 2s, m, n
    
    # compute the coefficients
    f_table = np.empty((s2, s2, m, n), dtype=np.float64)
    f_table[0, :] = f_hermite
    f_table[1, ::2] = yd
    f_table[1, 1:-1:2] = (f_table[0, 2::2] - f_table[0, :-2:2]) / (x[1:] - x[:-1])[:, :, None]
    for i in range(2, s2):
        f_table[i, :-i] = (f_table[i-1, 1:s2-i+1] - f_table[i-1, :s2-i]) / (x_hermite[i:] - x_hermite[:-i])[:, :, None]

    herm_coeffs = f_table[:, 0, :, :] # extract the first column

    # convert the hermite coeffs to a 'normal' polynomial
    coeffs = np.zeros((s2, s2, m, n))
    coeffs[0, 0] = 1
    for i in range(1, s2):
        coeffs[i, :i] = - x_hermite[i-1][None, :, None] * coeffs[i-1, :i] # we just expand the product (x-x0) (x-x1) (x-x2) ... 
        coeffs[i, 1:i+1] += coeffs[i-1, :i]

    # now the difficult part:
    
    herm_coeffs = np.expand_dims(herm_coeffs, axis=0) # (1, 2s, m, n)
    herm_coeffs = herm_coeffs.transpose([3, 2, 0, 1]) # (n, m, 1, 2s)
    coeffs = coeffs.transpose([3, 2, 0, 1])           # (n, m, 2s, 2s)
    poly_coeffs = herm_coeffs @ coeffs                # (n, m, 1, 2s)


    pp = scipy.interpolate.PPoly(poly_coeffs[:, :, 0, ::-1].transpose([2, 1, 0]), x_boundaries)
    return pp



def hermite_integration(x, y, yd):
    # we integrate by default from x[0] till x[1]

    # 'normalize' by starting from x=0
    # this does not affect the results
    start_x = x[0]
    x -= start_x

    x_hermite = np.repeat(x, 2)
    f_hermite = np.repeat(y, 2)
    n = len(x_hermite)

    # compute the hermite coefficients
    f_table = np.empty((n, n), dtype=np.float64)
    f_table[0, :] = f_hermite
    f_table[1, ::2] = yd
    f_table[1, 1:-1:2] = (f_table[0, 2::2] - f_table[0, :-2:2]) / (x[1:] - x[:-1])
    for i in range(2, n):
        f_table[i, :-i] = (f_table[i-1, 1:n-i+1] - f_table[i-1, :n-i]) / (x_hermite[i:] - x_hermite[:-i])


    herm_coeffs = f_table[:, 0]

    # convert the hermite coeffs to a 'normal' polynomial
    coeffs = np.zeros((n, n))
    coeffs[0, 0] = 1
    for i in range(1, n):
        coeffs[i, :i] = - x_hermite[i-1] * coeffs[i-1, :i]
        coeffs[i, 1:i+1] += coeffs[i-1, :i]

    poly_coeffs = herm_coeffs @ coeffs

    # the integral is equal to this:
    I = np.sum(poly_coeffs / np.arange(n+1)[1:] * x[-1] ** np.arange(n+1)[1:]) \
            - np.sum(poly_coeffs / np.arange(n+1)[1:] * x[0] ** np.arange(n+1)[1:]) 
        
    x += start_x
    return I




if __name__ == "__main__":


    # n = 2, s = 6, m = 2
    x = np.array([[0.2, 1.3, 2.1, 3.7, 4.2, 5.9],
              [6.2, 7.2, 8.3, 9.1, 9.3, 10]], dtype=float).T

    y = lambda x: np.array([x**2 + 2 * np.sin(x), 2 / (x + 1) + x**3 / 10]) 
    yd = lambda x: np.array([2*x + 2 * np.cos(x), - 2 / (x + 1)**2 + 3 * x**2 / 10])
    x_continuous = np.linspace(x[0][0], x[-1][-1], 1000)

    pp = hermite_interp(x, y(x), yd(x))

    x_continuous = np.linspace(x[0][0], x[-1][-1], 1000)
    plt.plot(x, y(x)[0], 'o')
    plt.plot(x, yd(x)[0], 'o')
    plt.plot(x_continuous, pp(x_continuous).T[0])
    plt.plot(x_continuous, pp(x_continuous, 1).T[0])

    plt.plot(x, y(x)[1], 'o')
    plt.plot(x, yd(x)[1], 'o')
    plt.plot(x_continuous, pp(x_continuous).T[1])
    plt.plot(x_continuous, pp(x_continuous, 1).T[1])
    plt.show()


    # n = 1, s = 6, m = 2
    x = np.array([[0.2, 1.3, 2.1, 3.7, 4.2, 5.9],
              [6.2, 7.2, 8.3, 9.1, 9.3, 10]], dtype=float).T
    y = lambda x: x**2 + 2 * np.sin(x)
    yd = lambda x: 2*x + 2 * np.cos(x)

    pp = hermite_interp(x, y(x), yd(x))

    x_continuous = np.linspace(x[0][0], x[-1][-1], 1000)
    plt.plot(x, y(x), 'o')
    plt.plot(x, yd(x), 'o')
    plt.plot(x_continuous, pp(x_continuous))
    plt.plot(x_continuous, pp(x_continuous, 1))
    plt.show()

    # n = 1, s = 6, m = 1
    x = np.array([0.2, 1.3, 2.1, 3.7, 4.2, 5.9], dtype=float).T
    y = lambda x: x**2 + 2 * np.sin(x)
    yd = lambda x: 2*x + 2 * np.cos(x)

    pp = hermite_interp(x, y(x), yd(x))

    x_continuous = np.linspace(x[0], x[-1], 1000)

    plt.plot(x, y(x), 'o')
    plt.plot(x, yd(x), 'o')
    plt.plot(x_continuous, pp(x_continuous))
    plt.plot(x_continuous, pp(x_continuous, 1))
    plt.show()

    print("Integration:")
    I = lambda x: x**3 / 3 - 2 * np.cos(x)
    print(I(x[-1]) - I(x[0]))
    print(hermite_integration(x, y(x), yd(x)))





    

