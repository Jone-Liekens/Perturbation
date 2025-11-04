
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
    print(x.shape, y.shape, yd.shape)
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

    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        yd = np.expand_dims(yd, axis=1)
    
    x_boundaries = np.array([xs[0] for xs in x.T] + [x[-1][-1]])
    x = x - x[0]

    x_hermite = np.repeat(x, 2, axis=0)
    f_hermite = np.repeat(y, 2, axis=0)
    n = len(x_hermite)
    m = x.shape[1]

    # compute the coefficients
    f_table = np.empty((n, n, m), dtype=np.float64)
    f_table[0, :] = f_hermite
    f_table[1, ::2] = yd
    f_table[1, 1:-1:2] = (f_table[0, 2::2] - f_table[0, :-2:2]) / (x[1:] - x[:-1])
    for i in range(2, n):
        f_table[i, :-i] = (f_table[i-1, 1:n-i+1] - f_table[i-1, :n-i]) / (x_hermite[i:] - x_hermite[:-i])

    herm_coeffs = f_table[:, 0, :]

    # convert the hermite coeffs to a 'normal' polynomial
    coeffs = np.zeros((n, n, m))
    coeffs[0, 0] = 1
    for i in range(1, n):
        coeffs[i, :i] = - x_hermite[i-1] * coeffs[i-1, :i] # we just expand the product (x-x0) (x-x1) (x-x2) ... 
        coeffs[i, 1:i+1] += coeffs[i-1, :i]


    herm_coeffs = np.expand_dims(herm_coeffs, axis=0) # (1, 12, 2)
    herm_coeffs = herm_coeffs.transpose([2, 0, 1])
    coeffs = coeffs.transpose([2, 0, 1])
    result = herm_coeffs @ coeffs # (2, 1, 12)

    poly_coeffs = result[:, 0, :].T # (12, 2)

    pp = scipy.interpolate.PPoly(poly_coeffs[::-1], x_boundaries)

    return pp


def hermite_integration(x, y, yd):
    print(x.shape, y.shape, yd.shape)

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


    print(f_table)
    herm_coeffs = f_table[:, 0]

    # convert the hermite coeffs to a 'normal' polynomial
    coeffs = np.zeros((n, n))
    coeffs[0, 0] = 1
    for i in range(1, n):
        coeffs[i, :i] = - x_hermite[i-1] * coeffs[i-1, :i]
        coeffs[i, 1:i+1] += coeffs[i-1, :i]

    print(coeffs)

    poly_coeffs = herm_coeffs @ coeffs
    print(poly_coeffs)

    # because we started from x=0 (by substracting x[0]), the integral is equal to this:
    return np.sum(poly_coeffs / np.arange(n+1)[1:] * x[-1] ** np.arange(n+1)[1:])




if __name__ == "__main__":
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


    x = np.array([0.2, 1.3, 2.1, 3.7, 4.2, 5.9], dtype=float)
    y = lambda x: x**2 + 2 * np.sin(x)
    yd = lambda x: 2*x + 2 * np.cos(x)

    pp = hermite_interp(x, y(x), yd(x))

    print(y(x))
    print(pp(x))
    print(yd(x))
    print(pp(x, 1))

    x_continuous = np.linspace(x[0], x[-1], 1000)
    plt.plot(x, y(x), 'o')
    plt.plot(x, yd(x), 'o')
    plt.plot(x_continuous, pp(x_continuous))
    plt.plot(x_continuous, pp(x_continuous, 1))
    plt.show()

    
    print("Integration:")
    I = lambda x: x**3 / 3 - 2 * np.cos(x)
    print(I(x[-1]))
    print(hermite_integration(x, y(x), yd(x)))





    

