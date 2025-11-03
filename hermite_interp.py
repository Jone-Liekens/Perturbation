
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(
    linewidth=200,   # allow long lines before wrapping
    precision=14,     # show only two decimals
    floatmode='maxprec',  # optional: consistent formatting
    # suppress=True    # avoid scientific notation for small numbers
)


def hermite_interp(x, y, yd):
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

    herm_coeffs = f_table[:, 0]

    # convert the hermite coeffs to a 'normal' polynomial
    coeffs = np.zeros((n, n))
    coeffs[0, 0] = 1
    for i in range(1, n):
        coeffs[i, :i] = - x_hermite[i-1] * coeffs[i-1, :i]
        coeffs[i, 1:i+1] += coeffs[i-1, :i]

    poly_coeffs = herm_coeffs @ coeffs

    # because we started from x=0 (by substracting x[0]), the integral is equal to this:
    return np.sum(poly_coeffs / np.arange(n+1)[1:] * x[-1] ** np.arange(n+1)[1:])




if __name__ == "__main__":
    x = np.array([0.2, 1.3, 2.1, 3.7, 4.2, 5.9], dtype=float)
    x_continuous = np.linspace(0, x[-1], 1000)

    y = lambda x: x**2 + 2 * np.sin(x)
    yd = lambda x: 2*x + 2 * np.cos(x)

    p = hermite_interp(x, y(x), yd(x))
    y_continuous = p(x_continuous)
    yd_continuous = p.deriv()(x_continuous)




    print(y(x))
    print(p(x))
    print(yd(x))
    print(p.deriv()(x))


    print("Integration:")
    I = lambda x: x**3 / 3 - 2 * np.cos(x)
    print(I(x[-1]))
    print(hermite_integration(x, y(x), yd(x)))

    plt.plot(x, y(x), 'o')
    plt.plot(x, yd(x), 'o')
    plt.plot(x_continuous, y_continuous)
    plt.plot(x_continuous, yd_continuous)
    plt.show()




    

