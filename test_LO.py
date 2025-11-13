
import numpy as np
from pcswe import * 


np.set_printoptions(
    linewidth=400,   # allow long lines before wrapping
    precision=2,     # show only two decimals
    floatmode='maxprec',  # optional: consistent formatting
    # suppress=True    # avoid scientific notation for small numbers
)

pcswe = PerturbationCSWE()
pcswe.H = 7.12
pcswe.A = 0.72
pcswe.L = 8e3
pcswe.r = 0.45
pcswe.h0 = 0.1

pcswe.tol = 1e-8
pcswe.small_number = 0

pcswe.set_derivative_vars()

# pcswe.solve()
# pcswe.visualize_components()

pcswe.solve_LO()
# pcswe.product_dxx(pcswe.y0.x, pcswe.y0.y, pcswe.deriv_LO(pcswe.y0.x, pcswe.y0.y))
# pcswe.visualize_LO()

print(pcswe.y0.c.shape)
print(pcswe.y0.c[:, -1, 3])






np.set_printoptions(
    linewidth=400,   # allow long lines before wrapping
    precision=14,     # show only two decimals
    floatmode='maxprec',  # optional: consistent formatting
    # suppress=True    # avoid scientific notation for small numbers
)
# visualize last polynomail
print(pcswe.y0.c[:, -1, 3])
print(pcswe.y0.x[-3:])





xc = pcswe.y0.x


bnd = 0.9999
bnd = xc[-8]
# bnd = xc[0]

x = linspace(bnd, 1, 100000)

st = np.argmin(abs(x - bnd))
stc = np.argmin(abs(xc - bnd))

fig, axs = plt.subplots(4, 4, figsize=(20, 20))
for nu in range(4):
    for i in range(4):
        axs[nu, i].plot(x[st:], pcswe.y0(x[st:], nu=nu)[i])
        axs[nu, i].plot(xc[stc:], pcswe.y0(xc[stc:], nu=nu)[i], 'o', color='red')

plt.show()

