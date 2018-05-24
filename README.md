# masterthesis

# To Do's:

2018 06 23 --> 2018 06 24
-------------------------

- [X] Merging convective scheme and diffusion scheme into residual function
    + tested --> working
- [ ] update solver function
- [X] update jacobian
	--> implemented just constant jacobian
	--> updated chapter in docu (roughly finished)
	--> image of jacobian structure
- [X] optimize residual funcion using numba
    + tested --> working
- [ ] test jacobian as class with an update method
- [X] start using boundary conditions in residual
    + into residual, solver, main, updated setup file
- [ ] test notebook about method for small frequencies in impedance measurements (fitting elipse and so on)
- [ ] add chapter to documentation about python implementation

2018 06 24 --> 2018 06 25
-------------------------


I = 3
# create x axis for simulation
xi = np.zeros(I+1, dtype = np.float64)
x_ = np.zeros(I+1, dtype = np.float64)

for i in range(0,I+1):

    xi[i] = i / I

# creating x axis with functional relation
x_ = xi
#x_ = np.sin( ( np.pi * xi  ) / 2 ) ** 2

# getting cell volumes
Dx = np.zeros(I, dtype = np.float64)
Dx = x_[1:] - x_[:I]
epsilon = np.ones(I + 1, dtype = np.float64)
DC = np.ones(I + 1, dtype = np.float64)
DA = np.ones(I + 1, dtype = np.float64)

chi2 = 2
Dt = 1e-4
Jac = constJac( I, Dt, Dx, DC, DA, epsilon, chi2)

print(Jac[0:I,:])