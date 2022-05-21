import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

import matplotlib.pyplot as plt
import pandas as pd


"""
Fit data of the inductance in a coil to the function
L(x) = A*exp(-B|x|^C) + D
"""

def L(x, A, B, C, D):
	return A*np.exp(-B*np.power(np.abs(x), C)) + D

def plot_relation(x, y, ax1, ax2):
	"""Find and plot the relation between x and y"""
	logx = np.log(x)
	logy = np.log(y)

	res = linregress(logx, logy)

	ax1.scatter(logx, logy, marker="o", c="b", label="Original data")
	logx_lin = np.linspace(np.min(logx), np.max(logx))
	ax1.plot(logx_lin, res.intercept + res.slope*logx_lin, 'r', label=f"Fitted line with a slope {res.slope:.4f}")
	ax1.legend()

	ax2.scatter(x, y, marker="o", c="b", label="Original data")
	x_lin = np.linspace(np.min(x), np.max(x))
	ax2.plot(x_lin, x_lin**res.slope, 'r', label=f"Fitted line with an exponent {res.slope:.4f}")
	ax2.legend()


data_file = "/Users/Paulsson/Documents/Private/Simon/Skola/Högskola/Bachelor/coilgun/src/ode_models/data/VarL_N_1000NEWEST.csv"
variable_name = "l (mm)"
# data_file = None

if data_file is None:
	xdata = []
	ydata = []
	variables = np.arange(5) + 1 
	for N in variables:

		x = np.linspace(-5, 5, 100)

		y0 = L(x, 5*N**2, 2, 2, N)

		rng = np.random.default_rng()
		y_noise = 0.2 * rng.normal(size=x.size)

		y = y0 + y_noise

		xdata.append(x)
		ydata.append(y)

else: 
	data = pd.read_csv(data_file, sep=',')

	data_set = []
	xdata = []
	ydata = []

	# Pick out all values of the variable being varied
	variables = np.unique(data[variable_name].to_numpy())

	for var in variables:
		data_N = data.loc[data[variable_name] == var]

		x = data_N['yPos (mm)'].to_numpy()
		y = data_N['Coil inductance (uH)'].to_numpy()

		# Convert to meter (m) and henry (H)
		x = x / 1e3
		y = y / 1e6

		# Add the reflected data to get more data points
		xdata.append(np.append(-x, x))
		ydata.append(np.append(y, y))

		data_set.append(data_N)

fig, axes = plt.subplots(1, len(xdata))

# Fit A, B, C, D to the data
ABCDs = []
for x, y, ax in zip(xdata, ydata, axes):
	# Good starting point
	D0 = np.min(y)
	A0 = np.max(y) - D0

	# Estimate B0 and C0
	L_fix_AD = lambda x, B, C: L(x, A0, B, C, D0)
	(B0, C0), _ = curve_fit(L_fix_AD, x, y, maxfev=10_000, bounds=(0, [np.inf, 10]), p0=[1e2, 1])

	# Fit ABCD
	ABCD, _ = curve_fit(L, x, y, maxfev=10_000, bounds=(0, [np.inf, np.inf, 10, np.inf]), p0=[A0, B0, C0, D0])
	
	ABCD = (A0, B0, C0, D0)
	ABCDs.append(ABCD)

	print(ABCD)

	# Plot fitted data
	ax.scatter(x, y, marker="o", c="b", label="Original data")
	xlin = np.linspace(np.min(x), np.max(x), num=500)
	ax.plot(xlin, L(xlin, *ABCD), 'r', label="Fitted line")
	ax.legend()

ABCDs = np.array(ABCDs)

fig1, ((ax_A1, ax_B1), (ax_C1, ax_D1)) = plt.subplots(2, 2)
fig2, ((ax_A2, ax_B2), (ax_C2, ax_D2)) = plt.subplots(2, 2)

# Find relation to A
# ax_A1.scatter(np.log(variables), np.log(ABCDs[:,0]))

# ax_A2.scatter(variables, ABCDs[:,0])
# ax_A2.set_ylim(bottom=0)
plot_relation(variables, ABCDs[:,0], ax_A1, ax_A2)
plot_relation(variables, ABCDs[:,1], ax_B1, ax_B2)
plot_relation(variables, ABCDs[:,2], ax_C1, ax_C2)
plot_relation(variables, ABCDs[:,3], ax_D1, ax_D2)

plt.show()

# plt.scatter(xdata, ydata, marker='*', c='b', label='data')
# plt.plot(
# 	xdata, 
# 	L(xdata, *popt), 
# 	'r-',
# 	label='fit: A=%5.3f, B=%5.3f, C=%5.3f, D=%5.3f' % tuple(popt)
# )
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

