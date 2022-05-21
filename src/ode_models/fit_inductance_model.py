import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd

from utils.path import src_path


"""
Fit data of the inductance in a coil to the function
L(x) = A*exp(-B|x|^C) + D
"""

def L(x, A, B, C, D):
	return A*np.exp(-B*np.power(np.abs(x), C)) + D

def func(x, C, a, b):
	"""
	Function to fit the parameters A, B, C and D to
	X = C*N^a*l^b
	"""
	N = x[:,0]
	l = x[:,1]

	return C * N**a * l**b

def B_func(x, C, a, b):
	N = x[:,0]
	l = x[:,1]
	return np.log(C * N**a * np.exp(b*l))

def func_Nl(N, l, C, a, b):
	return C * N**a * l**b

def B_func_Nl(N, l, C, a, b):
	return C * N**a * np.exp(b*l)

def fit_data_to_inductance_func(x, y):
	# Good starting point
	D0 = np.min(y)
	A0 = np.max(y) - D0

	# Estimate B0 and C0
	L_fix_AD = lambda x, B, C: L(x, A0, B, C, D0)
	(B0, C0), _ = curve_fit(L_fix_AD, x, y, maxfev=10_000, bounds=(0, [1e9, 10]), p0=[1e2, 1], ftol=3e-24, xtol=3e-24, gtol=3e-16, method='trf')

	# Fit ABCD
	ABCD, _ = curve_fit(L, x, y, maxfev=10_000, bounds=(0, [np.inf, 1e9, 10, np.inf]), p0=[A0, B0, C0, D0], ftol=3e-24, xtol=3e-24, gtol=3e-16, method='trf')
	
	return ABCD

def fit_param_to_fitted_curve(x, y):
	"""Fit a function to A, B, C or D from the data"""
	(C, a, b), _ = curve_fit(func, x, y, maxfev=10_000)
	return C, a, b

def fit_B(x, y):
	(C, a, b), _ = curve_fit(B_func, x, np.log(y), maxfev=10_000, p0=[2.5186e+08, -7.0269e-01, 2.6432e+02], ftol=3e-24, xtol=3e-24, gtol=3e-16, method='trf')
	return C, a, b

def fit_param_log(x, y):
	"""Fit with log plot"""
	line = lambda x, C, a, b: C + a*x[:,0] + b*x[:,1]
	(C, a, b), _ = curve_fit(line, np.log(x), np.log(y), maxfev=10_000)
	return np.exp(C), a, b

def fit_param_log_B(x, y):
	"""Fit with log plot"""
	line = lambda x, C, a, b: C + a*np.log(x[:,0]) + b*x[:,1]
	(C, a, b), _ = curve_fit(line, x, np.log(y), maxfev=10_000)
	return np.exp(C), a, b


def prepare_data(file, param_name, other_param):
	"""
	Prepare data that varies N
	
	param_name: "N" or "l (mm)"
	"""
	data = pd.read_csv(file, sep=',')

	# Pick out all values of the variable being varied
	parmas = np.unique(data[param_name].to_numpy())

	xdata, ydata = [], []
	ABCDs = []
	variables = []

	for var in parmas:
		data_N = data.loc[data[param_name] == var]

		x = data_N['yPos (mm)'].to_numpy()
		y = data_N['Coil inductance (uH)'].to_numpy()

		# Convert to meter (m) and henry (H)
		x = x / 1e3
		y = y / 1e6

		# Add the reflected data to get more data points
		x = np.append(-x, x)
		y = np.append(y, y)

		xdata.append(x)
		ydata.append(y)

		# Calculate ABCD
		ABCDs.append(fit_data_to_inductance_func(x, y))

		if param_name == "N":
			variables.append((var, other_param / 1e3))
		else:
			variables.append((other_param, var / 1e3))

	return np.array(xdata), np.array(ydata), np.array(ABCDs), np.array(variables)

def prepare_data_new(files):
	"""Prepare the new data"""
	print("Reading in data...")
	# Add every data frame read from a file here to concat later
	dfs = []
	for file in files:
		df = pd.read_csv(file, sep=',')

		coulmns = df.columns

		# Convert all the units
		for column in coulmns:
			if "(mm)" in column:
				df[column] = df[column] * 1e-3
			elif "(uH)" in column:
				df[column] = df[column] * 1e-6

		# Rename the columns
		df.columns = ['pos', 'l', 'N', 'L']

		dfs.append(df)

	# Concatenate all the data
	data = pd.concat(dfs, ignore_index=True)

	print("Done")
	print("Fitting data to inductance function")

	# Extract data
	xdata, ydata = [], []
	ABCDs = []
	Nls = []
	for l in range(10, 101, 10):
		for N in range(100, 1001, 100):
			# Get data with fix l and N (l in mm)
			data_Nl = data.loc[(data['l'] == l * 1e-3) & (data['N'] == N)]
			
			# Get position and inductance values and add the reflected points
			x = data_Nl['pos']
			y = data_Nl['L']

			x = np.append(-x, x)
			y = np.append(y, y)

			# Get constants A, B, C and D
			A, B, C, D = fit_data_to_inductance_func(x, y)

			# Add data to storage
			xdata.append(x)
			ydata.append(y)
			ABCDs.append([A, B, C, D])
			Nls.append([N, l * 1e-3])

	print("Done")
	
	return xdata, ydata, ABCDs, Nls




def plot_inductance_realtion(ax, x, y, ABCD):
	"""Plot inductance relation"""
	ax.scatter(x*1e3, 1e3 * y, marker="*", c="black", label="Data från COMSOL")
	xlin = np.linspace(np.min(x), np.max(x), num=500)
	ax.plot(xlin*1e3, 1e3 * L(xlin, *ABCD), label="Bästa anpassningen av\nL(x) till datan")
	ax.legend()

def plot_ABCD_relation_N(ax, N, y, params, l):
	ax.scatter(N, y, marker="o", c="b", label="Original data")
	x = np.expand_dims(np.linspace(np.min(N), np.max(N), num=100), axis=1)
	x = np.concatenate((x, l*np.ones_like(x)), axis=1)
	ax.plot(x[:,0], func(x, *params), 'r', label="Fitted line")
	ax.legend()

def plot_ABCD_relation_l(ax, l, y, params, N):
	ax.scatter(l, y, marker="o", c="b", label="Original data")
	x = np.expand_dims(np.linspace(np.min(l), np.max(l), num=100), axis=1)
	x = np.concatenate((N*np.ones_like(x), x), axis=1)
	ax.plot(x[:,1], func(x, *params), 'r', label="Fitted line")
	ax.legend()


def new_main():
	files = ["varXLN_L_10-30mm.csv", "varXLN_L_30-100mm.csv"]
	# Add full path
	files = [src_path() / "ode_models" / "data" / file for file in files]
	
	xdata, ydata, ABCDs, Nls = prepare_data_new(files)

	ABCDs = np.array(ABCDs)
	Nls = np.array(Nls)

	N = Nls[:,0]
	l = Nls[:,1]

	A = ABCDs[:,0]
	B = ABCDs[:,1]
	C = ABCDs[:,2]
	D = ABCDs[:,3]

	params_A = fit_param_log(Nls, A)
	params_B = fit_param_log_B(Nls, B)
	params_C = fit_param_log(Nls, C)
	params_D = fit_param_log(Nls, D)

	print(f"Fit for parmeters A = {params_A[0]:.4e} * N**{params_A[1]:.3f} * l**{params_A[2]:.3f}")
	print(f"Fit for parmeters B = {params_B[0]:.4e} * N**{params_B[1]:.3f} * e**({params_B[2]:.3f}*l)")
	print(f"Fit for parmeters C = {params_C[0]:.4e} * N**{params_C[1]:.3f} * l**{params_C[2]:.3f}")
	print(f"Fit for parmeters D = {params_D[0]:.4e} * N**{params_D[1]:.3f} * l**{params_D[2]:.3f}")

	# fig = plt.figure(figsize=plt.figaspect(1))
	# ax_A = fig.add_subplot(2, 2, 1, projection='3d')
	# ax_B = fig.add_subplot(2, 2, 2, projection='3d')
	# ax_C = fig.add_subplot(2, 2, 3, projection='3d')
	# ax_D = fig.add_subplot(2, 2, 4, projection='3d')

	# # Add data points
	# ax_A.scatter3D(np.log(N), np.log(l), np.log(A), color='red')
	# ax_B.scatter3D(np.log(N), l, np.log(B), color='red')
	# ax_C.scatter3D(np.log(N), np.log(l), np.log(C), color='red')
	# ax_D.scatter3D(np.log(N), np.log(l), np.log(D), color='red')

	# # Add approximation curve
	# N = np.linspace(100, 1000, 10)
	# l = np.linspace(10e-3, 100e-3, 10)

	# N, l = np.meshgrid(N, l)

	# A = func_Nl(N, l, *params_A)
	# B = B_func_Nl(N, l, *params_B)
	# C = func_Nl(N, l, *params_C)
	# D = func_Nl(N, l, *params_D)

	# ax_A.plot_wireframe(np.log(N), np.log(l), np.log(A), color='black')
	# ax_B.plot_wireframe(np.log(N), l, np.log(B), color='black')
	# ax_C.plot_wireframe(np.log(N), np.log(l), np.log(C), color='black')
	# ax_D.plot_wireframe(np.log(N), np.log(l), np.log(D), color='black')

	# fig = plt.figure(figsize=plt.figaspect(1))
	# ax_A = fig.add_subplot(2, 2, 1)
	# ax_B = fig.add_subplot(2, 2, 2)
	# ax_C = fig.add_subplot(2, 2, 3)
	# ax_D = fig.add_subplot(2, 2, 4)

	# Share a X axis with each column of subplots
	# fig, ((ax_A, ax_B), (ax_C, ax_D)) = plt.subplots(2, 2, sharex='col')

	# # Plot with fix l
	# l_fix = 60.0e-3
	# mask = l == l_fix
	# # Add data points
	# ax_A.scatter(np.log(N[mask]), np.log(A[mask]), color='black', marker='*', label="Anpassade värden på A")
	# ax_B.scatter(np.log(N[mask]), np.log(B[mask]), color='black', marker='*', label="Anpassade värden på B")
	# ax_C.scatter(np.log(N[mask]), np.log(C[mask]), color='black', marker='*', label="Anpassade värden på C")
	# ax_D.scatter(np.log(N[mask]), np.log(D[mask]), color='black', marker='*', label="Anpassade värden på D")

	# # Add approximation curve
	# Nn = np.linspace(100, 1000, 50)

	# Aa = np.log(func_Nl(Nn, l_fix, *params_A))
	# Bb = np.log(B_func_Nl(Nn, l_fix, *params_B))
	# Cc = np.log(func_Nl(Nn, l_fix, *params_C))
	# Dd = np.log(func_Nl(Nn, l_fix, *params_D))

	# ax_A.plot(np.log(Nn), Aa, color='black', label="Anpassad kurva för A")
	# ax_B.plot(np.log(Nn), Bb, color='black', label="Anpassad kurva för B")
	# ax_C.plot(np.log(Nn), Cc, color='black', label="Anpassad kurva för C")
	# ax_D.plot(np.log(Nn), Dd, color='black', label="Anpassad kurva för D")

	# ax_A.set_ylabel(r"$\ln(A)$")
	# # ax_A.set_xlabel(r"$\ln(N)$")
	# ax_B.set_ylabel(r"$\ln(B)$")
	# # ax_B.set_xlabel(r"$\ln(N)$")
	# ax_C.set_ylabel(r"$\ln(C)$")
	# ax_C.set_xlabel(r"$\ln(N)$")
	# ax_D.set_ylabel(r"$\ln(D)$")
	# ax_D.set_xlabel(r"$\ln(N)$")

	fig, ((ax_A, ax_B), (ax_C, ax_D)) = plt.subplots(2, 2)

	# Plot with fix l
	N_fix = 100
	mask = N == N_fix
	# Add data points
	ax_A.scatter(np.log(l[mask]), np.log(A[mask]), color='black', marker='*', label="Anpassade värden på A")
	ax_B.scatter(l[mask], np.log(B[mask]), color='black', marker='*', label="Anpassade värden på B")
	ax_C.scatter(np.log(l[mask]), np.log(C[mask]), color='black', marker='*', label="Anpassade värden på C")
	ax_D.scatter(np.log(l[mask]), np.log(D[mask]), color='black', marker='*', label="Anpassade värden på D")

	# Add approximation curve
	ll = np.linspace(10, 100, 50)*1e-3

	Aa = np.log(func_Nl(N_fix, ll, *params_A))
	Bb = np.log(B_func_Nl(N_fix, ll, *params_B))
	Cc = np.log(func_Nl(N_fix, ll, *params_C))
	Dd = np.log(func_Nl(N_fix, ll, *params_D))

	ax_A.plot(np.log(ll), Aa, color='black', label="Anpassad kurva för A")
	ax_B.plot(ll, Bb, color='black', label="Anpassad kurva för B")
	ax_C.plot(np.log(ll), Cc, color='black', label="Anpassad kurva för C")
	ax_D.plot(np.log(ll), Dd, color='black', label="Anpassad kurva för D")

	ax_A.set_ylabel(r"$\ln(A)$")
	ax_A.set_xlabel(r"$\ln(l)$")
	ax_B.set_ylabel(r"$\ln(B)$")
	ax_B.set_xlabel(r"$l$")
	ax_C.set_ylabel(r"$\ln(C)$")
	ax_C.set_xlabel(r"$\ln(l)$")
	ax_D.set_ylabel(r"$\ln(D)$")
	ax_D.set_xlabel(r"$\ln(l)$")

	ax_A.legend()
	ax_B.legend()
	ax_C.legend()
	ax_D.legend()

	plt.show()

	if input('Show all data? (y/n): ') == 'y':
		for x, y, ABCD, (N, l) in zip(xdata, ydata, ABCDs, Nls):
			plt.figure(num=1, figsize=(14, 8), dpi=80)
			ax = plt.subplot2grid((1,1),(0, 0))
			plot_inductance_realtion(ax, x, y, ABCD)

			A_approx = func_Nl(N, l, *params_A)
			B_approx = B_func_Nl(N, l, *params_B)
			C_approx = func_Nl(N, l, *params_C)
			D_approx = func_Nl(N, l, *params_D)

			xlin = np.linspace(np.min(x), np.max(x), num=500)

			ax.plot(xlin * 1e3, 1e3 * L(xlin, A_approx, B_approx, C_approx, D_approx), '--', label="L(x) med anpassade värden\npå A, B, C och D")

			ax.legend(fontsize=16)

			ax.set_xlabel(r"$x$ [mm]", fontsize=16)
			ax.set_ylabel(r"$L(x)$ [mH]", fontsize=16)

			plt.xticks(fontsize=16)
			plt.yticks(fontsize=16)

			A, B, C, D = ABCD
			print(f"{A=} : {A_approx=} : Error={(A_approx / A - 1) * 100:.02f}")
			print(f"{B=} : {B_approx=} : Error={(B_approx / B - 1) * 100:.02f}")
			print(f"{C=} : {C_approx=} : Error={(C_approx / C - 1) * 100:.02f}")
			print(f"{D=} : {D_approx=} : Error={(D_approx / D - 1) * 100:.02f}")
			print(f"{N=}")
			print(f"{l=}")
			plt.show()





def main():
	data_var_N_l_50 = "/Users/Paulsson/Documents/Private/Simon/Skola/Högskola/Bachelor/coilgun/src/ode_models/data/VarN_L_50mmNEWEST.csv"
	data_var_l_N_100 = "/Users/Paulsson/Documents/Private/Simon/Skola/Högskola/Bachelor/coilgun/src/ode_models/data/VarL_N_100NEWEST.csv"
	data_var_l_N_1000 = "/Users/Paulsson/Documents/Private/Simon/Skola/Högskola/Bachelor/coilgun/src/ode_models/data/VarL_N_1000NEWEST.csv"

	xdata1, ydata1, ABCDs1, variables1 = prepare_data(data_var_N_l_50, param_name="N", other_param=50)
	xdata2, ydata2, ABCDs2, variables2 = prepare_data(data_var_l_N_100, param_name="l (mm)", other_param=100)
	xdata3, ydata3, ABCDs3, variables3 = prepare_data(data_var_l_N_1000, param_name="l (mm)", other_param=1000)

	xdata = np.concatenate((xdata1, xdata2, xdata3), axis=0)
	ydata = np.concatenate((ydata1, ydata2, ydata3), axis=0)
	ABCDs = np.concatenate((ABCDs1, ABCDs2, ABCDs3), axis=0)
	variables = np.concatenate((variables1, variables2, variables3), axis=0)

	# Plot all inductance relations
	# yplots = 3
	# xplots = len(xdata) // yplots
	# fig, axes1 = plt.subplots(yplots, xplots)

	# i = 0
	# for yaxes in axes1:
	# 	for ax in yaxes:
	# 		x = xdata[i]
	# 		y = ydata[i]
	# 		ABCD = ABCDs[i]

	# 		plot_inductance_realtion(ax, x, y, ABCD)

	# 		i += 1

	A = ABCDs[:,0]
	B = ABCDs[:,1]
	C = ABCDs[:,2]
	D = ABCDs[:,3]

	params_A = fit_param_to_fitted_curve(variables, A)
	params_B = fit_param_to_fitted_curve(variables, B)
	params_C = fit_param_to_fitted_curve(variables, C)
	params_D = fit_param_to_fitted_curve(variables, D)

	print(f"Fit for parmeters A = {params_A[0]:.4e} * N**{params_A[1]:.4e} * l**{params_A[2]:.4e}")
	print(f"Fit for parmeters B = {params_B[0]:.4e} * N**{params_B[1]:.4e} * l**{params_B[2]:.4e}")
	print(f"Fit for parmeters C = {params_C[0]:.4e} * N**{params_C[1]:.4e} * l**{params_C[2]:.4e}")
	print(f"Fit for parmeters D = {params_D[0]:.4e} * N**{params_D[1]:.4e} * l**{params_D[2]:.4e}")

	# Plot relation for N
	fig1, ((ax_A1, ax_B1), (ax_C1, ax_D1)) = plt.subplots(2, 2)
	plot_ABCD_relation_N(ax_A1, variables1[:,0], ABCDs1[:,0], params_A, l=50e-3)
	plot_ABCD_relation_N(ax_B1, variables1[:,0], ABCDs1[:,1], params_B, l=50e-3)
	plot_ABCD_relation_N(ax_C1, variables1[:,0], ABCDs1[:,2], params_C, l=50e-3)
	plot_ABCD_relation_N(ax_D1, variables1[:,0], ABCDs1[:,3], params_D, l=50e-3)

	# Plot relation for l (N=100)
	fig2, ((ax_A2, ax_B2), (ax_C2, ax_D2)) = plt.subplots(2, 2)
	plot_ABCD_relation_l(ax_A2, variables2[:,1], ABCDs2[:,0], params_A, N=100)
	plot_ABCD_relation_l(ax_B2, variables2[:,1], ABCDs2[:,1], params_B, N=100)
	plot_ABCD_relation_l(ax_C2, variables2[:,1], ABCDs2[:,2], params_C, N=100)
	plot_ABCD_relation_l(ax_D2, variables2[:,1], ABCDs2[:,3], params_D, N=100)

	# Plot relation for l (N=1000)
	fig3, ((ax_A3, ax_B3), (ax_C3, ax_D3)) = plt.subplots(2, 2)
	plot_ABCD_relation_l(ax_A3, variables3[:,1], ABCDs3[:,0], params_A, N=1000)
	plot_ABCD_relation_l(ax_B3, variables3[:,1], ABCDs3[:,1], params_B, N=1000)
	plot_ABCD_relation_l(ax_C3, variables3[:,1], ABCDs3[:,2], params_C, N=1000)
	plot_ABCD_relation_l(ax_D3, variables3[:,1], ABCDs3[:,3], params_D, N=1000)

	plt.show()





if __name__ == '__main__':
	new_main()
