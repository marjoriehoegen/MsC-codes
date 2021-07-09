import numpy as np

# Function for the relative difference
def relative_dif(x,t1):
	mean = np.mean(x[:t1])
	difs = []
	for j in range(t1,len(x)):
		difs.append(abs((x[j] - mean)/mean))
	return difs

# Convertion to time series
# Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series
def X_to_TS(X_i):
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])

# SSA with base matrix in a initial fixed window
# Calculate D_1 and D_2 monitoring parameters
def ssa_for_detection_fixedw1(F, L, N_base, N_test):
	N_total = len(F)

	D1 = []
	D2 = []

	# Base matrix
	K = N_base - L + 1
	X = np.column_stack([F[i:i+L] for i in range(0,K)])
	d = np.linalg.matrix_rank(X)
	U, Sigma, V = np.linalg.svd(X)
	U_l = U[:,1]

	V = V.T
	X_elem = np.array( [Sigma[i] * np.outer(U[:,i], V[:,i]) for i in range(0,d)])
	Y = X_to_TS(X_elem[0])

	for k in range(N_base,N_total-N_base-N_test):

		# Test matrix
		Q = N_test - L + 1
		X_test = np.column_stack([F[i:i+L] for i in range(K+k,K+k+Q)])

		# D_1
		D = sum((np.dot(X_test[:,j].T,X_test[:,j]) - np.dot(np.dot(X_test[:,j].T,U_l),
			np.dot(U_l.T,X_test[:,j])))for j in range(Q))
		D_norm = D/(L*Q)
		D1.append(D_norm)
		
		U2, Sigma2, V2 = np.linalg.svd(X_test)
		V2 = V2.T
		d2 = np.linalg.matrix_rank(X_test)
		X_elem_test = np.array( [Sigma2[i] * np.outer(U2[:,i], V2[:,i]) for i in range(0,d2)] )
		Y_test = X_to_TS(X_elem_test[0])

		# D_2
		D_withY = np.sqrt(sum(np.square(Y_test[j] - Y[j]) for j in range(N_base)))
		D2.append(D_withY)

		if k==(N_total-N_base-N_test-1):
			return D1, D2

# Monitoring variables from a df with SSA through D_1 and D_2
def test(df,L,N_base,N_test,limit_D1,limit_D2,t1,npts_anomaly):
	'''
		Monitors a df with variables as columns
		Returns column indexes of the altered variables

		Parameters:
		L: number of points of trajectory matrix windows
		N_base: number of points in the base matrix
		N_test: number of points in the test matrix
		limit_D1: detection limit for D_1
		limit_D2: detection limit for D_2
		t1: number of points for initial reference used to calculate relative differences
		npts_anomaly: number of minimum anomaly points to consider an alteration
	'''
	
	altered_variables_D1 = []
	altered_variables_D2 = []

	for column in df:
		variable = df[column]
		F = variable.to_numpy()
		stat_D1, stat_D2 = ssa_for_detection_fixedw1(F, L, N_base, N_test)

		difs_D1 = dif_relativa(stat_D1,t1)
		difs_D2 = dif_relativa(stat_D2,t1)

		anomalies_D1 = []
		anomalies_D2 = []

		for j in range(len(difs_D1)):
			if abs(difs_D1[j]) > limit_D1:
				anomalies_D1.append(j)
			if abs(difs_D2[j]) > limit_D2:
				anomalies_D2.append(j)

		if np.sum(anomalies_D1) > npts_anomaly:
			altered_variables_D1.append(column)
		if np.sum(anomalies_D2) > npts_anomaly:
			altered_variables_D2.append(column)
			
	return altered_variables_D1, altered_variables_D2