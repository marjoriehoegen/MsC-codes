import numpy as np

# Standardization function
def standardization(df, mean, sd):
	df2 = df.copy()
	for i in range(df2.shape[1]):
		df2[:,i] = (df2[:,i] - mean[i])/sd[i]
	df_scaled = df2

	return df_scaled

# PCA model estimation
def pca_model(df_scaled):

	# Covariance matrix and decomposition to get eigenvalues and eigenvectors
	cov_matrix = np.cov(df_scaled, rowvar=False)
	eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
	eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
	eig_pairs = sorted(eig_pairs, key= lambda x: x[0], reverse=True)

	return eig_pairs, eigenvalues, eigenvectors

# Total explained variance
def explained_variance(eig_pairs,eigenvalues,n):
	total_variance = []
	for i in range(n):
		expl_var = eig_pairs[i][0]/np.sum(eigenvalues).real
		total_variance.append(expl_var)
	total = sum(total_variance)
	return total

# Feature vectors
def feature_vector(n_pcs, eig_pairs):
	pcs_values = []
	pcs_vectors = []
	
	for i in range(n_pcs):
		pcs_values.append(eig_pairs[i][0])
		pcs_vectors.append(eig_pairs[i][1])

	pcs_values = np.asarray(pcs_values)
	pcs_vectors = np.asarray(pcs_vectors)

	return pcs_values, pcs_vectors

# Data projection into pcs space
def transform(df_scaled, pcs_vectors):
	df_proj = df_scaled @ pcs_vectors.transpose()
	return df_proj

# Inverse data projection into original space
def inv_transform(df_proj, pcs_vectors):
	df_inv = df_proj @ pcs_vectors
	return df_inv

# SPE
def SPE(df_scaled,df_inv):
	spe = ((df_scaled - df_inv)**2).sum()
	return spe

# T^2
def T2(df_proj, pcs_values):
	hT2 = []
	for i in range(len(pcs_values)):
		T2 = (df_proj[0][i])**2/pcs_values[i]
		hT2.append(T2)
	return sum(hT2)

# Contribution to SPE and T^2

def contr_SPE(x_scaled,x_proj_inv,pcs_vectors):
	caj = []
	for j in range(pcs_vectors.shape[1]):
		caj.append((x_scaled[0,j] - x_proj_inv[0,j])**2)
	return caj

def contr_T2(x_scaled,x_proj,pcs_values,pcs_vectors):
	caj = []
	for j in range(pcs_vectors.shape[1]):
		contr = []
		for a in range(len(pcs_values)):
			c_a = (x_proj[0][a]/pcs_values[a]) *pcs_vectors[a][j] * x_scaled[0,j]
			contr.append(c_a)
			c = sum(x for x in contr if x > 0) # só pega valores positivos

		caj.append(c)

	return caj

# Monitoring variables from a df with PCA
def test(df_panda,t1,n_pcs,n_variables,limit_contrSPE,limit_contrT2,npts_anomaly):
	'''
		Monitors a df with variables as columns
		Returns column indexes of the altered variables

		Parameters:
		t1: number of initial points to calculate PCA model and obtain pcs
		n_pcs: number of pcs (principal components) to retain
		n_variables: number of variables in the original data
		limit_contrSPE: detection limit using contribution to SPE
		limit_contrT2: detection limit using contribution to T^2
		npts_anomaly: number of minimum anomaly points to consider an alteration
	'''

	df = df_panda.to_numpy()
	
	# PCA model calculation
	df_t1 = df[:t1,:].copy()
	mean1 = np.mean(df_t1,axis=0)
	sd1 = np.std(df_t1,axis=0)
	df_t1_scaled = standardization(df_t1,mean1,sd1)	
	eig_pairs, eigenvalues, eigenvectors = pca_model(df_t1_scaled)

	# Data transformation into the selected PCs space
	pcs_values, pcs_vectors = feature_vector(n_pcs,eig_pairs)
	total_expl_var = explained_variance(eig_pairs,eigenvalues,n_pcs)
	df_pca = transform(df_t1_scaled,pcs_vectors)

	# SPE, T2 and contributions
	df_t2 = df[t1:,].copy()
	spe = []
	hT2 = []

	caj_T2 = []
	caj_SPE = []

	for i in range(df_t2.shape[0]):
		sample = df_t2[i,:]
		sample = np.reshape(sample,(n_variables,1))
		sample = sample.T
		sample_scaled = standardization(sample,mean1,sd1)
		
		proj_sample = transform(sample_scaled,pcs_vectors)
		proj_sample_inv = inv_transform(proj_sample,pcs_vectors)

		spe1 = SPE(sample_scaled, proj_sample_inv)
		spe.append(spe1)
		
		h1 = T2(proj_sample,pcs_values)
		hT2.append(h1)

		caj_T2.append(contr_T2(sample_scaled,proj_sample,pcs_values,pcs_vectors))
		caj_SPE.append(contr_SPE(sample_scaled,proj_sample_inv,pcs_vectors))

	caj_T2 = np.array(caj_T2)
	caj_SPE = np.array(caj_SPE)

	# Variables monitoring with SPE and T^2 contribuitions
	altered_variables_SPE = []
	altered_variables_T2 = []

	for i in range(n_variables):
		contr_spe = caj_SPE[:,i]
		contr_t2 = caj_T2[:,i]
		
		anomalies_spe = []
		anomalies_t2 = []
		for j in range(len(contr_spe)):
			if abs(contr_spe[j]) > limit_contrSPE:
				anomalies_spe.append(j)
			if abs(contr_t2[j]) > limit_contrT2:
				anomalies_t2.append(j)

		if np.sum(anomalies_spe) > npts_anomaly:
			altered_variables_SPE.append(str((i+1)))
		if np.sum(anomalies_t2) > npts_anomaly:
			altered_variables_T2.append(str((i+1)))

	return altered_variables_SPE, altered_variables_T2