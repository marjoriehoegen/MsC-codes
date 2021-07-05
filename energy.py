import numpy  as np

# Calculate energy for the entire df
def energy_df(df,L):
	df_sq = df**2
	energy = df_sq.rolling(L).sum()
	energy.dropna(inplace=True)
	energy.reset_index(inplace = True,drop=True)
	return energy

# Function for relative difference
def relative_dif(x,t1):
	mean = np.mean(x[:t1])
	difs = []
	for j in range(t1,len(x)):
		difs.append(abs((x[j] - mean)/mean))
	return difs

# Monitoring variables from a df with energy analysis
def test(df, L, t1, limit, npts_anomaly):
	'''
		Monitors a df with variables as columns
		Returns column names of the altered variables

		Parameters:
		L: number points for the rolling window
		t1: number of initial points for reference to calculate relative difference
		limit: detection limit related to relative difference
		npts_anomaly: number of minimum anomaly points to consider an alteration

	'''
	altered_variables = []

	# Calculate energy for all variables in the df
	energydf = energy_df(df,L)

	# Monitor each df variable
	for h in df:
		energy = energydf[h]
		dif_en = relative_dif(energy,t1)

		anomalies_energy = []
		for i in range(len(dif_en)):
			if (dif_en[i] > limit):
				anomalies_energy.append(i)

		if np.sum(anomalies_energy) > npts_anomaly:
			altered_variables.append(h)

	return altered_variables