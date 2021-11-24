import numpy as np
import scipy.stats as stats

# Function for monitoring with maximum value
def control_max(x,nref,limit):
	anomalies = []

	# Select nref points of x and its maximum value
	xref = x[:nref]
	xmax = abs(max(xref))

	# Check if the other points of x are bigger than max(xref) * limit
	for i in range(nref,len(x)-nref):
		if abs(x[i]) > (limit*xmax):
			anomalies.append(i)
	return xmax,anomalies

# Kullback-Leibler Divergence
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# Calculate KLD between reference PDF (pdf1) and other pdfs
def kld_ref(variable, t1, window):

	klds = []
	# Select t1 points of variable for pdf1
	variable_t1 = variable[:t1]
	mean = np.mean(variable_t1)
	sd = np.std(variable_t1)
	pdf1 = stats.norm.pdf(variable_t1, mean, sd)

	# Calculate KLD between pdf1 and remaining points of the variable, with a rolling window
	variable_t2 = variable[t1:]
	for i in range(t1, len(variable_t2) - window):
		variable_t = variable_t2[i:i+window]
		mean = np.mean(variable_t)
		sd = np.std(variable_t)
		pdf2 = stats.norm.pdf(variable_t, mean, sd)
		klds.append(kl_divergence(pdf1, pdf2))

	return klds

# Monitoring variables from a df with KLD
def test(df,window,t1,nref,limit,npts_anomaly):
	'''
		Monitors a df with variables as columns
		Returns column names of the altered variables

		Parameters:
		window: number points for the rolling window
		t1: number of initial points for reference PDF (pdf1)
		nref: number of KLD points to obtain maximum reference value
		limit: detection limit related to max ref KLD
		npts_anomaly: number of minimum anomaly points to consider an alteration

	'''
	altered_variables = []
	for column in df:
		variable = df[column]
		klds = kld_ref(variable,t1,window)
		xmax,anomalies = control_max(klds,nref,limit)
		if np.sum(anomalies) > npts_anomaly:
			altered_variables.append(column)

	return altered_variables
