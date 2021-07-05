import numpy as np
import scipy.stats as stats

# Gaussian PDF
def prob(x, t1):

	# Separate x according to t1
	x_bt1 = x[:t1] # x before t1
	x_at1 = x[t1:] # x after t1
	
	# Calculate mean and variance of x before t1
	mean = np.mean(x_bt1)
	var = np.var(x_bt1)

	# Calculate probability density of x after t1
	density = []
	for x in x_at1:
		p_x = (1/(np.sqrt(2*np.pi) * np.sqrt(var))) * np.exp(-(((x-mean)**2)/(2*var)))
		density.append(p_x)

	return density

# Monitoring variables from a df with PDFs
def test(df, t1, limit, npts_anomaly):
	'''
		Monitors a df with variables as columns
		Returns column names of the altered variables

		Parameters:
		t1: number of initial points for reference PDF
		limit: detection limit
		npts_anomaly: number of minimum anomaly points to consider an alteration

	'''
	altered_variables = []
	for column in df:
		variable = df[column]
		density = prob(variable,t1)

		anomalies = []
		densities = np.array(density)
		anomalies = densities <= limit
		if np.sum(anomalies) > npts_anomaly:
			altered_variables.append(column)
	
	return altered_variables