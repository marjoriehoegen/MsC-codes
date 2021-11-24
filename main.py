# Python libraries
import numpy as np
import pandas as pd
import os

# Files with techniques code 
import anomaly_pdf
import kld
import pca
import ssa
import emd_energy
import autoencoder

# Preprocessing functions
def preprocessing(df):
	df = df.apply(lambda x: x.str.replace(',','.'))
	df[df.columns[1:]] =  df[df.columns[1:]].astype(float)
	return df

def drop_first_column(df):
	df = df.drop(df.columns[0],axis=1)
	return df

##############################################################################
# Tests with synthetic data
##############################################################################

# Directory definition
directory = 'data/'
file_names = []

# Read all .txt files in a directory
for file in os.listdir(directory):
	if (file.endswith(".txt")) and (file not in file_names):
		# Read data
		filepath = directory + os.sep + file
		file_names.append(file)

		df = pd.read_csv(filepath,encoding = "ISO-8859-1",sep='\t')
		df = preprocessing(df)
		df = drop_first_column(df)

		# Detection of alteration with PDFs
		h_pdf = anomaly_pdf.test(df,t1=100,limit=0,npts_anomaly=100)
		# Detection of alteration with KLDs
		h_kld = kld.test(df,window=100,t1=100,nref=500,limit=3,npts_anomaly=500)
		# Detection of alteration with PCA
		h_SPE, h_T2 = pca.test(
			df,t1=100,n_pcs=2,n_variables=135,limit_contrSPE=1000,limit_contrT2=50,
			npts_anomaly=100)
		# Detection of alteration with SSA
		h_D1, h_D2 = ssa.test(
			df,L=50,N_base=100,N_teste=100,limite_D1=0.1,limite_D2=200,t1=100,
			nptos_anomalia=100)
		# Detection of alteration with energy analysis
		h_energy = emd_energy.test(df,L=100,t1=100,limit=0.1,npts_anomaly=100)
		# # Detection of alteration with autoencoder
		h_autoencoder = autoencoder.test_all(df,t=500,limit=10,npts_anomaly=100)
