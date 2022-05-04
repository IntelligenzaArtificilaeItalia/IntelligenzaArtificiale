
import numpy as np
import pandas as pd

#make function to return percentage of missing value for each column in dataframe
def percentuale_nan(df):
    return(df.isnull().sum() / len(df) * 100)

#make function to return nan value of dataframe
def valori_nan(df):
    return(df.isnull().sum())

#make function to return statistics of dataframe
def statistiche(df):
    return(df.describe())

#make function to return statistics of dataframe column
def statistiche_colonna(df, colonna):
    return(df[colonna].describe())

#make function to return count of unique values in column in dataframe
def conta_valori_unici(df, colonna):
    return(df[colonna].nunique())

#make function to return mean on column in dataframe
def media(df, colonna):
    return(df[colonna].mean())

#make function to return variance of dataframe
def varianza(df):
    return(df.var())    

#make function to return covariance of dataframe
def covarianza(df):
    return(df.cov())

#make function to return variance of dataframe column
def varianza_colonna(df, colonna):
    return(df[colonna].var())

#make function to return covariance of dataframe column
def covarianza_colonna(df, colonna):
    return(df[colonna].cov())

#make function to return 0.25 quantile of dataframe column
def quantile_25(df, colonna):
    return(df[colonna].quantile(0.25))

#make function to return 0.50 quantile of dataframe column
def quantile_50(df, colonna):
    return(df[colonna].quantile(0.50))  

#make function to return 0.75 quantile of dataframe column
def quantile_75(df, colonna):
    return(df[colonna].quantile(0.75))


#make function to return min value of column in dataframe
def min(df, colonna):
    return(df[colonna].min())

#make function to return max value of column in dataframe
def max(df, colonna):
    return(df[colonna].max())

#make function to return correlation of dataframe
def correlazione(df):
    return(df.corr())

#make function to return correlation matrix of dataframe
def correlazione_matrice(df):
    return(df.corr().round(2))

#make function to return spearman correlation on column in dataframe
def correlazione_spearman(df, colonna, target):
    return(df[colonna].corr(df[target], method='spearman'))

#make function to return pearson correlation on column in dataframe
def correlazione_pearson(df, colonna, target):
    return(df[colonna].corr(df[target], method='pearson'))

#make function to return radio correlation on column in dataframe
def correlazione_radio(df, colonna, target):
    return(df[colonna].corr(df[target]))

#make function to return list of column with best correlation with passed column
def classifica_correlazione_colonna(df, colonna):
    return(df.corr()[colonna].sort_values(ascending=False))


#make function to open dtale on dataframe
def apri_dataframe_nel_browser(df):
    import dtale
    d = dtale.show(df)
    d.open_browser()

#make function to return pandas-profiling report on dataframe
def report_dataset(df):
    import pandas_profiling
    profile = pandas_profiling.ProfileReport(df, title="Report IntelligenzaArtificialeItalia.net", explorative=True)
    profile.to_file('profile_report_pandas.html')
    print("Report salvato in questa directory profile_report_pandas.html ")
    return(profile)

