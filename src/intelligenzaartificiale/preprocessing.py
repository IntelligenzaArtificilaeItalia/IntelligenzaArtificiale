import numpy as np
import pandas as pd


#make function to label encoding categorical values with scikit-learn
def label_encoding(df, colonna):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    return(le.fit_transform(df[colonna]))

#make function ro label encoding multiple columns of dataframe
def label_encoding_multiplo(df, lista_colonne):
    for colonna in lista_colonne:
        df[colonna] = label_encoding(df, colonna)
    return(df)

#make function to onehot encoding categorical values
def onehot_encoding(df, colonna):
    return(pd.get_dummies(df[colonna]))

#make function to remove duplicates from dataframe
def rimuovi_duplicati(df):
    return(df.drop_duplicates())

#make function to replace missing values with passed value
def sostituisci_nan(df, valore):
    return(df.fillna(valore))

#make function to replace missing values in column with mean value
def sostituisci_nan_media(df, colonna):
    return(df[colonna].fillna(df[colonna].mean()))

#make function to replace missing values in column with most frequent value
def sostituisci_nan_frequenti(df, colonna):
    return(df[colonna].fillna(df[colonna].mode()[0]))

#make function to remove from dataframe rows with missing values
def rimuovi_nan(df):
    return(df.dropna())

#make function to return dataframe without outliers
def rimuovi_outliers(df, colonna):
    return(df[(np.abs(df[colonna] - df[colonna].mean()) <= (3 * df[colonna].std()))])

#make function to return dataframe without outliers and missing values
def rimuovi_outliers_nan(df, colonna):
    return(df[(np.abs(df[colonna] - df[colonna].mean()) <= (3 * df[colonna].std())) & (df[colonna].isnull() == False)])

#make function to normalize dataframe
def normalizza(df):
    return(df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))))

#make function to normalize  specific columns of dataframe
def normalizza_colonne(df, lista_colonne):
    return(df[lista_colonne].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))))

#make function to scaling dataframe
def standardizza(df):
    return(df.apply(lambda x: (x - np.mean(x)) / (np.std(x))))

#make function to scaling  specific columns of dataframe
def standardizza_colonne(df, lista_colonne):
    return(df[lista_colonne].apply(lambda x: (x - np.mean(x)) / (np.std(x))))

#make function to return X_train, X_test, y_train, y_test train test split of dataframe
def dividi_train_test(df, colonna_y, test_size=0.20):
    from sklearn.model_selection import train_test_split
    X = df.drop(colonna_y, axis=1)
    y = df[colonna_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return(X_train, X_test, y_train, y_test)




