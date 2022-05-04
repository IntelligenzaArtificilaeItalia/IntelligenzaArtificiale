
import datatable as dt

#make a function to read .csv with datatable, print time taken to read and return dataframe
def leggi_csv(file_name):
    start = dt.now()
    df = dt.fread(file_name)
    end = dt.now()
    print("Tempo impiegato per leggere il file: ", end-start)
    return df

#make funtion to save dataset as feather file
def salva_feather(df, file_name):
    start = dt.now()
    df.to_feather(file_name + ".feather")
    end = dt.now()
    print("Tempo impiegato per salvare il file: ", end-start)
    return df


