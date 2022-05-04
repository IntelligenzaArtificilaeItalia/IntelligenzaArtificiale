import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import autoviz as av

#make function to plot of dataframe column
def grafico_colonna(df, colonna):
    plt.hist(df[colonna])
    plt.show()
    return(plt)

#makefuntion to plt scatter plot of dataframe
def grafico_scatter(df, colonna1, colonna2):
    plt.scatter(df[colonna1], df[colonna2])
    plt.show()
    return(plt)

#make function to line plot of two columns in dataframe
def grafico_line(df, colonna1, colonna2):
    plt.plot(df[colonna1], df[colonna2])
    plt.show()
    return(plt)

#make function to plot boxplot of dataframe two columns
def grafico_boxplot(df, colonna1, colonna2):
    plt.boxplot([df[colonna1], df[colonna2]])
    plt.show()
    return(plt)

#make function to plot histogram of two columns in dataframe
def grafico_hist(df, colonna1, colonna2):
    plt.hist(df[colonna1], bins=20, alpha=0.5, label=colonna1)
    plt.hist(df[colonna2], bins=20, alpha=0.5, label=colonna2)
    plt.legend(loc='upper right')
    plt.show()
    return(plt)


#make funtion to use autoviz to plot dataframe
def grafici(df):
    from autoviz.AutoViz_Class import AutoViz_Class
    AV = AutoViz_Class()
    AV.set_data(df)
    AV.plot()
    return(AV)

#make funtion to use autoviz to plot dataframe on depVar
def grafici_target(df, depVar):
    from autoviz.AutoViz_Class import AutoViz_Class
    AV = AutoViz_Class()
    AV.set_data(df)
    AV.set_depVar(depVar)
    AV.plot()
    return(AV)

#make function to create 3d plot of dataframe
def grafico_3d(df, colonna1, colonna2, colonna3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[colonna1], df[colonna2], df[colonna3])
    plt.show()
    return(plt)
