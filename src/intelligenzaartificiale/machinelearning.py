import numpy as np
import pandas as pd

#make function to return best sklearn regressor model for given dataframe 
def performance_modelli_regressione(df, lista_colonne_x, colonna_y):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import cross_val_score
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    regressor = [
        LinearRegression(),
        LogisticRegression(),
        SVR(),
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        ElasticNet(),
        Lasso(),
        Ridge(),
        DecisionTreeRegressor(),
        KNeighborsRegressor()
    ]
    scores = []
    for reg in regressor:
        scores.append(np.mean(cross_val_score(reg, X, Y, cv=5)))
    #add figure size 
    plt.figure(figsize=(30,10))
    plt.bar(range(len(regressor)), scores)
    plt.xticks(range(len(regressor)), [reg.__class__.__name__ for reg in regressor])
    plt.ylabel("Media CV Score")
    plt.title("Comparazione Algoritmi di Regressione")
    plt.show()
    return regressor[np.argmax(scores)]

#make function to return best sklearn classification model for given dataframe 
def performance_modelli_classificazione(df, lista_colonne_x, colonna_y):    
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    regressor = [
        LogisticRegression(),
        SVC(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier()
    ]
    scores = []
    for reg in regressor:
        scores.append(np.mean(cross_val_score(reg, X, Y, cv=5)))
    #add figure size 
    plt.figure(figsize=(30,10))
    plt.bar(range(len(regressor)), scores)
    plt.xticks(range(len(regressor)), [reg.__class__.__name__ for reg in regressor])
    plt.ylabel("Media CV Score")
    plt.title("Comparazione Algoritmi di Classificazione")
    plt.show()
    return regressor[np.argmax(scores)]   


#make funtion to return linear regression on multiple x columns
def regressione_lineare(df, lista_colonne_x, colonna_y):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    Y = df[colonna_y]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = Y.values.reshape(-1,1)
    reg = LinearRegression().fit(X, Y)
    return reg

##make funtion to return linear regression on multiple x columns
##input : dataframe, list of x columns, y column and fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False, random_state=None
def regressione_lineare_avanzata(df, lista_colonne_x, colonna_y, fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False, random_state=None):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    Y = df[colonna_y]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = Y.values.reshape(-1,1)
    reg = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs, positive=positive, random_state=random_state).fit(X, Y)
    return reg

#make funtion to return logistic regression on multiple x columns
def regressione_logistica(df, lista_colonne_x, colonna_y):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = LogisticRegression().fit(X, Y)
    return reg

##make funtion to return logistic regression on multiple x columns
##input : dataframe, list of x columns, y column and fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, C=1.0, class_weight=None, max_iter=100, multi_class='ovr', penalty='l2', random_state=None, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False
def regressione_logistica_avanzata(df, lista_colonne_x, colonna_y, fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, C=1.0, class_weight=None, max_iter=100, multi_class='ovr', penalty='l2', random_state=None, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = LogisticRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs, C=C, class_weight=class_weight, max_iter=max_iter, multi_class=multi_class, penalty=penalty, random_state=random_state, solver=solver, tol=tol, verbose=verbose, warm_start=warm_start).fit(X, Y)
    return reg

#make funtion to return SVR model on multiple x columns
def regressione_SVR(df, lista_colonne_x, colonna_y):
    from sklearn.svm import SVR
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = SVR().fit(X, Y)
    return reg

##make funtion to return SVR model on multiple x columns
##input : dataframe, list of x columns, y column and fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, C=1.0, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False
def regressione_SVR_avanzata(df, lista_colonne_x, colonna_y, fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, C=1.0, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False):
    from sklearn.svm import SVR
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = SVR(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs, C=C, epsilon=epsilon, gamma=gamma, kernel=kernel, max_iter=max_iter, shrinking=shrinking, tol=tol, verbose=verbose).fit(X, Y)
    return reg

#make funtion to return support vector machine model on multiple x columns
def regressione_SVC(df, lista_colonne_x, colonna_y):
    from sklearn.svm import SVC
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = SVC().fit(X, Y)
    return reg

##make funtion to return support vector machine model on multiple x columns
##input : dataframe, list of x columns, y column and fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False
def regressione_SVC_avanzata(df, lista_colonne_x, colonna_y, fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False):
    from sklearn.svm import SVC
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = SVC(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs, C=C, cache_size=cache_size, class_weight=class_weight, coef0=coef0, decision_function_shape=decision_function_shape, degree=degree, gamma=gamma, kernel=kernel, max_iter=max_iter, probability=probability, random_state=random_state, shrinking=shrinking, tol=tol, verbose=verbose).fit(X, Y)
    return reg

#make funtion to return random forest model on multiple x columns
def regressione_random_forest(df, lista_colonne_x, colonna_y):
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = RandomForestRegressor().fit(X, Y)
    return reg

##make funtion to return random forest model on multiple x columns
##input : dataframe, list of x columns, y column and n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False
def regressione_random_forest_avanzata(df, lista_colonne_x, colonna_y, n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start).fit(X, Y)
    return reg


#make function to return logistic classifier model on multiple x columns
def classificatore_logistico(df, lista_colonne_x, colonna_y):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = LogisticRegression().fit(X, Y)
    return reg

#make function to return logistic classifier model on multiple x columns with advanced parameters
def classificatore_logistico_avanzato(df, lista_colonne_x, colonna_y, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=None):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs).fit(X, Y)
    return reg

#make function to return naive bayes model on multiple x columns
def classificatore_naivebayes(df, lista_colonne_x, colonna_y):
    from sklearn.naive_bayes import GaussianNB
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = GaussianNB().fit(X, Y)
    return reg

#make function to return naive bayes model on multiple x columns with advanced parameters
def classificatore_naivebayes_avanzato(df, lista_colonne_x, colonna_y, priors=None):
    from sklearn.naive_bayes import GaussianNB
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = GaussianNB(priors=priors).fit(X, Y)
    return reg

#make function to return SVM model on multiple x columns
def classificatore_svm(df, lista_colonne_x, colonna_y):
    from sklearn.svm import SVC
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = SVC().fit(X, Y)
    return reg

#make function to return SVM model on multiple x columns with advanced parameters
def classificatore_svm_avanzato(df, lista_colonne_x, colonna_y, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None):
    from sklearn.svm import SVC
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, random_state=random_state).fit(X, Y)
    return reg

#make funtion to return random forest classifier model on multiple x columns
def classificatore_random_forest(df, lista_colonne_x, colonna_y):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = RandomForestClassifier().fit(X, Y)
    return reg

##make funtion to return random forest classifier model on multiple x columns
##input : dataframe, list of x columns, y column and n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False
def classificatore_random_forest_avanzato(df, lista_colonne_x, colonna_y, n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start).fit(X, Y)
    return reg

#make funtion to return gradient boosting model on multiple x columns
def regressione_gradient_boosting(df, lista_colonne_x, colonna_y):
    from sklearn.ensemble import GradientBoostingRegressor
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = GradientBoostingRegressor().fit(X, Y)
    return reg

##make funtion to return gradient boosting model on multiple x columns
##input : dataframe, list of x columns, y column and n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False
def regressione_gradient_boosting_avanzato(df, lista_colonne_x, colonna_y, n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
    from sklearn.ensemble import GradientBoostingRegressor
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = GradientBoostingRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start).fit(X, Y)
    return reg

#make funtion to return gradient boosting classifier model on multiple x columns
def classificatore_gradient_boosting(df, lista_colonne_x, colonna_y):
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = GradientBoostingClassifier().fit(X, Y)
    return reg

##make funtion to return gradient boosting classifier model on multiple x columns
##input : dataframe, list of x columns, y column and n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False
def classificatore_gradient_boosting_avanzato(df, lista_colonne_x, colonna_y, n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = GradientBoostingClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start).fit(X, Y)
    return reg

#make funtion to return decision tree model on multiple x columns
def regressione_decision_tree(df, lista_colonne_x, colonna_y):
    from sklearn.tree import DecisionTreeRegressor
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = DecisionTreeRegressor().fit(X, Y)
    return reg

#make funtion to return decision tree model on multiple x columns
##input : dataframe, list of x columns, y column and n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False
def regressione_decision_tree_avanzato(df, lista_colonne_x, colonna_y, n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
    from sklearn.tree import DecisionTreeRegressor
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = DecisionTreeRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start).fit(X, Y)
    return reg

#make funtion to return decision tree classifier model on multiple x columns
def classificatore_decision_tree(df, lista_colonne_x, colonna_y):
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = DecisionTreeClassifier().fit(X, Y)
    return reg

#make funtion to return decision tree classifier model on multiple x columns
##input : dataframe, list of x columns, y column and n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False
def classificatore_decision_tree_avanzato(df, lista_colonne_x, colonna_y, n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = DecisionTreeClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start).fit(X, Y)
    return reg

#make function to return KNN model on multiple x columns
def regressione_knn(df, lista_colonne_x, colonna_y):
    from sklearn.neighbors import KNeighborsRegressor
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = KNeighborsRegressor().fit(X, Y)
    return reg

#make funtion to return KNN model on multiple x columns
##input : dataframe, list of x columns, y column and n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs
def regressione_knn_avanzato(df, lista_colonne_x, colonna_y, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs):
    from sklearn.neighbors import KNeighborsRegressor
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs, **kwargs).fit(X, Y)
    return reg

#make function to return KNN classifier model on multiple x columns
def classificatore_knn(df, lista_colonne_x, colonna_y):
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = KNeighborsClassifier().fit(X, Y)
    return reg

#make funtion to return KNN classifier model on multiple x columns
##input : dataframe, list of x columns, y column and n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs
def classificatore_knn_avanzato(df, lista_colonne_x, colonna_y, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs):
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs, **kwargs).fit(X, Y)
    return reg


#make funtion to return elastic net classifier model on multiple x columns
def modello_elastic_net(df, lista_colonne_x, colonna_y):
    from sklearn.linear_model import ElasticNet
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = ElasticNet().fit(X, Y)
    return reg

#make funtion to return elastic net classifier model on multiple x columns
##input : dataframe, list of x columns, y column and alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute='auto', max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'
def modello_elastic_net_avanzato(df, lista_colonne_x, colonna_y, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute='auto', max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
    from sklearn.linear_model import ElasticNet
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, normalize=normalize, precompute=precompute, max_iter=max_iter, copy_X=copy_X, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection).fit(X, Y)
    return reg

#make funtion to return Lasso model on multiple x columns
def modello_lasso(df, lista_colonne_x, colonna_y):
    from sklearn.linear_model import Lasso
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = Lasso().fit(X, Y)
    return reg

#make funtion to return Lasso model on multiple x columns
##input : dataframe, list of x columns, y column and alpha=1.0, fit_intercept=True, normalize=False, precompute='auto', copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'
def modello_lasso_avanzato(df, lista_colonne_x, colonna_y, alpha=1.0, fit_intercept=True, normalize=False, precompute='auto', copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
    from sklearn.linear_model import Lasso
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = Lasso(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, precompute=precompute, copy_X=copy_X, max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection).fit(X, Y)
    return reg


#make funtion to return Ridge model on multiple x columns
def modello_ridge(df, lista_colonne_x, colonna_y):
    from sklearn.linear_model import Ridge
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = Ridge().fit(X, Y)
    return reg

#make funtion to return Ridge model on multiple x columns
##input : dataframe, list of x columns, y column and alpha=1.0, fit_intercept=True, normalize=False, precompute='auto', copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'
def modello_ridge_avanzato(df, lista_colonne_x, colonna_y, alpha=1.0, fit_intercept=True, normalize=False, precompute='auto', copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
    from sklearn.linear_model import Ridge
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    reg = Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, precompute=precompute, copy_X=copy_X, max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection).fit(X, Y)
    return reg

######################################################################################################


#make function to return accuracy score of model on multiple x columns
def valutazione_modello(reg, df, lista_colonne_x, colonna_y):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]
    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    return reg.score(X, Y)

#make funtion to predict y values from x colums values
def predizione_y(reg, df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df
    X = X.values.reshape(-1,len(df.columns))
    Y = reg.predict(X)
    return Y

#make funtion to save model
def salva_modello(reg, nome_modello):
    import pickle
    pickle.dump(reg, open(nome_modello, 'wb'))

#make funtion to load model
def carica_modello(nome_modello):
    import pickle
    reg = pickle.load(open(nome_modello, 'rb'))
    return reg

#make function to Explain Any Machine Learning Model with Shap
##plot summary of model with shap
def spiega_modello(reg, df, lista_colonne_x, colonna_y):
    import shap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    df = df.dropna()
    X = df[lista_colonne_x]

    for i in range(len(lista_colonne_x)):
        print("Feature ", i, " : ", lista_colonne_x[i])

    X = X.values.reshape(-1,len(lista_colonne_x))
    Y = df[colonna_y]
    Y = Y.values.reshape(-1,1)
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    shap.summary_plot(shap_values, X, plot_type='bar')

