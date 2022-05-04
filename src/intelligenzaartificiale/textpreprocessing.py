import pandas as pd
import texthero as hero

#make function to clean text column of dataframe with texthero
def pulisci_testo(df, colonna):
    return(hero.clean(df[colonna]))

#make function to lower text column of dataframe with texthero
def trasforma_in_minuscolo(df, colonna):
    return(df[colonna].apply(lambda x: x.lower()))

#make function to remove punctuation from text column of dataframe with texthero
def rimuovi_caratteri_speciali(df, colonna):
    return(df[colonna].apply(lambda x: x.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))))

#make function to remove punctuation and digits from text column of dataframe with texthero
def rimuovi_caratteri_speciali_e_cifre(df, colonna):
    return(df[colonna].apply(lambda x: x.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'))))

#make function to remove stopwords from text column of specific language
def rimuovi_stopwords(df, colonna, lingua):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words(lingua))
    return(df[colonna].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words])))

#make function to return tokenized text column of dataframe with texthero
def tokenizza_testo(df, colonna):
    return(hero.tokenize(df[colonna]))

#make function to vectorize text column of dataframe with texthero
def vettorizza_testo(df, colonna):
    return(hero.tfidf(df[colonna]))

#make function to add principal components from text column of dataframe with texthero
def componenti_principali(df, colonna):
    return(hero.pca(df[colonna]))

#make function to return bag of words from text column of dataframe with sklearn
def bag_of_words(df, colonna):
    from sklearn.feature_extraction.text import CountVectorizer
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(df[colonna])
    bag_of_words = count_vectorizer.transform(df[colonna])
    return(bag_of_words)

#make function to plot wordcloud from text column of dataframe
def crea_wordcloud(df, colonna):
    from wordcloud import WordCloud 
    import matplotlib.pyplot as plt
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='white').generate(str(df[colonna]))
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()