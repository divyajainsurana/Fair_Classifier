import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle

def loaddata():
    df = pd.read_csv('./german-credit-data.csv',sep=';')
    df = shuffle(df)
    x = df.ix[:,1:10]
    y = df.ix[:,-1]
    x = x.replace(['male', 'female'], [0, 1])
    x = x.replace(['free', 'rent', 'own'], [0, 1, 2])
    x = x.replace(['little', 'moderate', 'quite rich', 'rich'], [1, 2, 3, 4])
    x = x.replace(
        ['radio/TV', 'repairs', 'domestic appliances', 'vacation/others', 'furniture/equipment', 'car', 'education',
         'business'], [1, 2, 3, 4, 5, 6, 7, 8])
    x = x.fillna(0)
    y = y.replace(['good','bad'],[0,1])
    x = preprocessing.scale(x)
    #print (x,y)
    xtrain, xtext, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    return (xtrain,ytrain),(xtext,ytest)
