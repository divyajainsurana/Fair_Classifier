import pandas as pd
from collections import Counter
def load_data():
    filename = 'adult-all.csv'
    # load the csv file as a data frame
    dataframe = pd.read_csv(filename, header=None, na_values='?')
    dataframe = dataframe.dropna()
    print(dataframe.shape)
    # summarize the class distribution
    #target = dataframe.values[:,-1]
    dict_map = dict()
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    le = LabelEncoder()
    last_ix = len(dataframe.columns) - 1

    #y = LabelEncoder().fit_transform(y)
    dataframe[0] = LabelEncoder().fit_transform(dataframe[0])
    dataframe[1] = LabelEncoder().fit_transform(dataframe[1])
    dataframe[2] = LabelEncoder().fit_transform(dataframe[2])
    dataframe[3] = LabelEncoder().fit_transform(dataframe[3])
    dataframe[4] = LabelEncoder().fit_transform(dataframe[4])
    dataframe[5] = LabelEncoder().fit_transform(dataframe[5])
    dataframe[6] = LabelEncoder().fit_transform(dataframe[6])
    dataframe[7] = LabelEncoder().fit_transform(dataframe[7])
    dataframe[8] = LabelEncoder().fit_transform(dataframe[8])
    dataframe[9] = LabelEncoder().fit_transform(dataframe[9])
    dataframe[10] = LabelEncoder().fit_transform(dataframe[10])
    dataframe[11] = LabelEncoder().fit_transform(dataframe[11])
    dataframe[12] = LabelEncoder().fit_transform(dataframe[12])
    dataframe[13] = LabelEncoder().fit_transform(dataframe[13])
    dataframe[14] = LabelEncoder().fit_transform(dataframe[14])
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]

   # print (X)
   # print (y)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, 
                                                                random_state=42)
    from sklearn.preprocessing import StandardScaler                                                          
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return (X_train,y_train),(X_test,y_test)
#    print (X_train,y_train.shape)                                                         
#    dataframe.to_csv('adult_normalized.csv', index = False)




