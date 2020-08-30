import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_data():
    df = pd.read_csv('compas-scores-two-years.csv',sep=",")
    df = df.drop(['v_type_of_assessment','start','end','juv_fel_count','c_charge_desc','days_b_screening_arrest','id','name','first','last','compas_screening_date','dob','c_jail_in','c_jail_out','c_case_number','c_offense_date','event','c_arrest_date','c_days_from_compas','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out','violent_recid','vr_case_number','vr_charge_degree','vr_offense_date','vr_charge_desc','screening_date','v_screening_date','in_custody','out_custody'],axis=1)
        #df = pd.get_dummies(df,columns=['housing',"default","contact","loan","education"],drop_first=True)
    """
        def function_2 (row_1):
            if(row_1['age']<25 or row_1['age']>60):
                return -1;
            return 1;
        df['age']= df.apply(lambda row_1: function_2(row_1),axis=1)
    """
    dict_map = dict()
        #y_map = {'yes':1,'no':0}
        #dict_map['y'] = y_map
        #df = df.replace(dict_map)
        #label = df['y']
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    le = LabelEncoder()
    df.sex = le.fit_transform(df.sex)
    #df.race = le.fit_transform(df.race)
    #print (set(df.race))
    df = df[df.race != 'Other']
    df = df[df.race != 'Hispanic']
    df = df[df.race != 'Native American']
    dic = {'Asian':'Caucasian','Caucasian':'Caucasian','African-American':'African-American'}
    df['race']=df['race'].map(dic)
    #print (set(df.race))
    df.race = le.fit_transform(df[['race']])
    #print ((df['race']))
    #onehot_encoder = OneHotEncoder(sparse=False)
#df.race = pd.Series(df.race)
#df.race = df.race.values.reshape(7214, 1)
    #df.race = onehot_encoder.fit_transform(df[['race']])
#print (df.race)
    df.c_charge_degree = le.fit_transform(df.c_charge_degree)
    df.age_cat= le.fit_transform(df.age_cat)
    df.type_of_assessment = le.fit_transform(df.type_of_assessment)
    df.v_score_text= le.fit_transform(df.v_score_text)
    df.score_text = le.fit_transform(df.score_text)
    #df.v_type_of_assessment = le.fit_transform(df.v_type_of_assessment)
    #df.month = le.fit_transform(df.month)
    #df.day_of_week = le.fit_transform(df.day_of_week)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['two_year_recid'],axis=1), 
                                                            df['two_year_recid'], test_size=0.30, 
                                                            random_state=101)
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print (X_test.shape)
    df.to_csv('compas_normalized.csv', index = False)
    return (X_train,y_train),(X_test,y_test)
