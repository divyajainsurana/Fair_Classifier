import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_data():
    df = pd.read_csv('sqf-2019.csv',sep=",")
    df = df.drop(['SUSPECT_WEIGHT','SUSPECT_HEIGHT','SUPERVISING_OFFICER_RANK','OTHER_PERSON_STOPPED_FLAG','STOP_ID_ANONY','STOP_FRISK_DATE','STOP_FRISK_TIME','YEAR2','MONTH2','DAY2','STOP_WAS_INITIATED','RECORD_STATUS_CODE','ISSUING_OFFICER_RANK','STOP_LOCATION_SECTOR_CODE','ISSUING_OFFICER_COMMAND_CODE','SUPERVISING_OFFICER_COMMAND_CODE','LOCATION_IN_OUT_CODE','JURISDICTION_CODE','JURISDICTION_DESCRIPTION','OBSERVED_DURATION_MINUTES','STOP_DURATION_MINUTES','OFFICER_NOT_EXPLAINED_STOP_DESCRIPTION','SUSPECT_ARREST_OFFENSE','SUMMONS_OFFENSE_DESCRIPTION','ID_CARD_IDENTIFIES_OFFICER_FLAG','SHIELD_IDENTIFIES_OFFICER_FLAG','VERBAL_IDENTIFIES_OFFICER_FLAG','FIREARM_FLAG','KNIFE_CUTTER_FLAG','OTHER_WEAPON_FLAG','PHYSICAL_FORCE_CEW_FLAG','PHYSICAL_FORCE_DRAW_POINT_FIREARM_FLAG','PHYSICAL_FORCE_HANDCUFF_SUSPECT_FLAG','PHYSICAL_FORCE_OC_SPRAY_USED_FLAG','PHYSICAL_FORCE_OTHER_FLAG','PHYSICAL_FORCE_RESTRAINT_USED_FLAG','PHYSICAL_FORCE_VERBAL_INSTRUCTION_FLAG','PHYSICAL_FORCE_WEAPON_IMPACT_FLAG','BACKROUND_CIRCUMSTANCES_VIOLENT_CRIME_FLAG','BACKROUND_CIRCUMSTANCES_SUSPECT_KNOWN_TO_CARRY_WEAPON_FLAG','SUSPECTS_ACTIONS_CASING_FLAG','SUSPECTS_ACTIONS_CONCEALED_POSSESSION_WEAPON_FLAG','SUSPECTS_ACTIONS_DECRIPTION_FLAG','SUSPECTS_ACTIONS_DRUG_TRANSACTIONS_FLAG','SUSPECTS_ACTIONS_IDENTIFY_CRIME_PATTERN_FLAG','SUSPECTS_ACTIONS_LOOKOUT_FLAG','SUSPECTS_ACTIONS_OTHER_FLAG','SUSPECTS_ACTIONS_PROXIMITY_TO_SCENE_FLAG','SEARCH_BASIS_ADMISSION_FLAG','SEARCH_BASIS_CONSENT_FLAG','SEARCH_BASIS_HARD_OBJECT_FLAG','SEARCH_BASIS_INCIDENTAL_TO_ARREST_FLAG','SEARCH_BASIS_OTHER_FLAG','SEARCH_BASIS_OUTLINE_FLAG','DEMEANOR_CODE','SUSPECT_OTHER_DESCRIPTION','STOP_LOCATION_PRECINCT','STOP_LOCATION_APARTMENT','STOP_LOCATION_FULL_ADDRESS','STOP_LOCATION_STREET_NAME','STOP_LOCATION_X','STOP_LOCATION_Y','STOP_LOCATION_ZIP_CODE','STOP_LOCATION_PATROL_BORO_NAME','STOP_LOCATION_BORO_NAME'],axis=1)
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
    #df.SUSPECTED_CRIME_DESCRIPTION = le.fit_transform(df.SUSPECTED_CRIME_DESCRIPTION)
    #df.OFFICER_EXPLAINED_STOP_FLAG = le.fit_transform(df.OFFICER_EXPLAINED_STOP_FLAG)
    #df.OTHER_PERSON_STOPPED_FLAG= le.fit_transform(df.OTHER_PERSON_STOPPED_FLAG)
    df.SUMMONS_ISSUED_FLAG = le.fit_transform(df.SUMMONS_ISSUED_FLAG)
    df.OFFICER_IN_UNIFORM_FLAG= le.fit_transform(df.OFFICER_IN_UNIFORM_FLAG)
    df.FRISKED_FLAG = le.fit_transform(df.FRISKED_FLAG)
    df.SEARCHED_FLAG = le.fit_transform(df.SEARCHED_FLAG)
    df.ASK_FOR_CONSENT_FLG = le.fit_transform(df.ASK_FOR_CONSENT_FLG)
    df = df[df.CONSENT_GIVEN_FLG != '(']
    import numpy as np 
    #df = df[df.DEMEANOR_OF_PERSON_STOPPED != ' ']
    df['DEMEANOR_OF_PERSON_STOPPED'].replace('  ', np.nan, inplace=True)
    df.dropna(subset=['DEMEANOR_OF_PERSON_STOPPED'], inplace=True)
    df = df[df.SUSPECT_REPORTED_AGE != '(null)']
    df = df[df.SUSPECT_RACE_DESCRIPTION != 'BLACK HISPANIC']
    df = df[df.SUSPECT_RACE_DESCRIPTION != 'WHITE HISPANIC']
    df = df[df.SUSPECT_RACE_DESCRIPTION != 'ASIAN / PACIFIC ISLANDER']
    df = df[df.SUSPECT_RACE_DESCRIPTION != 'AMERICAN INDIAN/ALASKAN N']
    df = df[df.SUSPECT_RACE_DESCRIPTION != '(null)']
    df = df[df.CONSENT_GIVEN_FLG != '(']
    df = df[df.ASK_FOR_CONSENT_FLG != '(']
    df = df[df.SUSPECT_BODY_BUILD_TYPE != '(null)']
    df = df[df.SUSPECT_EYE_COLOR != '(null)']
    df = df[df.SUSPECT_HAIR_COLOR != '(null)']

    df.CONSENT_GIVEN_FLG = le.fit_transform(df.CONSENT_GIVEN_FLG)
    df.OTHER_CONTRABAND_FLAG = le.fit_transform(df.OTHER_CONTRABAND_FLAG)
    df.WEAPON_FOUND_FLAG = le.fit_transform(df.WEAPON_FOUND_FLAG)
    df.DEMEANOR_OF_PERSON_STOPPED = le.fit_transform(df.DEMEANOR_OF_PERSON_STOPPED)
    #df.SUSPECT_REPORTED_AGE = le.fit_transform(df.SUSPECT_REPORTED_AGE)
    df.SUSPECT_SEX = le.fit_transform(df.SUSPECT_SEX)
    df.SUSPECT_RACE_DESCRIPTION = le.fit_transform(df.SUSPECT_RACE_DESCRIPTION)
    #df.SUPERVISING_OFFICER_RANK = le.fit_transform(df.SUPERVISING_OFFICER_RANK)
    df.SUSPECTED_CRIME_DESCRIPTION = le.fit_transform(df.SUSPECTED_CRIME_DESCRIPTION)
    df.OFFICER_EXPLAINED_STOP_FLAG = le.fit_transform(df.OFFICER_EXPLAINED_STOP_FLAG)
    #df.OTHER_PERSON_STOPPED_FLAG = le.fit_transform(df.OTHER_PERSON_STOPPED_FLAG)
    df.SUSPECT_ARRESTED_FLAG = le.fit_transform(df.SUSPECT_ARRESTED_FLAG)
    df.SUMMONS_ISSUED_FLAG = le.fit_transform(df.SUMMONS_ISSUED_FLAG)
    df.OFFICER_IN_UNIFORM_FLAG = le.fit_transform(df.OFFICER_IN_UNIFORM_FLAG)
    df.FRISKED_FLAG = le.fit_transform(df.FRISKED_FLAG)
    df.SEARCHED_FLAG = le.fit_transform(df.SEARCHED_FLAG)
    df.ASK_FOR_CONSENT_FLG = le.fit_transform(df.ASK_FOR_CONSENT_FLG)
    df.CONSENT_GIVEN_FLG = le.fit_transform(df.CONSENT_GIVEN_FLG)
    df.OTHER_CONTRABAND_FLAG = le.fit_transform(df.OTHER_CONTRABAND_FLAG)
    df.WEAPON_FOUND_FLAG = le.fit_transform(df.WEAPON_FOUND_FLAG)
    #df.SUSPECT_HEIGHT = le.fit_transform(df.SUSPECT_HEIGHT)
    #df.SUSPECT_WEIGHT = le.fit_transform(df.SUSPECT_WEIGHT)
    df.SUSPECT_BODY_BUILD_TYPE = le.fit_transform(df.SUSPECT_BODY_BUILD_TYPE)
    df.SUSPECT_EYE_COLOR = le.fit_transform(df.SUSPECT_EYE_COLOR)
    df.SUSPECT_HAIR_COLOR = le.fit_transform(df.SUSPECT_HAIR_COLOR)



    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['SUSPECT_ARRESTED_FLAG'],axis=1), 
                                                                df['SUSPECT_ARRESTED_FLAG'], test_size=0.30, 
                                                                random_state=99)
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print (X_train.shape,y_train.shape)
    df.to_csv('nypd_normalized.csv', index = False)
    return (X_train,y_train),(X_test,y_test)

