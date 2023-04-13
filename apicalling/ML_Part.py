import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from fastapi import FastAPI
from fastapi import FastAPI
import uvicorn


df1 = pd.read_csv("https://www.dropbox.com/s/0d0oiufbyjz13x5/BurialMain1.csv?dl=1", on_bad_lines="skip")
df_main = df1[(((df1["headdirection"]=="W") | (df1["headdirection"]=="E") | (df1["headdirection"]=="N LL") | (df1["headdirection"]=="I")) & (df1['southtohead'] != 'U') & (df1['length'] != 'N LL'))]
df_main['ageatdeath'] = df_main['ageatdeath'].replace({'In':'IN'}, regex=True)

df_main['southtohead'] = pd.to_numeric(df_main['southtohead'])
df_main['depth'] = pd.to_numeric(df_main['depth'])
df_main['westtohead'] = pd.to_numeric(df_main['westtohead'])
df_main['length'] = pd.to_numeric(df_main['length'])
df_main['westtofeet'] = pd.to_numeric(df_main['westtofeet'])
df_main['southtofeet'] = pd.to_numeric(df_main['southtofeet'])

df_main.drop(columns=['text', 'id', 'sex', 'adultsubadult', 'goods', 'facebundles','dateofexcavation', 'burialmaterials', 'hair', 'photos', 'excavationrecorder', 'fieldbookexcavationyear', 'clusternumber', 'dataexpertinitials', 'burialid', 'samplescollected', 'area', 'squareeastwest', 'fieldbookpage', 'eastwest','burialnumber', 'northsouth', 'squarenorthsouth', 'shaftnumber','preservation'],inplace=True)

df_main = df_main.reset_index(drop=True)



# convert categorical columns to category type
df_main['headdirection'] = df_main['headdirection'].astype('category')
df_main['wrapping'] = df_main['wrapping'].astype('category')
df_main['ageatdeath'] = df_main['ageatdeath'].astype('category')
df_main['haircolor'] = df_main['haircolor'].astype('category')
# df_main['Robust'] = df_main['Robust'].astype('category')
# df_main['SupraorbitalRidges'] = df_main['SupraorbitalRidges'].astype('category')
# df_main['OrbitEdge'] = df_main['OrbitEdge'].astype('category')
# df_main['ParietalBossing'] = df_main['ParietalBossing'].astype('category')
# df_main['Gonion'] = df_main['Gonion'].astype('category')
# df_main['NuchalCrest'] = df_main['NuchalCrest'].astype('category')
# df_main['ZygomaticCrest'] = df_main['ZygomaticCrest'].astype('category')
# df_main['SphenooccipitalSynchrondrosis'] = df_main['SphenooccipitalSynchrondrosis'].astype('category')
# df_main['LamboidSuture'] = df_main['LamboidSuture'].astype('category')
# df_main['ToothAttrition'] = df_main['ToothAttrition'].astype('category')
# df_main['SubpubicAngle'] = df_main['SubpubicAngle'].astype('category')
# df_main['SciaticNotch'] = df_main['SciaticNotch'].astype('category')
# df_main['PubicBone'] = df_main['PubicBone'].astype('category')
# df_main['Medial_IP_Ramus'] = df_main['Medial_IP_Ramus'].astype('category')
# df_main['Femur'] = df_main['Femur'].astype('category')
# df_main['Humerus'] = df_main['Humerus'].astype('category')
# df_main['Caries_Periodontal_Disease'] = df_main['Caries_Periodontal_Disease'].astype('category')

# drop columns with object dtype
df_main = df_main.select_dtypes(exclude=['object'])

# split data into X (features) and y (target)
X = df_main.drop('headdirection' , axis=1)
y = df_main['headdirection']

# convert categorical values to binary values
y = pd.get_dummies(y, dtype='uint8', drop_first=True)
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14523)

# set parameters
params = {
    'objective': 'multi:softprob', # use softmax objective for multi-class classification
    'num_class': 2, # number of classes in the target variable
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    'seed': 14523
}

# create DMatrix for training/testing data
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=['wrapping', 'haircolor', 'ageatdeath'])
dtest = xgb.DMatrix(X_test, enable_categorical=['wrapping', 'haircolor', 'ageatdeath'])

# train model
model = xgb.train(params, dtrain)

# make predictions on test data
y_pred_proba = model.predict(dtest)

# convert probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

reg = model

with open('./model.pkl','wb') as file:
  pickle.dump(reg,file)
