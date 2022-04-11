import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Churn_Modelling.csv')
df.head(5)
df.sample(5)
print(df.shape)
# removing unnecessary columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)


# unique values in all columns
def print_unique_df(dataframe):
    for col in dataframe: print(dataframe[col].unique())


print_unique_df(df)

# from the above result, Gender and Geography are the only columns with strings and they need to be label encoded.
df.Gender = [1 if gen == 'Male' else 0 for gen in df.Gender]
# or df.Gender.replace({'Female':0, 'Male':1},inplace = True)
df.head(10)
# one hot encoding for Geography
df1 = pd.get_dummies(data=df, columns=['Geography'])

# scaling so that all the values fall in the range [0,1]
cols_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

scaler = MinMaxScaler()
df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])
# all unique values are in the range [0,1]
print_unique_df(df1)

# preprocessing and cleaning are done
X = df1.drop('Exited', axis=1)
y = df1.Exited

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))


# print(model.predict(X_test))

a = [0.302, 1, 0.351351, 0.4, 0.566170, 0.333333, 0, 1, 0.374680, 1, 0, 0]
test_array = [[0.516	,0	,0.310811	,0.1	,0.334031	,0.000000	,0	,1	,0.562709	,0	,0	,1]
	,[0.304	,0	,0.324324	,0.8	,0.636357	,0.666667	,1	,0	,0.569654	,1	,0	,0]
	,[0.698	,0	,0.283784	,0.1	,0.000000	,0.333333	,0	,0	,0.469120	,1	,0	,0]
	,[1.000	,0	,0.337838	,0.2	,0.500246	,0.000000	,1	,1	,0.395400	,0	,0	,1]
	,[0.590	,1	,0.351351	,0.8	,0.453394	,0.333333	,1	,0	,0.748797	,0	,0	,1]
	,[0.944	,1	,0.432432	,0.7	,0.000000	,0.333333	,1	,1	,0.050261	,1	,0	,0]
	,[0.052	,0	,0.148649	,0.4	,0.458540	,1.000000	,1	,0	,0.596733	,0	,1	,0]
	,[0.302	,1	,0.351351	,0.4	,0.566170	,0.333333	,0	,1	,0.374680	,1	,0	,0]
	,[0.668	,1	,0.121622	,0.2	,0.536488	,0.000000	,1	,1	,0.358605	,1	,0	,0]]

y_pred = model.predict(X_test)
cm = metrics.confusion_matrix(y_test,y_pred)
print(cm)
print(metrics.accuracy_score(y_test,y_pred))

# print(model.predict(test_array))