from pathlib import Path
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Cargar datos usando pandas
df = pd.read_csv('loan_data_set.csv')
df.tail(5)
# shape de dataframe
df.shape
df['Credit_History'].shape
df[['Credit_History']].shape
#propiedades
df.info()
df.describe()
df.describe(exclude=np.number)
### missing info y solving
df.isnull().sum()
#‘bfill’, ‘pad’, ‘ffill’, None, sin method: 0
df.fillna(method='bfill',inplace=True)
#formato de columnas
df.isnull().sum()
df['Credit_History'].unique()
df.info()
df['Credit_History']=df['Credit_History'].astype('category')
df['ApplicantIncome']=df['ApplicantIncome'].astype('int32')
# graficos boxplot
df['LoanAmount'].plot(kind='box')
df['sqrtCoapplicantIncome'] = np.sqrt(df['CoapplicantIncome'])
df.head()
# grafico de distribuciones de la variable
plt.plot(figsize=(15,5))
sns.displot(df['CoapplicantIncome'], label='CoapplicantIncome')
sns.displot(df['sqrtCoapplicantIncome'],label='sqrtCoapplicationIncome')
df.CoapplicantIncome.skew()
df.sqrtCoapplicantIncome.skew()

#Normalizacion

## Normalizacion via Z-score( (x- mean)/std)
mean_loan=df['LoanAmount'].mean()
std_loan=df['LoanAmount'].std()

df['zscoreloanamount']=(df['LoanAmount']-mean_loan)/std_loan
df.head()
from sklearn.preprocessing import StandardScaler

SS=StandardScaler()
#### con scikitlearn=>  X son matrices [[X]]; y son columnas [y]
scale_loan=SS.fit_transform(df[['LoanAmount']])

### Normalizacion via (x -min)/(max -min)
min_loan=df['LoanAmount'].min()
max_loan=df['LoanAmount'].max()
df['minmaxLoanAmount']=(df['LoanAmount']-min_loan)/(max_loan-min_loan)

from sklearn.preprocessing import MinMaxScaler

MS=MinMaxScaler()

minmaxloan=MS.fit_transform(df[['LoanAmount']])

## one hot encoder

df['Property_Area'].unique()
# # urban=0  ; rural=1; semiurban=2
# # urban_ind=0  ; rural_ind=1; semiurban_ind=2
# urban              1 0 0    
# urban              1 0 0
# rural              0 1 0
# semiurban          0 0 1
# rural              0 1 0
        
# # urban_ind=0  ; rural_ind=1
# urban              1 0     
# urban              1 0 
# rural              0 1 
# semiurban          0 0 
# rural              0 1 

Property_Area_1hot = pd.get_dummies(df['Property_Area'], drop_first=True)
 
##scikitlearn

from sklearn.preprocessing import OneHotEncoder

onehot=OneHotEncoder(drop='first')

Onehot_PA= onehot.fit(df[['Property_Area','Gender']])

Onehot_PA.categories_
Onehot_PA.drop_idx_
Onehot_PA.n_features_in_
Onehot_PA.feature_names_in_

#label encoding
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()

le_propertyA= LE.fit(df['Property_Area'])
le_propertyA.classes_
le_propertyA.transform(['Rural','Rural', 'Semiurban','Rural','Rural', 'Urban'])
le_propertyA.inverse_transform([0,0,1,0,0,2])

#tranformaciones features pd

df['LoanAmountcross']= df['LoanAmount']*df['Loan_Amount_Term']

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

def extract_year(x):
    y= x['date']
    year=y.year
    return year
df['year']= df.apply(lambda x : extract_year(x),axis=1)

## split de datos
from sklearn.model_selection import train_test_split
df_test=df[["LoanAmount","ApplicantIncome","CoapplicantIncome"]]
df_test.head()

Y=df_test['LoanAmount']
X=df_test.drop("LoanAmount", axis=1)
X.head()
Y.head()

X_train_h,X_test, Y_train_h, Y_test = train_test_split(X,Y,test_size=0.15,random_state=123)
X_train,X_val, Y_train, Y_val = train_test_split(X_train_h,Y_train_h,test_size=0.15,random_state=123)

##histogramas

sns.histplot(X_train.CoapplicantIncome,kde=True)

g=sns.FacetGrid(df, col="Property_Area", row="Gender")
g.map(sns.histplot,"LoanAmount")
plt.show()

sns.jointplot(data=df, x="ApplicantIncome", y="LoanAmount" , hue='Gender', color='b')
plt.show()

sns.jointplot(data=df, x="ApplicantIncome", y="LoanAmount" , kind='scatter', color='b')
plt.show()

sns.lmplot(x="ApplicantIncome", y="LoanAmount",hue='Property_Area',data=df)
plt.show()

corr_mat=np.corrcoef(df_test,rowvar=False)
corr_mat.shape
df_test.head()
corr_df=pd.DataFrame(corr_mat,columns=df_test.columns,index=df_test.columns)
sns.heatmap(corr_df,linewidths=1,cmap='plasma', fmt=".2f")

sns.pairplot(data=df_test,corner=True)

g= sns.PairGrid(df_test, corner=True)
g.map_lower(sns.kdeplot,hue=None, levels=5)
g.map_lower(sns.scatterplot,marker="+")
g.map_diag(sns.histplot, linewidth=0.1,kde=True)
