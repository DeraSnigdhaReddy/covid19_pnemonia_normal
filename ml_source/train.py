import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

df=pd.read_csv("D:\demo-covid\Data\pneumonia_covid_diagnosis_dataset.csv")

columns=["Gender","Fever","Cough","Fatigue","Breathlessness","Comorbidity","Type","Stage"]

for col in columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])

df=df.drop("Is_Curable",axis=1)
x=df.drop(columns=["Survival_Rate"],axis=1)
y=df['Survival_Rate']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=RandomForestRegressor()
model.fit(x_train,y_train)

prd=model.predict(x_test)
print(prd[0],y_test[0])

joblib.dump(model,"covid_diag.pkl")