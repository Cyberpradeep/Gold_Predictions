from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib

def train_model(df):
    # prepare features and target
    X = df[['Day', 'Month', 'Year']]
    y = df['Price']
    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train model
    model=RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(x_train,y_train)
    # evaluate
    prediction = model.predict(x_test)
    mae=mean_absolute_error(y_test,prediction)
    print(f"Model MAE: {mae}")
    joblib.dump(model,'gold_price.pkl')
    return model