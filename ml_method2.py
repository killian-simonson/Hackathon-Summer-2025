import pandas as pd
import numpy as np
import csv
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

from sklearn.model_selection import GridSearchCV

mean_rmse_total = 0
mean_rmse_count = 0

def load_data() :
    df = pd.read_csv("data/train_set.csv", index_col=0)
    return df

def read_parsed_data(dir, num) :
    data_list = []
    for i in range(1, num + 1):
        df = pd.read_csv(f"data/{dir}/" + str(i).zfill(4) + ".csv")
        data_list.append(df)
    return data_list

def train_model(inputData) : 
    param_grid = {
        'n_estimators' : [100, 200, 300],
        'max_depth' : [10, 20, 30],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4]
    } 

    X = pd.get_dummies(inputData[['p1', 'p2']])
    y = inputData.drop(['p1', 'p2'], axis = 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=57
    )

    rfr = RandomForestRegressor(
        n_estimators=200, random_state=19, n_jobs=-1
    )
    rfr.fit(X_train, y_train)

    # rfr_cv = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    # rfr_cv.fit(X_train, y_train)
    # rfr = rfr_cv.best_estimator_
    
    y_pred = rfr.predict(X_test)
    print("RMSE: " + str(root_mean_squared_error(y_test, y_pred)))
    return rfr, X.columns

def run_on_test_set(testDir, predictionDir, modelInfo) :

    model, feature_columns = modelInfo
    
    # list of all test perturbations
    lines = []
    with open(testDir, 'r') as file:
        for line in file:
            lines.append(line.strip())
        
    with open(predictionDir, 'w') as f:
        # write header
        f.write("gene,perturbation,expression\n")
        for i in range(0, 50) : # for each testing perturbation
            # set perturbation as 2 element array, with each element being one of the perturbation pieces
            perturbation = [lines[i][:5], lines[i][6:]]

            # adjust layout of dummies based off of current model
            df = pd.DataFrame([perturbation], columns=["p1", "p2"])  # adjust cols if needed
            df = pd.get_dummies(df)
            df = df.reindex(columns=feature_columns, fill_value=0)

            predicted_values = model.predict(df)[0]

            for j in range(1, 1001) : # for each model/gene
                # write to file
                f.write("g" + str(j).zfill(4) + "," + perturbation[0] + "+" + perturbation[1] + "," + str(predicted_values[j - 1]) + "\n")

df = load_data()
print(df)

df = df.transpose()
df = df.reset_index().rename(columns={'index': 'perturbation'})
print(df)

df[['p1', 'p2']] = df['perturbation'].str.split('+', expand=True)
df['p2'] = df['p2'].str.split('.').str[0]
df = df.drop(['perturbation'], axis = 1)
print(df)

model = train_model(df)

run_on_test_set("data/test_set.csv", "prediction/prediction.csv", model)

# rmse on untuned: 0.37320167870428955
# rmse on 2 param tune: 0.3741263663381296
# rmse on 3 param tune: 
