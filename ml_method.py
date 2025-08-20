import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error



def load_data() :
    df = pd.read_csv("data/train_set.csv", index_col=0)
    print("LOADED DATA:")
    print(df)
    return df


def reformat_data(inputData, outputDir) : 
    data_list = []
    for i in range(1, 1001):
        gene = 'g' + str(i).zfill(4)
        
        df = pd.DataFrame(columns=['p1', 'p2', 'exp'])
        
        for col_name in inputData.columns :
            p1 = col_name[:5]
            p2 = col_name[6:]
            idx = p2.find('.')
            p2 = p2[:idx] if idx != -1 else p2
            df.loc[len(df)] = [p1, p2, inputData.loc[gene][col_name]]
        
        print("TABLE " + str(i))
        print(df)
        data_list.append(df)
        df.to_csv(outputDir + "/" + str(i).zfill(4) + ".csv", index=False)

    return data_list

def read_parsed_data(dir, num):
    data_list = []
    for i in range(1, num + 1):
        df = pd.read_csv(f"data/{dir}/" + str(i).zfill(4) + ".csv")
        data_list.append(df)
    return data_list
        
def train_model(inputFrame, geneName) :
    input = pd.get_dummies(inputFrame)
    X = input.drop(['exp'], axis = 1)
    Y = input['exp']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 57)
    rfr = RandomForestRegressor(random_state = 19)
    rfr.fit(X_train, Y_train)
    Y_pred = rfr.predict(X_test)
    print(geneName + " mean absolute: " + str(mean_absolute_error(Y_test, Y_pred)))
    print(geneName + " mean squared: " + str(mean_squared_error(Y_test, Y_pred)))
    print(geneName + " r^2: " + str(r2_score(Y_test, Y_pred)))
    print(geneName + " RMSE: " + str(root_mean_squared_error(Y_test, Y_pred)) + "\n")
    return rfr, X.columns

def run_on_test_set(testDir, predictionDir, modelList) :
    # list of all test perturbations
    lines = []
    with open(testDir, 'r') as file:
        for line in file:
            lines.append(line.strip())
        
    with open(predictionDir, 'w') as f:
        # write header
        f.write("gene,perturbation,expression\n")
        for i in range(0, 50) : #for each testing perturbation
            # set perturbation as 2 element array, with each element being one of the perturbation pieces
            perturbation = [lines[i][:5], lines[i][6:]]


            for j in range (0, 1000) : #for each model/gene

                # retrieve model and layout for dummies
                curModel, feature_columns = modelList[j]

                # adjust layout of dummies based off of current model
                df = pd.DataFrame([perturbation], columns=["p1", "p2"])  # adjust cols if needed
                df = pd.get_dummies(df)
                df = df.reindex(columns=feature_columns, fill_value=0)

                 #predict a value!
                predicted_value = curModel.predict(df)[0]

                # write to file
                f.write("g" + str(j + 1).zfill(4) + "," + perturbation[0] + perturbation[1] + "," + str(predicted_value) + "\n")

# df = load_data()
# data_list = reformat_data(df, "data/parsed")

models = []

# list of dataframes, with each frame corresponding to one gene's file in parsed directory
data_list = read_parsed_data("parsed", 1000)

# train a model on each gene's file, so there's one model per gene
for i in range(1, 1001) :
    models.append(train_model(data_list[i - 1], "g" + str(i).zfill(4)))

# run all the models, write to prediction file
run_on_test_set("data/test_set.csv", "prediction/prediction.csv", models)





