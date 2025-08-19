import pandas as pd
import numpy as np
import csv

def load_data() :
    df = pd.read_csv("data/train_set.csv", index_col=0)
    print(df)
    return df

df = load_data()

def generate_prediction(dataframe, inputPath, outputPath) :
    with open(inputPath, 'r') as file:
        test_list = []
        csv_reader = csv.reader(file)
        for row in csv_reader:
            test_list.append([row[0][:5], row[0][6:11]])
    with open(outputPath, "w") as f :
        for i in range(0, 50) :
            perturbation = [test_list[i][0], test_list[i][1]]
            for j in range(1, 1001):
                val1 = dataframe.loc["g" + str(j).zfill(4)][str(perturbation[0]) + "+ctrl"]
                val2 = dataframe.loc["g" + str(j).zfill(4)][str(perturbation[1]) + "+ctrl"]
                prediction = val1 + val2
                f.write("g" + str(j).zfill(4) + "," + perturbation[0] + "+" + perturbation[1] + "," + str(prediction) + "\n")

generate_prediction(df, "data/test_set.csv", "prediction/prediction.csv")

