import os
from datetime import datetime
import dill
import pandas as pd
import glob
import json

path = os.environ.get('PROJECT_PATH', '.')

def predict():
    last_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{last_model}', 'rb') as file:
        model = dill.load(file)
    pred = pd.DataFrame(columns=['car_id', 'pred'])
    for file in glob.glob(f'{path}/data/test/*.json'):
        with open(file) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            X = {'car_id': df.id, 'pred': y}
            df2 = pd.DataFrame(X)
            pred = pd.concat([pred,df2], axis=0)

    pred.to_csv(f'{path}/data/predictions/cars_preds{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
