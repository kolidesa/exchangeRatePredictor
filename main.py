from sklearn import linear_model

import numpy
#import matplotlib.pyplot as plt

import json

dates = []
values = []
# Opening JSON file
with open('FXUSDCAD-sd-2020-01-02-ed-2022-12-30.json') as json_file:
    data = json.load(json_file)

    for i in data['observations']:
        dates.append(i['d'])
        values.append(i['FXUSDCAD']['v'])

logr = linear_model.LogisticRegression()
logr.fit(dates,values)

predicted = logr.predict(numpy.array(['2023-01-01']).reshape(-1,1))
print(predicted)