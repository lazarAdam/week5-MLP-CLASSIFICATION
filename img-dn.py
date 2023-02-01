# Author: Oussama El aallali
# program to download all the images using the links in cvs file
import requests
import pandas as pd

dataFrame = pd.read_csv('fish-cat.csv')
progressPercentage = 0.0

for i, row in dataFrame.iterrows():
    try:
        print(f'download progress: {"{:.2f}".format(progressPercentage)}%')
        r = requests.get(row['url'])

        if r.status_code == 200:

            if row['class'] == 'cat':

                open(f'./train/cat/cat{i}.jpg', 'wb').write(r.content)

            elif row['class'] == 'fish':

                open(f'./train/fish/fish{i}.jpg', 'wb').write(r.content)
            progressPercentage = (i * 100) / (dataFrame.shape[0])
        else:
            print('image not found 404 error')
            progressPercentage = (i * 100) / (dataFrame.shape[0])
            continue

    except (Exception, requests.exceptions.ConnectionError) as e:

        if type(e) == requests.exceptions.ConnectionError:

            print('connection error, continue to the next link...')
            continue
        else:
            print(e)
            continue

