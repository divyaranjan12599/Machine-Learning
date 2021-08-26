import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('Exercise.csv')

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])

plt.scatter(df['year'], df['per capita income (US$)'], marker = 'x', color = '#880000')
plt.xlabel('per capita income (US$)', size = 20)
plt.ylabel('year', size = 20)
plt.plot(df['year'], reg.predict(df[['year']]), color = '#000088')

plt.tight_layout()
plt.show()