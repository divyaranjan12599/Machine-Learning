import math
import numpy as np
import pandas as pd

def gradient_descent(x, y):
    theta = theta1 = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002
    cost_previous = 0
    for i in range(iterations):
        y_predicted = theta + theta1*x
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        theta_d = -(2/n)*sum(y-y_predicted)
        theta1_d = -(2/n)*sum(x*(y-y_predicted))
        theta = theta - learning_rate*theta_d
        theta1 = theta1 - learning_rate*theta1_d
        print(f'cost {cost}, theta {theta}, theta1 {theta1}, iteration {i}')
        if math.isclose(cost, cost_previous, rel_tol=1e-20, abs_tol=0.0):
            print(f'sol : cost {cost}, theta {theta}, theta1 {theta1}, iteration {i}')
            break
        cost_previous = cost

d = pd.read_csv('Exercisegd.csv')

x = np.array(d['math'])
y = np.array(d['cs'])

gradient_descent(x,y)
