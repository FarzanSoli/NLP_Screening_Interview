"""
Created on Thu May  4 21:07:26 2023

@author: Farzan Soleymani
"""
import numpy as np
from typing import Sequence
import matplotlib.pyplot as plt

def factorial(x):
    if x == 1:
        return 1
    else:
        return factorial(x-1)*x

def combination(K,i):
    return factorial(K)/((factorial(i))*(factorial(K-i)))


def prob_rain_more_than_n(p_i: Sequence[float], n: int) -> float:
    q_i = [1-x for x in p_i] # not raining probability
    rain_prob=0
    for i in range(n, 365):
        rain_prob += combination(365, n)*(p_i[i]**n)*(q_i[i]**(365-n))
    return print(f"The probability of rain at least {n} days in Vancouver is {rain_prob:.2%}")

""" TEST:
    Here I assumed each day is assigned with a random probability of raining (p_i).
"""

if __name__ == '__main__':
    np.random.seed(10)
    n = 100                           # least number of rainy days
    p_i = list(np.random.rand(365))   # rain probability
    prob_rain_more_than_n(p_i,n)
