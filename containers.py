#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'maximumContainers' function below.
#
# The function accepts STRING_ARRAY scenarios as parameter.
#

def maximumContainers(scenarios):
    # Write your code here
    for scenario in scenarios:
        scenario = scenario.split()
        n = int(scenario[0])
        cost = int(scenario[1])
        m = int(scenario[2])
        totalContainers = int(n/cost) #sets the original amount of containers
        remainingContainers = 0
        tradeBudget = totalContainers
        while tradeBudget >= m: #loop for determining amount of containers from trading
            newContainers =int( tradeBudget / m)
            totalContainers = totalContainers + newContainers

            tradeBudget = int (tradeBudget/m) + tradeBudget % m
        print(totalContainers)
if __name__ == '__main__':