import operator
import random
from collections import Counter

import numpy as np

ADVERTS = ['female_trust_cash', 'female_trust_car', 'female_familyhome_cash', 'female_familyhome_car',
           'family_trust_cash', 'family_trust_car', 'family_familyhome_cash', 'family_familyhome_car']

# THREE PREFERENCE CATEGORIES - VISIT www.thompsonsampling.com
def customer_preferences(first, second, third):
    female = 0
    family = 0
    cash = 0
    car = 0
    trust = 0
    familyhome = 0

    if first == "female":
        female = 7
        family = 2
    elif first == "family":
        female = 2
        family = 7

    if second == "trust":
        trust = 7
        familyhome = 2
    elif second == "family_home":
        trust = 2
        familyhome = 7

    if third == "cash":
        cash = 7
        car = 2
    elif third == "car":
        cash = 2
        car = 7

        ## Sum up Conversion rate
    female_trust_cash = (female + trust + cash)
    female_trust_car = (female + trust + car)
    female_familyhome_cash = (female + familyhome + cash)
    female_familyhome_car = (female + familyhome + car)
    family_trust_cash = (family + trust + cash)
    family_trust_car = (family + trust + car)
    family_familyhome_cash = (family + familyhome + cash)
    family_familyhome_car = (family + familyhome + car)

    scores = {'female_trust_cash': female_trust_cash,
              'female_trust_car': female_trust_car,
              'female_familyhome_cash': female_familyhome_cash,
              'female_familyhome_car': female_familyhome_car,
              'family_trust_cash': family_trust_cash,
              'family_trust_car': family_trust_car,
              'family_familyhome_cash': family_familyhome_cash,
              'family_familyhome_car': family_familyhome_car}

    return scores


def thompson_sampling(scores, percentage_scores):
    N = 10000  # total number of rounds (customers connecting to website)
    d = 8  # number of strategies

    # Creating Simulation

    conversion_rates = [element[1] for element in percentage_scores]  # get only the values
    X = np.array(np.zeros([N, d]))  # create zeros array

    for i in range(N):
        for j in range(d):  # Bernoulli distribution
            if np.random.rand() <= conversion_rates[j]:
                X[i, j] = 1

    # Implementing Random Strategy vs Thomson Sampling

    strategies_selected_rs = []
    strategies_selected_ts = []
    total_reward_rs = 0
    total_reward_ts = 0
    numbers_of_rewards_1 = [0] * d
    numbers_of_rewards_0 = [0] * d

    for n in range(0, N):  # for each round
        # Random Strategy
        strategy_rs = random.randrange(d)  # select random 0-8 strategy
        strategies_selected_rs.append(strategy_rs)  # append to list of random strategies
        reward_rs = X[n, strategy_rs]  # compare selected action with "real life simulation" X and get assigned reward
        total_reward_rs += reward_rs  # get total reward

        # Thomson Sampling
        strategy_ts = 0
        max_random = 0
        for i in range(0, d):  # for each strategy
            # compare how many times till now that strategy recieved 1 or 0 to get the Random Draw
            random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
            # update random beta for each strategy
            if random_beta > max_random:
                max_random = random_beta
                strategy_ts = i

        reward_ts = X[n, strategy_ts]  # compare selected action with "real life simulation" X and get assigned reward
        # update number of rewards
        if reward_ts == 1:
            numbers_of_rewards_1[strategy_ts] += 1
        else:
            numbers_of_rewards_0[strategy_ts] += 1
        # append to list of ts strategies
        strategies_selected_ts.append(strategy_ts)
        # accumulate total ts rewards
        total_reward_ts += reward_ts

    # For Histograms
    thompson_counter = Counter(strategies_selected_ts)
    thompson_strategies = dict(thompson_counter)
    top_ts = max(thompson_strategies.items(), key=operator.itemgetter(1))[0]
    top_ts_count = thompson_strategies.get(top_ts)

    random_counter = Counter(strategies_selected_rs)
    random_strategies = dict(random_counter)
    top_rs_count = random_strategies.get(top_ts)

    # Replace id's with ad names

    scores_list = []
    for key, value in scores.items():
        temp = [key, value]
        scores_list.append(temp)

    random_list = []
    for key, value in random_strategies.items():
        temp = [key, value]
        random_list.append(temp)

    thompson_list = []
    for key, value in thompson_strategies.items():
        temp = [key, value]
        thompson_list.append(temp)

    random_list.sort(key=lambda x: x[0])
    thompson_list.sort(key=lambda x: x[0])

    random_list = [a + b for a, b in zip(scores_list, random_list)]
    random_list.sort(key=lambda x: x[3], reverse=True)
    thompson_list = [a + b for a, b in zip(scores_list, thompson_list)]
    thompson_list.sort(key=lambda x: x[3], reverse=True)

    top_score = thompson_list[0][0]

    for element in thompson_list:
        element[0] = element[0].replace('_', ', ')
        element[0] = element[0].replace('female', 'girl')
        element[0] = element[0].title()

    for element in random_list:
        element[0] = element[0].replace('_', ', ')
        element[0] = element[0].replace('female', 'girl')
        element[0] = element[0].title()

    # Compute the Absolute and Relative Return

    absolute_return = int((total_reward_ts - total_reward_rs) * 1000)  # each customer converion = 1000 USD
    relative_return = int((total_reward_ts - total_reward_rs) / total_reward_rs * 100)

    algorithm_results = {'conversion_rates': conversion_rates, 'absolute_return': absolute_return,
                         'relative_return': relative_return, 'thompson_list': thompson_list, 'random_list': random_list,
                         'top_ts_count': top_ts_count, 'top_rs_count': top_rs_count, 'scores_list': scores_list,
                         "top_score": top_score}

    return algorithm_results
