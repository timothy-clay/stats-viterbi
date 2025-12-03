import numpy as np

colors = ['R', 'B']
bags = ['A', 'B']

t1 = float(input('Enter transition probability A -> A: '))
t2 = float(input('Enter transition probability A -> B: '))
t3 = float(input('Enter transition probability B -> A: '))
t4 = float(input('Enter transition probability B -> B: '))

assert t1 + t2 == 1 and t3 + t4 == 1, 'Invalid probability distribution'

o1 = float(input('Enter probability of seeing a red marble from bag A: '))
o2 = float(input('Enter probability of seeing a red marble from bag B: '))

assert o1 <= 1 and o2 <= 1, 'Invalid probability distribution'

obs_string = input('Enter sequence of observations (ex. RRBRBBR): ')

assert obs_string.count('R') + obs_string.count('B') == len(obs_string), 'Invalid sequence provided'

observations = [colors.index(obs) for obs in obs_string]

P = np.array([[t1, t2],
              [t3, t4]])

O = np.array([[o1, 1 - o1],
              [o2, 1 - o2]])

steady_state_dist = np.linalg.matrix_power(P, 1000)[0]
# print("Steady state distribution:", steady_state_dist)

dp = np.empty((P.shape[0], len(observations)), dtype=object)

def viterbi(P, O, observations):
    for i in range(len(observations)):
        # base case
        if i == 0:
            for s in range(P.shape[0]):
                dp[s][i] = (steady_state_dist[s] * O[s][observations[i]], None)
        # recursive case
        else:
            for s in range(P.shape[0]):
                # store both the probability and the previous state used to get here
                probabilities = [dp[prev_s][i-1][0] * P[prev_s][s] * O[s][observations[i]] for prev_s in range(P.shape[0])]
                dp[s][i] = (max(probabilities), np.argmax(probabilities))
viterbi(P, O, observations)
# print("DP Table:\n", dp)

# backtrack to find the most probable state sequence
most_probably_state_sequence = np.zeros(len(observations), dtype=int)
# start with the state that has the highest probability at the last observation
most_probably_state_sequence[-1] = np.argmax([dp[s][-1][0] for s in range(P.shape[0])])
# backtrack through the dp table and use the stored previous states
for i in range(len(observations) - 2, -1, -1):
    most_probably_state_sequence[i] = dp[most_probably_state_sequence[i + 1]][i + 1][1]
print("Most probable bag sequence:", ', '.join([bags[obs] for obs in most_probably_state_sequence]))