ntrials = 10000
import numpy as np
rng = np.random.default_rng()
avg_length = 0
for _ in range(ntrials):
    low = 0
    high = 1
    x = 50
    random_integers = rng.integers(low=low, high=high+1, size=x)
    max_consecutive = 0
    consecutive = 0
    for i in range(random_integers.size):
        if random_integers[i] == random_integers[i-1]:
            consecutive += 1
        else:
            consecutive = 1
            
        if consecutive > max_consecutive:
            max_consecutive = max(max_consecutive, consecutive)
    avg_length += max_consecutive
print(avg_length / ntrials)