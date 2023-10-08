import json
import random
from statistics import mean, stdev
from compute import wrapped

random.seed(35)

f = open("demo/demo_4932.json")
data = json.load(f)["demo"]
f.close()

data = list(zip(data, range(1272)))
# print(data)
# exit()

f = open("robust_sent.json")
robust = json.load(f)["sentList"]
f.close()

f = open("carb.json")
carb = json.load(f)
f.close()

testlist = random.sample(data, 10)
avg, v, m = [], [], []
for test in testlist:

    dists = []
    tsent = carb[test[1]]["ori_sent"]
    for didx in test[0]:
        dsent = robust[didx]["sent"]
        dist = wrapped(tsent, dsent)
        dists.append(dist)

    avg.append(mean(dists))
    v.append(stdev(dists))
    m.append(max(dists))

print(avg)
print(v)
print(m)