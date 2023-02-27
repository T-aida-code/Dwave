from pyqubo import Array, Constraint, Placeholder
from openjij import SQASampler
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
import copy
import matplotlib.pyplot as plt

N=4

solver = 'Dwave'
# solver = 'Openjij'

#都市座標
city = np.zeros((N, 2))
#region
# city[0] = [0, 0]
# city[1] = [5, 5]
# city[2] = [3, 7]
# city[3] = [1, 4]
# city[4] = [2, 10]
#endregion
city[0] = [0, 0]
city[1] = [1, 1]
city[2] = [0, 1]
city[3] = [1, 0]


# #region
# city[0] = [0, 0]
# city[1] = [5, 5]
# city[2] = [5, 0]
# city[3] = [0, 5]
# city[4] = [3, 3]
# #endregion

#Q:都市同士の距離を設定
Q = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            Q[i, j] = 1000
        else:
            Q[i, j] = ((city[i, 0] - city[j, 0])**2 + (city[i, 1] - city[j, 1])**2) ** 0.5

# Q = np.array([[1000, 20, 20, 50, 40],
#               [30, 1000, 10, 30, 20],
#               [20, 10, 1000, 30, 20],
#               [50, 30, 20, 1000, 10],
#               [40, 20, 20, 10, 1000]])

#x:変数設定
x = Array.create('x', shape = (N, N), vartype = 'BINARY')

#コスト計算式
cost = 0
for t in range(N):
    for i in range(N):
        for j in range(N):
            cost += Q[i][j] * x[t][i] * x[(t + 1) % N][j]

#制約計算式
const1 = 0
const2 = 0
for i in range(N):
    const1 += (np.sum(x[i]) - 1) ** 2
    const2 += (np.sum(x.T[i]) - 1) ** 2

#最終的なコスト計算:Placeholderはパラメータを示し、制約の影響力を変更する
cost_func = cost + Placeholder('lam') * Constraint(const1, label='const1') + Placeholder('lam') * Constraint(const2, label='const2')
model = cost_func.compile()

feed_dict = {'lam':1000}
qubo, offset = model.to_qubo(feed_dict = feed_dict)

if solver == 'Dwave':
    token = "DEV-36cbe434c6aa4ff98284964555b1c24fc4f7ea5e"
    endpoint = "https://cloud.dwavesys.com/sapi/"
    dw_sampler = DWaveSampler(solver = "Advantage2_prototype1.1", token = token, endpoint = endpoint)
    sampler = EmbeddingComposite(dw_sampler)
elif solver == "Openjij":
    sampler = SQASampler()

sampleset = sampler.sample_qubo(qubo, num_reads=10)
print(sampleset.record)
decoded_samples = model.decode_sampleset(sampleset=sampleset, feed_dict=feed_dict)
minst = 0
minstset = 0
i=0

for sample in decoded_samples:
    print(sampleset.record[i][0].reshape(N, N))
    print(sample.constraints(only_broken=True))
    i+=1

codes = int(input())

preset = sampleset.record[codes][0].reshape(N, N)
print(preset)
pllist = []
for i in range(N):
    for j in range(N):
        if preset[i,j] == 1:
            pllist.append(j)
for i in range(N):
    plt.plot([city[pllist[i], 0], city[pllist[i-1], 0]], [city[pllist[i], 1], city[pllist[i-1], 1]], color="red")