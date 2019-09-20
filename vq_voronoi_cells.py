import numpy as np

seed = 1548731457
np.random.seed(seed)

input_data = np.random.randn(30,2)
reconst_vectors = np.random.randn(12,2)

min_reconst = []

epsilon = 0.3
k = 0

for input in input_data:
    min_dis = np.linalg.norm(input-reconst_vectors[0])
    min_vec = reconst_vectors[0]
    for i in range(1,len(reconst_vectors)):
        dis = np.linalg.norm(input-reconst_vectors[i])
        if dis < min_dis:
            min_dis = dis
            min_vec = reconst_vectors[i]
    min_reconst.append(min_vec)

distortion = 0
for input , reconst in zip(input_data,reconst_vectors):
    distortion += np.linalg.norm(input - reconst) ** 2

distortion /= len(input_data)

print(distortion)
