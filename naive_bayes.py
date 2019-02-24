import numpy as np

d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

N = len(d1)

#Create train data set and train label set
train_data = np.array([d1,d2,d3,d4])
train_label = np.array(['B','B','B','N'])

#Probability for each p(B) and p(N) based on occurence
p_B = 3/4
p_N = 1/4

#Summing all word-occurence
sum_array_b = np.zeros(N)
sum_array_n = np.zeros(N)
#Each class has N words
sum_b = N
sum_n = N
for i in range(len(train_label)):
    if train_label[i] == 'B':
        sum_array_b += train_data[i]
    else:
        sum_array_n += train_data[i]
for i in sum_array_b:
    sum_b += i
for i in sum_array_n:
    sum_n += i
#Calculate probability for each word-occurence in N or B
prob_array_b = np.zeros(N)
prob_array_n = np.zeros(N)
for i in range(N):
        prob_array_b[i] = (sum_array_b[i] + 1) / sum_b
for i in range(N):
        prob_array_n[i] = (d4[i] + 1) / 13

#Predict on new data
d6 = [0,1,0,0,0,0,0,1,1]

def predict(data):
    p_B_data = p_B
    p_N_data = p_N
    for i in range(N):
        p_B_data *= (prob_array_b[i] ** data[i])
    print("Density for B: " + str(p_B_data))

    for i in range(N):
        p_N_data *= (prob_array_n[i] ** data[i])
    print("Density for N: " + str(p_N_data))

    prob_B_data = p_B_data / (p_B_data + p_N_data)
    prob_N_data = 1 - prob_B_data
    print("Probability for B: " + str(prob_B_data))
    print("Probability for N: " + str(prob_N_data))

predict(d6)
