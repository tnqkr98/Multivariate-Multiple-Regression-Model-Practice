import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(1)


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def remove_outliers(df, column_name, lower, upper):
    removed_outliers = df[column_name].between(df[column_name].quantile(lower), df[column_name].quantile(upper))

    print(str(df[column_name][removed_outliers].size) + "/" + str(sample_data[column_name].size) + " data points remain.")

    index_names = df[~removed_outliers].index
    return df.drop(index_names)

"""
 0   time           11964 non-null  int64
 1   age            11964 non-null  int64
 2   height         11964 non-null  int64
 3   weight         11964 non-null  int64
 4   depressive     11964 non-null  int64
 5   media          11964 non-null  int64
 6   liquor         11964 non-null  int64
 7   smoke          11964 non-null  int64
 8   caffeine       11964 non-null  int64
 9   exercise       11964 non-null  int64
 10  stress         11964 non-null  int64
 11  nap            11964 non-null  int64
 12  state_asleep   11964 non-null  uint8
 13  state_awake    11964 non-null  uint8
 14  gender_female  11964 non-null  uint8
 15  gender_male    11964 non-null  uint8
 16  disease_none   11964 non-null  uint8
 17  disorder_no    11964 non-null  uint8
 18  disorder_yes   11964 non-null  uint8
"""
# def predictEnv():


sample_data = pd.read_csv("sample.csv")
sample_data = remove_outliers(sample_data, "co2", 0.1, 0.9)
sample_data = pd.get_dummies(sample_data)                       # Embedding
sample_data.info()

x_data = sample_data.iloc[:, 6:]
y_data = sample_data.iloc[:, [1, 2, 3, 4, 5]]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
x_data.info(verbose=True, show_counts=True)

learning_rate = 0.02
iteration_number = 1000

sc_x = StandardScaler()
x_train_scaled = sc_x.fit_transform(x_train)
x_test_scaled = sc_x.transform(x_test)

sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train)
y_test_scaled = sc_y.transform(y_test)

x_train_scaled = np.array(x_train_scaled, dtype=np.float32)
y_train_scaled = np.array(y_train_scaled, dtype=np.float32)
x_test_scaled = np.array(x_test_scaled, dtype=np.float32)
y_test_scaled = np.array(y_test_scaled, dtype=np.float32)

inputs = torch.from_numpy(x_train_scaled)
targets = torch.from_numpy(y_train_scaled)

test_inputs = torch.from_numpy(x_test_scaled)
test_targets = torch.from_numpy(y_test_scaled)

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]

model = LinearRegression(input_dim, output_dim)
mse = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_list = []
test_loss_list = []
for iteration in range(iteration_number):
    optimizer.zero_grad()
    results = model(inputs)
    loss = mse(results, targets)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data)

    test_results = model(test_inputs)
    test_loss = mse(test_results, test_targets)
    test_loss_list.append(test_loss.data)

    if iteration % 50 == 0:
        print('epoch %3d, trian loss : %f, test loss : %f ' % (iteration, loss.data, test_loss.data))


plt.plot(range(iteration_number), loss_list, range(iteration_number), test_loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.show()


input_x_test = torch.from_numpy(x_test_scaled)
predicted = model(input_x_test.float()).data.numpy()

predicted = sc_y.inverse_transform(predicted)
print("%.2f" % predicted[0][0])
print("%.2f" % predicted[0][1])
print("%.2f" % predicted[0][2])
print("%.2f" % predicted[0][3])
print("%.2f" % predicted[0][4])
print(y_test['co2'])



