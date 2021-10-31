import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler


USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

torch.manual_seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.manual_seed_all(1)
    print('cuda index:', torch.cuda.current_device())


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression,self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        return x


def remove_outliers(df, column_name, lower, upper):
    removed_outliers = df[column_name].between(df[column_name].quantile(lower), df[column_name].quantile(upper))

    print(str(df[column_name][removed_outliers].size) + "/" + str(sample_data[column_name].size) + " data points remain.")

    index_names = df[~removed_outliers].index
    return df.drop(index_names)


def evaluateRegressor(true, predicted, message="    Test Set"):
    MSE = mean_squared_error(true, predicted, squared=True)
    MAE = mean_absolute_error(true, predicted)
    RMSE = mean_squared_error(true, predicted, squared=False)
    R_squared = r2_score(true, predicted)

    print(message)
    print("MSE :", MSE)
    print("MAE :", MAE)
    print("RMSE :", RMSE)
    print("R-Squared :", R_squared)


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
sample_data = remove_outliers(sample_data, "co2", 0.05, 0.95)
idx_zero_temp = sample_data[sample_data['temp'] == 0].index

sample_data = sample_data.drop(idx_zero_temp)
sample_data = pd.get_dummies(sample_data)                       # Embedding
sample_data.to_csv('dummy.csv', index=False)

# sample_data.info(verbose=True, show_counts=True)

x_data = sample_data.iloc[:, 6:]
y_data = sample_data.iloc[:, [1, 2, 3, 4, 5]]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
# x_data.info(verbose=True, show_counts=True)

learning_rate = 0.15
iteration_number = 12000

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

y_train_scaled = sc.fit_transform(y_train)
y_test_scaled = sc.transform(y_test)

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

"""predicted = sc.inverse_transform(predicted)
print("%.2f" % predicted[0][0])
print("%.2f" % predicted[0][1])
print("%.2f" % predicted[0][2])
print("%.2f" % predicted[0][3])
print("%.2f" % predicted[0][4])"""
# print(y_test['co2'])

predict_valid_y = model(input_x_test.float()).data.numpy()
evaluateRegressor(test_targets, predict_valid_y)
predict_valid_y = sc.inverse_transform(predict_valid_y)
evaluateRegressor(y_test, predict_valid_y,"    Valid Set")



