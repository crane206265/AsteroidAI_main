import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc


# load data
base_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/"
test_data_path = base_path + "data/pole_axis_RL_data_batches/unrolled/data_pole_axis_RL_preset_batch_filtered_3.npy"

test_data = np.load(test_data_path)[1:]
gc.collect()

train_data_paths = ["data_pole_axis_RL_preset_batch_0.npy",
                    "data_pole_axis_RL_preset_batch_1.npy",
                    "data_pole_axis_RL_preset_batch_2.npy",
                    "data_pole_axis_RL_preset_batch_filtered_4.npy"]
train_data_list = []
for data_name in train_data_paths[:]:
    train_data_path = base_path + "data/pole_axis_RL_data_batches/unrolled/" + data_name
    train_data_list.append(np.load(train_data_path)[1:])
train_data = np.concatenate(train_data_list, axis=0)
gc.collect()

print("[Data shapes]")
print("test_Data shape : ", test_data.shape)
print("train_Data shape : ", train_data.shape)
print("-"*20)


def lcInfoDistribution(train_data, bins=30):
    lc_info = train_data[::800, 1000:1006]
    Sdir = lc_info[:, 0:3]
    Edir = lc_info[:, 3:6]
    Stheta = np.arccos(Sdir[:, -1]) * 180 / np.pi - 90
    Etheta = np.arccos(Edir[:, -1]) * 180 / np.pi - 90

    plt.figure(figsize=(8, 6), dpi=100)
    sns.histplot(Stheta, bins=bins, color="orange", kde=True, label=r"$\theta_S$")
    sns.histplot(Etheta, bins=bins, color="dodgerblue", kde=True, label=r"$\theta_E$")
    plt.legend()
    plt.xlim(-90, 90)
    plt.xlabel(r"Polar Angle $[-90\degree, +90\degree]$")
    plt.title(r"Distribution of $\theta_S, \theta_E$")
    plt.show()

lcInfoDistribution(train_data)

