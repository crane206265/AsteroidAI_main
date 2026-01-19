import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import gc

def reward_p(lc_target_temp, lc_pred_temp):
    amp = amp_lc(lc_target_temp)
    loss_p = -torch.mean((80*(lc_target_temp - lc_pred_temp)/amp)**2)
    loss_p = (1.2*2/10) * loss_p
    return loss_p

def reward_d(lc_target_temp, lc_pred_temp):
    loss_d = -torch.mean((40*(torch.diff(lc_target_temp)-torch.diff(lc_pred_temp)))**2)
    loss_d = (1.0*2/10) * loss_d
    return loss_d

def reward_i(lc_target_temp, lc_pred_temp):
    amp = amp_lc(lc_target_temp)
    loss_i = -60*torch.trapezoid(torch.abs(lc_target_temp-lc_pred_temp))/(100*amp)
    loss_i = (1.0*2/10) * loss_i
    return loss_i

# Utils for LC-Related Calculation
def lc_mean(input_lc):
    """
    input_lc = [LC Length]
    """
    lc_len = input_lc.shape[-1]
    lc_mean0 = (torch.sum(input_lc, dim=-1) - (input_lc[..., 0] + input_lc[..., -1])/2) / lc_len
    return lc_mean0
    
def amp_lc(input_lc):
    lc_max = torch.amax(input_lc, dim=-1)
    lc_min = torch.amin(input_lc, dim=-1)
    return lc_max - lc_min

# --------------- plotter --------------- 

def plotter(state, rewards, grads, idx, ep_info):
    fig = plt.figure(figsize=(14, 4), dpi=200)
    row, col = 1, 3
    ax = [[fig.add_subplot(row, col, col*i+j) for j in range(1,col+1)] for i in range(row)]

    lc_target = state[800:900]
    lc_pred = state[900:1000]
    reward_p0, reward_d0, reward_i0 = rewards
    grad_p0, grad_d0, grad_i0 = grads
    state0, ep = ep_info
    lc_pred0 = state0[900:1000]

    # --------------- plot ax[0][0] ---------------
    # lightcurves
    ax[0][0].plot(lc_pred, label="lc_pred", color='royalblue')
    ax[0][0].plot(lc_pred0, color='royalblue', alpha=0.5, linestyle="dotted")
    ax[0][0].plot(lc_target, label="lc_target", color='orangered')#, linestyle='dotted')
    ax[0][0].set_title("Lightcurve at idx " + str(idx) + "(ref_reward = "+str(int(state[-1]*100)/100)+")")

    # --------------- plot ax[0][1] ---------------
    # LC - grad graph
    ax[0][1].plot(lc_pred, label="lc_pred", color='darkgray')
    ax[0][1].plot(lc_target, label="lc_target", color='black', linestyle="dashed")
    ylim = ax[0][1].set_ylim()
    dx = 2
    ax[0][1].plot(lc_pred+dx*grad_p0, label="grad_p0", color='royalblue')
    grad_d0_smooth = np.convolve(grad_d0, np.ones(5)/5, mode="same")
    ax[0][1].plot(lc_pred+dx*grad_d0, color='orangered', linestyle="dotted")
    ax[0][1].plot(lc_pred+dx*grad_d0_smooth, label="grad_d0", color='orangered')
    ax[0][1].plot(lc_pred+dx*grad_i0, label="grad_i0", color='gold')
    ax[0][1].legend()
    ax[0][1].set_ylim(ylim)
    ax[0][1].set_title("LC + grad*dx at idx " + str(idx) + "(dx="+str(dx)+")")

    # --------------- plot ax[0][2] ---------------
    # episode - reward & loss graph
    ax[0][2].plot(reward_p0, label="reward_p0", color='royalblue')
    ax[0][2].plot(reward_d0, label="reward_d0", color='orangered')
    ax[0][2].plot(reward_i0, label="reward_i0", color='gold')
    ax[0][2].set_ylabel("reward_elements", rotation = -90)
    ax02_1 = ax[0][2].twinx()
    reward_total0 = 100 + reward_p0 + reward_d0 + reward_i0
    ax02_1.plot(reward_total0, label="reward_total0", color='darkgray')
    ax02_1.set_ylabel("reward_total", rotation = 90)
    ax[0][2].legend()
    ax02_1.legend()
    ax[0][2].set_title("Episode - Reward Graph at ep " + str(ep))

    #plt.show()
    plt.tight_layout()
    plt.savefig(save_path+"img{:03d}.png".format(idx))
    plt.close()
    

# -------------------- Main Analysis --------------------

base_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/"
save_path = base_path + "data_analysis/reward/ideal/batch_1/"
test_data_path = base_path + "data/pole_axis_RL_data_batches/unrolled/ideal/ideal_data_pole_axis_RL_preset_batch_1.npy"

test_data = np.load(test_data_path)[1:]
gc.collect()

"""
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
"""

print("[Data shapes]")
print("test_Data shape : ", test_data.shape)
#print("train_Data shape : ", train_data.shape)
print("-"*20)

np.random.seed(206265)
sample_idx = list(np.random.randint(0, test_data.shape[0]//800, 100))
sample_idx = list(range(0, test_data.shape[0]//800))
print("sample idx : [", end='')
for idx in sample_idx:
    print(idx, end=' ')
print("]")

state_total = test_data[::800, :]
del test_data
gc.collect()

# Episode start idx list
# if direction data is same ==> one episode
ep_idx0_list = [0]
for i in range(1, state_total.shape[0]):
    dir_ref = state_total[ep_idx0_list[-1], 1000:1006]
    dir_temp = state_total[i, 1000:1006]
    if not np.array_equal(dir_ref, dir_temp):
        ep_idx0_list.append(i)


# --------------- calculate reward--------------- 

reward0 = np.zeros((state_total.shape[0], 3))
grad0 = np.zeros((state_total.shape[0], 3, 100))
for i in range(state_total.shape[0]):
    lc_target_t = torch.tensor(state_total[i, 800:900], dtype=torch.float32)
    lc_pred_t = torch.tensor(state_total[i, 900:1000], dtype=torch.float32, requires_grad=True)

    # Scaling lc_temp compared with target_lc_temp
    lc_target_mean = lc_mean(lc_target_t)
    lc_pred_temp = lc_pred_t * lc_target_mean / lc_mean(lc_pred_t)

    # Normalization for Loss Calculation
    lc_target_temp = lc_target_t - lc_target_mean
    lc_pred_temp = lc_pred_temp - lc_target_mean

    reward_p0_t = reward_p(lc_target_temp, lc_pred_temp)
    reward_d0_t = reward_d(lc_target_temp, lc_pred_temp)
    reward_i0_t = reward_i(lc_target_temp, lc_pred_temp)

    reward0[i, 0] = reward_p0_t.item()
    reward0[i, 1] = reward_d0_t.item()
    reward0[i, 2] = reward_i0_t.item()

    # Calculate Gradients
    lc_pred_t.grad = None
    reward_p0_t.backward(retain_graph=True)
    grad_p0 = lc_pred_t.grad.clone()
    
    lc_pred_t.grad = None
    reward_d0_t.backward(retain_graph=True)
    grad_d0 = lc_pred_t.grad.clone()

    lc_pred_t.grad = None
    reward_i0_t.backward(retain_graph=True)
    grad_i0 = lc_pred_t.grad.clone()

    grad0[i, :, :] = torch.stack((grad_p0, grad_d0, grad_i0), dim=0).numpy()


for num, i in tqdm(enumerate(sample_idx[:]), total=len(sample_idx)):
    state = state_total[i, :]
    ep_idx0_list += [2000]
    for ep in range(len(ep_idx0_list)):
        if(i < ep_idx0_list[ep]): break
    
    reward0_temp = reward0[ep_idx0_list[ep-1]:ep_idx0_list[ep], :]
    state0 = state_total[ep_idx0_list[ep-1], :]
    
    plotter(state,
            rewards=(reward0_temp[:, 0], reward0_temp[:, 1], reward0_temp[:, 2]),
            grads=(grad0[i, 0, :], grad0[i, 1, :], grad0[i, 2, :]),
            idx=i, ep_info=(state0, ep-1))