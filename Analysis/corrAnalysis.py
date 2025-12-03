import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import gc


class RewardMapModifier():
    def __init__(self, extends=(0, 1), blur_coef=(5, 3)):
        self.extends = extends
        self.blur_coef = blur_coef

    def extend_hori(self, reward_map, action_maps):
        left_reward = reward_map[..., :, -int(reward_map.shape[-2]*self.extends[1]/2):, :]
        right_reward = reward_map[..., :, :int(reward_map.shape[-2]*self.extends[1]/2), :]

        if action_maps is not None:
            left_actions = action_maps[..., :, -int(action_maps.shape[-2]*self.extends[1]/2):, :].copy()
            right_actions = action_maps[..., :, :int(action_maps.shape[-2]*self.extends[1]/2), :].copy()
            left_actions[..., :, :, 0] = left_actions[..., :, :, 0] - 1
            right_actions[..., :, :, 0] = right_actions[..., :, :, 0] + 1

        if self.extends[1] != 0:
            extended_reward = np.concatenate((left_reward, reward_map, right_reward), axis=-2)
            extended_actions = np.concatenate((left_actions, action_maps, right_actions), axis=-2) if action_maps is not None else action_maps
        else:
            extended_reward = reward_map
            extended_actions = action_maps

        return extended_reward, extended_actions

    def extend_vert(self, reward_map, action_maps):
        top_reward = np.roll(reward_map[..., :int(reward_map.shape[-3]*self.extends[0]/2), :, :], 20, axis=-2)
        bottom_reward = np.roll(reward_map[..., -int(reward_map.shape[-3]*self.extends[0]/2):, :, :], 20, axis=-2)
        top_reward = np.flip(top_reward, axis=-3)
        bottom_reward = np.flip(bottom_reward, axis=-3)

        if action_maps is not None:
            top_actions = np.flip(action_maps[..., :int(action_maps.shape[-3]*self.extends[0]/2), :, :].copy(), -3)
            bottom_actions = np.flip(action_maps[..., -int(action_maps.shape[-3]*self.extends[0]/2):, :, :].copy(), -3)
            top_actions[..., :, :, 1] = 2*0 - top_actions[..., :, :, 1]
            bottom_actions[..., :, :, 1] = 2*1 - bottom_actions[..., :, :, 1]

        if self.extends[0] != 0:
            extended_reward = np.concatenate((top_reward, reward_map, bottom_reward), axis=-3)
            extended_actions = np.concatenate((top_actions, action_maps, bottom_actions), axis=-3) if action_maps is not None else action_maps
        else:
            extended_reward = reward_map
            extended_actions = action_maps

        return extended_reward, extended_actions

    def blur(self, reward_map):
        #reward_map = 2.5 * np.tan( reward_map * (np.pi/2) / 6 )\n",
        #reward_map = 6 * 2*(1/(1+np.exp(-reward_map/7)) - 0.5)
        if len(reward_map.shape) == 3:
            reward_map[:, :, 0] = cv2.GaussianBlur(reward_map[:, :, 0], (self.blur_coef[0], self.blur_coef[0]), self.blur_coef[1])
        elif len(reward_map.shape) == 4:
            for i in range(reward_map.shape[0]):
                reward_map[i, :, :, 0] = cv2.GaussianBlur(reward_map[i, :, :, 0], (self.blur_coef[0], self.blur_coef[0]), self.blur_coef[1])
                #max_val = np.max(np.abs(reward_map[i, :, :, 0]))
                #reward_map[i, :, :, 0] = 6 * (2/np.pi) * np.arctan(reward_map[i, :, :, 0]/2) / ((2/np.pi) * np.arctan(max_val/2))
        reward_map = 6 * (2/np.pi) * np.arctan(reward_map/8)
        #reward_map = 6 * 2*(1/(1+np.exp(-reward_map/7)) - 0.5)
        return reward_map

    def operation(self, reward_map, action_maps, order=['extend_hori', 'extend_vert', 'blur']):
        result_reward = reward_map
        result_action = action_maps
        for op in order:
            if op == 'extend_hori':
                result_reward, result_action = self.extend_hori(result_reward, result_action)
            elif op == 'extend_vert':
                result_reward, result_action = self.extend_vert(result_reward, result_action)
            elif op == 'blur':
                if self.blur_coef == (0, 0):
                    reward_map = 6 * 2*(1/(1+np.exp(-reward_map/7)) - 0.5)
                else:
                    result_reward = self.blur(result_reward)
            else:
                raise NotImplementedError()
        return result_reward, result_action

    def ext_N_set(self, N_set):
        return (N_set[0]+2*int(N_set[0]*self.extends[1]/2), N_set[1]+2*int(N_set[1]*self.extends[0]/2))


def plotter(state, reward_map0, idx):
    """
    ## Plotter
    **: Plots the result of the model, with monitoring informations.**

    ### Contents

    """

    r_arr = state[:800].reshape(40, 20).T
    lc_target = state[800:900]
    lc_pred = state[900:1000]
    lc_info = state[1000:1006]

    Sdir = lc_info[0:3]
    Edir = lc_info[3:6]
    Stheta = np.arccos(Sdir[-1]) * 20 / np.pi
    Etheta = np.arccos(Edir[-1]) * 20 / np.pi

    fig = plt.figure(figsize=(12, 7))#, dpi=200)
    ax = [[fig.add_subplot(3, 3, 3*i+j) for j in range(1,3+1)] for i in range(3)]

    # --------------- plot ax[0][0] ---------------
    # lightcurves
    ax[0][0].plot(lc_pred, label="lc_pred", color='royalblue')
    ax[0][0].plot(lc_target, label="lc_target", color='orangered', linestyle='dotted')
    ax[0][0].set_title("Lightcurve at idx " + str(idx))
    ax[0][0].legend()
    _quaterLine(ax=ax[0][0], xlim=[0, len(lc_pred)], ylim=ax[0][0].set_ylim())

    # --------------- plot ax[1][0] ---------------
    # Fourier Transforms of LCs
    fft_target = np.fft.fft(lc_target)[1:lc_target.shape[0]//2+1]
    fft_pred = np.fft.fft(lc_pred)[1:lc_pred.shape[0]//2+1]
    fft_target_mag = np.log10(np.abs(fft_target))
    fft_pred_mag = np.log10(np.abs(fft_pred))
    lim = (np.min(fft_target_mag)-0.3, np.max(fft_target_mag)+0.3)

    ax[1][0].plot(fft_target_mag, color='royalblue')
    ax[1][0].plot(fft_pred_mag, color='orangered')
    ax[1][0].plot([np.argmax(fft_target_mag), np.argmax(fft_target_mag)], [lim[0], lim[1]], linestyle='dotted', color='gray')
    ax[1][0].set_title("FFT of LC at idx " + str(idx))
    ax[1][0].set_ylim(lim[0], lim[1])

    # --------------- plot ax[2][0] ---------------
    # lc_pred - lc_target
    FFT = True
    if not FFT:
        ax[2][0].plot(lc_pred-lc_target, label=r"$\Delta LC$", color='royalblue')
        ax[2][0].set_title(r"$\Delta LC$ at idx " + str(idx))
        ax[2][0].legend()
        _quaterLine(ax=ax[2][0], xlim=[0, len(lc_pred)], ylim=ax[2][0].set_ylim())
        ax[2][0].plot(ax[2][0].set_xlim(), [0, 0], color='gray', linestyle='dotted', alpha=0.4)
    else:
        fft_delta = np.fft.fft(lc_pred-lc_target)[1:lc_pred.shape[0]//2+1]
        fft_delta_mag = np.log10(np.abs(fft_delta))
        lim = (np.min(fft_delta_mag)-0.3, np.max(fft_delta_mag)+0.3)
        ax[2][0].plot(fft_delta_mag, color='royalblue')
        ax[2][0].plot(np.angle(fft_delta)/np.pi, color='orangered')
        ax[2][0].plot([np.argmax(fft_delta_mag), np.argmax(fft_delta_mag)], [lim[0], lim[1]], linestyle='dotted', color='gray')
        ax[2][0].set_title(r"FFT of $\Delta LC$ at idx " + str(idx))
        ax[2][0].set_ylim(lim[0], lim[1])

    # --------------- plot ax[0][1] ---------------
    # reward_map0
    reward_map0_img = ax[0][1].imshow(reward_map0, vmax=np.max(np.abs(reward_map0)), vmin=-np.max(np.abs(reward_map0)))#, vmax=6, vmin=-6)
    ax[0][1].set_title("Reward_Map at idx " + str(idx))
    #plt.colorbar(reward_map0_img, ax=ax[][], shrink=0.75)#, orientation='horizontal')
    _setRewardMapPlot(ax=ax[0][1], Etheta=Etheta, Stheta=Stheta)

    #rewardMapF = np.fft.fftshift(np.fft.fft2(reward_map0))
    rewardMapF = np.fft.rfft2(reward_map0)
    rewardMapF_mag = np.abs(rewardMapF)
    rewardMapF_arg = np.angle(rewardMapF)

    # --------------- plot ax[1][1] ---------------
    # Fourier Transform of reward_map0 - abs values
    ax[1][1].imshow(np.log10(rewardMapF_mag))
    ax[1][1].set_title("F.T. of reward_map0 (abs)")

    # --------------- plot ax[2][1] ---------------
    # Fourier Transform of reward_map0 - arg values
    ax[2][1].imshow(rewardMapF_arg/np.pi)
    ax[2][1].set_title("F.T. of reward_map0 (arg)")

    # --------------- plot ax[1][2] ---------------
    # Fourier Transform of reward_map0 - arg values
    filter = np.zeros_like(rewardMapF)
    f = 3
    filter[:f+1, :f] = 1
    filter[-f:, -f:] = 1
    filterRewardMap = np.fft.irfft2(rewardMapF * filter)
    ax[1][2].imshow(filterRewardMap, vmax=np.max(np.abs(filterRewardMap)), vmin=-np.max(np.abs(filterRewardMap)))




    plt.show()

def _quaterLine(ax:plt.Axes, xlim, ylim):
    quaters = [(xlim[1] - xlim[0])*i/4 for i in range(1,4)]
    for x in quaters:
        ax.plot((x, x), ylim, color='gray', linestyle='dotted', alpha=0.4)

def _setRewardMapPlot(ax:plt.Axes, Etheta, Stheta):
    """
    Draw optional informations for reward_map type plotting to ax
    """
    ax.plot([0, 40*(modifier0.extends[1]+1)-1], [Etheta*(modifier0.extends[0]+1), Etheta*(modifier0.extends[0]+1)], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax.plot([0, 40*(modifier0.extends[1]+1)-1], [Stheta*(modifier0.extends[0]+1), Stheta*(modifier0.extends[0]+1)], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]], color='gold', linewidth=0.8, linestyle='dotted')
    ax.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0]+20, 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax.plot([20*modifier0.extends[1], 20*modifier0.extends[1]], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax.plot([20*modifier0.extends[1]+40, 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax.set_xlim([0-0.5, 40-0.5])
    ax.set_ylim([20-0.5, 0-0.5])

# -------------------- Main Analysis --------------------

model_path = "C:/Users/dlgkr/Downloads/train1129_2/50model.pt"

base_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/"
save_path = base_path + "data_analysis/testset_model_analysis_imgs/train1129_2/"
test_data_path = base_path + "data/pole_axis_RL_data_batches/unrolled/data_pole_axis_RL_preset_batch_filtered_3.npy"

test_data = np.load(test_data_path)[1:]
gc.collect()

train_data_paths = ["data_pole_axis_RL_preset_batch_0.npy",
                    "data_pole_axis_RL_preset_batch_1.npy",
                    "data_pole_axis_RL_preset_batch_2.npy",
                    "data_pole_axis_RL_preset_batch_filtered_4.npy"]
train_data_list = []
for data_name in train_data_paths[:1]:
    train_data_path = base_path + "data/pole_axis_RL_data_batches/unrolled/" + data_name
    train_data_list.append(np.load(train_data_path)[1:])
train_data = np.concatenate(train_data_list, axis=0)
gc.collect()

print("[Data shapes]")
print("test_Data shape : ", test_data.shape)
print("train_Data shape : ", train_data.shape)
print("-"*20)

np.random.seed(206265)
sample_idx = list(np.random.randint(0, test_data.shape[0]//800, 10))
#sample_idx = list(range(0, test_data.shape[0]//800))
print("sample idx : [", end='')
for idx in sample_idx:
    print(idx, end=' ')
print("]")

modifier0 = RewardMapModifier((0, 0), (3, 2))

losses = np.zeros((len(sample_idx)))
pred_maps = np.zeros((len(sample_idx), 20, 40))
target_maps = np.zeros((len(sample_idx), 20, 40))
gc.collect()

filtered_num = 0
total_num = 0
filtered_percents = []

for num, i in tqdm(enumerate(sample_idx[:]), total=len(sample_idx)):
    state = test_data[i*800, :1006]
    target0 = test_data[i*800:(i+1)*800, -2].reshape(40, 20).T
    target, _ = modifier0.operation(np.expand_dims(target0, axis=-1), None, order=['extend_vert', 'extend_hori', 'blur'])
    target = target[:, :, 0]
    
    pred = np.zeros((20, 40))
    target_maps[num, :, :] = target.copy()

    total_num += 1
    plotter(state, target_maps[num, :, :], i)