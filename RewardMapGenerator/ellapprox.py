import numpy as np
import numpy.linalg as LA
from scipy.optimize import differential_evolution, minimize

import utils
from envs import AsteroidModel


PI = np.pi


# ============================================================
# Forward simulator for ellipsoid LC
# ============================================================

class EllipsoidForwardSimulator:
    def __init__(
        self,
        N_set=(40, 20),
        lc_unit_len=200,
        flux0=10.0,
        mean_radius=10.0,
    ):
        self.Nphi, self.Ntheta = N_set
        self.lc_unit_len = lc_unit_len
        self.flux0 = flux0
        self.mean_radius = mean_radius

    def make_ast_from_ellipsoid(self, axes, tilt):
        """
        axes = (a, b, c)
        tilt = (long, lat)

        Same convention as AsteroidModel:
            long: z-axis rotation
            lat : y-axis rotation in actual code
        """
        ast = AsteroidModel(
            axes=axes,
            N_set=(self.Nphi, self.Ntheta),
            tilt_mode="assigned",
            tilt=tilt,
        )

        ast.base_fitting_generator(mode="ellipsoid")

        # Same mean-radius normalization as AstEnv.step()
        r_mean = np.mean(ast.pos_sph_arr[:, :, 0])
        ast.pos_sph_arr[:, :, 0] *= self.mean_radius / r_mean
        ast.pos_cart_arr *= self.mean_radius / r_mean

        ast.surf_vec_cal()
        return ast

    def _get_R_eps(self, rot_axis):
        rot_axis = np.asarray(rot_axis, dtype=float)

        eps0 = np.arctan2(rot_axis[1], rot_axis[0])
        eps1 = np.arccos(rot_axis[2] / (LA.norm(rot_axis) + 1e-15))

        R_eps = utils.rotArr(-eps1, "y") @ utils.rotArr(-eps0, "z")
        return R_eps

    def _orb2geo(self, vec_orb, rot_angle, R_eps):
        return utils.rotArr(-rot_angle, "z") @ R_eps @ vec_orb

    def simulate_from_ast(self, ast, lc_info):
        """
        lc_info shape:
            [Sx, Sy, Sz, Ex, Ey, Ez, rot_x, rot_y, rot_z]
        """
        lc_info = np.asarray(lc_info, dtype=float)

        Sdir = lc_info[0:3]
        Edir = lc_info[3:6]
        rot_axis = lc_info[6:9]

        R_eps = self._get_R_eps(rot_axis)

        # Same as AstEnv.__lc_gen()
        # correct law
        surf = ast.surf_vec_arr
        area = LA.norm(surf, axis=-1, keepdims=True)
        N_arr = surf / np.sqrt(area + 1e-15)
        N_arr[area[..., 0] < 1e-12] = 0
        N_arr = N_arr.reshape(-1, 3)

        generated_lc = np.zeros(self.lc_unit_len)

        for t in range(self.lc_unit_len):
            theta_t = 2 * PI * t / self.lc_unit_len

            Edir_t = self._orb2geo(Edir.T, theta_t, R_eps)
            Sdir_t = self._orb2geo(Sdir.T, theta_t, R_eps)

            Edir_t = Edir_t / (LA.norm(Edir_t) + 1e-15)
            Sdir_t = Sdir_t / (LA.norm(Sdir_t) + 1e-15)

            generated_lc[t] = (
                utils.ReLU(N_arr @ Edir_t).T
                @ utils.ReLU(N_arr @ Sdir_t)
            )

        return self.flux0 * generated_lc

    def simulate_from_params(self, params, lc_info):
        """
        params = [ba, ca, long, lat]

        ba = b/a
        ca = c/a
        long, lat = ellipsoid orientation angles
        """
        ba, ca, long, lat = params

        a = self.mean_radius
        b = ba * a
        c = ca * a

        ast = self.make_ast_from_ellipsoid(
            axes=(a, b, c),
            tilt=(long, lat),
        )

        return self.simulate_from_ast(ast, lc_info)

    def make_ast_from_params(self, params):
        ba, ca, long, lat = params

        a = self.mean_radius
        b = ba * a
        c = ca * a

        return self.make_ast_from_ellipsoid(
            axes=(a, b, c),
            tilt=(long, lat),
        )


# ============================================================
# Loss utilities
# ============================================================

def lc_mean(lc):
    lc = np.asarray(lc, dtype=float)
    return (np.sum(lc) - 0.5 * (lc[0] + lc[-1])) / len(lc)


def lc_amp(lc, eps=1e-12):
    return max(np.max(lc) - np.min(lc), eps)


def fit_scale(obs, pred, eps=1e-12):
    """
    Find scalar s minimizing ||obs - s pred||^2.
    """
    return np.dot(obs, pred) / (np.dot(pred, pred) + eps)


def lc_loss(obs, pred, use_derivative=True):
    """
    Similar spirit to AstEnv.reward(), but returned as loss.
    Smaller is better.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if len(obs) != len(pred):
        raise ValueError("obs and pred must have the same length.")

    obs_mean = lc_mean(obs)
    pred_mean = lc_mean(pred)

    pred = pred * obs_mean / (pred_mean + 1e-12)

    obs0 = obs - obs_mean
    pred0 = pred - obs_mean

    amp = lc_amp(obs0)

    loss_p = np.mean((obs0 - pred0) ** 2) / (amp ** 2)

    loss_i = np.trapezoid(np.abs(obs0 - pred0)) / (len(obs0) * amp)

    if use_derivative:
        loss_d = np.mean((np.diff(obs0) - np.diff(pred0)) ** 2) / (amp ** 2)
    else:
        loss_d = 0.0

    return 1.2 * loss_p + 0.5 * loss_i + 0.3 * loss_d

def sinusoidal_score(lc, harmonic=2, eps=1e-12):
    """
    Return how close LC is to a single sinusoidal harmonic.
    
    harmonic=1 : one peak per period
    harmonic=2 : double-peaked ellipsoid-like LC
    """
    lc = np.asarray(lc, dtype=float)
    y = lc - lc_mean(lc)

    var = np.mean(y ** 2)
    if var < eps:
        return 0.0

    n = len(y)
    t = np.arange(n)
    phase = 2 * PI * harmonic * t / n

    X = np.stack([
        np.sin(phase),
        np.cos(phase),
    ], axis=1)

    # least-square fit: y ~ A sin + B cos
    coef, *_ = LA.lstsq(X, y, rcond=None)
    y_fit = X @ coef

    score = 1.0 - np.mean((y - y_fit) ** 2) / (var + eps)

    return float(np.clip(score, 0.0, 1.0))


def make_lc_weights_by_sinusoidality(
    obs_lcs,
    harmonic=2,
    w_min=0.5,
    w_max=3.0,
    sharpness=2.0,
):
    """
    More sinusoidal LC gets larger fitting weight.
    """
    scores = np.array([
        sinusoidal_score(lc, harmonic=harmonic)
        for lc in obs_lcs
    ], dtype=float)

    weights = w_min + (w_max - w_min) * (scores ** sharpness)

    return weights

# ============================================================
# Multi-view objective
# ============================================================

class MultiViewEllipsoidFitter:
    def __init__(
        self,
        obs_lcs,
        lc_infos,
        N_set=(40, 20),
        lc_unit_len=200,
        flux0=10.0,
        mean_radius=10.0,
        weights=None,
    ):
        self.obs_lcs = [np.asarray(lc, dtype=float) for lc in obs_lcs]
        self.lc_infos = [np.asarray(info, dtype=float) for info in lc_infos]

        if len(self.obs_lcs) != len(self.lc_infos):
            raise ValueError("obs_lcs and lc_infos must have the same length.")

        self.sim = EllipsoidForwardSimulator(
            N_set=N_set,
            lc_unit_len=lc_unit_len,
            flux0=flux0,
            mean_radius=mean_radius,
        )

        if weights is None:
            self.weights = np.ones(len(self.obs_lcs))
        else:
            self.weights = np.asarray(weights, dtype=float)

    def objective(self, params):
        """
        params = [ba, ca, long, lat]
        Balanced multi-view objective.
        Smaller is better.
        """
        ba, ca, long, lat = params

        # Constraint
        if not (0.05 < ca <= ba <= 1.0):
            return 1e9

        long = long % (2 * PI)
        lat = lat % PI
        params = np.array([ba, ca, long, lat])

        ast = self.sim.make_ast_from_params(params)

        losses = []
        weights = []

        for obs, info, w in zip(self.obs_lcs, self.lc_infos, self.weights):
            pred = self.sim.simulate_from_ast(ast, info)
            loss = lc_loss(obs, pred)

            losses.append(loss)
            weights.append(w)

        losses = np.asarray(losses, dtype=float)
        weights = np.asarray(weights, dtype=float)

        weights = weights / (np.sum(weights) + 1e-12)

        mean_loss = np.sum(weights * losses)

        # weighted max-loss
        # priority has mean 1, so if all weights are equal,
        # this reduces exactly to ordinary max_loss = max(losses)
        priority = weights * len(weights)
        max_loss = np.max(priority * losses)

        std_loss = np.sqrt(np.sum(weights * (losses - mean_loss) ** 2))

        # balanced objective
        return 0.7 * max_loss + 0.3 * mean_loss + 0.2 * std_loss

    def fit(
        self,
        ba_range=(0.5, 1.0),
        ca_range=(0.4, 1.0),
        long_range=(0.0, 2 * PI),
        lat_range=(0.0, PI),
        maxiter_global=25,
        popsize=8,
        local_refine=True,
        local_maxiter=300,
        seed=206265,
        verbose=True,
    ):
        bounds = [
            ba_range,
            ca_range,
            long_range,
            lat_range,
        ]

        de_result = differential_evolution(
            self.objective,
            bounds=bounds,
            maxiter=maxiter_global,
            popsize=popsize,
            tol=1e-4,
            polish=False,
            seed=seed,
            updating="immediate",
            workers=1,
        )

        x_best = de_result.x
        loss_best = de_result.fun

        if verbose:
            print("[Global]")
            print("params:", x_best)
            print("loss  :", loss_best)

        if local_refine:
            local = minimize(
                self.objective,
                x_best,
                method="Nelder-Mead",
                options={
                    "maxiter": local_maxiter,
                    "xatol": 1e-5,
                    "fatol": 1e-6,
                },
            )

            if local.fun < loss_best:
                x_best = local.x
                loss_best = local.fun

            if verbose:
                print("[Local]")
                print("params:", x_best)
                print("loss  :", loss_best)

        ba, ca, long, lat = x_best

        # Canonicalize
        long = long % (2 * PI)
        lat = lat % PI

        x_best = np.array([ba, ca, long, lat])

        ast_best = self.sim.make_ast_from_params(x_best)

        ell_init = np.array([
            self.sim.mean_radius,
            ba * self.sim.mean_radius,
            ca * self.sim.mean_radius,
            long,
            lat,
        ])

        return {
            "params": x_best,
            "ba": ba,
            "ca": ca,
            "long": long,
            "lat": lat,
            "axes": np.array([
                self.sim.mean_radius,
                ba * self.sim.mean_radius,
                ca * self.sim.mean_radius,
            ]),
            "tilt": np.array([long, lat]),
            "ell_init": ell_init,
            "ast": ast_best,
            "loss": loss_best,
        }

    def predict_lcs(self, fit_result):
        ast = fit_result["ast"]

        preds = []
        for info in self.lc_infos:
            preds.append(self.sim.simulate_from_ast(ast, info))

        return preds


# ============================================================
# Convenience function
# ============================================================

def fit_multiview_ellipsoid(
    obs_lcs,
    lc_infos,
    N_set=(40, 20),
    lc_unit_len=200,
    mean_radius=10.0,
    maxiter_global=25,
    popsize=8,
    local_refine=True,
    verbose=True,
    weights=None,   # 추가
):
    fitter = MultiViewEllipsoidFitter(
        obs_lcs=obs_lcs,
        lc_infos=lc_infos,
        N_set=N_set,
        lc_unit_len=lc_unit_len,
        mean_radius=mean_radius,
        weights=weights,   # 추가
    )

    result = fitter.fit(
        maxiter_global=maxiter_global,
        popsize=popsize,
        local_refine=local_refine,
        verbose=verbose,
    )

    preds = fitter.predict_lcs(result)
    result["pred_lcs"] = preds

    return result


def SNRNoise(SNR, lc, mode=True):
    if not mode: return lc
    sigma_A = np.std(lc, axis=-1)
    sigma_N1 = sigma_A / SNR
    sigma_N1 = sigma_N1.reshape(-1, 1)
    noise1 = np.random.normal(0, np.tile(sigma_N1, (1, lc.shape[-1])), lc.shape)

    sigma_B = np.mean(lc, axis=-1)
    sigma_N2 = sigma_B / (SNR*8)
    sigma_N2 = sigma_N2.reshape(-1, 1)
    noise2 = np.random.normal(0, np.tile(sigma_N2, (1, lc.shape[-1])), lc.shape)

    return lc + noise1 + noise2



# ------------------------- MAIN -------------------------

# obs_lcs: list[np.ndarray], each shape = (lc_unit_len,)
# lc_infos: list[np.ndarray], each shape = (9,)

DATA_PATH = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_pole_axis_total_preprocessed.npz"
DATA_PATH = r"C:\Users\dlgkr\OneDrive\Desktop\code\astronomy\asteroid_AI\data\data_pole_axis_total_preprocessed31.npz"
start_idx = 400
final_idx = start_idx + 100
local_i = 50

import matplotlib.pyplot as plt



total_data = np.load(DATA_PATH)
X_full = total_data["lc_arr"]
ell_full = total_data["ell_arr"]
#Y_full = total_data["r_arr"]

X_total = X_full[start_idx:final_idx].copy()
ell_total = ell_full[start_idx:final_idx].copy()
#Y_total = Y_full[start_idx:final_idx].copy()


# ============================================================
# Clean merged-format parser
# ============================================================
# New clean merge format:
#   lc_arr row = [LC_0 ... LC_{m-1} | lc_info_0 ... lc_info_{m-1}]
#   ell_arr row = [a, b, c, long, lat]
# where each LC has length lc_len and each lc_info has length 9.
def extract_clean_merged_sample(x, ell, merge_num, lc_len=100, use_num=None):
    x = np.asarray(x, dtype=float)
    ell = np.asarray(ell, dtype=float)

    if use_num is None:
        use_num = merge_num

    if use_num > merge_num:
        raise ValueError(f"use_num={use_num} cannot exceed merge_num={merge_num}")

    expected_len = merge_num * lc_len + merge_num * 9
    if len(x) != expected_len:
        raise ValueError(
            f"Clean merged format expected len(x)={expected_len}, but got {len(x)}. "
            f"Expected [LCs({merge_num * lc_len}) | infos({merge_num * 9})]. "
            "If this file is old format, use the old parser."
        )

    all_lcs = x[:merge_num * lc_len].reshape(merge_num, lc_len)

    info_start = merge_num * lc_len
    all_lc_infos = x[info_start:info_start + merge_num * 9].reshape(merge_num, 9)

    return all_lcs[:use_num], all_lc_infos[:use_num], ell
    #return all_lcs[1:1+use_num], all_lc_infos[1:1+use_num], ell


while True:
    x = X_total[local_i]
    ell = ell_total[local_i]

    merge_num = 3 
    use_num = 3
    lc_len = 100
    obs_lcs0, lc_infos, ell = extract_clean_merged_sample(
        x=x,
        ell=ell,
        merge_num=merge_num,
        lc_len=lc_len,
        use_num=use_num,
    )

    SNR = 3
    obs_lcs = SNRNoise(SNR, obs_lcs0, mode=True)

    # ==========================================
    # 2. Smoothing (Savitzky-Golay 필터)
    # ==========================================
    # 소행성 피크의 날카로운 형태를 잘 보존하면서 고주파 노이즈를 제거하기 위해 Sav-Gol 필터를 사용합니다.
    # window_length (홀수), polyorder (다항식 차수) 설정
    from scipy.signal import savgol_filter
    window_length = 21#21
    polyorder = 3
    obs_lcs = savgol_filter(obs_lcs, window_length, polyorder)
    
    ell_weights = make_lc_weights_by_sinusoidality(
        obs_lcs,
        harmonic=2,
        w_min=0.5,
        w_max=4.0,
        sharpness=2.0,
    )

    print("ell weights:", ell_weights)

    result = fit_multiview_ellipsoid(
        obs_lcs=obs_lcs,
        lc_infos=lc_infos,
        N_set=(20, 10),
        lc_unit_len=lc_len,
        mean_radius=10.0,
        maxiter_global=15,
        popsize=8,
        local_refine=False,
        weights=ell_weights,
    )

    if result["loss"] < 0.05: #0.15
        print("local i = %d | loss = %f"%(local_i, result["loss"]))
        local_i += 1
    else:
        print("local i = %d | loss = %f"%(local_i, result["loss"]))
        break

for i in range(use_num): plt.plot(obs_lcs[i])
plt.show()

print("ba:", result["ba"])
print("ca:", result["ca"])
print("tilt:", result["tilt"])
print("loss:", result["loss"])



# AstEnv 초기화용
ell_init = (True, result["ell_init"])



def scale_pred_to_obs(obs, pred):
    obs_mean = lc_mean(obs)
    pred_mean = lc_mean(pred)
    return pred * obs_mean / (pred_mean + 1e-12)

pred_lcs = result["pred_lcs"]

plt.figure(figsize=(10, 7))
for i in range(use_num):
    pred_scaled = scale_pred_to_obs(obs_lcs[i], pred_lcs[i])

    plt.plot(obs_lcs[i], label=f"obs #{i}", marker=".", linestyle='')
    plt.plot(obs_lcs0[i], linestyle="solid")

plt.title(f"Multi-view ellipsoid fit | global idx={start_idx + local_i}")
plt.xlabel("time index")
plt.ylabel("flux")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
for i in range(use_num):
    pred_scaled = scale_pred_to_obs(obs_lcs[i], pred_lcs[i])

    plt.plot(obs_lcs0[i], linestyle="--")
    plt.plot(obs_lcs[i], linestyle="", marker=".", label=f"obs #{i}")
    plt.plot(pred_scaled, linestyle="dotted", label=f"ellipsoid fit #{i}")

plt.title(f"Multi-view ellipsoid fit | global idx={start_idx + local_i}")
plt.xlabel("time index")
plt.ylabel("flux")
plt.legend()
plt.tight_layout()
plt.show()



from FinalAgent import AstEnv, MultiAgentRunner, QValueNet_CNN_B1
from FinalAgent_analyzer import MultiAgentAnalyzer
import torch
from torch import nn
import os

MODEL_PATH = "C:/Users/dlgkr/Downloads/train0521_1/40model.pt"
#MODEL_PATH = "C:/Users/dlgkr/Downloads/train0607_1/30model.pt"
SAVE_PATH = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data_analysis/final_agent/"

SAVE_FOLDER = str(start_idx)+"/"
if not os.path.exists(SAVE_PATH+SAVE_FOLDER): # 폴더가 없으면
    os.makedirs(SAVE_PATH+SAVE_FOLDER)
SAVE_PATH = SAVE_PATH + SAVE_FOLDER


N_SET = (40, 20)
LC_UNIT_LEN = 100
REWARD_DOMAIN = [-400, 95]

hidden_dim = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QValueNet_CNN_B1(input_dim=1010, hidden_dim=hidden_dim, activation=nn.ELU, dropout=0.15).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Build environment
envs = []
for i in range(use_num):
    env = AstEnv(
        target_lc=obs_lcs0[i],
        lc_info=lc_infos[i],
        reward_domain=REWARD_DOMAIN,
        N_set=N_SET,
        lc_unit_len=LC_UNIT_LEN,
        #ell_init=(True, result['ell_init'])#+0.3*(np.random.random(result['ell_init'].shape)-0.5))
        ell_init = (True, (1, 1, 1, 0, 0))
    )
    envs.append(env)

# If ellipsoid initialization fails, return flag
if env.ell_err:
    msg = "Ellipsoid Initialization Fails (reward0 = %.3f)"%(env.reward0)
    print(msg)
    raise

runner = MultiAgentRunner(envs, model)
msg = runner.run(local_i, SAVE_PATH)

raise

import gc
BASE_PATH = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/"
train_data_paths = [#"RL_domain1/data_pole_axis_RL_preset_batch_1.npy",
#                    "RL_domain1/data_pole_axis_RL_preset_batch_2.npy",
#                    "RL_domain1/data_pole_axis_RL_preset_batch_3.npy",
#                    "RL_domain1/data_pole_axis_RL_preset_batch_5.npy",
                    "RL_domain1/data_pole_axis_RL_preset_batch_6.npy"] # for corrected scattering law
#train_data_paths = ["RL_domain1/ideal_data_pole_axis_RL_preset_batch_0.npy",
#                    "RL_domain1/ideal_data_pole_axis_RL_preset_batch_1.npy",
#                    "RL_domain1/data_pole_axis_RL_preset_batch_3.npy",
#                    "RL_domain1/data_pole_axis_RL_preset_batch_5.npy",
#                    "RL_domain1/data_pole_axis_RL_preset_batch_6.npy"] # for corrected scattering law
train_data_list = []
for data_name in train_data_paths[:]:
    train_data_path = BASE_PATH + "data/pole_axis_RL_data_batches/unrolled/" + data_name
    print(train_data_path)
    train_data_list.append(np.load(train_data_path)[1:])
train_data = np.concatenate(train_data_list, axis=0)
gc.collect()



runner = MultiAgentAnalyzer(envs, model)
Stheta_distn, Etheta_distn, cosA_distn = runner.dirDistn(train_data) 


block_idx = np.arange(0, train_data.shape[0], 800)
lc_pred_dataset   = train_data[block_idx, 900:1000]
lc_target_dataset = train_data[block_idx, 800:900]
delta_lc_dataset = lc_pred_dataset - lc_target_dataset    
predF_dataset           = np.fft.fft(lc_pred_dataset, axis=1)[:, 1:LC_UNIT_LEN//2+1]
targetF_dataset         = np.fft.fft(lc_target_dataset, axis=1)[:, 1:LC_UNIT_LEN//2+1]
predF_dataset_mag       = np.log10(np.abs(predF_dataset))
predF_dataset_mag       -= predF_dataset_mag[1]
targetF_dataset_mag     = np.log10(np.abs(targetF_dataset))
targetF_dataset_mag     -= targetF_dataset_mag[1]

msg = runner.run(local_i, SAVE_PATH, dataset=(train_data, Stheta_distn, Etheta_distn, cosA_distn, predF_dataset_mag, targetF_dataset_mag))