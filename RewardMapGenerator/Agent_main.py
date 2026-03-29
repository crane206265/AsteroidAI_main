import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import gc

from FinalAgent import AstEnv, AgentRunner, QValueNet_CNN_B1

#################### Main Code for Agent Running ####################
    # UPDATED : 26.02.22
#############################################################################


#################### Settings ####################
DATA_PATH = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_pole_axis_total_preprocessed.npz"
SAVE_PATH = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data_analysis/final_agent/"
MODEL_PATH = "C:/Users/dlgkr/Downloads/train0208_1/40model.pt"

N_set = (40, 20)
lightcurve_unit_len = 100
reward_domain = [-300, 60]

# Global Variables for each Worker Processors
X_total = None
ell_total = None
START_IDX = None
REWARD_DOMAIN = None
N_SET = None
LC_UNIT_LEN = None

hidden_dim = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################### Functions ####################

def init(data_path, start_idx, final_idx, reward_domain, n_set, lc_unit_len, hidden_dim=4069):
    global X_total, ell_total, START_IDX, REWARD_DOMAIN, N_SET, LC_UNIT_LEN, model

    START_IDX = start_idx
    REWARD_DOMAIN = reward_domain
    N_SET = n_set
    LC_UNIT_LEN = lc_unit_len

    total_data = np.load(data_path)
    X_full = total_data["lc_arr"]
    ell_full = total_data["ell_arr"]

    X_total = X_full[start_idx:final_idx].copy()
    ell_total = ell_full[start_idx:final_idx].copy()

    model = QValueNet_CNN_B1(input_dim=1010, hidden_dim=hidden_dim, activation=nn.ELU, dropout=0.15).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("MODEL LOADED")


def run_one(local_i, save_path):
    global X_total, ell_total, START_IDX, REWARD_DOMAIN, N_SET, LC_UNIT_LEN, model

    global_i = START_IDX + local_i
    x = X_total[local_i]
    ell = ell_total[local_i]

    target_lc = x[:-9]
    lc_info = x[-9:]

    # FT Filter
    fft_coef_zip = np.abs(np.fft.fft(target_lc))[:target_lc.shape[0]//2+1]
    fft_coef_zip = np.log10(fft_coef_zip)
    log_thr = np.log10(2)#4
    if not np.all(fft_coef_zip[2] - log_thr >= fft_coef_zip[3:]):
        msg = "Filtered by FT Filter"
        print(msg)
        return msg

    # Build environment
    env = AstEnv(
        target_lc=target_lc,
        lc_info=lc_info,
        reward_domain=REWARD_DOMAIN,
        N_set=N_SET,
        lc_unit_len=LC_UNIT_LEN,
        ell_init=(True, ell)
    )

    # If ellipsoid initialization fails, return flag
    if env.ell_err:
        msg = "Ellipsoid Initialization Fails (reward0 = %.3f)"%(env.reward0)
        print(msg)
        return msg

    runner = AgentRunner(env, model)
    msg = runner.run(global_i, save_path)

    return msg

def main():
    # ---- Range of global indices to compute ----
    start_idx = 740#740#1300       # inclusive
    final_idx = 1200#1500     # exclusive (global index)
    num_samples = final_idx - start_idx

    init(DATA_PATH, start_idx, final_idx, reward_domain, N_set, lightcurve_unit_len, hidden_dim)
    

    for i in range(num_samples):
        print("\n---------- Current idx : "+str(i+start_idx)+" ----------")
        msg = run_one(i, SAVE_PATH)
        with open("AgentRecords.txt", "a", encoding="utf-8") as f:
            f.write("Current idx : "+str(i+start_idx)+" ||  "+msg+"\n")
    

if __name__ == "__main__":
    # Required on Windows for multiprocessing to work correctly
    main()
