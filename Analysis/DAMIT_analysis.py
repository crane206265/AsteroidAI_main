import numpy as np
import matplotlib.pyplot as plt

import tarfile


file_path = "C:/Users/dlgkr/Downloads/damit-20200211T010301Z.tar.gz"

def data_extract(tar:tarfile.TarFile, lc_path, spin_path):
    # tar : TarFile objects of tar.gzip file
    # lc_path : lc file name
    # spin_path : spin file name
    # 
    # return : time_arr list, intensity_arr list, period(unit of days)  

    spin_member = tar.getmember(spin_path)
    spin_f = tar.extractfile(spin_member)
    spin_content = spin_f.read().decode('utf-8')
    spin_rows = spin_content.split("\n")

    period = float(spin_rows[0].split(' ')[-1])

    lc_member = tar.getmember(lc_path)
    lc_f = tar.extractfile(lc_member)
    lc_content = lc_f.read().decode('utf-8')
    lc_rows = lc_content.split("\n")
    
    num_lc = int(lc_rows[0])
    t_arr_list = []
    intensity_arr_list = []
    _ = lc_rows.pop(0) # delete the first row

    print("Total # of LCs : %02d"%(num_lc))
    for idx in range(num_lc):
        lc_point_num, mode = lc_rows[0].split(" ")
        lc_point_num = int(lc_point_num)
        mode = int(mode)
        _ = lc_rows.pop(0)

        t_arr = np.array([])
        intensity_arr = np.array([])
        for i in range(lc_point_num):
            t_arr = np.append(t_arr, float(lc_rows[0].split(" ")[0]))
            for j in range(1, 10):
                if lc_rows[0].split(" ")[j] != '':
                    intensity_arr = np.append(intensity_arr, float(lc_rows[0].split(" ")[j]))
                    break
            _ = lc_rows.pop(0)

        t_arr_list.append(t_arr)
        intensity_arr_list.append(intensity_arr)

    return t_arr_list, intensity_arr_list, period/24

#############################################
#------------------- GPT --------------------
#############################################
import numpy as np

def phase_fold(t_arr, period):
    phase = (t_arr % period) / period
    order = np.argsort(phase)
    return phase[order], order

def robust_amplitude(y_arr):
    q10, q90 = np.percentile(y_arr, [10, 90])
    return q90 - q10

def largest_phase_gap(phase_sorted):
    if len(phase_sorted) < 2:
        return 1.0
    gaps = np.diff(phase_sorted)
    wrap_gap = (phase_sorted[0] + 1.0) - phase_sorted[-1]
    return max(np.max(gaps), wrap_gap)

def occupied_bin_fraction(phase_sorted, n_bins=20):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(phase_sorted, bins=bins)
    return np.mean(counts > 0)

def local_scatter(phase_sorted, y_sorted, window=0.05):
    """
    phase window 안의 local median 기준 scatter
    """
    scatters = []
    for p, y in zip(phase_sorted, y_sorted):
        dphi = np.abs(phase_sorted - p)
        dphi = np.minimum(dphi, 1.0 - dphi)  # circular distance
        mask = dphi < window
        if np.sum(mask) >= 3:
            y_med = np.median(y_sorted[mask])
            scatters.append(np.abs(y - y_med))
    if len(scatters) == 0:
        return np.inf
    return 1.4826 * np.median(scatters)  # robust scatter estimate

def fourier_fit(phase, y, n_harm=2):
    """
    간단한 low-order Fourier fit
    """
    X = [np.ones_like(phase)]
    for k in range(1, n_harm + 1):
        X.append(np.cos(2 * np.pi * k * phase))
        X.append(np.sin(2 * np.pi * k * phase))
    X = np.column_stack(X)

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_fit = X @ coef
    return y_fit, coef

def filter_one_lc(
    t_arr,
    y_arr,
    period,
    min_points=20,
    min_amp=0.02,
    min_occ=0.6,
    max_gap=0.25,
    max_scatter_ratio=0.35,
    max_nrmse=0.4,
    n_bins=20,
    n_harm=2,
):
    result = {
        "passed": False,
        "reason": [],
    }

    if len(t_arr) != len(y_arr) or len(t_arr) == 0:
        result["reason"].append("invalid_array")
        return result

    if len(t_arr) < min_points:
        result["reason"].append("too_few_points")

    phase_sorted, order = phase_fold(t_arr, period)
    y_sorted = y_arr[order]

    amp = robust_amplitude(y_sorted)
    occ = occupied_bin_fraction(phase_sorted, n_bins=n_bins)
    gap = largest_phase_gap(phase_sorted)
    scat = local_scatter(phase_sorted, y_sorted, window=0.05)

    if amp <= 0:
        scatter_ratio = np.inf
    else:
        scatter_ratio = scat / amp

    # Fourier pre-fit
    try:
        y_fit, coef = fourier_fit(phase_sorted, y_sorted, n_harm=n_harm)
        rmse = np.sqrt(np.mean((y_sorted - y_fit) ** 2))
        nrmse = rmse / amp if amp > 0 else np.inf
    except Exception:
        y_fit = None
        nrmse = np.inf
        result["reason"].append("fit_failed")

    result.update({
        "n_points": len(t_arr),
        "amp": amp,
        "occ": occ,
        "largest_gap": gap,
        "scatter": scat,
        "scatter_ratio": scatter_ratio,
        "nrmse": nrmse,
    })

    if amp < min_amp:
        result["reason"].append("too_small_amplitude")
    if occ < min_occ:
        result["reason"].append("poor_phase_coverage")
    if gap > max_gap:
        result["reason"].append("large_phase_gap")
    if scatter_ratio > max_scatter_ratio:
        result["reason"].append("too_noisy")
    if nrmse > max_nrmse:
        result["reason"].append("bad_fourier_fit")

    result["passed"] = (len(result["reason"]) == 0)
    return result

def reconstruct_uniform_lc(t_arr, y_arr, period, n_harm=3, n_samples=100):
    phase = (t_arr % period) / period

    # 🔥 phase alignment (핵심)
    idx_max = np.argmax(y_arr)
    phase_shift = phase[idx_max]
    phase = (phase - phase_shift) % 1

    order = np.argsort(phase)
    phase = phase[order]
    y = y_arr[order]

    # Fourier fit
    X = [np.ones_like(phase)]
    for k in range(1, n_harm + 1):
        X.append(np.cos(2 * np.pi * k * phase))
        X.append(np.sin(2 * np.pi * k * phase))
    X = np.column_stack(X)

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    phase_uniform = np.linspace(0, 1, n_samples, endpoint=False)

    X_u = [np.ones_like(phase_uniform)]
    for k in range(1, n_harm + 1):
        X_u.append(np.cos(2 * np.pi * k * phase_uniform))
        X_u.append(np.sin(2 * np.pi * k * phase_uniform))
    X_u = np.column_stack(X_u)

    y_uniform = X_u @ coef

    return phase_uniform, y_uniform

# ---------- main ----------

with tarfile.open(file_path, 'r:gz') as tar:
    total_files = tar.getnames() # type : list
    lc_txts0 = [name for name in total_files if name.endswith('lc.txt')]
    spin_txts0 = [name for name in total_files if name.endswith('spin.txt') and name.count('IAU') == 0] #all spin.txt files
    
    asts_idx_list = [int(name.split('/')[2].removeprefix('asteroid_')) for name in lc_txts0]

    # filtering lc_txts if they have corr spin.txt files
    spin_txts = []
    spin_txts_include = []
    for name in spin_txts0:
        ast_idx = int(name.split('/')[2].removeprefix('asteroid_'))
        if ast_idx in asts_idx_list and ast_idx not in spin_txts_include:
            spin_txts.append(name)
            spin_txts_include.append(ast_idx)

    lc_txts = []
    for ast_idx in spin_txts_include:
        lc_txts = lc_txts + [name for name in lc_txts0 if ('asteroid_'+str(ast_idx)+'/') in name]

    # sorting
    lc_txts.sort(key=lambda name: int(name.split('/')[2].removeprefix('asteroid_')))
    spin_txts.sort(key=lambda name: int(name.split('/')[2].removeprefix('asteroid_'))) 
    

    # print all routes of lc
    check_all = False
    if check_all:
        for i, (lc_name, spin_name) in enumerate(zip(lc_txts, spin_txts)):
            print("%04d | "%(i) + str(lc_name) +" | " + str(spin_name))

        plt.hist(asts_idx_list, bins=len(lc_txts))
        plt.title("# of lc.txt Files for each Asteroids")
        plt.show()
    
    idx = 10
    t_arr_list, intensity_arr_list, period = data_extract(tar, lc_txts[idx], spin_txts[idx])

    for lc_idx, (t_arr, intensity_arr) in enumerate(zip(t_arr_list, intensity_arr_list)):
        res = filter_one_lc(t_arr, intensity_arr, period,
                            min_points=20, min_amp=0.02, min_occ=0.65,
                            max_gap=0.20, max_scatter_ratio=0.35, max_nrmse=0.4,
                            n_bins=20, n_harm=4)
        print(lc_idx, res["passed"], res["reason"])
        if not res["passed"]: continue
        print("-"*20)
        print("Period : %.5f (hour) / %.5f (days)"%(period*24, period))
        print("Time Range : [%.5f, %.5f] --> dt = %.5f (%.5f P)"%(t_arr[0], t_arr[-1], t_arr[-1]-t_arr[0], (t_arr[-1]-t_arr[0])/period))
        print(" ")
        
        plt.plot(t_arr, intensity_arr, marker='.', color='royalblue', linestyle='none')
        ylims = plt.ylim()
        plt.plot([t_arr[0]]*2, ylims, linestyle='dotted', color='orangered', alpha=0.7)
        plt.plot([t_arr[0]+period]*2, ylims, linestyle='dotted', color='orangered', alpha=0.7)

        phase_u, y_u = reconstruct_uniform_lc(t_arr, intensity_arr, period, n_harm=4, n_samples=100)
        t_uniform = phase_u * period
        t0_max_idx = np.argmax(intensity_arr)
        t_max_idx = np.argmax(y_u)
        t_uniform = t_uniform+t_arr[t0_max_idx]-t_uniform[t_max_idx] # phase align
        t_uniform = np.where(t_uniform < t_arr[0]+period, t_uniform, t_uniform-period)
        t_uniform = np.where(t_uniform > t_arr[0], t_uniform, t_uniform+period)
        plt.plot(t_uniform, y_u, marker='.', color='firebrick', linestyle='none', alpha=0.7)
        
        title = lc_txts[idx].split('/')[2].replace('asteroid', 'Asteroid') + "_%02dth LC"%(lc_idx)
        save_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/DAMIT_analysis/"

        plt.title(title)
        #plt.show()
        plt.savefig(save_path + title + ".png")
        plt.close()


    

raise NotImplementedError

lc_arr = data[i*800+1, 800:900]
fft_coef_zip = np.abs(np.fft.fft(lc_arr))[:lc_arr.shape[0]//2+1]
fft_coef_zip = np.log10(fft_coef_zip)
log_thr = np.log10(4)#4
if np.all(fft_coef_zip[2] - log_thr >= fft_coef_zip[3:]):
    filtered_data_temp = np.concatenate((filtered_data_temp, data[i*800+1:(i+1)*800+1, :]), axis=0)