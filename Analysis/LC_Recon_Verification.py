import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


# -----------------------------
# Basic geometry utilities
# -----------------------------
def rotArr(angle, axis):
    if axis == "x" or axis == 0:
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])
    elif axis == "y" or axis == 1:
        return np.array([
            [ np.cos(angle), 0, np.sin(angle)],
            [0,              1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == "z" or axis == 2:
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
    else:
        raise ValueError("axis must be x/y/z or 0/1/2")


def sph2cart(sph_coord):
    r, phi, theta = sph_coord
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])


def relu(x):
    return x * (x > 0)


# -----------------------------
# Rebuild asteroid surface from r_arr
# -----------------------------
def build_surface_vectors_from_rarr(
    r_arr,
    N_set=(40, 20),
    south_mode="last_row_mean",
):
    """
    Rebuild surf_vec_arr from flattened r_arr.

    r_arr is assumed to have shape (Nphi*Ntheta,)
    corresponding to pos_sph_arr[:-1, :-1, 0].

    Since the true code uses Ntheta+1 rows for vertices,
    we must reconstruct the missing theta=pi row.
    """
    Nphi, Ntheta = N_set
    dphi = 2 * np.pi / Nphi
    dtheta = np.pi / Ntheta

    r_grid = np.asarray(r_arr, dtype=float).reshape(Nphi, Ntheta)

    pos_sph_arr = np.zeros((Nphi + 1, Ntheta + 1, 3), dtype=float)
    pos_cart_arr = np.zeros((Nphi + 1, Ntheta + 1, 3), dtype=float)

    if south_mode == "last_row_mean":
        south_r = float(np.mean(r_grid[:, -1]))
    elif south_mode == "last_row_by_phi":
        south_r = None
    else:
        raise ValueError("south_mode must be 'last_row_mean' or 'last_row_by_phi'")

    for i in range(Nphi):
        for j in range(Ntheta + 1):
            phi_ij = (j % 2) * (dphi / 2) + i * dphi
            theta_ij = j * dtheta

            if j < Ntheta:
                r_ij = r_grid[i, j]
            else:
                if south_mode == "last_row_mean":
                    r_ij = south_r
                elif south_mode == "last_row_by_phi":
                    r_ij = r_grid[i, -1]

            pos_sph_arr[i, j] = np.array([r_ij, phi_ij, theta_ij])
            pos_cart_arr[i, j] = sph2cart(pos_sph_arr[i, j])

            if i == 0:
                pos_sph_arr[Nphi, j] = pos_sph_arr[0, j]
                pos_cart_arr[Nphi, j] = pos_cart_arr[0, j]

    surf_vec_arr = np.zeros((Nphi, Ntheta, 2, 3), dtype=float)

    # Same logic as Asteroid_Model.surf_vec_cal()
    for i in range(Nphi):
        for j in range(Ntheta):
            if j % 2 == 0:
                v11 = pos_cart_arr[i + 1, j]     - pos_cart_arr[i, j + 1]
                v12 = pos_cart_arr[i + 1, j + 1] - pos_cart_arr[i, j + 1]
                v21 = pos_cart_arr[i + 1, j]     - pos_cart_arr[i, j]
                v22 = pos_cart_arr[i, j + 1]     - pos_cart_arr[i, j]
            else:
                v11 = pos_cart_arr[i + 1, j + 1] - pos_cart_arr[i, j]
                v12 = pos_cart_arr[i, j + 1]     - pos_cart_arr[i, j]
                v21 = pos_cart_arr[i + 1, j]     - pos_cart_arr[i, j]
                v22 = pos_cart_arr[i + 1, j + 1] - pos_cart_arr[i, j]

            surf_vec_arr[i, j, 0] = -0.5 * np.cross(v11, v12)
            surf_vec_arr[i, j, 1] = -0.5 * np.cross(v21, v22)

    return surf_vec_arr, pos_sph_arr, pos_cart_arr


# -----------------------------
# Regenerate LC from r_arr + lc_info
# -----------------------------
def regenerate_lc_from_rarr(
    r_arr,
    lc_info,
    lc_unit_len=100,
    N_set=(40, 20),
    flux0=1.0,
    south_mode="last_row_mean",
):
    """
    lc_info format:
        [Sx, Sy, Sz, Ex, Ey, Ez, rot_x, rot_y, rot_z]

    This follows the envs.py convention:
        orb2geo(vec, theta_t) = Rz(-theta_t) @ R_eps @ vec
    """
    lc_info = np.asarray(lc_info, dtype=float)
    Sdir = lc_info[0:3]
    Edir = lc_info[3:6]
    rot_axis = lc_info[6:9]

    rot_axis = rot_axis / (LA.norm(rot_axis) + 1e-15)

    initial_phi = np.arctan2(rot_axis[1], rot_axis[0])
    initial_theta = np.arccos(np.clip(rot_axis[2], -1.0, 1.0))
    R_eps = rotArr(-initial_theta, "y") @ rotArr(-initial_phi, "z")

    surf_vec_arr, _, _ = build_surface_vectors_from_rarr(
        r_arr,
        N_set=N_set,
        south_mode=south_mode
    )

    # Same vectorized Lambert-like formula as envs.py __lc_gen()
    surf = surf_vec_arr
    area = LA.norm(surf, axis=-1, keepdims=True)
    N_arr = surf / np.sqrt(area + 1e-15)
    N_arr[area[..., 0] < 1e-12] = 0
    N_arr = N_arr.reshape(-1, 3)

    generated_lc = np.zeros(lc_unit_len, dtype=float)

    for t in range(lc_unit_len):
        theta_t = 2 * np.pi * t / lc_unit_len

        Edir_t = rotArr(-theta_t, "z") @ R_eps @ Edir
        Sdir_t = rotArr(-theta_t, "z") @ R_eps @ Sdir

        Edir_t = Edir_t / (LA.norm(Edir_t) + 1e-15)
        Sdir_t = Sdir_t / (LA.norm(Sdir_t) + 1e-15)

        generated_lc[t] = relu(N_arr @ Edir_t).T @ relu(N_arr @ Sdir_t)

    return flux0 * generated_lc


# -----------------------------
# Error metrics
# -----------------------------
def lc_mean_scale(pred, target, eps=1e-15):
    return pred * (np.mean(target) / (np.mean(pred) + eps))


def normalized_rmse(pred, target, eps=1e-15):
    amp = np.max(target) - np.min(target)
    return np.sqrt(np.mean((pred - target) ** 2)) / (amp + eps)


def best_circular_shift(pred, target):
    """
    Finds k such that np.roll(pred, k) best matches target.
    Useful for detecting time/phase shift.
    """
    best_k = 0
    best_err = np.inf

    for k in range(len(target)):
        pred_k = np.roll(pred, k)
        err_k = normalized_rmse(pred_k, target)
        if err_k < best_err:
            best_err = err_k
            best_k = k

    return best_k, best_err

def extract_from_preprocessed_row(x, ell, lc_len=100, merge_num=1):
    """
    Works for your Preprocessing Data.py output.

    x = data["lc_arr"][idx]
    ell = data["ell_arr"][idx]

    For merge_num=1:
        x = [LC | Sdir+Edir+rot_axis]

    For merge_num=m:
        x = [LC_0 ... LC_{m-1} | info_0 ... info_{m-2} | last_info]
        where info_k originally means [Sdir+Edir+rot_axis+ell]
        but the final ell of the last block was saved separately as ell_arr.
    """
    x = np.asarray(x, dtype=float)
    ell = np.asarray(ell, dtype=float)

    expected_len = merge_num * (lc_len + 9 + 5) - 5
    if len(x) != expected_len:
        raise ValueError(
            f"Unexpected row length: len(x)={len(x)}, expected={expected_len}. "
            f"Check lc_len or merge_num."
        )

    lcs = x[:merge_num * lc_len].reshape(merge_num, lc_len)

    lc_infos = []
    for k in range(merge_num):
        start = merge_num * lc_len + k * (9 + 5)
        lc_info_k = x[start:start + 9]
        lc_infos.append(lc_info_k)

    return lcs, np.asarray(lc_infos), ell



# -----------------------------
# Main verification function
# -----------------------------
def verify_preprocessed_data(
    preprocessed_path,
    indices=None,
    N_set=(40, 20),
    lc_unit_len=100,
    flux0=1.0,
    south_mode="last_row_mean",
    plot=True,
):
    """
    preprocessed npz format assumed:
        lc_arr : [LC(100), lc_info(9)]
        r_arr  : [Nphi*Ntheta]
        ell_arr
    """
    data = np.load(preprocessed_path)

    X = data["lc_arr"]
    R = data["r_arr"]

    if indices is None:
        indices = list(range(min(10, len(X))))

    results = []

    for idx in indices:
        x = X[idx]
        target_lc = x[:lc_unit_len]
        lc_info = x[lc_unit_len:lc_unit_len + 9]
        r_arr = R[idx]

        pred_lc = regenerate_lc_from_rarr(
            r_arr=r_arr,
            lc_info=lc_info,
            lc_unit_len=lc_unit_len,
            N_set=N_set,
            flux0=flux0,
            south_mode=south_mode
        )

        pred_lc_scaled = lc_mean_scale(pred_lc, target_lc)

        raw_err = normalized_rmse(pred_lc, target_lc)
        scaled_err = normalized_rmse(pred_lc_scaled, target_lc)

        best_k_raw, best_err_raw = best_circular_shift(pred_lc, target_lc)
        best_k_scaled, best_err_scaled = best_circular_shift(pred_lc_scaled, target_lc)

        results.append({
            "idx": idx,
            "raw_nrmse": raw_err,
            "mean_scaled_nrmse": scaled_err,
            "best_shift_raw": best_k_raw,
            "best_shift_raw_nrmse": best_err_raw,
            "best_shift_scaled": best_k_scaled,
            "best_shift_scaled_nrmse": best_err_scaled,
        })

        print(
            f"[idx {idx}] "
            f"raw={raw_err:.6g}, "
            f"scaled={scaled_err:.6g}, "
            f"best_shift_scaled={best_k_scaled}, "
            f"best_shift_scaled_err={best_err_scaled:.6g}"
        )

        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(target_lc, label="saved LC", linewidth=2)
            plt.plot(pred_lc_scaled, "--", label="regenerated from r_arr + lc_info")
            plt.plot(
                np.roll(pred_lc_scaled, best_k_scaled),
                ":",
                label=f"best shifted regenerated, shift={best_k_scaled}"
            )
            plt.title(f"LC regeneration check | idx={idx}")
            plt.xlabel("time index")
            plt.ylabel("flux")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return results

def infer_merge_num_from_row_length(row_len, lc_len=100):
    unit = lc_len + 9 + 5
    val = (row_len + 5) / unit

    if abs(val - round(val)) > 1e-8:
        raise ValueError(
            f"Cannot infer merge_num: row_len={row_len}, lc_len={lc_len}, "
            f"(row_len+5)/(lc_len+14)={val}"
        )

    return int(round(val))


def extract_from_preprocessed_row_auto(x, ell, lc_len=100, merge_num=None):
    x = np.asarray(x, dtype=float)
    ell = np.asarray(ell, dtype=float)

    if merge_num is None:
        merge_num = infer_merge_num_from_row_length(len(x), lc_len=lc_len)
        print(f"[auto] inferred merge_num = {merge_num}")

    expected_len = merge_num * (lc_len + 9 + 5) - 5
    if len(x) != expected_len:
        actual_merge_num = infer_merge_num_from_row_length(len(x), lc_len=lc_len)
        raise ValueError(
            f"Unexpected row length: len(x)={len(x)}, expected={expected_len}. "
            f"You passed merge_num={merge_num}, but this file looks like "
            f"merge_num={actual_merge_num}."
        )

    lcs = x[:merge_num * lc_len].reshape(merge_num, lc_len)

    lc_infos = []
    info_start0 = merge_num * lc_len
    for k in range(merge_num):
        start = info_start0 + k * (9 + 5)
        lc_info_k = x[start:start + 9]
        lc_infos.append(lc_info_k)

    return lcs, np.asarray(lc_infos), ell, merge_num

def extract_from_clean_merged_row(x, ell, lc_len=100, merge_num=3):
    x = np.asarray(x, dtype=float)
    ell = np.asarray(ell, dtype=float)

    expected_len = merge_num * lc_len + merge_num * 9
    if len(x) != expected_len:
        raise ValueError(
            f"len(x)={len(x)}, expected={expected_len}. "
            f"This parser is for clean merged format."
        )

    lcs = x[:merge_num * lc_len].reshape(merge_num, lc_len)

    info_start = merge_num * lc_len
    lc_infos = x[info_start:info_start + merge_num * 9].reshape(merge_num, 9)

    return lcs, lc_infos, ell


def verify_and_plot_merged_idx(
    preprocessed_path,
    idx,
    merge_num=None,
    lc_len=100,
    N_set=(40, 20),
    flux0=1.0,
    south_mode="last_row_mean",
    show_shifted=True,
    save_path=None,
):
    data = np.load(preprocessed_path) if type(preprocessed_path) is str else preprocessed_path
    X = data["lc_arr"]
    R = data["r_arr"]
    ELL = data["ell_arr"]

    x = X[idx]
    r_arr = R[idx]
    ell = ELL[idx]

    lcs, lc_infos, ell = extract_from_clean_merged_row(
        x,
        ell,
        lc_len=lc_len,
        merge_num=merge_num
    )

    results = []

    fig, axes = plt.subplots(
        merge_num,
        1,
        figsize=(12, 3.2 * merge_num),
        sharex=True
    )

    if merge_num == 1:
        axes = [axes]

    for k in range(merge_num):
        target_lc = lcs[k]
        lc_info = lc_infos[k]

        pred_lc = regenerate_lc_from_rarr(
            r_arr=r_arr,
            lc_info=lc_info,
            lc_unit_len=lc_len,
            N_set=N_set,
            flux0=flux0,
            south_mode=south_mode
        )

        pred_lc_scaled = lc_mean_scale(pred_lc, target_lc)
        err = normalized_rmse(pred_lc_scaled, target_lc)
        shift, shift_err = best_circular_shift(pred_lc_scaled, target_lc)
        shifted_pred = np.roll(pred_lc_scaled, shift)

        results.append({
            "idx": idx,
            "k": k,
            "merge_num": merge_num,
            "err": err,
            "best_shift": shift,
            "shift_err": shift_err,
            "target_lc": target_lc,
            "pred_lc": pred_lc,
            "pred_lc_scaled": pred_lc_scaled,
            "shifted_pred_lc": shifted_pred,
            "lc_info": lc_info,
        })

        ax = axes[k]
        ax.plot(target_lc, label="saved LC", linewidth=2)
        ax.plot(pred_lc_scaled, "--", label="regen scaled")

        if show_shifted:
            ax.plot(
                shifted_pred,
                ":",
                label=f"regen scaled + roll({shift})"
            )

        ax.set_title(
            f"idx={idx}, merged LC #{k} | "
            f"NRMSE={err:.4g}, best_shift={shift}, shifted_NRMSE={shift_err:.4g}"
        )
        ax.set_ylabel("flux")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time index")

    fig.suptitle(
        f"LC regeneration check from r_arr + lc_info | idx={idx}, merge_num={merge_num}",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")

    #plt.show()
    plt.close()

    print("\nSummary")
    for res in results:
        print(
            f"idx={res['idx']}, LC #{res['k']}: "
            f"err={res['err']:.6g}, "
            f"best_shift={res['best_shift']}, "
            f"shifted_err={res['shift_err']:.6g}"
        )

    return results


def cross_check_merged_idx(
    preprocessed_path,
    idx,
    merge_num=3,
    lc_len=100,
    N_set=(40, 20),
    flux0=1.0,
    south_mode="last_row_mean",
    plot=True,
):
    data = np.load(preprocessed_path)
    X = data["lc_arr"]
    R = data["r_arr"]
    ELL = data["ell_arr"]

    x = X[idx]
    r_arr = R[idx]
    ell = ELL[idx]

    lcs, lc_infos, ell = extract_from_clean_merged_row(
        x,
        ell,
        lc_len=lc_len,
        merge_num=merge_num
    )

    err_mat = np.zeros((merge_num, merge_num))
    shift_mat = np.zeros((merge_num, merge_num), dtype=int)
    shift_err_mat = np.zeros((merge_num, merge_num))

    pred_cache = []

    for info_j in range(merge_num):
        pred_lc = regenerate_lc_from_rarr(
            r_arr=r_arr,
            lc_info=lc_infos[info_j],
            lc_unit_len=lc_len,
            N_set=N_set,
            flux0=flux0,
            south_mode=south_mode
        )
        pred_cache.append(pred_lc)

    for target_i in range(merge_num):
        target_lc = lcs[target_i]

        for info_j in range(merge_num):
            pred_lc = pred_cache[info_j]
            pred_lc_scaled = lc_mean_scale(pred_lc, target_lc)

            err = normalized_rmse(pred_lc_scaled, target_lc)
            shift, shift_err = best_circular_shift(pred_lc_scaled, target_lc)

            err_mat[target_i, info_j] = err
            shift_mat[target_i, info_j] = shift
            shift_err_mat[target_i, info_j] = shift_err

    print("\nNRMSE matrix: rows = target LC, cols = lc_info used")
    print(err_mat)

    print("\nBest shift matrix")
    print(shift_mat)

    print("\nShifted NRMSE matrix")
    print(shift_err_mat)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        im0 = axes[0].imshow(err_mat)
        axes[0].set_title("NRMSE: target LC i vs lc_info j")
        axes[0].set_xlabel("lc_info j")
        axes[0].set_ylabel("target LC i")
        axes[0].set_xticks(range(merge_num))
        axes[0].set_yticks(range(merge_num))
        for i in range(merge_num):
            for j in range(merge_num):
                axes[0].text(j, i, f"{err_mat[i,j]:.3g}", ha="center", va="center")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(shift_err_mat)
        axes[1].set_title("Best-shift NRMSE")
        axes[1].set_xlabel("lc_info j")
        axes[1].set_ylabel("target LC i")
        axes[1].set_xticks(range(merge_num))
        axes[1].set_yticks(range(merge_num))
        for i in range(merge_num):
            for j in range(merge_num):
                axes[1].text(
                    j, i,
                    f"{shift_err_mat[i,j]:.3g}\nsh={shift_mat[i,j]}",
                    ha="center",
                    va="center"
                )
        fig.colorbar(im1, ax=axes[1])

        fig.suptitle(f"Cross-check matrix | idx={idx}")
        fig.tight_layout()
        plt.show()

    return err_mat, shift_mat, shift_err_mat



PREPROCESSED_PATH = r"C:\Users\dlgkr\OneDrive\Desktop\code\astronomy\asteroid_AI\data\data_pole_axis_total_preprocessed31.npz"
datapp = np.load(PREPROCESSED_PATH)

idx = 404
for idx in range(200, 300):
    merge_num = 3
    lc_len = 100

    #data = np.load(PREPROCESSED_PATH)
    #
    #print(data.files)
    #
    #if "source_indices" in data.files:
    #    src = data["source_indices"][idx]
    #    print("source_indices:", src)
    #    print("asteroid block:", src // 10)
    #    print("local LC index:", src % 10)
    #else:
    #    print("source_indices not saved")

    results = verify_and_plot_merged_idx(
        #preprocessed_path=PREPROCESSED_PATH,
        preprocessed_path=datapp,
        idx=idx,
        merge_num=merge_num,      # 헷갈리면 None으로 둬도 자동 추정함
        lc_len=lc_len,
        N_set=(40, 20),
        flux0=1.0,
        south_mode="last_row_mean",
        show_shifted=True,
        save_path=r"C:\Users\dlgkr\OneDrive\Desktop\code\astronomy\asteroid_AI\data_analysis\recon_valid"+"/idx"+str(idx)+".png"            # 저장하려면 r"...\check_idx240.png"
    )

raise
err_mat, shift_mat, shift_err_mat = cross_check_merged_idx(
    preprocessed_path=PREPROCESSED_PATH,
    idx=idx,
    merge_num=3,
    lc_len=100,
    N_set=(40, 20),
    flux0=1.0,
    south_mode="last_row_mean",
    plot=True
)



raise
# -----------------------------------------

SINGLE_PATH = r"C:\Users\dlgkr\OneDrive\Desktop\code\astronomy\asteroid_AI\data\data_pole_axis_total_preprocessed.npz"

results_single = verify_preprocessed_data(
    preprocessed_path=SINGLE_PATH,
    indices=[23, 26, 29],
    N_set=(40, 20),
    lc_unit_len=100,
    flux0=1.0,
    south_mode="last_row_mean",
    plot=True
)