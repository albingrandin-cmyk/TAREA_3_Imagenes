import cv2
import matplotlib.pyplot as plt
import numpy as np

# Part A
def generate_degraded_observation(x, h, noise_sigma):
    """
    A function to generate a observation y = h * x + n.

    Parameters:
        x (np.ndarray): ideal real image
        h (np.ndarray): PSF-kernel
        noise_sigma (float): The standarddeviation of the noise

    Returns:
        y (np.ndarray): Distorted image in the spatial domain
        Y (np.ndarray): FFT2 of the distorted image
        H (np.ndarray): Transferfunktion (FFT2) for the centralised PSF
    """
    M, N = x.shape
    m, n = h.shape

    # Normalizing PSF
    h_norm = h / np.sum(h)

    # Padding
    h_padded = np.zeros((M, N), dtype=np.float64)
    h_padded[:m, :n] = h_norm

    # Centralising
    h_padded = np.roll(h_padded, shift=(-(m // 2), -(n // 2)), axis=(0, 1))

    # The transfer function in the frequency domain
    X = np.fft.fft2(x)
    H = np.fft.fft2(h_padded)

    # circular convolution
    Y_blur = X * H
    y_blur = np.real(np.fft.ifft2(Y_blur))

    # Gaussian noise
    noise = np.random.normal(0, noise_sigma, size=(M, N))
    y = y_blur + noise

    # FFT of the noisy observation
    Y = np.fft.fft2(y)

    return y, Y, H


def wiener_filter(Y, H, K):
    """
    A function for the Wiener filter:

    X_hat = (H* / (|H|^2 + K)) * Y
    """
    # H* (complexconjugate)
    H_conj = np.conj(H)

    # |H|^2 (magnitude squared)
    H_sq_mag = np.abs(H) ** 2

    # Wiener-filtret's transfer function W(u,v)
    W = H_conj / (H_sq_mag + K)

    # filtration in frequency domain
    X_hat = W * Y

    # Inverse FFT and real part
    x_hat = np.real(np.fft.ifft2(X_hat))

    return x_hat, X_hat


# Part B
import matplotlib.pyplot as plt
import numpy as np


def apply_H(x, H):
    return np.real(np.fft.ifft2(H * np.fft.fft2(x)))


def apply_HT(r, H):
    return np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(r)))


def tikhonov_gradient_descent(
    y, H, x_true, lam=0.01, tau=0.5, max_iter=200, tol=1e-6
):
    """
    Function which solves Tikonov-regularisation via gradientdescent

    min_x  0.5 * ||H x - y||_2^2 + 0.5 * lam * ||x||_2^2

    Parameters:
        y (np.ndarray): Observerved distorted image
        H (np.ndarray): PSF-transfer function (FFT2)
        x_true (np.ndarray): Idealimage for RMSE-calculation
        lam (float): Regulariserisation parameter for lambda
        tau (float): Steplength
        max_iter (int): Max iterations
        tol (float): The termination criterion based on relative change in cost

    Returns:
        x (np.ndarray): Restored image
        history (dict): Includes cost and rmse per iteration
    """
    # Initialisation with y as a start guess
    x = y.copy()

    num_pixels = x_true.size
    cost_history = []
    rmse_history = []

    for k in range(max_iter):
        # calculating residual: r = H x - y
        r = apply_H(x, H) - y

        # Calculating cost fuction J(x)
        data_fit = 0.5 * np.sum(r**2)
        reg_term = 0.5 * lam * np.sum(x**2)
        J = data_fit + reg_term
        cost_history.append(J)

        # Caculating RMSE comparing to the true image
        rmse = np.sqrt(np.mean((x - x_true) ** 2))
        rmse_history.append(rmse)

        if k > 0 and abs(cost_history[-2] - J) / cost_history[-2] < tol:
            print(f"Convergence at iteration: {k} (Relative tolerance exceeded).\nlambda = {lam}\ntau = {tau}")
            break

        # Calculating gradient: grad = H^T(H x - y) + lam * x
        grad = apply_HT(r, H) + lam * x

        # x^{k+1} = x^k - tau * grad
        x = x - tau * grad

    history = {"cost": np.array(cost_history), "rmse": np.array(rmse_history)}

    return x, history

img_path = "tarea3_files/cameraman.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
x = img.astype(np.float64) / 255.0  # x in [0, 1]

# The PSF-kenel h
psf_path = "tarea3_files/psf.png"
h_img = cv2.imread(psf_path, cv2.IMREAD_GRAYSCALE)
h = h_img.astype(np.float64)


# Experiment 1
# Defining parameters
sigmas = [0.01, 0.05]
K_values = np.logspace(-6, 0, 100)

results = {}

# Experiments on the two noise levels
for sigma in sigmas:
    y, Y, H = generate_degraded_observation(x, h, noise_sigma=sigma)

    rmse_list = []
    for K in K_values:
        x_hat, _ = wiener_filter(Y, H, K)
        x_hat_clipped = np.clip(x_hat, 0.0, 1.0)
        rmse = np.sqrt(np.mean((x_hat_clipped - x) ** 2))
        rmse_list.append(rmse)

    rmse_array = np.array(rmse_list)
    opt_idx = np.argmin(rmse_array)
    opt_K = K_values[opt_idx]
    min_rmse = rmse_array[opt_idx]

    
    x_opt, _ = wiener_filter(Y, H, opt_K)
    results[sigma] = {
        "y": y,
        "rmse": rmse_array,
        "opt_K": opt_K,
        "min_rmse": min_rmse,
        "x_opt": np.clip(x_opt, 0.0, 1.0),
    }

    print(
        f"Noise sigma = {sigma:0.2f} -> Optimal K* = {opt_K:.2e} (Minimum"
        f" RMSE = {min_rmse:.4f})"
    )

plt.figure(figsize=(9, 5))

for sigma in sigmas:
    res = results[sigma]
    plt.semilogx(
        K_values,
        res["rmse"],
        linewidth=2,
        label=(
            f"Noise $\sigma={sigma}$ (Opt. $K^*={res["opt_K"]:.2e}$, RMSE="
            f" {res["min_rmse"]:.4f})"
        ),
    )
    plt.plot(res['opt_K'], res['min_rmse'], 'ro')

plt.title("Wiener-filter: RMSE as a function of $K$", fontsize=13, fontweight='bold')
plt.xlabel("Regularisationparameter $K$ (log-scale)")
plt.ylabel("RMSE")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Experiment_Wiener_RMSE_vs_K.png")
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Row 1: Low noise
axes[0, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title("Ideal image ($x$)")
axes[0, 0].axis("off")

axes[0, 1].imshow(results[sigmas[0]]['y'], cmap="gray", vmin=0, vmax=1)
axes[0, 1].set_title(f"Distorted ($y$), $\sigma={sigmas[0]}$")
axes[0, 1].axis("off")

axes[0, 2].imshow(results[sigmas[0]]["x_opt"], cmap="gray", vmin=0, vmax=1)
axes[0, 2].set_title(
    f"Wiener $K^*={results[sigmas[0]]["opt_K"]:.2e}$\nRMSE ="
    f" {results[sigmas[0]]["min_rmse"]:.4f}"
)
axes[0, 2].axis("off")

# Row 2: High Noise
axes[1, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
axes[1, 0].set_title("Ideal image ($x$)")
axes[1, 0].axis("off")

axes[1, 1].imshow(results[sigmas[1]]['y'], cmap="gray", vmin=0, vmax=1)
axes[1, 1].set_title(f"Distorted ($y$), $\sigma={sigmas[1]}$")
axes[1, 1].axis("off")

axes[1, 2].imshow(results[sigmas[1]]['x_opt'], cmap='gray', vmin=0, vmax=1)
axes[1, 2].set_title(
    f"Wiener $K^*={results[sigmas[1]]["opt_K"]:.2e}$\nRMSE ="
    f" {results[sigmas[1]]["min_rmse"]:.4f}"
)
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("Experiment_Wiener_Rekonstruktioner.png")
plt.show()

# Experiment 2
def perturb_psf_rotation(h, angle=15.0):
    """
    Rotates the PSF-kernel with angle degrees around its center
    """
    m, n = h.shape
    center = (n // 2, m // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    h_pert = cv2.warpAffine(
        h,
        M,
        (n, m),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return h_pert / np.sum(h_pert)

# Creating the faulty PSF-kernel
h_pert = perturb_psf_rotation(h, angle=90.0)

# Generating distorted image with the true PSF
sigma = 0.01
y, Y, H_true = generate_degraded_observation(x, h, noise_sigma=sigma)

# Calculating transfer function for faulty PSF
M_pad, N_pad = x.shape
m, n = h_pert.shape
h_pert_padded = np.zeros((M_pad, N_pad), dtype=np.float64)
h_pert_padded[:m, :n] = h_pert
h_pert_padded = np.roll(
    h_pert_padded, shift=(-(m // 2), -(n // 2)), axis=(0, 1)
)
H_pert = np.fft.fft2(h_pert_padded)

# Weiner-reconstruction with true vs faulty PSF
K_opt = results[0.01]["opt_K"]
x_hat_correct, _ = wiener_filter(Y, H_true, K_opt)
x_hat_pert, _ = wiener_filter(Y, H_pert, K_opt)

x_hat_correct = np.clip(x_hat_correct, 0.0, 1.0)
x_hat_pert = np.clip(x_hat_pert, 0.0, 1.0)

# Calculating RMSE
rmse_correct = np.sqrt(np.mean((x_hat_correct - x) ** 2))
rmse_pert = np.sqrt(np.mean((x_hat_pert - x) ** 2))

print(f"RMSE with correct PSF:{rmse_correct:.4f}")
print(f"RMSE with faulty PSF:{rmse_pert:.4f}")

# Defining zoomed in area (head of cameraman)
crop_y, crop_x = slice(30, 130), slice(100, 200)


fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Upper row: full image
axes[0, 0].imshow(y, cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title(f"Distorted image ($y$)\n$\sigma={sigma}$")
axes[0, 0].axis("off")

axes[0, 1].imshow(x_hat_correct, cmap="gray", vmin=0, vmax=1)
axes[0, 1].set_title(f"Correct PSF\nRMSE = {rmse_correct:.4f}")
axes[0, 1].axis("off")

axes[0, 2].imshow(x_hat_pert, cmap="gray", vmin=0, vmax=1)
axes[0, 2].set_title(f"Faulty PSF (Rotated ${90}^\circ$)\nRMSE = {rmse_pert:.4f}")
axes[0, 2].axis("off")

# Lower row: cropped zoom
axes[1, 0].imshow(y[crop_y, crop_x], cmap="gray", vmin=0, vmax=1)
axes[1, 0].set_title("Zoom: Distorted ($y$)")
axes[1, 0].axis("off")

axes[1, 1].imshow(x_hat_correct[crop_y, crop_x], cmap="gray", vmin=0, vmax=1)
axes[1, 1].set_title("Zoom: Correct PSF")
axes[1, 1].axis("off")

axes[1, 2].imshow(x_hat_pert[crop_y, crop_x], cmap="gray", vmin=0, vmax=1)
axes[1, 2].set_title("Zoom: Faulty PSF")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("Experiment_PSF_Mismatch.png")
plt.show()

# Experiment 3


sigma = 0.01
y, Y, H = generate_degraded_observation(x, h, noise_sigma=sigma)

# The optimal K value wegot in earlier experiments
K_opt = 4.98e-3

# lambdas
lambdas = [
    1e-5,
    K_opt,
    1e-1,
]  # [Underregulating, Optimal (Wiener-K*), overregulating]
tau_fixed = 0.8
max_iter = 150

results_lambda = {}
for lam in lambdas:
    x_res, hist = tikhonov_gradient_descent(
        y, H, x_true=x, lam=lam, tau=tau_fixed, max_iter=max_iter, tol=1e-8
    )
    results_lambda[lam] = {"x_hat": np.clip(x_res, 0.0, 1.0), "hist": hist}


# Theoretical limit for stability: tau < 2 / (1 + lambda) ≈ 1.998
taus = [
    0.05,
    0.9,
    2.02,
]  # [Slow convergens, Optimal, instable/divergence]
results_tau = {}

for tau in taus:
    x_res, hist = tikhonov_gradient_descent(
        y, H, x_true=x, lam=K_opt, tau=tau, max_iter=max_iter, tol=1e-8
    )
    results_tau[tau] = {'x_hat': np.clip(x_res, 0.0, 1.0), "hist": hist}


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: RMSE vs Iterations for different lambdas
for lam in lambdas:
    axes[0, 0].plot(
        results_lambda[lam]["hist"]["rmse"],
        linewidth=2,
        label=f"$\lambda = {lam}$",
    )
axes[0, 0].set_title(
    r"RMSE evolution for different $\lambda$ ($\tau = 0.8$)", fontweight="bold"
)
axes[0, 0].set_xlabel("Iteration $k$")
axes[0, 0].set_ylabel("RMSE")
axes[0, 0].grid(True, linestyle="--", alpha=0.6)
axes[0, 0].legend()

# Plot 2: Cost J(x) vs Iterations for different lambda (Log-scale)
for lam in lambdas:
    axes[0, 1].semilogy(
        results_lambda[lam]["hist"]["cost"],
        linewidth=2,
        label=f"$\lambda = {lam}$",
    )
axes[0, 1].set_title(
    "Costfunction $J(x^k)$ for different $\lambda$", fontweight="bold"
)
axes[0, 1].set_xlabel("Iteration $k$")
axes[0, 1].set_ylabel("Cost $J(x)$ (log-scale)")
axes[0, 1].grid(True, linestyle='--', alpha=0.6)
axes[0, 1].legend()

# Plot 3: RMSE vs Iterations for different Tau
for tau in taus:
    rmse_vals = np.minimum(results_tau[tau]["hist"]["rmse"], 1.0)
    axes[1, 0].plot(rmse_vals, linewidth=2, label=rf"$\tau = {tau}$")
axes[1, 0].set_title(
    r"RMSE evolution for different $\tau$ ($\lambda = 10^{-3}$)", fontweight="bold"
)
axes[1, 0].set_xlabel("Iteration $k$")
axes[1, 0].set_ylabel("RMSE")
axes[1, 0].grid(True, linestyle='--', alpha=0.6)
axes[1, 0].legend()

# Plot 4: Cost J(x) for different Tau (Log-scale)
for tau in taus:
    cost_vals = np.minimum(results_tau[tau]["hist"]["cost"], 1e7)
    axes[1, 1].semilogy(cost_vals, linewidth=2, label=rf"$\tau = {tau}$")
axes[1, 1].set_title(
    r"Cost $J(x^k)$ for different $\tau$", fontweight="bold"
)
axes[1, 1].set_xlabel("Iteration $k$")
axes[1, 1].set_ylabel("Cost $J(x)$ (log-scale)")
axes[1, 1].grid(True, linestyle='--', alpha=0.6)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("Tikhonov_Parameterexperiment.png")
plt.show()


def closed_form_tikhonov(Y, H, lam):
  H_conj = np.conj(H)
  H_sq = np.abs(H) ** 2
  # Adding a very small epsilon incase of division with lambda = 0
  eps = 1e-12 if lam == 0 else 0.0
  W = H_conj / (H_sq + lam + eps)
  X_hat = W * Y
  return np.real(np.fft.ifft2(X_hat))

# Calculating reconstructions
# Case 1: Optimal regularisation (lambda = K*)
x_closed_opt = np.clip(closed_form_tikhonov(Y, H, lam=K_opt), 0.0, 1.0)
x_iter_opt, _ = tikhonov_gradient_descent(
    y, H, x_true=x, lam=K_opt, tau=0.8, max_iter=200, tol=1e-8
)
x_iter_opt = np.clip(x_iter_opt, 0.0, 1.0)

# Case 2: not regulrized (lambda = 0)
x_closed_zero = np.clip(closed_form_tikhonov(Y, H, lam=0.0), 0.0, 1.0)
x_iter_zero, _ = tikhonov_gradient_descent(
    y, H, x_true=x, lam=0.0, tau=0.8, max_iter=100, tol=1e-8
)
x_iter_zero = np.clip(x_iter_zero, 0.0, 1.0)

# RMSE
rmse_closed_opt = np.sqrt(np.mean((x_closed_opt - x) ** 2))
rmse_iter_opt = np.sqrt(np.mean((x_iter_opt - x) ** 2))

rmse_closed_zero = np.sqrt(np.mean((x_closed_zero - x) ** 2))
rmse_iter_zero = np.sqrt(np.mean((x_iter_zero - x) ** 2))

print("Cuantitive comparison(RMSE)")
print(f"Lambda = K* ({K_opt:.2e}):")
print(f"Closed solution: RMSE = {rmse_closed_opt:.4f}")
print(f"Iterative solution: RMSE = {rmse_iter_opt:.4f}")
print(f"Lambda = 0 (not regularized):")
print(f"Closed (Inverse): RMSE = {rmse_closed_zero:.4f}")
print(f"Iterative (100 it): RMSE = {rmse_iter_zero:.4f}")

# Visualisation
crop_y, crop_x = slice(30, 130), slice(100, 200)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Row 1: Distorted image and reference
axes[0, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title("Ideal image ($x$)")
axes[0, 0].axis("off")

axes[0, 1].imshow(y, cmap="gray", vmin=0, vmax=1)
axes[0, 1].set_title(f"Distorted ($y$), $\sigma={sigma}$")
axes[0, 1].axis("off")

axes[0, 2].imshow(y[crop_y, crop_x], cmap="gray", vmin=0, vmax=1)
axes[0, 2].set_title("Zoom $y$")
axes[0, 2].axis("off")

# Row 2: Optimal regularisation (lambda = K*)
axes[1, 0].imshow(x_closed_opt, cmap="gray", vmin=0, vmax=1)
axes[1, 0].set_title(f"Closed Wiener/Tikhonov ($\lambda=K^*$)\nRMSE = {rmse_closed_opt:.4f}")
axes[1, 0].axis("off")

axes[1, 1].imshow(x_iter_opt, cmap="gray", vmin=0, vmax=1)
axes[1, 1].set_title(f"Iterative Tikhonov GD ($\lambda=K^*$)\nRMSE = {rmse_iter_opt:.4f}")
axes[1, 1].axis('off')

axes[1, 2].imshow(x_iter_opt[crop_y, crop_x], cmap="gray", vmin=0, vmax=1)
axes[1, 2].set_title("Zoomed: $\lambda=K^*$ (Iterative)")
axes[1, 2].axis("off")

# Row 3: Not regularized (lambda = 0)
axes[2, 0].imshow(x_closed_zero, cmap="gray", vmin=0, vmax=1)
axes[2, 0].set_title(f"Closed Inverse filter ($\lambda=0$)\nRMSE = {rmse_closed_zero:.4f}")
axes[2, 0].axis('off')

axes[2, 1].imshow(x_iter_zero, cmap="gray", vmin=0, vmax=1)
axes[2, 1].set_title(f"Iterative Landweber ($\lambda=0$, 100 it)\nRMSE = {rmse_iter_zero:.4f}")
axes[2, 1].axis('off')

axes[2, 2].imshow(x_iter_zero[crop_y, crop_x], cmap="gray", vmin=0, vmax=1)
axes[2, 2].set_title("Zoomed: $\lambda=0$ (Iterative)")
axes[2, 2].axis("off")

plt.tight_layout()
plt.savefig("Sammanställning_Sluten_vs_Iterativ.png")
plt.show()