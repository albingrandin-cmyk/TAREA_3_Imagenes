import cv2
import matplotlib.pyplot as plt
import numpy as np

# Implementera en komplett bearbetningskedja
"""
Padding:
- What is it?
    We are changing the dimensions of the foto from MxN to PxQ.
    We can do this either by "zero padding" which means filling
    the new rows and columns with constant zero values. "Reflect",
    which means we mirror the corner pixels outwards. "Replicate",
    which means we copy the value of the edge pixels.
- Why do we do it?
    To avoid circular convolution overlap (wraparound artifacts).
    To create a buffer for the filter's impulse response. By increaseing
    the size of the image the image artifacts occur in the padded
    area which we can easily cut away from the image. 
"""

def apply_padding(img, pad_factor=2.0, mode="reflect", constant_value=0):
    """
    A function to add padding to an image where you can select
    amount of padding and the mode of padding.

    Parameters:
        img(np.ndarray): The image we want to add padding to
        pad_factor(float): The multiplicative factor of padding
        mode(string): The mode of padding
        constant_value(int): If the mode is constant this is the
        constant value for the pixels added.

    Return:
        img_padded(np.ndarray): The padded image
        pad_info(tuple(top,bottom,left,right)): The amount of
        padding in each direction
    """



    # defining the padding
    M, N = img.shape

    M_pad = int(M*pad_factor)
    N_pad = int(N*pad_factor)

    pad_m = M_pad - M
    pad_n = N_pad - N

    # mEqually distributed
    pad_top = pad_m // 2
    pad_bottom = pad_m - pad_top
    pad_left = pad_n // 2
    pad_right = pad_n - pad_left

    # defining pad_width for np.pad
    pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))

    # Adding padding
    if mode == "constant":
        img_padded = np.pad(img, pad_width, mode="constant", constant_values=constant_value)
    else:
        img_padded = np.pad(img, pad_width, mode=mode)

    pad_info = (pad_top, pad_bottom, pad_left, pad_right)
    return img_padded, pad_info


def crop_to_original(img_padded, pad_info):
  """
  Returning the padded image to its original dimensions

  Parameters:
    img_padded(np.ndarray): The padded image
    pad_info(tuple(top,bottom,left,right)): The amount of
    padding in each direction

Return:
    original_img(np.ndarray): The image after we have croped
    away the padding
  """
  pad_top, pad_bottom, pad_left, pad_right = pad_info
  M_padded, N_padded = img_padded.shape
  original_img = img_padded[pad_top : M_padded - pad_bottom, pad_left : N_padded - pad_right]
  return original_img

"""
2D-FFT:
We are going to the spatial domain to the frequency domain to
to have easier calculations. We are also shifting the zero frequency (0,0)
to (M_pad/2, N_pad/2) to better visualize and have easier calculations
when applying the Notch filter
"""

def compute_fft(img_padded):
    """
    Calculating the 2D-FFT and shifting the zero frequency

    Parameter:
        img_padded(np.ndarray): The padded image

    Return:
        F_shifted(np.ndarray): The complex frequency spectrum
        with the zero frequency (DC-component) at the centrum
        magnitude_spectrum(np.ndarray): The logaritmized magnitude
        spectrum for visualization
    """


    F = np.fft.fft2(img_padded)
    F_shifted = np.fft.fftshift(F)
    magnitude_spectrum = np.log(1 + np.abs(F_shifted))

    return F_shifted, magnitude_spectrum

"""
Notch-reject Filter: Design to attune or fully block specific
frequencies and let all others through. The noise occurs as
symmetrical pairs (u0, v0) and (-u0, -v0). 
The Guassian Notch-reject Filter: Creates a smother transition
to avoid halo-artifacts. If there are multiple noise peaks we
multiply the transferfuctions with eachother
"""

def notch_filter(shape, centers, D0, filter_type="gaussian"):

    """
    Function to implement a Gaussian and a ideal Notch-reject filter

    Parameters:
        shape(tuple(M_pad, N_pad)): The shape of the padded image
        centers(list of tuples[(u0,v0)...]): Relative coordinates
        for the noise from centrum
        D0(float): The filter ratio
        filter_type(string): the type of nothc-reject filter we want
    Return:
        H(np.ndarray): The filter mask
    """


    M_pad, N_pad = shape

    # Finding centrum after fft
    v_c = M_pad // 2  # Y-row
    u_c = N_pad // 2  # X-row

    # Creating coordinate sysyem to avoid for loops
    u_coords = np.arange(N_pad)
    v_coords = np.arange(M_pad)
    U, V = np.meshgrid(u_coords, v_coords)

    # 
    H = np.ones((M_pad, N_pad), dtype=np.float64)

    for u0, v0 in centers:
        # The positions for noise
        u1, v1 = u_c + u0, v_c + v0
        u2, v2 = u_c - u0, v_c - v0

        # The distance from every pixel and the notch-centrum
        D1 = np.sqrt((U - u1) ** 2 + (V - v1) ** 2)
        D2 = np.sqrt((U - u2) ** 2 + (V - v2) ** 2)

        if filter_type == "gaussian":
            H1 = 1.0 - np.exp(-0.5 * (D1 / D0) ** 2)
            H2 = 1.0 - np.exp(-0.5 * (D2 / D0) ** 2)
            H_pair = H1 * H2  
        elif filter_type == "ideal":
            H_pair = np.ones((M_pad, N_pad), dtype=np.float64)
            H_pair[(D1 <= D0) | (D2 <= D0)] = 0.0


        H *= H_pair

    return H

def notch_filtering_pipeline(img, centers, D0, filter_type="gaussian", pad_factor=2.0, pad_mode="reflect"):
    """
    A function to perform the notch-filter pipeline
    1. Padding 
    2. 2D-FFT and shift to centrum
    3. construction of Notch-filter (H)
    4. Multiplication in frequencydomain (G = F * H)
    5. Inverse FFT and extraction of real part
    6. Croping to original size
    """

    # Padding
    img_padded, pad_info = apply_padding(img, pad_factor, mode=pad_mode)

    # 2D-FFT and shifting
    F_shifted, mag_spectrum_before = compute_fft(img_padded)

    # The transfer function H 
    H = notch_filter(img_padded.shape, centers, D0, filter_type=filter_type)

    # Multiplication of the complex-spectrum
    G_shifted = F_shifted * H

    # Mag spectrum for visulisation
    mag_spectrum_after = np.log(1 + np.abs(G_shifted))

    # invers transform
    G = np.fft.ifftshift(G_shifted)
    img_filtered_padded = np.real(np.fft.ifft2(G))

    # croping to original
    img_filtered = crop_to_original(img_filtered_padded, pad_info)

    return img_filtered, mag_spectrum_before, mag_spectrum_after, H


# Experiment and analysis

# 1:
img_path = "tarea3_files/cameraman.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

pad_factor = 2.0
img_padded, pad_info = apply_padding(img, pad_factor=pad_factor, mode="reflect")
F_shifted, magnitude_spectrum = compute_fft(img_padded)

# Finding centrum for the padded image
M_pad, N_pad = img_padded.shape
v_c, u_c = M_pad // 2, N_pad // 2  # DC-component

# Masking low frequencies around centrum to isolate noise
Y, X = np.ogrid[:M_pad, :N_pad]
dist_from_center = np.sqrt((X - u_c) ** 2 + (Y - v_c) ** 2)

mag_search = magnitude_spectrum.copy()
mag_search[dist_from_center < 20] = (0)

# Identifying the largest noise peak
v_peak, u_peak = np.unravel_index(np.argmax(mag_search), mag_search.shape)

# Calculating relative coordinates
u0 = int(u_peak - u_c)
v0 = int(v_peak - v_c)

print(f"DC-centrum in the padded image: ({u_c}, {v_c})")
print(
    f"Identified noise peak at: (X={u_peak}, Y={v_peak}), Relative"
    f" (u0, v0): ({u0}, {v0})"
)
print(f"Symmetrical mirror point (-u0, -v0): ({-u0}, {-v0})")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original image (cameraman.jpg)")
axes[0].axis("off")

# Spektrum med origo i mitten
extent = [-u_c, N_pad - u_c, M_pad - v_c, -v_c]
axes[1].imshow(magnitude_spectrum, cmap="gray", extent=extent)
axes[1].scatter(
    [u0, -u0],
    [-v0, v0],
    color="red",
    marker="x",
    s=100,
    linewidths=2,
    label=f"Notch-pair ({u0}, {v0})",
)
axes[1].axhline(0, color="white", linestyle="--", alpha=0.3)
axes[1].axvline(0, color="white", linestyle="--", alpha=0.3)
axes[1].set_title("Magnitudespectrum with identified noisepairs")
axes[1].set_xlabel("Frequency $u$ (columns)")
axes[1].set_ylabel("Frequency $v$ (rows)")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("Experiment1")
plt.show()

# 2: 

D0_values = [3, 10, 25]
filter_types = ["ideal", "gaussian"]

# A common colourscale for difference image (0 to 50 pixellevels)
v_max_diff = 50.0
n = 0
for f_type in filter_types:
  for D0 in D0_values:
    
    img_filt, mag_before, mag_after, H = notch_filtering_pipeline(
        img, [(u0, v0)], D0, filter_type=f_type, pad_factor=1.5
    )

    # Calculating difference image
    diff_img = np.abs(img.astype(np.float64) - img_filt.astype(np.float64))

    # Visualisation
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Filtrerad bild
    axes[0].imshow(img_filt, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Filtered image\n({f_type.capitalize()}, $D_0={D0}$)")
    axes[0].axis("off")

    # Spectrum before
    axes[1].imshow(mag_before, cmap="gray")
    axes[1].set_title("Spectrum BEFORE")
    axes[1].axis("off")

    # Spectrum after
    axes[2].imshow(mag_after, cmap="gray")
    axes[2].set_title("Spectrum AFTER")
    axes[2].axis("off")

    # Difference image (common scale)
    im_diff = axes[3].imshow(
        diff_img, cmap="magma", vmin=0, vmax=v_max_diff
    )
    axes[3].set_title("Difference image\n(|Noisy - Filtrered|)")
    axes[3].axis("off")
    fig.colorbar(im_diff, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Filter: {f_type.upper()} | Notch-width $D_0 = {D0}$",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"Experiment2.{n}.png")
    n +=1
    plt.show()


# 3: 
# 
pad_factors = [1.0, 1.5, 2.0]
D0_test = 3.0
centers = [(20, -6)]
pad_mode = "reflect"

filtered_results = {}

# Running simulations
for pf in pad_factors:
  img_filt, _, _, _ = notch_filtering_pipeline(
      img,
      centers,
      D0=D0_test,
      filter_type="gaussian",
      pad_factor=pf,
      pad_mode=pad_mode,
  )
  filtered_results[pf] = img_filt

# Visualisation
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Coordinates for zoom
crop_slice = (slice(0, 60), slice(0, 100))

for i, pf in enumerate(pad_factors):
  res_img = filtered_results[pf]

  # The filtered image
  axes[0, i].imshow(res_img, cmap="gray", vmin=0, vmax=255)
  title_str = (
      "Without padding (1.0x)" if pf == 1.0 else f"Padding with factor {pf}x"
  )
  axes[0, i].set_title(f"{title_str}\n{res_img.shape[0]}x{res_img.shape[1]}")
  axes[0, i].axis("off")

  # The zoom
  crop_zoom = res_img[crop_slice]
  axes[1, i].imshow(crop_zoom, cmap="gray", vmin=0, vmax=255)
  axes[1, i].set_title(f"Zoomed in corner (Upper-left)\n(Pad factor = {pf})")

plt.tight_layout()
plt.savefig("Experiment3")
plt.show()


"""
Seleccione una frecuencia afectada por uno de sus notches e indique el valor complejo del espectro antes del
filtrado, el valor H(u,v) aplicado y el valor resultante después de la multiplicación.
"""
centers = [(20, -6)]
D0 = 10.0


img_padded, pad_info = apply_padding(img, pad_factor=2.0, mode='reflect')
F_shifted, _ = compute_fft(img_padded)
H = notch_filter(img_padded.shape, centers, D0=D0, filter_type='gaussian')

#Choosing matrix coordinates
M_pad, N_pad = img_padded.shape
v_c, u_c = M_pad // 2, N_pad // 2
u0, v0 = centers[0]

v_target = v_c + v0 + 4  # row 254
u_target = u_c + u0 + 3  # column 279

# Getting values
F_val = F_shifted[v_target, u_target]  # Complex spectrum BEFORE
H_val = H[v_target, u_target]  # Transfer function value
G_val = F_val * H_val  # Result AFTER

# 5. Skriv ut resultat och verifiera fasen
print(f'Position in the matrix: rad (v)={v_target}, column (u)={u_target}')
print(f'1. Spectrumvalue BEFORE [F(u,v)]:  {F_val.real:.2f} + {F_val.imag:.2f}j')
print(f'2. Filtervalue [H(u,v)]:         {H_val:.4f}')
print(f'3. Spectrumvalue AFTER [G(u,v)]: {G_val.real:.2f} + {G_val.imag:.2f}j')

print(f'\nPhase BEFORE:  {np.angle(F_val):.6f} row')
print(f'Phase AFTER: {np.angle(G_val):.6f} row')