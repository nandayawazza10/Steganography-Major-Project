"""
Visually Adaptive Steganography - Initial Implementation & Testing
-------------------------------------------------------------------
Idea:
  1. Human eyes are most sensitive to GREEN, then RED, least to BLUE.
  2. Smooth areas show changes easily; textured/edge areas hide them.
  So we hide MORE bits in BLUE and in TEXTURED pixels, fewer in GREEN / SMOOTH pixels.

Key trick (so the receiver can extract without extra info):
  The "texture map" is computed from the image with its 2 lowest bit-planes
  cleared. Embedding only touches those 2 bit-planes, so the map is identical
  before and after embedding -> decoder rebuilds the same map.
"""

import os, glob, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ------------------------- 1. SETTINGS (edit these) -------------------------
IMAGE_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")  # folder next to this script
NUM_IMAGES    = 10            # how many images to test
IMG_SIZE      = (256, 256)    # resize for quick testing (None = keep original)
PAYLOAD_RATIO = 1.0         # use 100% of each image's adaptive capacity
OUT_DIR       = "results"
SHOW_DEMO     = "random"      # which image's demo to open: an index (0-9), "random", or "all"
np.random.seed(42)            # reproducible random payloads

# Bits hidden per colour channel for each texture level. Order = (B, G, R) (OpenCV).
# Level 0 = smooth, 1 = medium, 2 = highly textured.
# Blue gets the most, green the least (human sensitivity: G > R > B).
BITS_TABLE = np.array([
    [1, 0, 0],   # smooth   : only 1 bit in blue
    [2, 0, 1],   # medium   : 2 bits blue, 1 bit red
    [2, 1, 1],   # textured : 2 bits blue, 1 bit green, 1 bit red
], dtype=np.uint8)
GRAD_T1, GRAD_T2 = 15, 40     # gradient thresholds separating the 3 levels


# ------------------------- 2. LOADING IMAGES -------------------------
def load_images(folder, n, size):
    """Load n images from folder; fill the rest with synthetic ones if needed."""
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")   # USC-SIPI uses .tiff
    print(f"Looking for images in: {os.path.abspath(folder)}")
    files = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    paths = [os.path.join(folder, f) for f in files if f.lower().endswith(exts)]  # case-insensitive
    images, names = [], []
    for p in paths:
        if len(images) == n: break
        img = cv2.imread(p, cv2.IMREAD_COLOR)           # BGR uint8 (grayscale -> 3 channels)
        if img is None:
            print(f"  Could not read {os.path.basename(p)}, skipping"); continue
        if size: img = cv2.resize(img, size)
        images.append(img); names.append(os.path.basename(p))
    print(f"Loaded {len(images)} real image(s); "
          f"{max(0, n - len(images))} synthetic image(s) will be added.\n")

    # Fallback: synthetic images so the script always runs (smooth -> noisy mix)
    k = 0
    while len(images) < n:
        h, w = size or (256, 256)
        x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        base = np.stack([x, y, (x + y) / 2], axis=2) * 255          # smooth gradient
        noise = np.random.randn(h, w, 3) * (5 + 6 * k)              # growing texture
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        images.append(img); names.append(f"synthetic_{k}.png"); k += 1
    return images, names


# ------------------------- 3. TEXTURE MAP & SLOTS -------------------------
def texture_level_map(img):
    """Return per-pixel level (0/1/2). Uses image with 2 LSB planes cleared."""
    masked = img & 0xFC                                  # clear lowest 2 bits (embedding-invariant)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = cv2.GaussianBlur(np.hypot(gx, gy) / 8.0, (5, 5), 0)   # smoothed gradient magnitude
    level = np.zeros(grad.shape, np.uint8)
    level[grad >= GRAD_T1] = 1
    level[grad >= GRAD_T2] = 2
    return level

def build_slots(img):
    """
    Decide WHERE bits go. Returns pixel order and the list of usable slots.
    Pixels are sorted most-textured first, so the payload fills the best regions first.
    A slot = (pixel position in sorted order, channel, bit-plane).
    """
    level = texture_level_map(img)
    # NOTE: cast to int16 before negating; negating uint8 wraps around (1 -> 255) and flips the order
    order = np.argsort(-level.ravel().astype(np.int16), kind="stable")  # textured pixels first
    bpc = BITS_TABLE[level.ravel()[order]]               # (N,3) bits per channel per pixel
    valid = bpc[:, :, None] > np.arange(2)[None, None, :]   # (N,3,2): is plane p usable?
    slot_idx = np.flatnonzero(valid.ravel())             # flat indices of usable slots
    return order, slot_idx, level


# ------------------------- 4. ADAPTIVE EMBED / EXTRACT -------------------------
def adaptive_embed(img, bits):
    order, slot_idx, _ = build_slots(img)
    assert len(bits) <= len(slot_idx), "Payload larger than capacity!"
    n, c, p = np.unravel_index(slot_idx[:len(bits)], (len(order), 3, 2))
    pix = img.reshape(-1, 3)[order].copy()               # pixels in embedding order
    # Write one bit-plane at a time (a channel can have 2 slots; writing both
    # in a single fancy-index assignment would make one overwrite the other)
    for plane in (0, 1):
        s = (p == plane)
        b = bits[s].astype(np.uint8)
        pix[n[s], c[s]] = (pix[n[s], c[s]] & np.uint8(0xFF ^ (1 << plane))) | (b << plane)
    out = np.empty_like(pix); out[order] = pix           # put pixels back in place
    return out.reshape(img.shape)

def adaptive_extract(stego, n_bits):
    order, slot_idx, _ = build_slots(stego)              # same map as encoder
    n, c, p = np.unravel_index(slot_idx[:n_bits], (len(order), 3, 2))
    pix = stego.reshape(-1, 3)[order]
    return (pix[n, c] >> p) & 1


# ------------------------- 5. BASELINE: TRADITIONAL LSB -------------------------
def lsb_embed(img, bits):
    flat = img.reshape(-1).copy()
    flat[:len(bits)] = (flat[:len(bits)] & 0xFE) | bits.astype(np.uint8)
    return flat.reshape(img.shape)

def lsb_extract(stego, n_bits):
    return stego.reshape(-1)[:n_bits] & 1


# ------------------------- 6. METRICS -------------------------
def evaluate(cover, stego):
    return (psnr(cover, stego, data_range=255),
            ssim(cover, stego, channel_axis=2, data_range=255))


# ------------------------- 7. MAIN EXPERIMENT -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    images, names = load_images(IMAGE_DIR, NUM_IMAGES, IMG_SIZE)
    rows = []
    print(f"{'Image':<18}{'Bits':>8}{'bpp':>6} | {'LSB PSNR':>9}{'LSB SSIM':>9} | "
          f"{'ADP PSNR':>9}{'ADP SSIM':>9} | Extract OK")
    print("-" * 88)

    for img, name in zip(images, names):
        _, slot_idx, _ = build_slots(img)
        capacity = len(slot_idx)
        n_bits = int(capacity * PAYLOAD_RATIO)           # same payload for both methods
        payload = np.random.randint(0, 2, n_bits)        # random secret bits

        stego_lsb = lsb_embed(img, payload)
        stego_adp = adaptive_embed(img, payload)

        ok = (np.array_equal(lsb_extract(stego_lsb, n_bits), payload) and
              np.array_equal(adaptive_extract(stego_adp, n_bits), payload))

        p1, s1 = evaluate(img, stego_lsb)
        p2, s2 = evaluate(img, stego_adp)
        bpp = n_bits / (img.shape[0] * img.shape[1])
        rows.append([name, capacity, n_bits, round(bpp, 3), p1, s1, p2, s2, ok])
        print(f"{name[:17]:<18}{n_bits:>8}{bpp:>6.2f} | {p1:>9.2f}{s1:>9.4f} | "
              f"{p2:>9.2f}{s2:>9.4f} | {ok}")

    # ---- averages ----
    arr = np.array([[r[4], r[5], r[6], r[7]] for r in rows])
    m = arr.mean(axis=0)
    print("-" * 88)
    print(f"{'AVERAGE':<32} | {m[0]:>9.2f}{m[1]:>9.4f} | {m[2]:>9.2f}{m[3]:>9.4f} |")

    # ---- save CSV ----
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "capacity_bits", "payload_bits", "bpp",
                    "LSB_PSNR", "LSB_SSIM", "ADAPTIVE_PSNR", "ADAPTIVE_SSIM", "extract_ok"])
        w.writerows(rows)

    # ---- plot PSNR & SSIM comparison ----
    x = np.arange(len(rows)); wd = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(x - wd/2, arr[:, 0], wd, label="LSB")
    ax[0].bar(x + wd/2, arr[:, 2], wd, label="Adaptive")
    ax[0].set_title("PSNR (dB) - higher is better"); ax[0].legend()
    ax[1].bar(x - wd/2, arr[:, 1], wd, label="LSB")
    ax[1].bar(x + wd/2, arr[:, 3], wd, label="Adaptive")
    ax[1].set_title("SSIM - higher is better"); ax[1].set_ylim(arr[:, [1, 3]].min() - 0.01, 1.0)
    for a in ax: a.set_xlabel("Image index"); a.set_xticks(x)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "metrics_comparison.png"), dpi=150)

    # ---- visual demo: where did each method change pixels? (one picture per image) ----
    figs = []
    for idx, (img, name) in enumerate(zip(images, names)):
        _, slot_idx, level = build_slots(img)
        n_bits = int(len(slot_idx) * PAYLOAD_RATIO)
        payload = np.random.randint(0, 2, n_bits)
        d_lsb = np.abs(img.astype(int) - lsb_embed(img, payload)).sum(2) * 60       # amplified
        d_adp = np.abs(img.astype(int) - adaptive_embed(img, payload)).sum(2) * 60
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title(f"Cover: {name}")
        ax[1].imshow(level, cmap="gray"); ax[1].set_title("Texture level map (0-2)")
        ax[2].imshow(np.clip(d_lsb, 0, 255), cmap="hot"); ax[2].set_title("LSB changes (x60)")
        ax[3].imshow(np.clip(d_adp, 0, 255), cmap="hot"); ax[3].set_title("Adaptive changes (x60)")
        for a in ax: a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"visual_{idx}_{os.path.splitext(name)[0]}.png"), dpi=150)
        figs.append(fig)

    print(f"\nSaved CSV + plots in '{OUT_DIR}/' (one visual_*.png per image)")

    # ---- which demo figure(s) to open on screen ----
    if SHOW_DEMO == "all":
        pass                                              # leave every figure open
    else:
        keep = SHOW_DEMO if SHOW_DEMO != "random" else int(np.random.randint(len(figs)))
        for i, f in enumerate(figs):
            if i != keep: plt.close(f)                    # close the rest, show only one
    plt.show()

if __name__ == "__main__":
    main()
