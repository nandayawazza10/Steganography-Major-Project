# Adaptive Steganography: A Perceptual Approach to Secure Data Embedding

> A research-oriented Python proof of concept for **content-adaptive
> image steganography**. The implementation uses local image gradients
> to guide where payload bits are embedded, then reconstructs the same
> embedding map during extraction.

```{=html}
<p align="center">
```
`<strong>`{=html}Adaptive embedding · Deterministic extraction ·
Image-quality evaluation`</strong>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [The Problem](#the-problem)
-   [Our Approach](#our-approach)
-   [How It Works](#how-it-works)
-   [Adaptive Allocation Rule](#adaptive-allocation-rule)
-   [Project Structure](#project-structure)
-   [Requirements](#requirements)
-   [Getting Started](#getting-started)
-   [Configuration](#configuration)
-   [Evaluation and Metrics](#evaluation-and-metrics)
-   [Current Experimental Results](#current-experimental-results)
-   [Limitations](#limitations)
-   [Future Work](#future-work)
-   [Research Background](#research-background)
-   [Disclaimer](#disclaimer)

## Overview

Image steganography is the practice of concealing information inside a
digital image so that the presence of the information is not obvious to
an observer.

This project explores an **adaptive spatial-domain embedding strategy**.
Instead of assigning the same embedding behaviour to every pixel, it
analyses local image structure and allocates payload bits according to a
pixel's gradient-based texture level.

The current implementation is a Python proof of concept built with
OpenCV, NumPy, scikit-image and Matplotlib. It includes:

-   Image loading, normalization and resizing for controlled
    experiments.
-   A three-level gradient/texture map using Sobel operators.
-   Content-aware payload-slot allocation across BGR channels.
-   Adaptive embedding and corresponding extraction routines.
-   A traditional sequential LSB replacement baseline.
-   Payload recovery checks and image-quality measurements.
-   CSV results, metric comparison plots and visual change maps.

**Project status:** Experimental proof of concept. The implementation
demonstrates a working adaptive embedding/extraction pipeline; it is not
a production-ready or cryptographically secure steganography system.

## The Problem

A basic Least Significant Bit (LSB) method embeds information by
changing the least significant bit of image-channel values. It is simple
and can preserve visual quality at modest payloads, but a uniform
embedding strategy does not account for differences between image
regions.

A change in a smooth area can be easier to notice than a comparable
change in a textured or edge-rich area. Similarly, colour channels do
not contribute equally to perceived image differences.

Research in adaptive image steganography motivates using image content,
local complexity and perceptual distortion to guide embedding. The
problem explored here is:

> **Can a lightweight, reproducible embedding rule use local gradient
> information and channel-dependent bit allocation to place payload bits
> preferentially in visually complex regions, while allowing the
> receiver to reconstruct the same embedding locations without
> transmitting a separate texture map?**

This proof of concept addresses the placement and synchronization parts
of that problem. Its current quality metrics are preliminary and do not
establish resistance to statistical steganalysis or common image
transformations.

## Our Approach

The implementation follows three design ideas:

1.  **Content awareness:** estimate local image structure from a
    gradient-magnitude map and classify pixels as smooth, medium or
    highly textured.
2.  **Channel-aware allocation:** use a fixed BGR allocation table that
    assigns fewer bits to smooth regions and more bits to textured
    regions, with the blue channel receiving the largest share in this
    prototype.
3.  **Reproducible extraction:** compute the texture map after clearing
    the two lowest bit planes. Since embedding modifies only those two
    planes, the decoder can rebuild the same map from the stego image.

The algorithm is deliberately heuristic and lightweight. It is inspired
by research on local complexity, human visual sensitivity and adaptive
distortion, but it is **not an implementation of WOW, STCs, or any one
referenced paper**.

## How It Works

### 1. Image loading and preprocessing

-   Images are read using OpenCV in three-channel BGR format.
-   By default, each image is resized to `256 × 256` for controlled,
    quick experiments.
-   The script looks for supported image formats inside the `images/`
    directory.
-   If fewer than the configured number of images are found, synthetic
    gradient/noise images are generated as a fallback.

### 2. Texture-level map generation

Before analysing image structure, the implementation clears the two
lowest bit planes of every channel:

``` python
masked = img & 0xFC
```

It then:

1.  Converts the masked image to grayscale.
2.  Computes horizontal and vertical Sobel gradients.
3.  Calculates gradient magnitude.
4.  Smooths the magnitude map with a Gaussian blur.
5.  Classifies each pixel using two thresholds.

The result is a per-pixel map with three levels:

  ------------------------------------------------------------------------
                         Level Interpretation        Embedding behaviour
  ---------------------------- --------------------- ---------------------
                           `0` Smooth region         Lowest payload
                                                     allocation

                           `1` Medium gradient       Intermediate
                                                     allocation

                           `2` Highly textured /     Highest payload
                               edge-rich region      allocation
  ------------------------------------------------------------------------

### 3. Embedding-slot construction

Pixels are stably sorted by texture level, with highly textured pixels
first. The allocation table determines which channel bit planes are
eligible for payload data at each pixel.

A slot represents one writable bit position:

``` text
(pixel position, colour channel, bit plane)
```

Only the two lowest bit planes (plane 0 and plane 1) are considered.

### 4. Adaptive embedding

The payload is written into the generated slots. The implementation
writes plane 0 and plane 1 separately to avoid one advanced-indexing
assignment overwriting another when both planes of a channel are used.

After embedding, pixels are restored to their original spatial order to
produce the stego image.

### 5. Extraction

The extraction routine:

1.  Receives the stego image and the expected payload length (`n_bits`).
2.  Clears the two lowest bit planes implicitly through the same
    texture-map routine.
3.  Rebuilds the pixel ordering and slot list.
4.  Reads the selected bit positions in the same order.

Because the texture map is based on the image with those two bit planes
cleared, embedding does not alter the map used to select slots.

**Important:** the current extraction function requires the payload
length to be supplied externally. A future version should store or
transmit payload length and other required metadata in a defined format.

## Adaptive Allocation Rule

The current implementation uses this allocation table. Channel order is
**B, G, R**, matching OpenCV's image representation.

  -----------------------------------------------------------------------
  Texture          Blue bits     Green bits       Red bits Total bits per
  level                                                             pixel
  ----------- -------------- -------------- -------------- --------------
  Smooth                   1              0              0              1
  (`0`)                                                    

  Medium                   2              0              1              3
  (`1`)                                                    

  Textured                 2              1              1              4
  (`2`)                                                    
  -----------------------------------------------------------------------

The two gradient thresholds are:

``` python
GRAD_T1 = 15
GRAD_T2 = 40
```

For each channel, the number in the table indicates how many of its two
lowest bit planes can be used. For example, a value of `2` allows bit
planes 0 and 1, while `1` allows only bit plane 0.

These thresholds and allocations are experimental parameters, not
universal perceptual constants. Their suitability needs to be tested
across more image types and payload rates.

## Project Structure

The experiment expects the following basic layout:

``` text
project/
├── images/
│   ├── image_01.tiff
│   ├── image_02.png
│   └── ...
├── <your_script>.py
└── results/                  # generated when the script runs
    ├── results.csv
    ├── metrics_comparison.png
    ├── visual_0_*.png
    └── ...
```

The current script writes the output directory relative to the working
directory. The image directory is resolved relative to the Python
script.

## Requirements

-   Python 3.10 or newer recommended
-   NumPy
-   OpenCV for Python
-   Matplotlib
-   scikit-image

Install the dependencies:

``` bash
python -m pip install numpy opencv-python matplotlib scikit-image
```

A virtual environment is recommended:

``` bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy opencv-python matplotlib scikit-image
```

## Getting Started

1.  Clone the repository:

    ``` bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  Create an `images/` directory next to the Python script and place
    your test images inside it.

3.  Run the experiment using the actual Python script filename in your
    checkout:

    ``` bash
    python <your_script>.py
    ```

4.  Inspect the terminal summary and the generated `results/` directory.

The script compares the adaptive method and a sequential LSB replacement
baseline using the same payload length for each image. It also checks
whether both extraction routines recover the generated payload exactly.

## Configuration

The main experiment settings are defined near the top of the script:

  ------------------------------------------------------------------------
  Parameter                          Current default Purpose
  --------------------- ---------------------------- ---------------------
  `NUM_IMAGES`                                  `10` Number of images to
                                                     process

  `IMG_SIZE`                            `(256, 256)` Resize dimensions;
                                                     set to `None` to
                                                     retain original
                                                     dimensions

  `PAYLOAD_RATIO`                              `1.0` Fraction of adaptive
                                                     slot capacity used

  `OUT_DIR`                              `"results"` Output directory

  `SHOW_DEMO`                             `"random"` Display one random
                                                     demo, a chosen index,
                                                     or all demos

  `GRAD_T1`                                     `15` Lower gradient
                                                     threshold

  `GRAD_T2`                                     `40` Upper gradient
                                                     threshold

  `BITS_TABLE`                       See table above Per-level BGR
                                                     allocation
  ------------------------------------------------------------------------

The random seed is set to `42` in the current script to make payload
generation reproducible.

## Evaluation and Metrics

The experiment reports:

-   **Payload capacity (bits):** number of available adaptive embedding
    slots.
-   **Payload size (bits):** number of bits embedded in the image.
-   **Bits per pixel (bpp):** payload bits divided by image height ×
    width.
-   **PSNR (Peak Signal-to-Noise Ratio):** pixel-level reconstruction
    quality; higher values generally indicate lower error.
-   **SSIM (Structural Similarity Index):** structural similarity
    between cover and stego images; values closer to 1 indicate greater
    similarity.
-   **Extraction check:** exact equality between the generated payload
    and the extracted payload.

The script generates:

  -----------------------------------------------------------------------
  Output                              Description
  ----------------------------------- -----------------------------------
  `results.csv`                       Per-image capacity, payload, bpp,
                                      PSNR, SSIM and extraction status

  `metrics_comparison.png`            LSB vs adaptive PSNR and SSIM
                                      comparison

  `visual_*.png`                      Cover image, texture map, and
                                      amplified change maps for both
                                      methods
  -----------------------------------------------------------------------

The visual difference maps amplify channel changes by a factor of 60 to
make small modifications easier to inspect. They are diagnostic
visualizations, not a measure of security.

## Current Experimental Results

The current recorded run used **10 real USC-SIPI Miscellaneous images**,
resized to `256 × 256`, with a payload ratio of `1.0`.

  Metric        Sequential LSB baseline   Adaptive method
  ----------- ------------------------- -----------------
  Mean PSNR                    54.56 dB          53.19 dB
  Mean SSIM                      0.9980            0.9985

Additional observations:

-   Exact payload extraction succeeded for all 10 images in the recorded
    run.
-   Payload sizes ranged from approximately `74,726` to `111,035` bits.
-   Payload rate ranged from approximately `1.14` to `1.69` bpp, with a
    mean of approximately `1.38` bpp.
-   Adaptive SSIM was marginally higher in this run, while adaptive PSNR
    was lower by approximately `1.37 dB`.

### Interpreting these results

These results demonstrate that the current encoder and decoder can
operate together and recover the generated payload for the tested
images. They do **not** demonstrate that the adaptive method is
universally more imperceptible or more secure.

The comparison is preliminary: the current baseline is a simple
sequential LSB replacement implementation, and the adaptive method uses
two bit planes in some slots. A stronger, properly controlled comparison
is needed before drawing broad conclusions.

## Limitations

The current proof of concept has several important limitations:

-   It is a spatial-domain research prototype, not a production
    application.
-   It uses fixed gradient thresholds and a hand-designed allocation
    table.
-   Its perceptual model is a heuristic; it does not implement a
    complete human-visual-system model.
-   The current baseline is sequential LSB replacement and should be
    supplemented with stronger and better-controlled baselines.
-   The extraction routine requires the payload length as an input.
-   The current experiment evaluates PSNR, SSIM and exact extraction,
    but does not yet test JPEG compression, resizing, noise, format
    conversion or statistical steganalysis.
-   Robustness against image transformations is not currently
    guaranteed.
-   The method does not provide encryption or cryptographic key
    management. Steganography should not be treated as a replacement for
    encryption.

## Future Work

Planned extensions include:

-   Evaluate on a broader dataset, including the USC-SIPI Textures
    volume and diverse real-world photographs.
-   Test multiple payload rates rather than only the current
    full-capacity setting.
-   Improve the baseline and add additional reference methods for fair
    comparison.
-   Evaluate robustness under JPEG compression, resizing, noise and
    format conversion.
-   Add statistical steganalysis-oriented evaluation.
-   Investigate more principled distortion-cost models and improved
    perceptual allocation.
-   Build a self-hosted interface for image upload, embedding,
    extraction and experiment visualization.
-   Add a defined payload format with metadata handling and optional
    encryption before embedding.

## Research Background

The project is informed by research on adaptive image steganography,
local complexity and perceptual distortion:

1.  **Lou et al. (2010)** --- *A novel adaptive steganography based on
    local complexity and human vision sensitivity.*
2.  **Luo et al. (2010)** --- *Edge Adaptive Image Steganography Based
    on LSB Matching Revisited.*
3.  **Holub and Fridrich (2012)** --- *Designing Steganographic
    Distortion Using Directional Filters.*

These works provide research context and motivation. The present
implementation is a smaller, heuristic proof of concept and should not
be described as a direct reproduction of their algorithms.

## Disclaimer

This repository is intended for academic study and experimentation in
image processing and information hiding. Use only images and data that
you have permission to process. The current implementation has not been
independently security-audited and should not be relied upon to protect
sensitive information.
