# Audio Source Separation with Non-negative Matrix Factorization

Splitting a vocal + instrumental mix into its two sources by factorizing the spectrogram matrix, $V \approx WH$ with $W, H \ge 0$. The pipeline is built step by step from linear algebra: RREF, the four fundamental subspaces, Gram–Schmidt, projection, least squares, eigen-decomposition and diagonalization.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-SciPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)](NMF_Source_Separation.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aadi-prakash/nmf-audio-source-separation/blob/main/NMF_Source_Separation.ipynb)

> Mini project for **UE25MA242A – Mathematical Foundations for AI & Data Science**, PES University (B.Tech CSE, Semester 3, 2026).

![True vs estimated spectrograms for the vocal and instrumental sources](assets/separation.png)

---

## Results

The test clip is a 6-second mix of a sung vowel melody and a band (bass, plucked chords, kick, hi-hat), with both sources at equal loudness. The clip is synthesized in code, so the true sources are known and separation can be measured objectively with **SI-SDR** (scale-invariant signal-to-distortion ratio; higher is better).

| Method | Vocal SI-SDR | Instrumental SI-SDR |
|---|---:|---:|
| Mixture (no processing) | 0.2 dB | 0.2 dB |
| Projection onto source subspaces | 15.5 dB | 13.7 dB |
| Unconstrained least squares | 21.6 dB | 21.5 dB |
| **NMF (final method)** | **21.3 dB** | **21.3 dB** |
| Oracle mask (upper bound) | 23.1 dB | 22.9 dB |

That is a **+21 dB improvement** for each source, within 2 dB of the best any masking method could achieve on this clip.

![SI-SDR by method](assets/si_sdr.png)

**Listen** (the WAV files download when clicked): [mixture](outputs/mixture.wav) · [estimated vocal](outputs/vocal_estimated.wav) · [estimated instrumental](outputs/instrumental_estimated.wav) · [true vocal](outputs/vocal_true.wav) · [true instrumental](outputs/instrumental_true.wav)

### Why NMF instead of plain least squares?

On the main clip, least squares scores as well as NMF, but 41% of its coefficients are negative ("negative amounts of sound"), and they have to be clipped before they can form a mask. The difference appears when the model does not fit perfectly. The test below keeps the dictionary learned from an "ah" vowel and has the singer sing vowels the dictionary never saw:

| Sung vowel | LS vocal | NMF vocal | LS instrumental | NMF instrumental | NMF − LS |
|---|---:|---:|---:|---:|---:|
| "ah" (as trained) | 21.6 dB | 21.3 dB | 21.5 dB | 21.3 dB | −0.3 dB |
| "ee" (unseen) | 13.7 dB | 15.4 dB | 13.7 dB | 15.2 dB | **+1.6 dB** |
| "oo" (unseen) | 8.6 dB | 10.0 dB | 9.1 dB | 10.2 dB | **+1.3 dB** |

With signed coefficients, least squares can imitate an unfamiliar spectrum by cancelling atoms against each other, including atoms from the other source. NMF can only add, so the mismatch stays as unexplained error instead of leaking into the wrong track.

---

## How it works

A magnitude spectrogram $V$ (frequencies × time frames, all entries $\ge 0$) is approximately a non-negative combination of a few spectral building blocks:

$$V \approx WH, \qquad W \ge 0,\ H \ge 0.$$

Dictionaries are learned per source from isolated training material: $W_v$ for the voice and $W_i$ for the band. The mixture is then explained with both dictionaries at once:

$$V \approx W_v H_v + W_i H_i,$$

where $H$ is found by the Lee–Seung multiplicative update, which is non-negative least squares:

$$H \leftarrow H \odot \frac{W^\top V}{W^\top W H}.$$

Soft masks $M_v = \dfrac{W_vH_v}{W_vH_v + W_iH_i}$ and $M_i = 1 - M_v$ scale the complex STFT of the mixture. The inverse STFT then gives the two separated waveforms, which add back exactly to the mixture.

<p align="center">
  <img src="assets/dictionaries.png" alt="Learned vocal and instrumental dictionaries" width="100%">
  <img src="assets/nmf_activations.png" alt="NMF cost curve and activations H" width="100%">
</p>

### The 10-stage pipeline

The notebook follows the course's mini-project framework. Each stage ends with a printed **Concept → Purpose → Outcome** box containing that run's numbers.

| # | Stage | Linear algebra | Result on the test clip |
|---|---|---|---|
| 0 | Real-world data | signals as vectors in $\mathbb{R}^L$ | 6 s at 16 kHz, vocal + band |
| 1 | Matrix representation | STFT as the DFT matrix applied to windowed frames, $F^HF = NI$ | $V$ is 513 × 376, non-negative, invertible transform |
| 2 | Matrix simplification | Gauss–Jordan → RREF, column–row factorization $V \approx CR$ | 89 pivot frames out of 376 |
| 3 | Structure of the space | four fundamental subspaces, rank–nullity | 89 + 287 = 376 ✓ |
| 4 | Remove redundancy | linear independence, basis selection, NMF dictionaries | 48 independent atoms (RREF of $W$: 48/48 pivots) |
| 5 | Orthogonalization | modified Gram–Schmidt, principal angles between subspaces | smallest angle 41.8°, median 86.1° |
| 6 | Projection | $P = QQ^\top$, symmetric and idempotent | projection-only separation: 15.5 / 13.7 dB |
| 7 | Prediction / approximation | normal equations $W^\top W h = W^\top v$ via LU; NMF = NNLS | 41% of LS coefficients negative; NMF keeps $H \ge 0$ |
| 8 | Pattern discovery | eigen-analysis of $VV^\top$ and $W^\top W$, Perron–Frobenius | 28 eigen-directions hold 99% of the energy |
| 9 | System simplification | orthogonal diagonalization $C = Q\Lambda Q^\top$, Eckart–Young | best rank-48 error 5.0% vs NMF 18.6% (but signed) |
| 10 | Final output | soft masking, inverse STFT, SI-SDR | +21 dB per source, WAV files written |

<details>
<summary><b>More figures</b></summary>

**Mixture spectrogram (Stage 1)**
![Mixture spectrogram](assets/spectrogram.png)

**RREF staircase of the spectrogram (Stage 2)**
![RREF of V](assets/rref.png)

**Principal angles between the vocal and instrumental subspaces (Stage 5)**
![Principal angles](assets/principal_angles.png)

**Eigen-analysis of $VV^\top$ (Stage 8)**
![Eigen-analysis](assets/eigen_analysis.png)

**Approximation error vs number of basis vectors (Stage 9)**
![Error vs k](assets/error_vs_k.png)

</details>

---

## Quick start

```bash
git clone https://github.com/aadi-prakash/nmf-audio-source-separation.git
cd nmf-audio-source-separation
pip install -r requirements.txt
jupyter notebook NMF_Source_Separation.ipynb
```

Then run **Kernel → Restart & Run All**. The full pipeline runs in about 10 seconds on a laptop and writes the separated WAV files to `outputs/`. Results are deterministic, because every random step is seeded.

No setup needed: use the **Open in Colab** badge above (Colab already has all the dependencies).

## Using your own audio

In the configuration cell, set `USE_SYNTHETIC = False` and pick one of two modes.

```python
USE_SYNTHETIC = False
MIX_PATH = "my_song.wav"

# Supervised: isolated recordings of each source are available
VOCAL_TRAIN_PATH = "vocals_only.wav"
INST_TRAIN_PATH  = "instrumental_only.wav"

# Semi-supervised: no stems; give the instrumental-only part of the song (e.g. the intro)
VOCAL_TRAIN_PATH = None
INST_TRAIN_PATH  = None
INST_SEGMENT     = (0.0, 8.0)   # seconds

# Optional: true stems, used only to compute SI-SDR scores
VOCAL_TRUE_PATH = None
INST_TRUE_PATH  = None
```

Any WAV file works: mono or stereo, at any sample rate (it is resampled to 16 kHz). In semi-supervised mode, the instrumental dictionary is learned from the given segment and the vocal dictionary from whatever that dictionary cannot explain. On a test with only a 3-second instrumental intro, this mode reached about 11–12 dB SI-SDR, against about 21 dB with stems.

## Repository structure

```
nmf-audio-source-separation/
├── NMF_Source_Separation.ipynb   # the full project, with outputs
├── requirements.txt
├── assets/                       # figures used in this README
└── outputs/                      # audio written by the notebook
    ├── mixture.wav
    ├── vocal_true.wav
    ├── instrumental_true.wav
    ├── vocal_estimated.wav
    └── instrumental_estimated.wav
```

## Implementation notes

- **Written from scratch:** Gauss–Jordan RREF with partial pivoting, LU decomposition with forward and back substitution, modified Gram–Schmidt, NMF multiplicative updates, overlap-add inverse STFT, and SI-SDR. NumPy's `lstsq`, `qr`, `svd` and `matrix_rank` are used only to cross-check the hand-written versions.
- **Library routines:** the FFT (`numpy.fft`) and the symmetric eigensolver (`numpy.linalg.eigh`) are taken from NumPy.
- **Synthetic test data:** the vocal is a harmonic series shaped by the four formants of the vowel /a/, with vibrato and breath noise. The band is additive synthesis plus noise-based drums. Training clips use different notes from the test clip, so the dictionaries are not fitted to the test material.
- **STFT settings:** 1024-sample Hann window (64 ms), hop 256 (16 ms), 513 frequency bins, 24 NMF atoms per source.

## Limitations

- The dictionaries must resemble the sources. Accuracy drops when the singer's timbre or vowel differs from the training material (see the robustness table).
- Harmonics shared by the voice and an instrument are ambiguous to the model.
- Only magnitudes are modelled. The mixture's phase is reused, and magnitudes do not add exactly where the two sources overlap in a time–frequency cell.
- The Euclidean cost weights loud frequencies more heavily than human hearing does. The KL or Itakura–Saito divergences are common alternatives.

## References

1. D. D. Lee and H. S. Seung, "Learning the parts of objects by non-negative matrix factorization," *Nature* 401, 788–791, 1999.
2. D. D. Lee and H. S. Seung, "Algorithms for non-negative matrix factorization," *NIPS* 13, 2001.
3. P. Smaragdis and J. C. Brown, "Non-negative matrix factorization for polyphonic music transcription," *IEEE WASPAA*, 2003.
4. P. Smaragdis, B. Raj and M. Shashanka, "Supervised and semi-supervised separation of sounds from single-channel mixtures," *ICA*, 2007.
5. T. Virtanen, "Monaural sound source separation by nonnegative matrix factorization with temporal continuity and sparseness criteria," *IEEE Trans. Audio, Speech, and Language Processing* 15(3), 2007.
6. J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR – half-baked or well done?," *IEEE ICASSP*, 2019.
7. C. Eckart and G. Young, "The approximation of one matrix by another of lower rank," *Psychometrika* 1(3), 1936.
8. G. Strang, *Introduction to Linear Algebra*, 6th ed., Wellesley-Cambridge Press, 2023.

## Author

**Aaditya Prakash** · B.Tech CSE, PES University, Bengaluru · [github.com/aadi-prakash](https://github.com/aadi-prakash)
