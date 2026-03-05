import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks

# R1.a)

# Define the sequence duration.
M = 512
# Generate the discrete time array.
n = np.arange(M)
# Calculate the fundamental angular frequency.
w0 = 4.6 * (2 * np.pi / M) # = 0.0564505 rad/sec.
# Compute the synthetic signal based on the sum of the three sinusoids.
x = 5 * np.cos(w0 * n + 1) + 2 * np.cos(2 * w0 * n + 2) + 3 * np.cos(5 * w0 * n + 3)

# R1.b)

# Plot the signal.
plt.figure(figsize=(10, 4))
plt.plot(n, x)
plt.title('Synthetic Signal x(n)')
plt.xlabel('Sample index n')
plt.ylabel('Amplitude')
plt.show()

# Observe the plotted synthetic signal.
# Since the frequency multiplier is given as 4.6/M, this means that the first sinusoid completes 4.6 cycles within an M-sized window,
# the second sinusoid completes 9.2 cycles, and the third sinusoid completes 23 cycles. 
# Because of this, the observed plot does not contain a complete number of cycles because only the third sinusoid completes an integer 
# number of cycles within the M-sized window.

# R1.c)

# Define the DFT length for the first part of the analysis.
N = 512
# Compute the Discrete Fourier Transform of the signal.
X = fft(x, N)
# Compute the normalized frequencies for the frequency axis.
freqs = fftfreq(N)
# Calculate the magnitude spectrum.
mag = np.abs(X)
# Calculate the phase spectrum.
phase = np.angle(X)

# Plot the data.
# Shift the zero-frequency component to the center of the spectrum for proper plotting.
freqs_shifted = fftshift(freqs)
mag_shifted = fftshift(mag)
phase_shifted = fftshift(phase)

# Plot the requested data.
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(freqs_shifted, mag_shifted, color='darkorange')
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Magnitude Spectrum')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude')
plt.subplot(2, 1, 2)
plt.plot(freqs_shifted, phase_shifted, color='teal')
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Phase Spectrum')
plt.xlabel('Normalized Frequency')
plt.ylabel('Phase in radians')
plt.tight_layout()
plt.show()

# The DFT works by assuming that the M-sized block repeats infinitely in time. When a sinusoid completes an exact, integer number of 
# cycles within that specific window, the end of one block lines up perfectly with the beginning of the next.
# However, in our case since we have a non-integer number of complete cycles within the M-sized window, the cutoff at the 
# right edge of the window creates discontinuities where the blocks connect. The FT will then interpret these discontinuities as 
# a different frequency content, creating spectral leakage into surrounding frequency bins. 
# To better understand this, we need to understand the underlying math. What we have in the time domain as a finite sequence signal is 
# the multiplication of an infinite sinusoid signal with a rectangular signal (M = 512) that is 1 for 0 <= n <= N-1 and 0 everywhere else.
# Since we also know that multiplication in the time domain is a convolution in the frequency domain, the resulting FT of this signal
# is the convolution of the dirac deltas at frequencies +-w0, +-2w0 and +-5w0 with a Sinc centered at w = 0. Since the convolution of a spectrum with 
# dirac deltas is just the shift of the spectrums to the dirac deltas' shift value, we will have 3 sinc functions centered at +-w0, +-2w0 and +-5w0.
# The Discrete Fourier Transform computes specific, equally spaced samples of this continuous frequency spectrum. The DFT analysis equation 
# calculates these discrete frequency bins at integer multiples of (2pi)/N. With N = M = 512 and since w0 and 2w0 are not integers, the frequency bins
# will not capture the true spike of the first two sinusoids because their frequency does not match with any frequency bin. Due to frequency offset, 
# the bins will capture the decaying spikes of the Sinc functions instead of the points where its value is zero (crosses the axis).
# To better understand this, we have created the following code:

########################################################## DFT Visualization ##########################################################

# Compute the standard DFT without zero-padding to obtain the discrete frequency bins.
X_discrete = fft(x)
freqs_discrete = fftfreq(M)
mag_discrete = np.abs(X_discrete)

# Compute a high-resolution DFT by heavily zero-padding the signal to approximate the continuous spectrum.
padding_factor = 16
N_hires = M * padding_factor
X_continuous = fft(x, N_hires)
freqs_continuous = fftfreq(N_hires)
mag_continuous = np.abs(X_continuous)

# Isolate the positive frequencies for the total arrays.
half_M = M // 2
freqs_discrete_pos = freqs_discrete[:half_M]
mag_discrete_pos = mag_discrete[:half_M]

half_N_hires = N_hires // 2
freqs_continuous_pos = freqs_continuous[:half_N_hires]
mag_continuous_pos = mag_continuous[:half_N_hires]

# Define the individual components of the synthetic signal to visualize their separate spectra.
x1 = 5 * np.cos(w0 * n + 1)
x2 = 2 * np.cos(2 * w0 * n + 2)
x3 = 3 * np.cos(5 * w0 * n + 3)

# Compute high-resolution DFTs for each individual component.
X1_continuous = fft(x1, N_hires)
X2_continuous = fft(x2, N_hires)
X3_continuous = fft(x3, N_hires)

# Calculate the positive magnitude for each individual component.
mag1_continuous = np.abs(X1_continuous)[:half_N_hires]
mag2_continuous = np.abs(X2_continuous)[:half_N_hires]
mag3_continuous = np.abs(X3_continuous)[:half_N_hires]

# Create a figure to overlay the individual components, the overall continuous spectrum, and the discrete points.
plt.figure(figsize=(12, 6))

# Plot the individual continuous spectra with distinct colors to show the separate Sinc functions.
plt.plot(freqs_continuous_pos, mag1_continuous, color='mediumseagreen', alpha=0.8, label='Component 1 (4.6 cycles)')
plt.plot(freqs_continuous_pos, mag2_continuous, color='mediumpurple', alpha=0.8, label='Component 2 (9.2 cycles)')
plt.plot(freqs_continuous_pos, mag3_continuous, color='crimson', alpha=0.8, label='Component 3 (23 cycles)')
plt.plot(freqs_continuous_pos, mag_continuous_pos, color='royalblue', alpha=0.4, linestyle='dashed', linewidth=2, label='Total Continuous Spectrum')
plt.plot(freqs_discrete_pos, mag_discrete_pos, color='darkorange', marker='o', linestyle='-', zorder=5, label='DFT Output (Connected Bins)')
plt.axhline(0, color='black', linewidth=0.8)
plt.xlim(0, 0.06)
plt.title('Individual Sinc Functions vs Total Spectrum and DFT Output')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# In this code one can visualize that the sampled frequencies from the DFT do not match the tip of the Sinc peaks for the first
# two sinusoids (because of the non-integer number of cycles within the window). However, the third sinusoid sampling perfectly matches
# the peak of the Sinc signal and perfectly the axis-crossings where the sinc = 0 (no spectral leakage).

#######################################################################################################################################

# R1.d)

# Isolate the positive frequencies for the one-sided spectrum.
half_N = N // 2
mag_onesided = mag[:half_N]
freqs_onesided = freqs[:half_N]
phase_onesided = phase[:half_N]

# Find the peaks in the one-sided magnitude spectrum.
peaks_indices, _ = find_peaks(mag_onesided)
# Sort the identified peaks by their magnitude in descending order.
sorted_peaks = sorted(peaks_indices, key=lambda idx: mag_onesided[idx], reverse=True)
# Select the indices of the three largest isolated peaks.
top_3_peaks = sorted_peaks[:3]

# Extract the magnitude frequency and phase for the top three peaks.
# Normalize the magnitude by dividing by half the sequence length to get the true amplitude of the sinusoids.
peak_mags = mag_onesided[top_3_peaks] * (2 / M)
peak_freqs = freqs_onesided[top_3_peaks]
peak_phases = phase_onesided[top_3_peaks]

# Print the extracted values for verification.
print('Identified Peaks for N=512.')
for idx in range(3):
    m = peak_mags[idx]
    f = peak_freqs[idx]
    p = peak_phases[idx]
    print(f'Peak index: {top_3_peaks[idx]}, Magnitude: {m:.3f}, Frequency: {f:.4f}, Phase: {p:.3f}')

# Initialize the reconstructed signal array with zeros.
x_r = np.zeros(M)
# Reconstruct the signal by summing the three identified sinusoidal components.
for idx in range(3):
    m = peak_mags[idx]
    f = peak_freqs[idx]
    p = peak_phases[idx]
    x_r += m * np.cos(2 * np.pi * f * n + p)

# Consider why the reconstruction quality differs among the three sinusoids.
# Since the frequency multiplier for the third sinusoid equals exactly 23 when divided by the fundamental frequency resolution, 
# this sinusoid aligns perfectly with the DFT bin 23 and suffers no spectral leakage.
# As explained previously, also note that the first and second sinusoids fall at non-integer bin locations 4.6 and 9.2 respectively.
# This misalignment causes spectral leakage for only the first two sinusoids which degrades their reconstruction quality because the
# sampled frequency is not the same as the true frequency from the original sinusoids.

# Plot the requested data.
plt.figure(figsize=(10, 4))
plt.plot(n, x, label='Original x(n)', alpha=0.7)
plt.plot(n, x_r, label='Reconstructed x_r(n)', linestyle='dashed')
plt.title('Original vs Reconstructed Signal (N=512)')
plt.xlabel('Sample index n')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# R1.e)

# Calculate the mean square error of the reconstruction.
mse = np.mean((x - x_r)**2)
print(f'Mean Square Error (N=512): {mse:.4f}')

# Visualize both the original and reconstructed signals on the same plot.
# It is obvious that the reconstructed signal deviates from the original signal and mean square error quantifies this discrepancy.
# The error comes from using inaccurate amplitude frequency and phase values derived from the leaky spectral peaks.

# R1.f)

# Repeat the previous items using a DFT of length N=2048 and comment on the effect of the DFT length.

# Define the larger DFT length for the zero-padded analysis.
N_large = 2048

# Compute the DFT of the original signal with zero-padding to the new length.
X_large = fft(x, N_large)
# Compute the normalized frequencies for the larger DFT.
freqs_large = fftfreq(N_large)
# Calculate the new magnitude spectrum.
mag_large = np.abs(X_large)
# Calculate the new phase spectrum.
phase_large = np.angle(X_large)

# Shift the zero-frequency component to the center of the spectrum for proper plotting.
freqs_shifted_large = fftshift(freqs_large)
mag_shifted_large = fftshift(mag_large)
phase_shifted_large = fftshift(phase_large)

# Plot the magnitude and phase spectra for N=2048.
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(freqs_shifted_large, mag_shifted_large, color='darkorange')
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Magnitude Spectrum (N=2048)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude')
plt.subplot(2, 1, 2)
plt.plot(freqs_shifted_large, phase_shifted_large, color='teal')
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Phase Spectrum (N=2048)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Phase in radians')
plt.tight_layout()
plt.show()

# We can observe that the magnitude spectrum appears much smoother than the N=512 case.

# Isolate the positive frequencies for the larger one-sided spectrum.
half_N_large = N_large // 2
mag_onesided_large = mag_large[:half_N_large]
freqs_onesided_large = freqs_large[:half_N_large]
phase_onesided_large = phase_large[:half_N_large]

# Find the peaks in the new one-sided magnitude spectrum.
peaks_indices_large, _ = find_peaks(mag_onesided_large)
# Sort the identified peaks by magnitude to find the most prominent ones.
sorted_peaks_large = sorted(peaks_indices_large, key=lambda idx: mag_onesided_large[idx], reverse=True)
# Select the indices of the three largest peaks from the padded signal.
top_3_peaks_large = sorted_peaks_large[:3]

# Extract the magnitude frequency and phase for the new peaks.
# Normalize the magnitude using the original sequence length rather than the padded length.
# Use M instead of N_large because zero-padding does not add energy to the signal.
peak_mags_large = mag_onesided_large[top_3_peaks_large] * (2 / M)
peak_freqs_large = freqs_onesided_large[top_3_peaks_large]
peak_phases_large = phase_onesided_large[top_3_peaks_large]

# Print the newly extracted values for verification.
print('Identified Peaks for N=2048.')
for idx in range(3):
    m = peak_mags_large[idx]
    f = peak_freqs_large[idx]
    p = peak_phases_large[idx]
    print(f'Peak index: {top_3_peaks_large[idx]}, Magnitude: {m:.3f}, Frequency: {f:.4f}, Phase: {p:.3f}')

# Initialize the newly reconstructed signal array.
x_r_large = np.zeros(M)
# Reconstruct the signal using the improved parameter estimates.
for idx in range(3):
    m = peak_mags_large[idx]
    f = peak_freqs_large[idx]
    p = peak_phases_large[idx]
    x_r_large += m * np.cos(2 * np.pi * f * n + p)

# Plot data.
plt.figure(figsize=(10, 4))
plt.plot(n, x, label='Original x(n)', alpha=0.7)
plt.plot(n, x_r_large, label='Reconstructed x_r_large(n)', linestyle='dashed')
plt.title('Original vs Reconstructed Signal (N=2048)')
plt.xlabel('Sample index n')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Compute the new mean square error.
mse_large = np.mean((x - x_r_large)**2)
# Print the resulting error.
print(f'Mean Square Error (N=2048): {mse_large:.4f}')

# Evaluate the effect of increasing the DFT length to N=2048.
# The denser grid provides peak locations that are much closer to the true underlying frequencies of the sinusoids.
# This reduces the estimation errors for the signal parameters and consequently yields a significantly lower mean square error for the reconstruction.

########################################################## DFT Visualization ##########################################################

# Isolate the positive frequencies for the N=2048 discrete arrays.
freqs_large_pos = freqs_large[:half_N_large]
mag_large_pos = mag_large[:half_N_large]

# Create a figure to overlay the individual components, the overall continuous spectrum, and the dense discrete points.
plt.figure(figsize=(12, 6))

# Plot the individual continuous spectra with distinct colors to show the separate Sinc functions.
plt.plot(freqs_continuous_pos, mag1_continuous, color='mediumseagreen', alpha=0.8, label='Component 1 (4.6 cycles)')
plt.plot(freqs_continuous_pos, mag2_continuous, color='mediumpurple', alpha=0.8, label='Component 2 (9.2 cycles)')
plt.plot(freqs_continuous_pos, mag3_continuous, color='crimson', alpha=0.8, label='Component 3 (23 cycles)')

# Plot the overall continuous DTFT approximation using a dashed line to show how the complex values combine.
plt.plot(freqs_continuous_pos, mag_continuous_pos, color='royalblue', alpha=0.4, linestyle='dashed', linewidth=2, label='Total Continuous Spectrum')

# Plot the denser N=2048 discrete bins and connect them.
# Reduce the marker size slightly so the dense dots do not clutter the plot.
plt.plot(freqs_large_pos, mag_large_pos, color='darkorange', marker='o', markersize=3, linestyle='-', zorder=5, label='DFT Output N=2048 (Connected Bins)')

# Add a line for the axis.
plt.axhline(0, color='black', linewidth=0.8)
# Set the x-limit to match previous views.
plt.xlim(0, 0.06)
# Set the title and labels.
plt.title('Individual Sinc Functions vs Total Spectrum and DFT Output (N=2048)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# In this code one can visualize that the sampled frequencies from the DFT do not match the tip of the Sinc peaks for the first
# two sinusoids perfectly (because of the non-integer number of cycles within the window), although they are much closer dure to the 
# increased number of bins in the same window. The third sinusoid sampling continues to  perfectly match the peak of the Sinc signal 
# and perfectly the axis-crossings where the sinc = 0 (no spectral leakage).

#######################################################################################################################################