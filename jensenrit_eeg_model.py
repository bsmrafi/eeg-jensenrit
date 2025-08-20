import numpy as np
from scipy.signal import hilbert, welch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
from matplotlib import pyplot as plt
import sys
from signal_comparison import compare_eeg_signals

class Synth_eeg:
    def __init__(self, N=32, T=1000, dt=0.01, lambda_threshold=0.2):
        """
        Initialize the framework with model parameters (now using Jansen-Rit dynamics)
        """
        self.N = N
        self.T = T
        self.dt = dt
        self.lambda_threshold = lambda_threshold

        # Jansen-Rit parameters
        self.A = 3.25     # Excitatory synaptic gain
        self.B = 22       # Inhibitory synaptic gain
        self.a = 100      # Excitatory time constant
        self.b = 50       # Inhibitory time constant
        self.C = 135      # Connectivity scaling factor
        self.C1 = self.C
        self.C2 = 0.8 * self.C
        self.C3 = 0.25 * self.C
        self.C4 = 0.25 * self.C

        self.e0 = 2.5     # Maximum firing rate
        self.v0 = 6       # Firing threshold
        self.r = 0.56     # Steepness of sigmoid

        self.sigma_noise = 0.05

        np.random.seed(42)
        self.y = np.random.rand(self.N, self.N) * 0.3
        np.fill_diagonal(self.y, 1.0)

    def S(self, v):
        """Sigmoid function to convert membrane potential to firing rate"""
        return 2 * self.e0 / (1 + np.exp(self.r * (self.v0 - v)))

    def process_eeg(self, E):
        E_A = hilbert(E, axis=1)
        phi = np.unwrap(np.angle(E_A))
        omega = np.mean(np.diff(phi, axis=1) / self.dt, axis=1)

        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                delta_phi = phi[i] - phi[j]
                R_ij = np.abs(np.mean(np.exp(1j * delta_phi)))
                A[i, j] = A[j, i] = max(0, R_ij - self.lambda_threshold)

        return phi, omega, E_A

    def initialize_neural_mass(self, *_):
        """Initialize state variables for Jansen-Rit model"""
        # State variables: y0, y1, y2 (membrane potentials) and derivatives
        y = np.zeros((self.N, 6))  # y0, y1, y2, y3, y4, y5
        return y

    def jansen_rit_dynamics(self, y, p=120):
        """
        y: (N, 6) state variables
        p: input drive (can be noisy)
        """
        y0, y1, y2, y3, y4, y5 = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5]

        dy0 = y3
        dy1 = y4
        dy2 = y5

        s1 = self.S(y1 - y2)         # output from pyramidal cells
        s2 = self.S(self.C1 * y0)    # input to excitatory interneurons
        s3 = self.S(self.C3 * y0)    # input to inhibitory interneurons

        dy3 = self.A * self.a * (p + self.C2 * s2) - 2 * self.a * y3 - self.a**2 * y0
        dy4 = self.A * self.a * s1 - 2 * self.a * y4 - self.a**2 * y1
        dy5 = self.B * self.b * self.C4 * s3 - 2 * self.b * y5 - self.b**2 * y2

        dy = np.stack([dy0, dy1, dy2, dy3, dy4, dy5], axis=1)
        return dy

    def reconstruct_signal(self, y):
        """Reconstruct EEG from pyramidal output (y1 - y2) and apply volume conduction"""
        V = y[:, 1, :] - y[:, 2, :]  # y1 - y2 is pyramidal output
        E_synth = np.zeros_like(V)
        for t in range(V.shape[1]):
            for i in range(self.N):
                E_synth[i, t] = V[i, t]
                for j in range(self.N):
                    if i != j:
                        E_synth[i, t] += self.y[i, j] * V[j, t]
        E_synth += self.sigma_noise * np.std(E_synth) * np.random.randn(*E_synth.shape)
        return E_synth

    def simulate(self, E):
        _, _, E_A = self.process_eeg(E)
        envelope=abs(E_A)
        y = self.initialize_neural_mass()
        Y = np.zeros((self.N, 6, self.T))
        Y[:, :, 0] = y
        p = 120 + 60 * (envelope - np.mean(envelope)) / np.std(envelope)
        print(p.shape)
        for t in tqdm(range(self.T - 1), desc="Simulating"):
            dy = self.jansen_rit_dynamics(Y[:, :, t], p[:,t]) # p=120 + 10 * np.random.randn(self.N))
            Y[:, :, t+1] = Y[:, :, t] + self.dt * dy

        E_synth = self.reconstruct_signal(Y)
        return E_synth, Y

def get_band_signals(data, fs, bands):
    nyq = 0.5 * fs
    band_signals = {}
    for band, (low, high) in bands.items():
        b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
        band_signals[band] = np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 0, data)
    return band_signals

eeg_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 40)
}

if __name__ == "__main__":
    fs = 128
    N = 8
    T = 6 * fs
    synth = 0
    bands = 1

    print('Plotting EEG band: ' + sys.argv[1])

    if synth:
        t = np.linspace(0, 10, T)
        E = np.zeros((N, T))
        for i in range(N):
            freq = 10 + i
            E[i] = np.sin(2*np.pi*freq*t) + 0.5*np.random.randn(T)
    else:
        data = np.loadtxt('eeg_eye_state.csv', delimiter=',', dtype=np.float32)
        data1 = data[0:T, :N]
        scaler = MinMaxScaler()
        data1 = scaler.fit_transform(data1)
        E = data1 - np.mean(data1, axis=0)
        if bands:
            band_signals = get_band_signals(E, fs, eeg_bands)
            E = band_signals[sys.argv[1]]
        E = E.T
    print(E.shape)
    mvf = Synth_eeg(N=N, T=T, dt=1/fs)
    E_synth, Y = mvf.simulate(data1.T)
    #plt.figure()
    #plt.subplot(211);plt.plot(data1);plt.subplot(212);plt.plot(E_synth.T);plt.show()
    if bands:
        band_signals = get_band_signals(E_synth.T, fs, eeg_bands)
        E_synth = band_signals[sys.argv[1]]
        E_synth = E_synth.T
    fig, axes = plt.subplots(N, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i in range(N):
        axes[2*i].plot(E[i, :])
        axes[2*i].set_title('Original')
        axes[2*i+1].plot(E_synth[i, fs:])
        axes[2*i+1].set_title('Synthesized')
    plt.tight_layout()
    plt.savefig("jensenrit_orig_synth_"+sys.argv[1]+".png")
    plt.show()
    
    def compare_connectivity(original, synthesized):
        orig_corr = np.corrcoef(original)
        synth_corr = np.corrcoef(synthesized)
        return np.linalg.norm(orig_corr - synth_corr, 'fro')
    connectivity_diff = compare_connectivity(E, E_synth)
    print(f"\nConnectivity matrix difference: {connectivity_diff:.3f}")
    def plot_psd_comparison(E, E_synth, fs):
        plt.figure(figsize=(10, 5))
        f, Pxx = welch(E.mean(axis=0), fs, nperseg=fs*2)
        f_synth, Pxx_synth = welch(E_synth.mean(axis=0), fs, nperseg=fs*2)
        plt.semilogy(f, Pxx, label='Original EEG')
        plt.semilogy(f_synth, Pxx_synth, label='Synthesized EEG')
        plt.title('Power Spectral Density Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V^2/Hz)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("jensenrit_psd_"+sys.argv[1]+".png")
        plt.show()

    plot_psd_comparison(E, E_synth, fs)
    
    print("Simulation complete!")

