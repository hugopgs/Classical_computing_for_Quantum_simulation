import numpy as np
import matplotlib.pyplot as plt

def calculate_ramsey_probability(delta, Omega, tau, T):
    """
    Calculates the transition probability P_e(delta) for a Ramsey sequence 
    (two pulses separated by free evolution time T).
    
    Parameters:
        delta (float or np.array): Detuning from resonance.
        Omega (float): Rabi frequency.
        tau (float): Pulse duration.
        T (float): Free evolution time between pulses.
        
    Returns:
        float or np.array: Excitation probability.
    """
    # Effective Rabi frequency
    Omega_eff = np.sqrt(Omega**2 + delta**2)
    
    # Pulse evolution components (half-rotations)
    c = np.cos(Omega_eff * tau / 2)
    s = np.sin(Omega_eff * tau / 2)
    
    # Dimensionless parameters
    alpha = delta / Omega_eff
    beta = Omega / Omega_eff
    
    # Combined evolution term (Pulse + Free Evolution)
    # This formula accounts for finite pulse duration effects
    term = c * np.cos(delta * T / 2) + alpha * s * np.sin(delta * T / 2)
    
    return 4 * (beta**2) * (s**2) * (term**2)


def plot_ramsey_fringes(Omega=1.0):
    """
    Plots Ramsey fringes to visualize the effect of free evolution time T 
    and pulse duration tau.
    """
    # Detuning range: -10*Omega to 10*Omega
    delta = np.linspace(-10 * Omega, 10 * Omega, 2000)
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # ---- (1) Compare different T values for a pi/2 pulse ----
    tau_pi_half = np.pi / (2 * Omega)  # Duration for a pi/2 pulse
    
    # Loop to compare T = 5/Omega and T = 10/Omega
    for i in range(1, 3):
        T_val = i * (5 / Omega)
        Pe_values = calculate_ramsey_probability(delta, Omega, tau_pi_half, T_val)
        axes[0].plot(delta / Omega, Pe_values, lw=2, label=fr'$T = {i*5}/\Omega$')
    
    axes[0].set_ylabel(r'$P_e(\Delta)$', fontsize=12)
    axes[0].set_title(r'Ramsey fringes: Varying free evolution time $T$ ($\tau=\pi/2\Omega$)', fontsize=13)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.7)

    # ---- (2) Compare pi/2 vs pi pulses for fixed T ----
    T_fixed = 10 / Omega
    pulse_configs = [
        (r"$\pi/2$", np.pi / (2 * Omega)), 
        (r"$\pi$", np.pi / Omega)
    ]
    
    for label, tau_val in pulse_configs:
        Pe_values = calculate_ramsey_probability(delta, Omega, tau_val, T_fixed)
        axes[1].plot(delta / Omega, Pe_values, lw=2, label=fr'$\tau\Omega = {label}$')
        
    axes[1].set_xlabel(r'Detuning $\Delta / \Omega$', fontsize=12)
    axes[1].set_ylabel(r'$P_e(\Delta)$', fontsize=12)
    axes[1].set_title(r'Effect of pulse area for fixed $T=10/\Omega$', fontsize=13)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_ramsey_fringes()