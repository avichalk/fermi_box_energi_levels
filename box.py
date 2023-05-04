import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import astropy.units as u
import astropy.constants as const

def fplus(beta, mu, eta):
    """
    Fermi-Dirac Distribution for fermions (f+).

    Inputs:
        beta : `float`
            Inverse temperature 1/(k_B*T)
        eta : `float`
            Energy levels of the fermion.
        mu : `float`
            Chemical potential

    Outputs:
        fplus : float
            Fermi-dirac distribution.
    """
    fplus = 1/(np.exp(beta.value*(eta.value - mu.value)) + 1) 
    return fplus

def eta_nvec(n, m, L):
    """
    Calculates the energy levels of a single fermion.

    Inputs:
        n : `3d numpy array of ints`
            Number of fermions along the x, y, z axes.

    Outputs:
        eta_nvec : `3d numpy array of floats`
            Energy levels of the fermion.
    
    """
    eta_nvec = (np.pi * const.hbar)**2 / (2 * m * L**2) * np.sum(np.linalg.norm(n, axis=1)**2)
    return eta_nvec

def total(n_vec, m, L, beta, mu):
    """
    Total energy of the system as per the Grand Canonical Ensemble.

    Inputs:
        n : `3d numpy array of ints`
            Number of fermions along the x, y, z axes.

    Outputs:
        E : `float`
            Total energy of the system.
    """

    ## note: factor of two out front as each possible particle state
    ## has two spin states (spin up, spin down)
    eta = eta_nvec(n_vec, m, L)
    ## there's a problem with the summation
    ## fix that here

    E = 2 * eta * fplus(beta, mu, eta) ## returns a 3d array for n_x, n_y, n_z
    return E 


def test_func(L, rho_e, sigma_e, dE):
    """
    Fit function for scipy curvefit. We pass in E/L^2 vs L,
    giving us rho_e*L + sigma_e / 6 + dE / L**2. 
    """
    return rho_e*L + 6*sigma_e + dE / L**2
    

def main():
    m = const.m_e ## mass of electron
    n_max = 10000 # max no. of particles in each coordinate direction
    n_vec = np.linspace((1, 1, 1), (n_max, n_max, n_max), n_max) ## generates a 3d numpy array from 1 to n_max

    ## values for beta and mu that satisfy:
    ##      beta * mu = 10
    ##      beta * mu = 1
    ##      beta * mu = 0.1
    mu = np.array([1, 1, 1]) * 7 * u.eV  # value of mu for copper 
    beta = np.array([10, 1, 0.1]) * 1 / mu

    L = np.linspace(1e-8, 9e-9, 1000,) * u.m ## values for box size (in m)
    
    fig, ax = plt.subplots(3,figsize=[20, 15], sharex=True)
    for i in range(3): ## iterating over all combos of beta*mu
        ## E/L^2
        E = total(n_vec, m, L, beta[i], mu[i]) / L**2

        ax[i].plot(L, E, label=f'Total Energy. $\\beta$= {beta[i]:.2e}. $\\mu$= {mu[i]}, $\\beta \\mu$={beta[i]*mu[i]}', color='pink', )

        ## the curve fit requires our input parameters (order of 1e-20 for L and 1e40 for y) to be
        ## rescaled. this is accomplished by dividing by the means of both L and y
        ## the data is rescaled once it is returned. We also only fit to the left half
        param, param_cov = curve_fit(test_func, L / L.mean(), E/E.mean(),) # diag=(1./L.mean(),1./y.mean())) 
        param_edit = (param * E.mean()).to(u.J/u.m**2)
        print(f'Parameters: {", ".join(str(x) for x in param_edit)}.')
        print(param_cov)
        ## NOTE: it says in the documentation that diags (commented out) will rescale the input data. 
        ## It does not. For some reason.
        ## Plot the generated fit
        ax[i].plot(L, test_func(L / L.mean(), *param) * E.mean(), label=f'Generated fit. $\\rho_E$={param_edit[0]:.2e}, $\\sigma_E$={param_edit[1]:.2e}, $\\delta E$ ={param_edit[2]:.2e}', linestyle='dotted', )

        ax[i].set_ylabel(r'$E/L^2 [\frac{J}{m^2}]$')
        ax[i].legend(title_fontsize=16)
    
    ax[2].set_xlabel('L [m]')
    ax[0].set_ylabel(r'$E/L^2 [\frac{J}{m^2}]$')
    ax[2].set_ylabel(r'$E/L^2 [\frac{J}{m^2}]$')
    plt.savefig('fit.eps', format='eps')

if __name__ == "__main__":
    main()
