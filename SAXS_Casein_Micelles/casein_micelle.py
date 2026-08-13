r"""
Definition
----------

This model calculates the scattering from casein micelles

The total scattering intensity $I(q)$ has eight contribtuions: 

* $I_{aggr}$: Large aggregates, substantially larger than the casein micelles (power law)
* $I_{micelle}$: The overall casein micelle (polydisperse spheres with hard-sphere structure factor )
* $I_{int}$: Intermediate sized submicelles  (polydisperse spheres)
* $I_{P}$: protein (star of rods with hard-sphere structure factor)
* $I_{C}$: calcium phosphate (oblate ellipsoids with hard-sphere structure factor)
* $I_{PC}$: protein-calcium phosphate cross terms 
* $I_{L}$: protein stacking (Lorentzian peak) 
* $I_{B}$: imperfect background subtraction correction (constant)

and the total scattering is the sum of these terms: 

.. math:: I_{total} = I_{aggr} + I_{micelle} + I_{int} + I_{P} + I_{C} + I_{PC} + I_{L} + I_{B}

the aggregate contribution is given as a power law: 

.. math:: scale_power q^{-power}

the scattering from the casein micelle overall shape is modelled as a polydipserse sphere with Schulz size distribution

.. math:: P_{sphere}

the scattering from the submicelles were modelled with the same polydisperse sphere form factor

the scattering from the protein was modelled with the form factor of a star of rods with $n_a$ arms: 

.. math:: P_P(q) = (P_rod/n_a + (1-1/n_a) A_rod^2) A_{xs}^2

$P_{rod}$ is the form factor of an infinitely thin rod, $P_{rod} = 2Si(qL)/qL - sinc(qL/2)$, where $sinc(x) = sin(x)/x$ and $Si(x)$ is the sine integral. The arm cross section was taken into account by $A_{xs} = 2J_1(qR_{xs})/(qR_{xs})$, where $J_1$ is the first order Bessel function of the first kind. 

where L is the arm length, n_a is the number of arms and... 

the proteins interacts with a hard-sphere structure factor, given as usual. The structure factor was weighted by the the fraction of subparticles being proteins, x. The fraction of calcium phosphate is consequently (1-x). The structure factor was also and weighted with the relative volume: 

.. math:: V_{rel,P} = R_{HS,P}^3/R_{HS,av}^3

where R_{HS,av} =  ((1-x)*R_{HS,C}**3 + x*R_{HS,P}**3)**(1/3). The final weighted structure factor is then: 

.. math:: S_P = 1 + fraction * weight * (S_{HS}(q,eta,R_HS)-1)

Due to aniosotropy, the structure factor was modified by the decoupling approximation: 

.. math:: S_{eff,P} = 1 + beta * (S_P-1)

where beta is given in terms of the form factor amplitude: 

.. math:: beta = <A>^2/<A^2>

where <..> is the orientation average, and <A^2> is the form factor. 

The scattering from the protein is weighted with the total number of proteins, in terms of the fraction of proteins (x), and the total number density of subparticles (N_p), as well as the protein scattering mass (M_P, which is the contrast or excess scattering length density times the volume of one protein subparticle):

.. math:: I_P = N_p * x * M_P**2 * P_P * S_{eff,P}

The scattering from the calcium phosphate subparticles is similar, but with the form factor described as oblate ellipsoids with axes $R,R,\vareps R$ and averaged all orientations ($\alpha$):

.. math:: P_C = int_0^(pi/2) psi_{sphere}(q,R_{eff})^2 sin(\alpha)d\alpha

where R_{eff} is the effective radius, $R_{eff} = R \sqrt(\sin^2(\alpha) + \vareps \cos^2(\alpha))$, and psi_{sphere}(x) is the form factor amplitude of a sphere, $psi_{sphere}(x) = 3(\sin(x) - x \cos(x))/x^3$.

The fraction of calcium phosphates is (1-x), so the scattering from calcium phosphate is: 

.. math:: I_C = N_p * x * M_C**2 * P_C * S_{eff,C}

with the effective and weighted structure factor provided as described for the protein. 

The calcium phosphate and proteins are similar is size, so we must include their cross term. Their effective form factor is provided as a product of the form factor amplitudes:

.. math:: P_{PC} = A_P A_C

likewise for the scattering mass: M_{PC} = M_P M_C. The effective fraction is sqrt(x*(1-x)). Using the approximations R_{HS,PC} \approx R_{HS_P} and eta_{HS,PC} \approx eta_{HS_P} and beta_{PC} = 1, the effective and weighted structure factor can be calculated as described for the protein. The scattering is

.. math:: I_{PC} = 2 N_p sqrt(x(1-x)) M_PC S_{eff_PC}

Finally, the protein form repeated stacking, modelled as a Lorentzian peak at q_{Lorentz} with width w_{Lorentz}: 

.. math:: I_{Lorentz} = scale_{Lorentz} = scale_{Lorentz}/[1 + (q-q_{Lorentz})^2/w_{Lorentz}^2]

Use the SasView built in constant background to model imperfect background subtraction. Do NOT use SasView built-in scaling. 

References
----------

Jan Skov Pedersen, Thea Lykkegaard Møller, Norbert Raak, and Milena Corredig. Soft Matter (2022) 18: 8613–8625. DOI: 10.1039/d2sm00724j

Authorship and Verification
----------------------------

* **Author:** Andreas Haahr Larsen **Date:** 10 December 2025
"""

import numpy as np
from numpy import inf
from scipy.special import jv,sici,gamma

name = "casein_micelle"
title = "casein micelle"

description = """casein micelles"""

category = "spheres"

parameters = [["scale power",    "--",  0, [0, inf], "", "Power law scale factor"],
              ["power",      "",      3.5, [2, 5], "", "Exponent of power law"],
            #   ["n_sub",    "",      28200, [3000, 200000], "", "Ratio casein micelle forward scattering to subparticle scattering"],
              ["n_sub",    "",      22000, [3000, 200000], "", "Ratio casein micelle forward scattering to subparticle scattering"],
            #   ["R_L",  "Å",     528.0, [200, 800], "", "Radius of casein micelles"],
              ["R_L",  "Å",     350.0, [200, 800], "", "Radius of casein micelles"],
              ["sigma_L", "",  0.37, [0.1, 0.8], "", "Relative polydispersity casein micelles"],
            #   ["eta_L",    "",      0.1, [0.01, 0.8], "", "Volume fraction of casein micelles in solution"],
              ["eta_L",    "",      0.05, [0.01, 0.8], "", "Volume fraction of casein micelles in solution"],
              ["R_HS_L",  "Å",     600.0, [200, 1600], "", "Hard-sphere radius of casein micelles"],
            #   ["n_int",       "", 431, [10, 1000], "", "Number of submicelles per micelle"],
              ["n_int",       "", 100, [10, 1000], "", "Number of submicelles per micelle"],
            #   ["R_int",    "Å",      127, [50, 500], "", "Radius of submicelles"],
              ["R_int",    "Å",      60, [30, 500], "", "Radius of submicelles"],
              ["sigma_int",    "",      0.31, [0.1, 0.8], "", "Relative polydispersity submicelles"],
              ["L",    "Å",      45, [20, 70], "", "Length of arms in protein star polyer"],
              ["n_a",    "",      6, [4, 30], "", "Number of arms in protein star of rods"],
              ["R_xs",    "Å",      7, [2, 20], "", "Radius of arms in protein star of rods"],
              ["MW_P",    "Da",      10888, [3000, 30000], "", "Molecular weight protein"],
              ["eta_P",    "",      0.27, [0.01, 0.5], "", "Volume fraction of protein in the casein micelles"],
              ["R_HS_P",    "Å",      40.0, [20, 80], "", "Hard-sphere radius protein"],
              ["R",    "Å",      23.7, [10, 50], "", "Radius calcium phosphate"],
              ["eps",    "",      0.4, [0.2, 0.8], "", "Ellipticity calcium phosphate"],
              ["R_HS_C",    "Å",      100, [50, 200], "", "Hard-sphere radius calsium phosphate"],
              ["eta_C",    "",      0.20, [0.01, 0.5], "", "Volume fraction of calcium phosphate in the casein micelles"],
            #   ["c_P",    "g/ml",      0.027, [0.0027, 0.27], "", "Protein concentration"],
              ["c_P",    "g/ml",      0.01, [0.0027, 0.27], "", "Protein concentration"],
            #   ["c_C",    "g/ml",      0.0025, [0.00025, 0.025], "", "Calcium phosphate concentration"],
              ["c_C",    "g/ml",      0.001, [0.00025, 0.025], "", "Calcium phosphate concentration"],
              ["rho_C",    "g/cm3",      2.21, [2.0, 2.4], "", "Density calcium phosphate"],
              ["factor_C",    "",      3.2, [1.0, 6.0], "", "Increase calcium phosphate structure factor contribution"],
              ["scale_lorentz",    "1/cm",      0.0017, [0.0002, 0.02], "", "Scale of Lorentzian peak"],
              ["q_lorentz",    "1/Å",      0.52, [0.3, 0.9], "", "Position of Lorentzian peak"],
              ["w_lorentz",    "1/Å",      0.3, [0.05, 0.5], "", "Width of Lorentzian peak"],
              ["d_rho_e_P",    "e/Å^3",      0.100, [0.090, 0.110], "", "Excess electron density protein"],
              ["d_rho_e_C",    "e/Å^3",      0.377, [0.3, 0.46], "", "Excess electron density calcium phosphate"],
              ["v_spec_P",    "cm^3/g",      0.71, [0.60, 0.85], "", "partial specific volume, protein"],
              ["expansion_C",    "",      1.06, [1.00, 1.20], "", "Increase of calcium phosphate density"],
              ["N_integration",    "",      300, [150, 800], "", "Point in polydisperse sphere integration (not af fit param)"],
             ]

def Iq(q,scale_power,power,n_sub,R_L,sigma_L,eta_L,R_HS_L,n_int,R_int,sigma_int,L,n_a,R_xs,MW_P,eta_P,R_HS_P,R,eps,R_HS_C,eta_C,c_P,c_C,rho_C,factor_C,scale_lorentz,q_lorentz,w_lorentz,d_rho_e_P,d_rho_e_C,v_spec_P,expansion_C,N_integration):
    """
    Description
    """

    ##################################################################################
    ### set constants
    ##################################################################################

    N_A = 6.023E23      # Avogadros number, mol-1
    r_e = 2.82E-13      # Thomson (or classical) radius, cm
    Å3_per_cm3 = 1e24   # unit conversion, Å3/cm3

    ##################################################################################
    ### derived parameters
    ##################################################################################

    # scattering masses
    V_C = 4/3 * np.pi * eps * R**3 # volume of calcium phosphate, Å3
    rho_C /= expansion_C # mail JSK
    d_rho_e_C /= expansion_C # mail JSK
    M_P = d_rho_e_P * Å3_per_cm3 * r_e * MW_P * v_spec_P / N_A # [M] = 1/Å3 * Å3/cm3 * cm * g/mol cm3/g * mol = cm
    M_C = d_rho_e_C * r_e * V_C # [M] = 1/Å3 * cm * Å3 = cm

    # number densities
    n_P = c_P * N_A / MW_P # "number of Ps" according text (below fig 5), but has units of density: [n_P] = g/ml * 1/mol * mol/g = ml-1 = cm-3
    mass_C = V_C/Å3_per_cm3 * rho_C # mass of C, [g] # mail from JSK
    #n_C = c_C/(V_C*rho_C)*Å3_per_cm3 # number density, also given as 6.3E16 ml-1 in the text - is this value consistent with that?? OBS: units: [n_C] = g/ml * 1/Å3 * cm3/g * Å3/cm3 = ml-1 = cm-3
    n_C = c_C/mass_C # number density ml-1
    x = n_P/(n_P + n_C) # below eqn 35: "x is the fraction of PPs, expected to be quite close to unity", defined between eqn 45 and eqn 46
    N_p = n_C + n_P # overall scaling, cm-3, below eqn (35): "Np is the total number [density, red] of objects (CCPs and PPs)" N_p gives total intensity units of cm-1 (eqn 35), as [M] = cm and [I] = cm-1 = [N_p] * [M][M] = [N_p] cm2 -> [N_p] = cm-3

    # prefactor for low-q and intermediate q scattering - excluding overall scaling, N_p
    prefactor = (n_sub/(1+n_int)) * (x*M_P**2 + (1-x)*M_C**2) # eqn 46 [prefactor] = [M]**2 = cm**2, between eqn 46 and 47: "a3 can be regarded as the number of intermediate structures in a micelle and we use the symbol nint",  "a2 parameter can with some care be regarded as the number of subparticles in the structure, and we, therefore, use the symbol nsub"

    # average R: used for scaling of structure factors at high q
    R_HS_P_3 = R_HS_P**3
    R_HS_C_3 = R_HS_C**3
    R_HS_av_3 = (1-x)*R_HS_P_3 + x*R_HS_C_3
    #R_HS_av = ((1-x)*R_HS_C_3 + x*R_HS_P_3)**(1/3)
    rel_V_P = R_HS_P_3/R_HS_av_3
    rel_V_C = R_HS_C_3/R_HS_av_3
    rel_V_C *= factor_C # tune weight of structure factor to improve the match to fig 5

    ##################################################################################
    ### calculate intensities
    ##################################################################################

    # intensity at very low q: power law
    I_power = power_law(q, scale_power, power)    

    # intensity at low q: large polydisperse sphere
    # scale_L = N_p * prefactor * a_3 # eqn 46
    scale_L = N_p * prefactor * n_int # eqn 46, between eqn 46 and 47: "a3 can be regarded as the number of intermediate structures in a micelle and we use the symbol nint"
    P_L,beta = P_poly_sph(q,R_L,sigma_L,N=int(N_integration),return_beta=True) 
    S_eff_L = 1 + beta * (S_HS(q,eta_L,R_HS_L)-1)
    I_L = scale_L * P_L * S_eff_L

    # intensity at intermediate q: medium polydisperse spheres
    scale_int = N_p*prefactor
    P_int = P_poly_sph(q,R_int,sigma_int,N=int(N_integration)) 
    I_int = scale_int*P_int

    # intensity at high q: protein
    fraction_P = x
    A_P, P_P, beta_P = P_star(q,L,R_xs,n_a) # formfactor ampltitude, form factor, and beta for protein star of rods
    S_P = S_HS(q,eta_P,R_HS_P)
    S_eff_P = S_eff_X(fraction_P, rel_V_P, S_P, beta_P)
    I_high_q_P  = N_p * fraction_P * M_P**2 * P_P * S_eff_P

    # intensity at high q: calcium phosphate
    fraction_C = 1-x
    A_C, P_C, beta_C = P_ellipsoid(q,R,eps) # formfactor and and beta for calcium phosphate ellipsoid
    S_C = S_HS(q,eta_C,R_HS_C)
    S_eff_C = S_eff_X(fraction_C, rel_V_C, S_C, beta_C)
    I_high_q_C  = N_p * fraction_C * M_C**2 * P_C * S_eff_C
        
    # intensity at hiqh q: cross section
    fraction_PC = (fraction_P*fraction_C)**0.5
    P_PC = A_P*A_C
    # R_HS_PC = R_HS_P # or R_HS_PC = (R_HS_P + R_HS_C)/2 # using this value: Rij = (Ri + Rj)/2 (between eqn 38 and 39), instead of R_PC = R_P (table 2)
    # eta_PC = eta_P # or eta_PC = (eta_P + eta_C)/2 # using this value instead - similar to R_HS_PC value
    rel_V_PC = rel_V_P
    M_PC = M_P*M_C
    S_PC = S_P
    beta_PC = 1 # maybe use beta_PC = beta_P instead?
    S_eff_PC = S_eff_X(fraction_PC, rel_V_PC, S_PC, beta_PC)
    I_high_q_PC = N_p * 2 * fraction_PC * M_PC * P_PC * S_eff_PC

    # collect high-q intensity terms
    I_high_q = (I_high_q_P + I_high_q_C + I_high_q_PC) # eqn 35

    # intensity at very high q: Lorentzian peak
    I_lorentz = lorentz(q,scale_lorentz,q_lorentz,w_lorentz)
        
    # constant background
    # I_background = np.ones_like(q) * constant_background

    # collected intensity
    # I_total = I_power + I_L + I_int + I_high_q + I_lorentz + I_background
    I_total = I_power + I_L + I_int + I_high_q + I_lorentz

    return I_total

Iq.vectorized = True  # Iq accepts an array of q values

##################################################################################
### basic functions
##################################################################################

def bessj1(x):
    """"
    Bessel function of the first kind of first order
    """
    return jv(1,x)

def my_sinc(x):
    """
    function for calculating sinc = sin(x)/x
    numpy.sinc is defined as sinc(x) = sin(pi*x)/(pi*x)
    """
    return np.sinc(x/np.pi)

def Si(x):
    return sici(x)[0]

def Sic(x):
    return Si(x)/x

def schulz_pdf(R, R0, sigma):
    """
    Schulz distribution with mean R0 and relative width sigma
    sigma = sqrt(<R^2> - <R>^2) / <R>
    """
    z = 1/sigma**2 - 1
    z1 = z + 1
    coef = (z1**z1) / (R0 * gamma(z1))
    pdf = coef * (R/R0)**z * np.exp(-z1 * R / R0)
    pdf[R <= 0] = 0
    return pdf

##################################################################################
### form factors and structure factors
##################################################################################

def psi_sphere(q,r):
    """
    Form factor amplitude of a sphere
    """
    x = q*r     
    return 3*(np.sin(x)-x*np.cos(x))/x**3

def P_sphere(q,r):
    """
    Form factor of a sphere
    """
    return psi_sphere(q,r)**2 

def P_poly_sph(q, R0, sigma, Rmin=1e-3, Rmax_factor=6, N=300, return_beta=False):
    """
    Polydisperse sphere form factor computed by numerical Schulz averaging.
    No asymptotic expansions. No q-dependent switching.
    R0 in same units as used for q.
    """
    # choose R-grid (log spaced)
    Rmax = Rmax_factor * R0 * (1 + 3*sigma)
    R = np.logspace(np.log10(Rmin), np.log10(Rmax), N)

    # Schulz PDF
    D = schulz_pdf(R, R0, sigma)

    # Volume^2 weight
    V = 4*np.pi*R**3/3
    V2 = V**2

    # denominator (normalization)
    denom_P = np.trapezoid(D * V2, R) # form factor

    # shape: (N_q, N_R)
    q = np.atleast_1d(q)
    qR = np.outer(q, R)

    # form factor
    A = 3*(np.sin(qR) - qR*np.cos(qR))/qR**3 # psi_sphere(q,R) vectorized
    P = A**2
    num_P = np.trapzoid(D * V2 * P, R, axis=1)
    P_q = num_P / denom_P
    
    if not return_beta:
        return P_q
    else:
        denom_A = np.trapezoid(D * V, R) # form factor amplitude
        num_A = np.trapezoid(D * V * A, R, axis=1)
        A_q = num_A / denom_A
        beta_q = A_q**2 / P_q
        return P_q, beta_q

def P_star(q,L,R_xs,n_a):
    """form factor for star of rods"""
    # rod form factor
    qL = q*L
    qL_half = qL/2
    A_rod = Sic(qL)
    P_rod = 2*A_rod - my_sinc(qL_half)**2 # eqn 42, rewritten as function of qL/2

    # cross section of arms
    qR_xs = q*R_xs
    A_xs = 2 * bessj1(qR_xs)/qR_xs
    P_xs = A_xs**2

    # collect terms
    A = A_rod*A_xs 
    A_rod_2 =A_rod**2
    P = (P_rod/n_a + A_rod_2 - A_rod_2/n_a) * P_xs
    beta = A**2/P
    return A,P,beta

def S_HS(q,eta,R_HS):
    """
    Calculate the hard-sphere structure factor using the Percus-Yevick approximation.
    Implements the stable version with Taylor expansion for small A = 2*R*q.
    adapted from SasView's hardsphere.c and vectorized for python
    """

    q = np.asarray(q, dtype=float)
    if eta <= 0.0:
        return np.ones_like(q)

    vf = eta
    X = np.abs(2.0 * R_HS * q)

    denom = 1.0 - vf
    if denom < 1e-12:
        return np.ones_like(q)

    Xinv = 1.0 / denom
    D = Xinv * Xinv
    A = (1.0 + 2.0 * vf) * D
    A *= A
    B = (1.0 + 0.5 * vf) * D
    B *= B
    B *= -6.0 * vf
    G = 0.5 * vf * A

    cutoff_tiny = 5e-6
    cutoff_series = 0.05

    S = np.empty_like(q)
    Xabs = X

    mask0 = Xabs < cutoff_tiny
    mask1 = (Xabs >= cutoff_tiny) & (Xabs < cutoff_series)
    mask2 = ~ (mask0 | mask1)

    # q -> 0 limit
    S[mask0] = 1.0 / A

    # Taylor series region
    x = Xabs[mask1]
    x2 = x * x
    FF = (8.0 * A + 6.0 * B + 4.0 * G
          + (-0.8 * A - B/1.5 - 0.5 * G
             + (A/35.0 + 0.0125 * B + 0.02 * G) * x2) * x2)
    S[mask1] = 1.0 / (1.0 + vf * FF)

    # General expression
    x = Xabs[mask2]
    x2 = x * x
    x4 = x2 * x2
    s = np.sin(x)
    c = np.cos(x)
    FF = ((G * ((4.0 * x2 - 24.0) * x * s
                - (x4 - 12.0 * x2 + 24.0) * c + 24.0) / x2
           + B * (2.0 * x * s - (x2 - 2.0) * c - 2.0)) / x
          + A * (s - x * c)) / x
    S[mask2] = 1.0 / (1.0 + 24.0 * vf * FF / x2)

    return S

def psi_ellipsoid(q,R,eps,alpha):
    """
    Form factor amplitude: ellipsoid of revolution  
    axes R,R,epsR (Pedersen1997)
    """
    r_effective = R*np.sqrt(np.sin(alpha)**2 + eps**2*np.cos(alpha)**2)
    psi = psi_sphere(q,r_effective)

    return psi

def P_ellipsoid(q,R,eps):
    """
    Form factor amplitude: ellipsoid of revolution 
    axes R,R,epsR (Pedersen1997)
    """
    N_alpha = 15
    q = np.atleast_1d(q)
    alpha_max = np.pi/2
    da = alpha_max / N_alpha
    alpha = np.linspace(da/2, alpha_max - da/2, N_alpha)

    # broadcast q and alpha
    q2, a2 = np.meshgrid(q, alpha, indexing="ij")
    r_eff = R*np.sqrt(np.sin(a2)**2 + eps**2*np.cos(a2)**2)
    x = q2 * r_eff
    psi = 3*(np.sin(x) - x*np.cos(x))/x**3

    weights = np.sin(a2) * da
    P = np.sum(psi**2 * weights, axis=1)
    A = np.sum(psi * weights, axis=1)
    beta = A**2 / P
    return A, P, beta

def S_eff_X(fraction,rel_V,S,beta):
    """
    volume-weighted effective hard-sphere structure factor
    if beta = 1 then S_eff = S
    """
    S_w = 1 + fraction * rel_V * (S-1)
    S_eff = 1 + beta * (S_w-1)
    return S_eff

##################################################################################
### Scattering intensities
##################################################################################

def power_law(q,A,power):
    """
    scattering from large aggregates, expressed via a power law
    """
    return A*q**(-power)

def lorentz(q,A,q0,w):
    """
    scattering from repeated structures in proteins strands by Lorentzian peak
    """
    P_lorentz = 1/(1+((q-q0)/w)**2)
    I_lorentz = A * P_lorentz
    return I_lorentz

# ##################################################################################
# ### TEST:
# ##################################################################################

# import time

# print(f'name: {name}')
# print(f'title: {title}')
# print(f'description: {description}')

# start_time = time.time()
# q = np.linspace(0.0001,1.2,5000)
# p0 = []
# for p in parameters:
#     p_name,p_units,p_guess,p_bounds,p_empty,p_description = p
#     p0.append(p_guess)
# I = Iq(q,*p0) + 1e-3

# end_time = time.time()-start_time

# print(f'Finished, time: {end_time:.2f} s')
# if 1:
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(q,I,label='total intensity')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.legend ()
#     plt.show()
