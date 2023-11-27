import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy.integrate import quad
#from astropy.cosmology import Planck15   # You can choose a different cosmology if needed
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from numba import jit

 
def lx_func(func, fixed_value):
    def f(x):
        return func(fixed_value, x)
    return f
 
def z_func(func, fixed_value):
    def f(x):
        
        return func(x,fixed_value)
    return f



def simpsons_integration(func, a, b, n=500):
    h = (b - a) / n
    s = func(a) + func(b)

    for i in range(1, n, 2):
        s += 4 * func(a + i * h)
    for i in range(2, n, 2):
        s += 2 * func(a + i * h)

    return s * h / 3


import pandas as pd
 
def Psi_Ueda(Lx):
     return 1/np.log(10)*1/Lx*((Lx/L_star)**(gamma1)+(Lx/L_star)**(gamma2))**(-1) 


 
def Psi_Ajello(Lx):
    return 1/Lx*(Lx/L_star)**(1-gamma2) #Function from Ajello et al 

 
def z_star(Lx): #Calc Z_star used for caculation of XLF for RG
    if (Lx < L_c):
        return z_c *(Lx/L_c)**alpha
    else:
        return z_c
 
def e_z(z,Lx): #Used in calculation of XLF for RG. redshift evolution is of type PDF
    z_s =z_star(Lx)
    if z<=z_s:
        return (1+z)**(v_1)
    else: 
        return e_z(z_s,Lx) *((1+z)/(1+z_s))**(v_2)
 
def e_z_PL(z): #Used in calculation of XLF for Blazars, BLlac, and FSRQ. redshift evolution is of type PLF
    
    return (1+z)**(v_1 + v_2*z)





H0 = 70.0  # Hubble constant in km/s/Mpc
Om0 = 0.3  # Matter density parameter (ΩM)
Ode0 = 0.7  # Dark energy density parameter (ΩΛ)

# Create a FlatLambdaCDM cosmology object
cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)  # Replace with your desired H0 value


 
def integrandRG(z, lx): #different integrands, Blazar and FSRQ use the same integrand
   
    
    return Psi_Ueda(lx)*cosmo1.differential_comoving_volume(z).value*e_z(z,lx)
 
def integrandBlazar(z, lx):
   
    return Psi_Ueda(lx/e_z_PL(z))*cosmo1.differential_comoving_volume(z).value
 
def integrandBLlac(z, lx):
 
   
    return Psi_Ajello(lx/e_z_PL(z))*cosmo1.differential_comoving_volume(z).value



model_constants = { #Constants for all models, no unit attached yet
    'SLDDE_RG': {
        'A': 10**(-6.077),
        'L_star': 10**(44.33),
        'gamma1':2.15,
        'gamma2':1.10,
        'v1':4,
        'v2':-1.5,
        'z_c': 1.9,
        'Lc':10**(44.6),
        'alpha':   0.317,
        'corr_fac': 1,
        'L_lower':42,
         'L_higher':47,
        'integrand':integrandRG,
        
    },
    'AMPLE_Blazar': {
        'A': 1.379*10**(-7),
        'L_star': 1.81*10**(44),
        'gamma1':-0.87,
        'gamma2':2.73,
        'v1':3.45,
        'v2':-0.25,
        'z_c': 0,
        'Lc':0,
        'alpha':   0,
        'corr_fac': 1,
        'L_lower':44,
         'L_higher':48.5,
        'integrand':integrandBlazar,
        # Add more constants for model 1
    },
     'AMPLE_FSRQ': {
        'A': 0.175*10**(-7),
        'L_star': 2.42*10**(44),
        'gamma1':-50,
        'gamma2':2.49,
        'v1':3.67,
        'v2':-0.30,
        'z_c': 0,
        'Lc':0,
        'alpha':   0,
        'corr_fac': 1,
        'L_lower':46,
         'L_higher':48.5,
        'integrand':integrandBlazar,
        # Add more constants for model 1
    },
     'APLE_BLlac': {
        'A': 0.830*10**(-7),
        'L_star': 1*10**(44),
        'gamma1':0,
        'gamma2':2.61,
        'v1':-0.79,
        'v2':0,
        'z_c': 0,
        'Lc':0,
        'alpha':   0,
        'corr_fac': 1,
         'L_lower':44.5,
         'L_higher':48.5,
         'integrand':integrandBLlac,
        # Add more constants for model 1
    },}


def set_model_constants(model):
    global selected_model, A, L_star, gamma1, gamma2, v_1, v_2, z_c, L_c, alpha, corr_fac, N_steps, L_x, integrand, z_lower, z_upper, a, x, L_x_lower, L_x_upper, Z_list,E_v_list, E1, E2
    
    selected_model = model
    
    constants = model_constants[selected_model]
    A = constants['A']
    L_star = constants['L_star']
    gamma1 = constants['gamma1']
    gamma2 = constants['gamma2']
    v_1 = constants['v1']
    v_2 = constants['v2']
    z_c = constants['z_c']
    L_c = constants['Lc']
    alpha = constants['alpha']
    corr_fac = constants['corr_fac']

    N_steps = 40
    L_x = np.logspace(constants["L_lower"], constants["L_higher"], N_steps)

    integrand = constants["integrand"]
    z_lower = 0.01
    z_upper = 2.5

    a = np.log10(L_x[-1]) - np.log10(L_x[0])
    x = L_x / (10.0 ** (44))

    L_x_lower = np.log10(L_x[0])
    L_x_upper = np.log10(L_x[0]) + a / 3
    Z_list = np.linspace(z_lower, 9, N_steps)
    E_v_list = np.linspace(1000,10000000,10) #GeV
    E1 = E_v_list[0]
    E2 = E_v_list[-1]
    
    return 

    # Access constants for the selected model and giving all constants their unit






 
def prefrom_int(func, list,z_lower,z_upper):

    results = []
    for h in list:
        results.append(A*corr_fac*4*np.pi*simpsons_integration(func(integrand,h), z_lower, z_upper))

    return results

def lum_den_calc():
    integral_results_z_1 = prefrom_int(z_func, L_x,z_lower,z_upper)
    integral_results_z_1  = np.array(integral_results_z_1)
    print("done 1")
    integral_results_z_2 =  prefrom_int(z_func, L_x,z_lower+ 2.5,z_upper+2.5)
    integral_results_z_2 = np.array(integral_results_z_2)
    print("done 2")
    integral_results_z_3 = prefrom_int(z_func, L_x,z_lower+5,z_upper+5)
    integral_results_z_3 = np.array(integral_results_z_3)
    print("done 3")
    integral_results_z_4 =  prefrom_int(z_func, L_x,z_lower+7.5,z_upper+7.5)
    integral_results_z_4 = np.array(integral_results_z_4)

    # Save the data in columns
    data = np.column_stack((integral_results_z_1, integral_results_z_2, integral_results_z_3, integral_results_z_4, L_x, Z_list))
    header = " "+str(z_lower)+ "<" + "z" + "<" + str(z_upper) + ", " + str(z_lower+2.5)+ "<" + "z" + "<" + str(z_upper+2.5) + ", " + str(z_lower+5)+ "<" + "z" + "<" + str(z_upper+5) + ", " + str(z_lower+7.5)+ "<" + "z" + "<" + str(z_upper+7.5) + ", " + "L_x" + ", " + "Z_list"
    #header = "integral_results_z_1,integral_results_z_2,integral_results_z_3,integral_results_z_4,L_x,Z_list"
    np.savetxt('lum_den_data/LD_' + selected_model + '.txt', data, delimiter=',', header=header)



    return integral_results_z_1, integral_results_z_2, integral_results_z_3, integral_results_z_4




def red_evo_calc():
    global a1, a2, a3, a4, a5, a6

    if (selected_model == "AMPLE_Blazar"):
        a1 = 44
        a2 = 46.5
        a3 = 46.5
        a4 = 47.5
        a5 = 47.5
        a6 = 48.5

        print("AMPLE_Blazar")
    elif(selected_model == "SLDDE_RG"):
        a1 = 42
        a2 = 43.5
        a3 = 43.5
        a4 = 45
        a5 = 45
        a6 = 47

        print("SLDDE_RG")

    elif(selected_model == "AMPLE_FSRQ"):
        a1 = 46
        a2 = 46.75
        a3 = 46.75
        a4 = 47.5
        a5 = 47.5
        a6 = 48.5
        
        print("AMPLE_FSRQ")
    else:
        a1 = 44.5
        a2 = 45.5
        a3 = 45.5
        a4 = 46.5
        a5 = 47.5
        a6 = 48.5
        print("APLE_BLlac")


    integral_results_lx1 = prefrom_int(lx_func, Z_list,10**a1,10**a2)
    integral_results_lx1 = np.array(integral_results_lx1)
    print("done 1")
    integral_results_lx2 =   prefrom_int(lx_func, Z_list,10**a3,10**a4)
    integral_results_lx2 = np.array(integral_results_lx2)
    print("done 2")
    integral_results_lx3 =  prefrom_int(lx_func, Z_list,10**a5,10**a6)
    integral_results_lx3 = np.array(integral_results_lx3)
    print("done 3")
    integral_results_lx4 =  prefrom_int(lx_func, Z_list,10**a1,10**a6)
    integral_results_lx4 = np.array(integral_results_lx4)


    # Save the data in columns
    data = np.column_stack((integral_results_lx1, integral_results_lx2, integral_results_lx3, integral_results_lx4, L_x, Z_list))
    header = str(a1) + "<" + "log(Lx)" + "<" + str(a2) + ", " + str(a3)+ "<" + "log(Lx)" + "<" + str(a4) + ", " + str(a5)+ "<" + "log(Lx)" + "<" + str(a6) + ", " + str(a1)+ "<" + "log(Lx)" + "<" + str(a6) + ", " + "L_x" + ", " + "Z_list"
    #header = "integral_results_z_1,integral_results_z_2,integral_results_z_3,integral_results_z_4,L_x,Z_list"
    np.savetxt('red_den_data/RD_' + selected_model + '.txt', data, delimiter=',', header=header)

    return integral_results_lx1, integral_results_lx2, integral_results_lx3, integral_results_lx4


def em_j_numerator(z,lx):
    return A*lx*integrand(z,lx)

def em_j_denominator(z,lx):
    return A*integrand(z,lx)

def em_j(z, lowerlim, upperlim):
    a = simpsons_integration(lx_func(em_j_numerator,z), lowerlim, upperlim,n=2000)
    b = simpsons_integration(lx_func(em_j_denominator,z), lowerlim, upperlim,n=2000)
    avg_lx = a/b


    return avg_lx 
def em_avg_calc(arr1,arr2,arr3,arr4):



    n_1 = (arr1)/cosmo1.differential_comoving_volume(Z_list).value
    n_2 = (arr2)/cosmo1.differential_comoving_volume(Z_list).value
    n_3 = (arr3)/cosmo1.differential_comoving_volume(Z_list).value

    n_tot =(arr4)/cosmo1.differential_comoving_volume(Z_list).value



    #print(L_x)

    em_j1 = []
    em_j2 = []
    em_j3 = []
    em_tot = []
    avg_lx1 = []
    avg_lx2 = []
    avg_lx3 = []
    avg_tot = []
    for z in Z_list:
        res1 = em_j(z,10**a1,10**a2)
        res2 = em_j(z,10**a3,10**a4)
        res3 = em_j(z,10**a5,10**a6)
        res = em_j(z,10**a1,10**a6)

        em_j1.append(res1 * n_1[z == Z_list])
        em_j2.append(res2 * n_2[z == Z_list])
        em_j3.append(res3 * n_3[z == Z_list])
        em_tot.append(res * (n_tot[z == Z_list]))
        avg_lx1.append(res1)
        avg_lx2.append(res2)
        avg_lx3.append(res3)
        avg_tot.append(res)
    
    em_j1 = np.array(em_j1)
    em_j1 = np.reshape(em_j1, -1)

    em_j2 = np.array(em_j2)
    em_j2 = np.reshape(em_j2, -1)

    em_j3 = np.array(em_j3)
    em_j3 = np.reshape(em_j3, -1)

    em_tot = np.array(em_tot)
    em_tot = np.reshape(em_tot, -1)

    avg_lx1 = np.array(avg_lx1)
    avg_lx2 = np.array(avg_lx2)
    avg_lx3 = np.array(avg_lx3)
    avg_tot = np.array(avg_tot)
    return em_j1, em_j2, em_j3, em_tot, avg_lx1, avg_lx2, avg_lx3, avg_tot


from scipy.optimize import curve_fit



def power_law(z,a, n):
    
    return a * (1 + z) ** n

# Fit the first array
#popt1, pcov1 = curve_fit(power_law, Z_list, avg_lx1, p0=[avg_lx1[0], 1])
#print(f"Fit parameters for array 1: a={popt1[0]:.2f}, n={popt1[1]:.2f}")

# Fit the second array
#popt2, pcov2 = curve_fit(power_law, Z_list, avg_lx2, p0=[avg_lx2[0], 1])
#print(f"Fit parameters for array 2: a={popt2[0]:.2f}, n={popt2[1]:.2f}")

# Fit the third array
#popt3, pcov3 = curve_fit(power_law, Z_list, avg_lx3, p0=[avg_lx3[0], 1])
#print(f"Fit parameters for array 3: a={popt3[0]:.2f}, n={popt3[1]:.2f}")
# Define the broken power law function
def broken_power_law(z, a, b, c, e):
    

    return a*((z/c)**b+(z/c)**e)**(-1)

def return_fit_varaibles(avg_lx1, avg_lx2, avg_lx3, avg_tot, em_tot, n_tot,c):

    print("The model in question is: ", selected_model)
    # Fit the first array
    popt1, pcov1 = curve_fit(broken_power_law, Z_list, avg_lx1, p0=[avg_lx1[0], 1, c, -1])
    print(f"Fit parameters for array 1: a={popt1[0]:.2f}, b={popt1[1]:.2f}, c={popt1[2]:.2f}, e={popt1[3]:.2f}")

    # Fit the second array
    popt2, pcov2 = curve_fit(broken_power_law, Z_list, avg_lx2, p0=[avg_lx2[0], 1, c, -1])
    print(f"Fit parameters for array 2: a={popt2[0]:.2f}, b={popt2[1]:.2f}, c={popt2[2]:.2f}, e={popt2[3]:.2f}")

    # Fit the third array
    popt3, pcov3 = curve_fit(broken_power_law, Z_list, avg_lx3, p0=[avg_lx3[0], 1, c, -1])
    print(f"Fit parameters for array 3: a={popt3[0]:.2f}, b={popt3[1]:.2f}, c={popt3[2]:.2f}, e={popt3[3]:.2f}")



    # Fit the first array
    popt_tot, pcov_tot= curve_fit(broken_power_law, Z_list, avg_tot, p0=[avg_tot[0], 1, c, -1])
    print(f"Fit parameters for avg_tot: a={popt_tot[0]:.2f}, b={popt_tot[1]:.2f}, c={popt_tot[2]:.2f}, e={popt_tot[3]:.2f}")



    popt_tot_em, pcov_tot_em= curve_fit(broken_power_law, Z_list, em_tot, p0=[em_tot[0], 1, c, -1])
    print(f"Fit parameters for em_tot: a={popt_tot_em[0]:.2f}, b={popt_tot_em[1]:.2f}, c={popt_tot_em[2]:.2f}, e={popt_tot_em[3]:.2f}")


    popt_tot_n, pcov_tot_n= curve_fit(broken_power_law, Z_list, n_tot, p0=[n_tot[0], 1, c, -1])
    print(f"Fit parameters for n_tot: a={popt_tot_n[0]:.2f}, b={popt_tot_n[1]:.2f}, c={popt_tot_n[2]:.2f}, e={popt_tot_n[3]:.2f}")

    return popt1, popt2, popt3, popt_tot, popt_tot_em, popt_tot_n




from scipy.optimize import minimize

def SED_neut(E_v,D): #general SED for all neutrino emmision from the tpye of AGN considered
    return D*E_v**(-2.37) #2.2 comes from some paper i think


def integrate_SED(D,func): #Want this to equate the luminosity of our AGN, This luminosity is redshift dependent. 
    integral, _ = quad(func, E1, E2, args=(D))
    #print(integral)
    return integral


def objective_function(D, target_luminosity,func):
   
    return abs(integrate_SED(D,func) - target_luminosity)

def D_func(z,a,b,c,d):
    return a*((z/c)**b+(z/c)**d)**(-1)


#print(integrate_SED(2.829371152471951e+42))

#D_init = 2.829371152471951e+42
#z_trial = 0.01
#print(broken_power_law(z_trial, *popt_tot))
def D_list_calc(c):
    
    D_list = []

    for i in range(len(Z_list)):
        z_trial = Z_list[i]
        D_init = broken_power_law(z_trial, *popt_tot)* 624.15 #conversion from ergs to GeV
        Neutrino_luminosity = broken_power_law(z_trial, *popt_tot)* 624.15
        result = minimize(objective_function, D_init, args=(Neutrino_luminosity,SED_neut),method='Nelder-Mead')
        D_list.append(result.x[0])

    #result = minimize(objective_function, D_init, args=(broken_power_law(z_trial, *popt_tot)),method='Nelder-Mead')
    D_list = np.array(D_list)

    popt_tot_D, pcov_tot_D= curve_fit(D_func, Z_list, D_list, p0=[D_list[0], 1, c, 1]) #c param very important and volatile
    print(f"Fit parameters for D_list: a={popt_tot_D[0]:.2f}, b={popt_tot_D[1]:.2f}, c={popt_tot_D[2]:.8f}, d={popt_tot_D[3]:.2f}")
    return D_list, popt_tot_D



def flux_integrand(z):
    Dh = cosmo.hubble_distance.value  # Hubble distance in Mpc
    E_z = cosmo.efunc(z)  # E(z) function from astropy
 
    return (Dh / E_z) * SED_neut(Ev*(1+z),(D_func(z,*popt_tot_D))) / (Ev**2 *4 * np.pi * (1 + z)**2 )* broken_power_law(z, *popt_tot_n)

def calc_phi():
    global Ev
    d_phi = []

    for i in range(len(E_v_list)):
        Ev = E_v_list[i]

        

        results_flux = simpsons_integration(flux_integrand, 0.001, 9,n=2000)
        d_phi.append(results_flux/(3.0857*10**24)**2) #convert from Mpc^2 to cm^2 

    d_phi = np.array(d_phi)
    return d_phi

def ICECUBE(E_v, E_cut,gamma,phi_cut):
    return phi_cut*(E_v/100000)**(-gamma)*np.exp(-(E_v/E_cut))


E_cut_ice = 1250000 #GeV
gamma = 2.37 #dimensionless
phi_cut = 1.64*10**(-18) #C units
#E_v_list = np.linspace(0.01,2000,400)*(10**3) #GeV


from scipy.integrate import dblquad


def SED_neut_2(E_v,D,Lx): #general SED for all neutrino emmision from the tpye of AGN considered
    E_cut = func(Lx)
    return D*E_v**(-2.37)*np.exp(-E_v/E_cut) #2.2 comes from some paper i think

def func(Lx):
    below = 10**42
    return 100000*(Lx/below)**(0.25)

def integrate_SED_2(D,lx,func): #Want this to equate the luminosity of our AGN, This luminosity is redshift dependent. 
    integral, _ = quad(func, E1, E2, args=(D,lx))
    #print(integral)
    return integral


def objective_function_2(D, target_luminosity,func):
   
    return abs(integrate_SED_2(D,target_luminosity,func) - target_luminosity)




def lin_func(lx,a,b):
    return a*lx+b
def D_fit_2():

    D_list_2 = []
    for i in range(len(L_x)):
        lx = L_x[i]
        D_init =lx* 624.15 #conversion from ergs to GeV
        Neutrino_luminosity = lx* 624.15
        result = minimize(objective_function_2, D_init, args=(Neutrino_luminosity,SED_neut_2),method='Nelder-Mead')
        #print(lx, np.log10(result.x[0]))
        D_list_2.append(result.x[0])

    popt_tot_D_2, pcov_tot_D_2= curve_fit(lin_func, L_x, D_list_2, p0=[ D_list_2[0],1]) 
    print(f"Fit parameters for D_list: a={popt_tot_D_2[0]:.2f}, b={popt_tot_D_2[1]:.2f}")

    return D_list_2, popt_tot_D_2


def flux_integrand_2(z,lx):
    Dh = cosmo.hubble_distance.value  # Hubble distance in Mpc
    E_z = cosmo.efunc(z)  # E(z) function from astropy
    D = lin_func(lx, *popt_tot_D_2)
    
    return (Dh / E_z) * SED_neut_2(Ev*(1+z),D,lx*624.15) / (Ev**2 *4 * np.pi * (1 + z)**2 )* integrand(z,lx)/cosmo1.differential_comoving_volume(z).value



def calc_phi_2():
    print(len(E_v_list))
    global Ev

    # Define the integration limits for z and lx
    z_min = z_lower
    z_max = 9
    lx_min = L_x[0]  # Replace with the minimum value for lx
    lx_max = L_x[-1]  # Replace with the maximum value for lx

    # Perform the double integration
    #result, error = dblquad(flux_integrand_2, lx_min, lx_max, lambda lx: z_min, lambda lx: z_max)

    #print(result)


    d_phi_2 = []
    print("Start")
    for i in range(len(E_v_list)):
        Ev = E_v_list[i]

        
        
        result, error = dblquad(flux_integrand_2, lx_min, lx_max,  z_min,  z_max,epsabs=1e-01, epsrel=1.49e-01)
        print(result)
        #print(i)
        d_phi_2.append(result/(3.0857*10**24)**2) #convert from Mpc^2 to cm^2 

    print("End")
    d_phi_2 = np.array(d_phi_2)
    return d_phi_2

selected_model = 'AMPLE_Blazar'
#different models 'SLDDE_RG', 'AMPLE_Blazar', 'AMPLE_FSRQ', 'APLE_BLlac'
set_model_constants(selected_model)

blaz_ir_z1, blaz_ir_z2, blaz_ir_z3, blaz_ir_z4 = lum_den_calc()
blaz_ir_lx1, blaz_ir_lx2, blaz_ir_lx3, blaz_ir_lx4 = red_evo_calc()
blaz_em_j1, blaz_em_j2, blaz_em_j3, blaz_em_tot, blaz_avg_lx1, blaz_avg_lx2, blaz_avg_lx3, blaz_avg_tot = em_avg_calc(blaz_ir_lx1, blaz_ir_lx2, blaz_ir_lx3, blaz_ir_lx4)

blaz_n_tot =(blaz_ir_lx1)/cosmo1.differential_comoving_volume(Z_list).value
popt1, popt2, popt3, popt_tot, popt_tot_em, popt_tot_n = return_fit_varaibles(blaz_avg_lx1, blaz_avg_lx2, blaz_avg_lx3, blaz_avg_tot, blaz_em_tot, blaz_n_tot,4)
D_list, popt_tot_D = D_list_calc(4)
blaz_d_phi = calc_phi()
D_list_2, popt_tot_D_2 = D_fit_2()
blaz_d_phi_2 = calc_phi_2()
