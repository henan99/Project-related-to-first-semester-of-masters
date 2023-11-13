import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy.integrate import quad
from astropy.cosmology import Planck15   # You can choose a different cosmology if needed
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM






N = 100
L_x  = np.logspace(42,47,1000)
x = L_x/(10.0**(44))


model_constants = {
    'SLDDE': {
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
        'corr_fac': 400,
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
        # Add more constants for model 1
    },
    # Add constants for other models as needed
}


selected_model = 'SLDDE'

# Access constants for the selected model
constants = model_constants[selected_model]
#print(constants)

A = constants['A']
L_star = constants['L_star']
gamma1 = constants['gamma1']
gamma2 = constants['gamma2']
v_1 = constants['v1']
v_2 = constants['v2']
z_c = constants['z_c']
L_c = constants['Lc']
alpha =constants['alpha']
corr_fac = constants['corr_fac']

#sigma = A*np.log(10)*(L_x/L_star)**(1-gamma2)
#sigma2 = A*((L_x/L_star)**(gamma1)+(L_x/L_star)**(gamma2))**(-1)

#corr_fac = 400
#redshift = 1.0




# Calculate the comoving volume up to the given redshift
#comoving_volume = Planck15.comoving_volume(redshift)

# Print the comoving volume (in Mpc^3)
#print(f"Comoving volume at z={redshift}: {comoving_volume:.2e}")

def Psi_Ajello(A,Lx,L_star,gamma2):
    return A*np.log(10)*(Lx/L_star)**(1-gamma2)

def Psi_Ueda(A,Lx,L_star,gamma2,gamma1):
     return A*np.log(10)*((Lx/L_star)**(gamma1)+(Lx/L_star)**(gamma2))**(-1)


#z = np.linspace(0,2.5,N)
def z_star(Lx):
    if (Lx < L_c):
        return z_c *(Lx/L_c)**alpha
    else:
        return z_c

def e_z(z,Lx):
    z_s =z_star(Lx)
    if z<=z_s:
        return (1+z)**(v_1)
    else: 
        return e_z(z_s, Lx) *(1+z/(1+z_s))**(v_2)

def e_z_PL(z,v1,v2):
    return (1+z)**(v1 + v2*z)


    #return (1+z)**(v_1+v_2*z)

#sigma_e = A*((L_x/L_star)**(gamma1)+(L_x/L_star)**(gamma2))**(-1)

#sigma_tot =  A*np.log(10)*L_x*((L_x/L_star)**(gamma1)+(L_x/L_star)**(gamma2))**(-1)
#for i in range(N):  
#    sigma_tot += A*np.log(10)*L_x*((L_x/e_z[i]/L_star)**(gamma1)+(L_x/e_z[i]/L_star)**(gamma2))**(-1)



Omega_matter = 1.0  # For a matter-dominated universe, Ω_m = 1
H0 = 70 * u.km / (u.s * u.Mpc)  # Replace with your desired H0 value

# Create a custom cosmology for a flat matter-dominated universe
cosmo1 = FlatLambdaCDM(H0=H0, Om0=Omega_matter)







# Define the function you want to integrate, which depends on 'z' and 'Lx'
def integrand(z, Lx):
    # Define your function here, for example:
    #return A*np.log(10)*((L_x/e_z(z,Lx)/L_star)**(gamma1)+(L_x/e_z(z,Lx)/L_star)**(gamma2))**(-1)*Planck15.comoving_volume(z).value
    return Psi_Ueda(A,Lx*1/e_z(z,Lx),L_star,gamma2, gamma1)*cosmo1.comoving_volume(z).value


# Define the limits of integration for 'z'
z_lower = 0
z_upper = 2.5


L_x_lower = 10**(42)
L_x_upper = 10**(43.5)
# Perform the integration

integral_results_z = []

# Perform the integration for each 'Lx' value
for Lx in L_x:
    result, error = quad(lambda z: integrand(z, Lx), z_lower, z_upper)
    integral_results_z.append(corr_fac*result)

# Convert the integral results to a NumPy array
integral_results_z = np.array(integral_results_z)
#print(integral_results_z)
Z_list = np.arange(0,9,50)
integral_results_lx = []

# Perform the integration for each 'Lx' value
for z_list in Z_list:
    result, error = quad(lambda lx: integrand(z_list, lx), L_x_lower, L_x_upper)
    integral_results_lx.append(corr_fac*result)

# Convert the integral results to a NumPy array
integral_results_lx = np.array(integral_results_lx)

#x = L_x/(10**(44))
x= np.array(x)

plt.figure(1)
plt.plot(np.log10(x),np.log10(integral_results_z))
plt.ylabel("dN/dlog(Lx)")
plt.xlabel("log[Lx/L44]")
plt.grid()
plt.savefig('my_plot_newfunc_z.jpg', format='jpeg')

plt.figure(2)
plt.plot(Z_list,np.log10(integral_results_lx))
plt.ylabel("dN/dz")
plt.xlabel("redshift")
plt.grid()
plt.savefig('my_plot_newfunc_lx.jpg', format='jpeg')

print("finished")


plt.close()


