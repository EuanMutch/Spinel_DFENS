# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:56:36 2017

@author: ejfm2
"""
#FENICS
# from dolfin import *

# May need to rewrite so that intensive parameters are required for classes?
# read command line arguments
import os
import sys
import time

# numerical functions
import numpy as np
import scipy.integrate as integrate
import math as m
from scipy import special as sp
from scipy.interpolate import interp1d
from dolfin import *


# data
import pandas as pd

# MC
import pymultinest
import json

# plotting
import matplotlib.pyplot as plt
#import seaborn as sb

# multi thread
from mpi4py import MPI
from petsc4py import PETSc



# custom
import pmc
import KC_fO2 as kc

set_log_active(False)

# SETUP MPI variables-------------------------------------------------
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
#comm = PETSc.Comm(MPI.COMM_SELF)
comm = mpi_comm_self()

# FUNCTIONS : DIFFUSION-------------------------------------------------
# Diffusion Functions
       
def D(C, T_i, P, lnfO2, lnD0, clnfO2, cCr, cT_i, cP, cT_iP):
    
    lnD = lnD0 + clnfO2*lnfO2 + cCr*C + cT_i*T_i + cP*P + cT_iP*T_i*P
    
    D = exp(lnD)*Constant(1e12)
    
    return D

        
def modc(model, fcoords, dist):
    mod_i = model #[::-1]
    intp = interp1d(fcoords, mod_i,bounds_error=False, fill_value= mod_i[0]) #[:,0]
    modx = intp(dist)
    return modx
        

def ICs_import(dist_init, IC, xs, rc, Q):
    i_int = interp1d(dist_init, IC ,bounds_error=False, fill_value=rc)
    ic = i_int(xs)
    Cinit = ic[dof_to_vertex_map(Q)] 
    return Cinit
        
# Model function

def mod_diff(cube, nparams):
    
    t, T, fe3, P, DCr = cube[0], cube[1], cube[2], cube[3], cube[4:nparams]
    #t, T, aSiO2, A_PlMg, DMgpl = cube[0], cube[1], cube[2], cube[3], cube[4:nparams]
    t *= (86400.0*365.0) # convert time into seconds # may have to start from years rather than days
    dt = t/300.0 # only 50 time steps
    dT = Constant(dt)
    P *= 1.0e8
    T += 273.0
    T_i = 1/T
#==============================================================================
#     L = max(dist)
#     n = 299
#==============================================================================
    
#==============================================================================
#     L_pl = max(dist_pl)
#     n_pl = 299
#==============================================================================
    R = Constant(0.008314)
    
    lnfO2 = kc.fO2calc_eq7(melt_comp, T, P, fe3)  # fO2 in bars from Kress and Carmichael
    #A_Mg = Constant(A_PlMg)
    #B_Mg = Constant(B_PlMg)
    
    # 2D mesh will be defined outside loop 
    
    # Adjust mesh depending on numerical stability
#==============================================================================
#     Dd = m.exp(DMgpl[0] + DMgpl[1]*0.7 + DMgpl[2]*aSiO2 + DMgpl[3]*T_i)*1e12
#     
#     ms = L/n
#     #print(ms)
#     CFL = (dt*Dd)/(ms**2)
#     
#     if CFL > 0.5:
#         Dx = m.sqrt((1/0.49)*dt*Dd)
#         n = int(max(dist_pl)/Dx)
#         n_pl = n
#         #L = max(dist)
#         L_pl = max(dist_pl)
#         #olM = Mesh(n1, L1)  
#         
#     if n_pl < 1:
#         n_pl = 2
# 
#     mesh = IntervalMesh(comm, n, 0.0, L)
#     xs = np.linspace(0, L, n+1)
#==============================================================================

    Q = FunctionSpace(mesh_r, "CG", 1)
    # Define boundaries
#==============================================================================
#     def left_boundary(x):
#         return near(x[0], 0.0)
# 
#     def right_boundary(x):
#         return near(x[0], L)
#==============================================================================
        
    # Construct weak form
    C0 = Function(Q)       # Composition at current time step
    C1 = Function(Q)  # Composition at next time step
    S = TestFunction(Q)
        
    theta = Constant(0.5)
    C_mid = theta*C1 + Constant((1.0-theta))*C0

    T_i = Constant(T_i)
    P = Constant(P)
    lnfO2 = Constant(lnfO2)
    
    F = S*(C1-C0)*dx + dT*(inner(D(C_mid, T_i, P, lnfO2, Constant(DCr[0]), Constant(DCr[1]), Constant(DCr[2]), Constant(DCr[3]), Constant(DCr[4]), Constant(DCr[5]))*grad(S), grad(C_mid)))*dx
    
#==============================================================================
#     Cbc0 = DirichletBC(Q, Constant(rim_comp), left_boundary) 
#     Cbcs = [Cbc0_fo]
#==============================================================================

    u0 = Constant(rim_comp)
    OB = Outer()

    Cbcs = DirichletBC(Q, u0, OB)
        
    if IC_i == True:
        Cinit = ICs_import(dist_init, inicon['Cr_no'].values, xs, rim_comp, Q)
        C0.vector()[:] = Cinit

    else:
        Cinit = Constant(core_comp)
        C0.interpolate(Cinit)


    # Timestepping
    i = 0
    
    while i < t:
        solve(F==0, C1, Cbcs)
        C0.assign(C1)
        
        i += dt # Need to decide on time increment
        
# Convert FEniCS output into numpy array interpolated at observation distances        
    #rcoords = mesh.coordinates()

    s = []
    f = []
    
    for p in zip(x, y):
        q = Point(p[0], p[1])
        s.append((q-p0).norm())
        f.append(C0(q))
    Crn_mod = f
        
    Cr_mod = modc(Crn_mod, s, dist)
    
    gl_mod = Cr_mod
       
    return gl_mod

def mod_diff_plot(cube, nparams):
    
    t, T, fe3, P, DCr = cube[0], cube[1], cube[2], cube[3], cube[4:nparams]
    #t, T, aSiO2, A_PlMg, DMgpl = cube[0], cube[1], cube[2], cube[3], cube[4:nparams]
    t *= (86400.0*365.0) # convert time into seconds # may have to start from years rather than days
    dt = t/150.0 # only 50 time steps
    dT = Constant(dt)
    P *= 1.0e8
    T += 273.0
    T_i = 1/T
#==============================================================================
#     L = max(dist)
#     n = 299
#==============================================================================
    
#==============================================================================
#     L_pl = max(dist_pl)
#     n_pl = 299
#==============================================================================
    R = Constant(0.008314)
    
    lnfO2 = kc.fO2calc_eq7(melt_comp, T, P, fe3)  # fO2 in bars from Kress and Carmichael
    #A_Mg = Constant(A_PlMg)
    #B_Mg = Constant(B_PlMg)
    
    # 2D mesh will be defined outside loop 
    
    # Adjust mesh depending on numerical stability

    Q = FunctionSpace(mesh_r, "CG", 1)
    # Define boundaries
        
    # Construct weak form
    C0 = Function(Q)       # Composition at current time step
    C1 = Function(Q)  # Composition at next time step
    S = TestFunction(Q)
        
    theta = Constant(0.5)
    C_mid = theta*C1 + Constant((1.0-theta))*C0

    T_i = Constant(T_i)
    P = Constant(P)
    lnfO2 = Constant(lnfO2)
    
    F = S*(C1-C0)*dx + dT*(inner(D(C_mid, T_i, P, lnfO2, Constant(DCr[0]), Constant(DCr[1]), Constant(DCr[2]), Constant(DCr[3]), Constant(DCr[4]), Constant(DCr[5]))*grad(S), grad(C_mid)))*dx
    

    u0 = Constant(rim_comp)
    OB = Outer()

    Cbcs = DirichletBC(Q, u0, OB)
        
    if IC_i == True:
        Cinit = ICs_import(dist_init, inicon['Cr_no'].values, xs, rim_comp, Q)
        C0.vector()[:] = Cinit

    else:
        Cinit = Constant(core_comp)
        C0.interpolate(Cinit)


    # Timestepping
    i = 0
    
    while i < t:
        solve(F==0, C1, Cbcs)
        C0.assign(C1)
        
        i += dt # Need to decide on time increment
        
# Convert FEniCS output into numpy array interpolated at observation distances        
    #rcoords = mesh.coordinates()
    
    s = []
    f = []
    for p in zip(x, y):
        q = Point(p[0], p[1])
        s.append((q-p0).norm())
        f.append(C0(q))
    Crn_mod = f
        
    Cr_mod = modc(Crn_mod, s, dist)

    fig = plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    p = plot(C0, cmap='viridis')
    plt.plot(x,y, linewidth=2, color='black')
    plt.colorbar(p)
    plt.subplot(1,2,2)
    plt.errorbar(dist, obs['Cr_no'].values, yerr = obs['Cr_no_sd'].values,fmt='o', color='red', label='data')
    plt.plot(disti, inicon_cr, color = 'black')
    plt.plot(dist, Cr_mod, color = 'blue')
    plt.xlabel(r'Distance ($\mu$m)')
    plt.ylabel('Cr#') 
    plt.tight_layout()

    plt.savefig(f_out + 'fit.png')
    plt.close()

    plt.plot(s, Crn_mod, color ='blue')
    plt.errorbar(dist, obs['Cr_no'].values, yerr = obs['Cr_no_sd'].values,fmt='o', color='red', label='data')
    plt.savefig(f_out + 'fit2.png')
    plt.close()
    
    gl_mod = Cr_mod
       
    return gl_mod

    
# loglikelihood function
    
# ---------------- log likelihood function
def loglike(model, data, data_err):
    return np.sum( -0.5*((model - data)/data_err)**2 )
    

#------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------    
if __name__ == "__main__":
    
    
        ## DATA-------------------------------------------------
        
        # import data from files
        
    f_dat = sys.argv[1]
    f_melt = sys.argv[2]
    f_modpar = sys.argv[3]
    f_mesh = sys.argv[4]
    #f_err = sys.argv[6]
    f_mk = sys.argv[5]
    f_out = sys.argv[6]
    
    if len(sys.argv) == 8:
        f_inicon = sys.argv[7]
        inicon = pd.read_csv(f_inicon) #np.genfromtxt(f_inicon,delimiter=',',dtype=None, names=True)
        dist_init = inicon['Distance'].values
        IC_i = True
    else:
        IC_i = False
    
    
    obs = pd.read_csv(f_dat) # Import analytical model comps
    
    mc = np.genfromtxt(f_melt,delimiter=',',dtype=None, names=True)
    obs_mpar = np.genfromtxt(f_modpar,delimiter=',',dtype=None, names=True)
    
    
    #ele_err = np.genfromtxt(f_err,delimiter=',',dtype=None, names=True)
    df_mk = pd.read_csv(f_mk)
    
    # Import vcov files
    
    CrAl_all_vcov = pd.read_csv('./D_vcov/Spinel_CrAl_all_ss_vcov.csv')
    
    
    # Consider combining uncertainties and observations into a single file
    # Also for plag, probably convenient to combine all data sources into a single file with uncertainties as well.
    
    # combine different elements into a single global array
    
       
    gl_obs = obs['Cr_no'].values 
    gl_err = obs['Cr_no_sd'].values
    
    
    dist = obs['Distance'].values
    
    melt_comp = mc['Composition']
    
    rim_comp, core_comp = df_mk['Cr_no_markers'][0], df_mk['Cr_no_markers'][1]
    
    
    # Generate and refine mesh
    
    mesh = Mesh(comm, f_mesh)
    
    cell_markers =  CellFunction("bool", mesh)
    cell_markers.set_all(False)
    
    pc = Point(float(df_mk['mesh_centre'][0]), float(df_mk['mesh_centre'][1]))
    
    x = np.linspace(df_mk['P_start'][0],df_mk['P_end'][0], 30)
    y = np.linspace(df_mk['P_start'][1], df_mk['P_end'][1], 30)
    p0 = Point(x[0], y[0])
    
    for cell in cells(mesh):
        if cell.distance(pc) > df_mk['C_dist'][0]:
                cell_markers[cell] = True
    
    mesh_r = refine(mesh, cell_markers)
    
    class Outer(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    
    
    #n, L = 299, max(dist)
    #olM = Mesh(n, L)  # olivine mesh
    # check for output
    dir = '/'.join(f_out.split('/')[:-1])
    if not os.path.exists(dir):
        print("Making directory for output:", dir)
        os.makedirs(dir)
    
    # PARAMETER SETUP-------------------------------------------------
    # number of weight parameters (melt region sections = N(weights) + 1)
    
    parameters = ["t", "T", "fe_3", "P", "lnD0", "clnfO2", "cCr_no", "cT_i", "cP", "cT_iP"] 
    pti = np.empty([len(parameters)])
    pti[:] = np.nan
    pti[4:] = 0
    #pti[4:] = 0
    
    cov_Cr = CrAl_all_vcov.as_matrix(columns=['(Intercept)', 'lnfO2', 'Cr_no', 'T_i', 'P_Pa', 'T_i:P_Pa'])
    
    cov_s = np.array([cov_Cr])
    
    #==============================================================================
    # for i in Ds.index.values:
    #     parameters.append(i)
    #==============================================================================
    n_dim = len(parameters)
    n_params = n_dim
    
    # MCMC-------------------------------------------------
    # setup prior cube
    #tprior = str(obs_mpar['tprior'][0])
    
    ptype = ["MG"] * n_dim
    ptype[0:1] = ["LU"]   #type of distribution LU = ln uniform, U = uniform
    ptype[1:4] = ["G"]*3
    
    pcube = np.full((n_dim,2), np.nan)
    pcube[0,:] = [2.0, 6.0] #[float(obs_mpar['t'][0]), float(obs_mpar['t'][1])]     # time (days)
    pcube[1,:] = [1215.0, 30.0] #[float(obs_mpar['T'][0]), float(obs_mpar['T'][1])]      # T 
    pcube[2,:] = [0.14, 0.02] #[float(obs_mpar['fe3_fet'][0]), float(obs_mpar['fe3_fet'][1])]      # fe3/fet
    pcube[3,:] = [8.0, 1.4] #[float(obs_mpar['P_kbar'][0]), float(obs_mpar['P_kbar'][1])] # P (kbar)
    pcube[4:,0] = np.array([-8.595, 9.158e-3, 3.611, -5.283e4, -6.034e-10, -9.586e-7]) # DFo
    
    invMC = pmc.Pmc(n_dim, ptype, pcube, cov_s, pti, loglike, mod_diff,
                gl_obs, gl_err,
                evidence_tolerance = 0.5, sampling_efficiency = 0.8,
                n_live_points = 400, n_params= n_params)
    
    json.dump(parameters, open(f_out + 'params.json', 'w')) # save parameter names
    
    invMC.run_mc(f_out)
    result = invMC.result(f_out)
    
    # fiddle around with this to gaussian with variance and covariance, manually change prior cube
    
    if rank == 1 :
            # PLOT-------------------------------------------------
        # Prevents having to generate 4 plots when weaving on 4 separate processors
        bf_params = result.get_best_fit()['parameters']
        #print(bf_params)
        
        # extract best fit parameter for time and rerun in model, then plot up
        
        #C_Fo_bf = C_bf
        
        if IC_i == True:
            inicon_cr = inicon['Cr_no'].values
            disti = dist_init
        
        else:
            inicon_cr = np.ones(len(obs['Cr_no'].values))*core_comp
            disti = dist
        
        C_bf = mod_diff_plot(bf_params, len(bf_params))
        
        # WRITE-------------------------------------------------
        #==============================================================================
        #     pd.DataFrame(dat_cov).to_csv(f_out + 'data_cv.dat', sep=' ')
        #     pd.DataFrame(dat_cvE).to_csv(f_out + 'data_cvE.dat', sep=' ')
        #==============================================================================
        pd.DataFrame({'CrAl_bf': C_bf}).to_csv(f_out + 'mod_cv.csv', sep=',')
            


        
        



