import torch
from torch.fft import fftn as fft, ifftn as ifft
#-------------------------------------------------------------------------------
import parameters
import piezo_strain_tensor
import energy
from fourier_frequency import fourier_frequencies
from write_vtk import write_to_vtk3D
from greens_function import Green_Operator
from solver import Solve_Piezoelectricity
from store_hdf5 import Write_to_HDF5

def Evolve_Sponteneous_Polarization_Anisotropic(device:torch.device, FOLDER:str, grid_points:tuple, grid_space:tuple, time:tuple,
                                elec_field_ext:tuple, eps_ext_applied:tuple, domain_type:str, save_data:str):

    """
    Polarization evolution in 3D with an anisotropic gradient energy.

    device - the torch device

    FOLDER - directory where you want to save results

    grid_points - number of grid points in each diection; (Nx, Ny, Nz)

    grid_space - grid spacing in each direction; (dx, dy, dz)

    time - simulation time setup

    elec_field_ext - applied external electric field in each direction; (x, y, z)

    eps_ext_applied - applied external strain in each direction; (x, y, z)

    domain_type - what is you domain type; "180", "90", "random". see parameters.py

    save_data - save the data into VTK file; "YES" or "NO"
    """

    # Parameters ---------------------------------------------------------------
    C0, K0, P0, e00, e31, e33, e15, G, l, mu, mob = parameters.BTO_MD()
    a1, a2, a3, a4 = parameters.Landau_Polynomial_Coeffs()

    # Simulation and time setup ------------------------------------------------
    Nx, Ny, Nz = grid_points
    dx, dy, dz = grid_space
    nsteps, nt, dt = time

    # Scaling parameters -------------------------------------------------------
    l_scale = l  # length scale   m
    G_scale = G   # energy scale   J/m³
    P_scale = P0         # polarization scale   C/m²
    E_scale = G_scale/P_scale   # (N/m²)/(C/m²) = N/C = J/(mC) = V/m: Electric Field scale
    D_scale = P_scale           # C/m²: Electric Displacement scale
    K_scale = D_scale/E_scale   # (C/m²)/(V/m) = C/(Vm) Dielectric Permittivity scale
    e_scale = P_scale           # C/m²: Piezoelectric scale
    C_scale = G_scale           # N/m²: Elastic Modulus scale
    t_scale = 1E-12           # s: Time scale
    v_scale = l_scale/t_scale   # m/s   velocity scale
    mob_scale = (P_scale/t_scale)/(G_scale/P_scale) # Mobility scale

    # Non dimensionalize -------------------------------------------------------
    C0 = C0/C_scale
    e31, e33, e15 = e31/e_scale, e33/e_scale, e15/e_scale
    K0 = K0/K_scale
    mob = mob/mob_scale
    P0 = P0/P_scale
    dt = dt/t_scale
    dx = dx/l_scale
    dy = dy/l_scale
    dz = dz/l_scale

    # Send to device -----------------------------------------------------------
    C0 = C0.to(device)
    K0 = K0.to(device)

    # Initial polarization -----------------------------------------------------
    P = parameters.initial_polarization(Nx, Ny, domain_type, Nz).to(device)

    # Frequencies --------------------------------------------------------------
    freq = fourier_frequencies(Nx, dx, Ny, dy, Nz, dz, device)

    # Save inital polarization -------------------------------------------------
    if save_data == "YES":
        # zero step
        write_to_vtk3D(FOLDER, 0, "Polarization", P.to(torch.device("cpu")),
                            Nx, Ny, Nz, dx, dy, dz)
        Write_to_HDF5(FOLDER, 0, P.to(torch.device("cpu")))

    # Applied external fields --------------------------------------------------
    eps_ext  = torch.zeros((3,3,Nx,Ny,Nz)).to(device)             # applied elas. field
    eps_ext[0, 0] = eps_ext_applied[0] # in x
    eps_ext[1, 1] = eps_ext_applied[1] # in y
    eps_ext[2, 2] = eps_ext_applied[2] # in z

    E_ext = torch.zeros((3,Nx,Ny,Nz)).to(device)              # applied elec. field
    E_ext[0] = elec_field_ext[0]   # in x
    E_ext[1] = elec_field_ext[1]   # in y
    E_ext[2] = elec_field_ext[2]   # in z

    # Needed for the polarization evolve
    denom = torch.zeros_like(P, dtype=torch.complex64)
    denom[0] = 1 + dt * mob * (freq[0]**2 * (mu+2) + freq[1]**2 + freq[2]**2)
    denom[1] = 1 + dt * mob * (freq[0]**2 + freq[1]**2 * (mu+2) + freq[2]**2)
    denom[2] = 1 + dt * mob * (freq[0]**2 + freq[1]**2 + freq[2]**2 * (mu+2))
    num = torch.zeros_like(P, dtype=torch.complex64)

    # Start the evolution
    for step in range(nsteps):

        # Print the time step
        print('\n---------------------- Time step:\t' + str(step)+'\t----------------------')
        # -------------------------------------------------------------------------
        eP, e0, eps0 = piezo_strain_tensor.Piezo_Strain_Tensor(P, e33, e31, e15, e00, device)
        deP_dPx, deP_dPy, deP_dPz, deps_elas_dPx, deps_elas_dPy, deps_elas_dPz = piezo_strain_tensor.Piezo_Strain_Tensor_Derivative(P, e33, e31, e15, e00, device)
        # -------------------------------------------------------------------------
        G_elas, G_piezo, G_dielec = Green_Operator(C0, e0, K0, freq, Nx, Ny, Nz, device)
        # ------------------------------------------------------------------------
        sigma, D, eps_elas, E = Solve_Piezoelectricity(C0, K0, e0, eP, G_elas, G_piezo, G_dielec,
                                                        eps0, eps_ext, E_ext, Nx, Ny, Nz, P)
        # ------------------------------------------------------------------------
        del G_elas, G_piezo, G_dielec
        # ------------------------------------------------------------------------
        dH_bulk_dP = energy.Bulk_Energy_Derivative(C0, eps_elas, eP, E, deP_dPx, deP_dPy, deP_dPz, deps_elas_dPx, deps_elas_dPy, deps_elas_dPz)
        # ------------------------------------------------------------------------
        dPsi_dP = energy.Landau_Polynomial_Derivative(P, a1, a2, a3, a4)
        # ------------------------------------------------------------------------
        dH_dP = dPsi_dP + dH_bulk_dP
        # ------------------------------------------------------------------------
        dH_dP_k = fft(dH_dP, dim=(1,2,3))
        # ------------------------------------------------------------------------
        del dH_dP, dPsi_dP, dH_bulk_dP
        # ------------------------------------------------------------------------
        P_k = fft(P, dim=(1,2,3))
        # ------------------------------------------------------------------------
        num[0] = P_k[0] - dt * mob * (freq[0] * freq[1] * P_k[1] + freq[2] * freq[0] * P_k[2] + dH_dP_k[0])
        num[1] = P_k[1] - dt * mob * (freq[0] * freq[1] * P_k[0] + freq[2] * freq[1] * P_k[2] + dH_dP_k[1])
        num[2] = P_k[2] - dt * mob * (freq[0] * freq[2] * P_k[1] + freq[2] * freq[1] * P_k[1] + dH_dP_k[2])
        #P_k = (P_k - dt * mob * dH_dP_k) / denom
        P_k = num / denom
        # ------------------------------------------------------------------------
        P = ifft(P_k, dim=(1,2,3)).real
        # ------------------------------------------------------------------------
        if ((step + 1) % nt == 0 and save_data == "YES"):
            # --------------------------------------------------------------------
            write_to_vtk3D(FOLDER, step + 1, "Polarization", P.to(torch.device("cpu")),
                                Nx, Ny, Nz, dx, dy, dz)
            Write_to_HDF5(FOLDER, 0, P.to(torch.device("cpu")))
        # --------------------------------------------------------------------
        del sigma, D, eps_elas, E
    # ----------------------------------------------------------------------------


def Evolve_Sponteneous_Polarization_Isotropic(device:torch.device, FOLDER:str, grid_points:tuple, grid_space:tuple, time:tuple,
                                elec_field_ext:tuple, eps_ext_applied:tuple, domain_type:str, save_data:str):
    # ----------------------------------------------------------------------------
    """
    Polarization evolution in 3D with an isotropic gradient energy.

    device - the torch device

    FOLDER - directory where you want to save results

    grid_points - number of grid points in each diection; (Nx, Ny, Nz)

    grid_space - grid spacing in each direction; (dx, dy, dz)

    time - simulation time setup

    elec_field_ext - applied external electric field in each direction; (x, y, z)

    eps_ext_applied - applied external strain in each direction; (x, y, z)

    domain_type - what is you domain type; "180", "90", "random". see parameters.py

    save_data - save the data into VTK file; "YES" or "NO"
    """

    # import material data
    G, l_wall, P0, mob, e00, k_sep, k_grad, e31, e33, e15, C0, K0 = parameters.BTO_3D_Schrade()
    a1, a2, a3, a4 = parameters.Landau_Polynomial_Coeffs_Schrade()
    # ----------------------------------------------------------------------------
    Nx, Ny, Nz = grid_points
    dx, dy, dz = grid_space
    nsteps, nt, dt = time
    # ----------------------------------------------------------------------------
    # Scaling parameters
    l_scale = l_wall  # length scale   m
    G_scale = G/l_wall   # energy scale   J/m³
    P_scale = P0         # polarization scale   C/m²
    E_scale = G_scale/P_scale   # (N/m²)/(C/m²) = N/C = J/(mC) = V/m: Electric Field scale
    D_scale = P_scale           # C/m²: Electric Displacement scale
    K_scale = D_scale/E_scale   # (C/m²)/(V/m) = C/(Vm) Dielectric Permittivity scale
    e_scale = P_scale           # C/m²: Piezoelectric scale
    C_scale = G_scale           # N/m²: Elastic Modulus scale
    t_scale = 1E-12           # s: Time scale
    v_scale = l_scale/t_scale   # m/s   velocity scale
    mob_scale = (P_scale/t_scale)/(G_scale/P_scale) # Mobility scale
    # ----------------------------------------------------------------------------
    # Non dimensionalize
    C0 = C0/C_scale
    e31, e33, e15 = e31/e_scale, e33/e_scale, e15/e_scale
    K0 = K0/K_scale
    mob = mob/mob_scale
    P0 = P0/P_scale
    dt = dt/t_scale
    dx = dx/l_scale
    dy = dy/l_scale
    dz = dz/l_scale
    elec_field_ext[0] = elec_field_ext[0]/E_scale
    elec_field_ext[1] = elec_field_ext[1]/E_scale
    elec_field_ext[2] = elec_field_ext[2]/E_scale
    # ----------------------------------------------------------------------------
    C0 = C0.to(device)
    K0 = K0.to(device)
    # ----------------------------------------------------------------------------
    # initial polarization
    P = parameters.initial_polarization(Nx, Ny, domain_type, Nz).to(device)
    # ----------------------------------------------------------------------------
    # frequencies
    freq = fourier_frequencies(Nx, dx, Ny, dy, Nz, dz, device)
    # Save inital polarization -------------------------------------------------
    if save_data == "YES":
        # zero step
        write_to_vtk3D(FOLDER, 0, "Polarization", P.to(torch.device("cpu")),
                            Nx, Ny, Nz, dx, dy, dz)
        Write_to_HDF5(FOLDER, 0, P.to(torch.device("cpu")))
        # ----------------------------------------------------------------------------
    # Applied external fields --------------------------------------------------
    eps_ext  = torch.zeros((3,3,Nx,Ny,Nz)).to(device)             # applied elas. field
    eps_ext[0, 0] = eps_ext_applied[0] # in x
    eps_ext[1, 1] = eps_ext_applied[1] # in y
    eps_ext[2, 2] = eps_ext_applied[2] # in z

    E_ext = torch.zeros((3,Nx,Ny,Nz)).to(device)              # applied elec. field
    E_ext[0] = elec_field_ext[0]   # in x
    E_ext[1] = elec_field_ext[1]   # in y
    E_ext[2] = elec_field_ext[2]   # in z
    # ----------------------------------------------------------------------------
    denom = 1 + dt * mob * k_grad * (freq[0]**2 + freq[1]**2 + freq[2]**2)
    # ----------------------------------------------------------------------------
    for step in range(nsteps):
        # applied as sine function
        # ------------------------------------------------------------------------
        print('\n---------------------- Time step:\t' + str(step)+'\t----------------------')
        # -------------------------------------------------------------------------
        eP, e0, eps0 = piezo_strain_tensor.Piezo_Strain_Tensor(P, e33, e31, e15, e00, device)
        deP_dPx, deP_dPy, deP_dPz, deps_elas_dPx, deps_elas_dPy, deps_elas_dPz = piezo_strain_tensor.Piezo_Strain_Tensor_Derivative(P, e33, e31, e15, e00, device)
        # -------------------------------------------------------------------------
        G_elas, G_piezo, G_dielec = Green_Operator(C0, e0, K0, freq, Nx, Ny, Nz, device)
        # ------------------------------------------------------------------------
        sigma, D, eps_elas, E = Solve_Piezoelectricity(C0,K0,e0,eP,G_elas,G_piezo,G_dielec,eps0,eps_ext,E_ext,Nx,Ny,Nz,P)
        # ------------------------------------------------------------------------
        del G_elas, G_piezo, G_dielec
        # ------------------------------------------------------------------------
        dH_bulk_dP = energy.Bulk_Energy_Derivative(C0, eps_elas, eP, E, deP_dPx, deP_dPy, deP_dPz, deps_elas_dPx, deps_elas_dPy, deps_elas_dPz)
        # ------------------------------------------------------------------------
        dPsi_dP = energy.Landau_Polynomial_Derivative(P, a1, a2, a3, a4)
        # ------------------------------------------------------------------------
        dH_dP =  k_sep * dPsi_dP + dH_bulk_dP
        # ------------------------------------------------------------------------
        dH_dP_k = fft(dH_dP, dim=(1,2,3))
        del dH_dP, dPsi_dP, dH_bulk_dP
        # ------------------------------------------------------------------------
        P_k = fft(P, dim=(1,2,3))
        # ------------------------------------------------------------------------
        P_k = (P_k - dt * mob * dH_dP_k) / denom
        # ------------------------------------------------------------------------
        P = ifft(P_k, dim=(1,2,3)).real
        # ------------------------------------------------------------------------
        if ((step + 1) % nt == 0 and save_data == "YES"):
            # --------------------------------------------------------------------
            write_to_vtk3D(FOLDER, step + 1, "Polarization", P.to(torch.device("cpu")),
                                Nx, Ny, Nz, dx, dy, dz)
            Write_to_HDF5(FOLDER, step + 1, P.to(torch.device("cpu")))
            # --------------------------------------------------------------------
        del sigma, D, eps_elas, E
    # ----------------------------------------------------------------------------
    # print('Length scale in m = ', l_scale)
    # print('Energy scale   J/m³ = ', G_scale)
    # print('Polarization scale   C/m² = ', P_scale)
    # print('Electric Field scale V/m = ', E_scale)
    # print('Electric Displacement scale C/m² = ', D_scale)
    # print('Dielectric Permittivity scale C/(Vm) = ', K_scale)
    # print('Piezoelectric scale C/m² = ', e_scale)
    # print('Time scale s = ', t_scale)
    # print('Mobility scale= ', mob_scale)
