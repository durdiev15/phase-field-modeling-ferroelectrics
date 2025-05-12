"""
All data can be stored as .h5 file

Write_to_HDF5 -> in this function you always need to give 5 KEYWORD ARGUMENTS:

    - folder -> directory to save results
    - step   -> the time step
    - P -> polarization
    - Elas_strain  ->  elastic strain
    - Elec_field  ->  electric field
"""

import h5py

def Write_to_HDF5(HDF_RESULTS_FILE, step, P, **kwargs):

    print(step)

    hdf = h5py.File(HDF_RESULTS_FILE,'a')
    time = '/time_'+str(int(step))

    # Polarization tensor
    hdf.create_dataset('Polarization/Px'+str(time), data = P[0])
    hdf.create_dataset('Polarization/Py'+str(time), data = P[1])
    hdf.create_dataset('Polarization/Pz'+str(time), data = P[2])

    for key, value in kwargs.items():

        if key == 'ElasticStrain':

            # Elastic strain tensor
            hdf.create_dataset('Elastic strain/strain_XX'+str(time), data = value[0,0])
            hdf.create_dataset('Elastic strain/strain_XY'+str(time), data = value[0,1])
            hdf.create_dataset('Elastic strain/strain_XZ'+str(time), data = value[0,2])
            hdf.create_dataset('Elastic strain/strain_YY'+str(time), data = value[1,1])
            hdf.create_dataset('Elastic strain/strain_YZ'+str(time), data = value[1,2])
            hdf.create_dataset('Elastic strain/strain_ZZ'+str(time), data = value[2,2])

        elif key == 'ElectricField':

            # Electric field tensor
            hdf.create_dataset('Electric field/Ex'+str(time), data = value[0])
            hdf.create_dataset('Electric field/Ey'+str(time), data = value[1])
            hdf.create_dataset('Electric field/Ez'+str(time), data = value[2])

        elif key=='SimulationParams':
            sim_params = value
            # Saving simulation parameters to HDF5
            sim_params_group = hdf.create_group('Simulation_Parameters')  # Group for simulation parameters
            sim_params_group.create_dataset('P_scale', data=sim_params['P_scale'])
            sim_params_group.create_dataset('l_scale', data=sim_params['l_scale'])
            sim_params_group.create_dataset('G_scale', data=sim_params['G_scale'])
            sim_params_group.create_dataset('E_scale', data=sim_params['E_scale'])
            sim_params_group.create_dataset('K_scale', data=sim_params['K_scale'])
            sim_params_group.create_dataset('t_scale', data=sim_params['t_scale'])
            sim_params_group.create_dataset('a1', data=sim_params['a1'])
            sim_params_group.create_dataset('a2', data=sim_params['a2'])
            sim_params_group.create_dataset('a3', data=sim_params['a3'])
            sim_params_group.create_dataset('a4', data=sim_params['a4'])
            sim_params_group.create_dataset('C11', data=sim_params['C11'])
            sim_params_group.create_dataset('C12', data=sim_params['C12'])
            sim_params_group.create_dataset('C44', data=sim_params['C44'])
            sim_params_group.create_dataset('eps_spont0', data=sim_params['eps_spon0'])
            sim_params_group.create_dataset('mob', data=sim_params['mob'])
            sim_params_group.create_dataset('Nx', data=sim_params['Nx'])
            sim_params_group.create_dataset('Ny', data=sim_params['Ny'])
            sim_params_group.create_dataset('Nz', data=sim_params['Nz'])
            sim_params_group.create_dataset('dx', data=sim_params['dx'])
            sim_params_group.create_dataset('dy', data=sim_params['dy'])
            sim_params_group.create_dataset('dz', data=sim_params['dz'])
            sim_params_group.create_dataset('nsteps', data=sim_params['nsteps'])
            sim_params_group.create_dataset('nt', data=sim_params['nt'])
            sim_params_group.create_dataset('dt', data=sim_params['dt'])



    hdf.close()
