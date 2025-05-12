import numpy as np
import itertools
import torch

# Functions to convert Voigt to full tensors
def full_3x3_to_Voigt_6_index_3D(i, j):
    if i == j:
        return i
    return 6-i-j

def Voigt_to_full_Tensor_3D(C11, C12, C44):

    C_cub = np.array([[C11, C12, C12, 0, 0, 0],
                      [C12, C11, C12, 0, 0, 0],
                      [C12, C12, C11, 0, 0, 0],
                      [0,   0,   0, C44, 0, 0],
                      [0,   0,   0, 0, C44, 0],
                      [0,   0,   0, 0, 0, C44]])
    
    C_out = np.zeros((3,3,3,3), dtype=np.float32)

    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index_3D(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index_3D(k, l)
        C_out[i, j, k, l] = C_cub[Voigt_i, Voigt_j]

    return C_out

# Inital polarization
def Initial_Polarization(Nx, Ny, domain_type, Nz):

    torch.manual_seed(51254)
    print("Initial polarization for 3D simulations")

    if domain_type == 'random':

        # generates random initial polarization
        Px = (0.1 * (2.0 * torch.rand(Nx, Ny, Nz) - 1.0))#.requires_grad_()
        Py = (0.1 * (2.0 * torch.rand(Nx, Ny, Nz) - 1.0))#.requires_grad_()
        Pz = (0.1 * (2.0 * torch.rand(Nx, Ny, Nz) - 1.0))#.requires_grad_()

    elif domain_type == '180':

        # generates a single 180° DW
        print("180° domain type")
        Px = torch.zeros((Nx,Ny,Nz))
        Py = torch.zeros((Nx,Ny,Nz))
        Pz = torch.ones((Nx,Ny,Nz))
        Pz[:,int(Ny/2):,:]=-1

    elif domain_type == '90':

        # generates 90° DW structure
        from scipy import linalg
        first_row = torch.zeros(Nz)
        nn=Nz/2
        first_row[:int(nn)]=1
        first_row[2*int(nn)-1:3*int(nn)]=1

        first_col = torch.ones(Ny)
        first_col[:int(nn)]=0
        first_col[2*int(nn)-1:3*int(nn)]=0

        Pyy = torch.from_numpy(linalg.toeplitz(first_col, first_row))
        Pxx = (1-Pyy)*-1
        # create 90° DW in yz plane
        Px = torch.zeros((Nx,Ny,Nz))
        Py = torch.zeros((Nx,Ny,Nz))
        Py[:] = Pxx
        Pz = torch.zeros((Nx,Ny,Nz))
        Pz[:] = Pyy

        print(Px.shape, Py.shape, Pz.shape)

    elif domain_type == 'minus_z':
        Px = torch.zeros((Nx,Ny,Nz))
        Py = torch.zeros((Nx,Ny,Nz))
        Pz = -torch.ones((Nx,Ny,Nz))

    elif domain_type == 'plus_z':
        Px = torchp.zeros((Nx,Ny,Nz))
        Py = torch.zeros((Nx,Ny,Nz))
        Pz = torch.ones((Nx,Ny,Nz))

    return torch.stack((Px, Py, Pz))


# C0 = torch.tensor(Voigt_to_full_Tensor_3D(C_cub))