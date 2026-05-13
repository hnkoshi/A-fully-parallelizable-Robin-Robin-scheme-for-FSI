from dolfin import *
from mpi4py import MPI
from Fluid import Fluid_Solver
from Structure import Nonlinear_Structure_Solver

if __name__ == "__main__":

    # physical constant
    para = {'skip'     : 10,
            'rho_s'    : 1000.0,
            'mu_s'     : 2.0e+6,
            'lambda_s' : 8.0e+6,
            'rho_f'    : 1000.0,
            'mu_f'     : 1.0,
            'L_1'      : 4500.0,
            'L_2'      : 4500.0,
            'T'        : 10.001,
            'dt'       : 5e-4,
            'recompute': 4,
            'Structure_Type': 'Nonlinear', # Linear -- Drop F and det(F)
            'path_f'  : "./benchmark_mesh/fluid.xdmf",
            'path_s'  : "./benchmark_mesh/beam.xdmf",
            'boundary_f': "./benchmark_mesh/fluid_boundary.xdmf",
            'boundary_s': "./benchmark_mesh/beam_boundary.xdmf"}
    
    # MPI Communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # midd = int(size/2)
    midd = size - 2
    color = 0 if rank < midd else 1

    if color == 0:
        # Fluid Subproblem
        group_comm = comm.Split(color)
        group_rank = group_comm.Get_rank()
        Fluid_Solver(group_comm, comm, group_rank, midd, para)
    
    else:
        # Structure Subproblem
        group_comm = comm.Split(color)
        group_rank = group_comm.Get_rank()
        Nonlinear_Structure_Solver(group_comm, comm, group_rank, para)





