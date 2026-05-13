from dolfin import *
from mpi4py import MPI
import numpy as np
import sys
import time
from scipy.spatial import KDTree
from dolfin import PETScOptions

def Newton_Solver_Recompute_Robust(group_rank,
    F, u, bcs, J, recompute, lin_solver,
    max_it=20, atol=1e-8, rtol=1e-6,
    # --- robustness options (defaults chosen to be safe) ---
    recompute_on_increase=True,      # like github: if residual increases -> recompute J
    use_raw_residual=True,           # avoid false convergence caused by bc.apply on b
    ident_zeros=True,                # like github: avoid zero diagonals
    lmbda=1.0,                       # damping factor
    diverge_guard=1e20,              # detect blow-up / NaN
    verbose=False
):

    it = 0
    A = None          # Jacobian matrix
    b = None          # RHS used for solve (BC applied)
    b_raw = None      # raw residual vector for convergence check (no BC applied)
    du = u.copy(deepcopy=True)  # same space; will be overwritten each iter
    du.vector().zero()

    r0 = None
    last_r = None

    while it < max_it:

        # -------- assemble residual (raw) for convergence / recompute logic --------
        if b is None:
            b = assemble(-F)
        else:
            assemble(-F, tensor=b)

        for bc in bcs:
            bc.apply(b, u.vector())

        r = b.norm("l2")

        if r0 is None:
            r0 = r if r > 0.0 else 1.0

        rel_r = r / r0

        # divergence checks (like github's NaN/huge guards)
        if (not np.isfinite(r)) or r > diverge_guard:
            raise RuntimeError(
                f"Newton diverged: residual norm = {r:.3e} at iteration {it}"
            )

        # stopping criteria based on residual
        if r <= atol or rel_r <= rtol:
            if verbose and group_rank == 0:
                print(f"[Newton] converged by residual at it={it}, r={r:.3e}, rel={rel_r:.3e}")
            break

        # decide whether to (re)assemble Jacobian
        recompute_due_to_freq = (
            it == 0 or (recompute is not None and recompute > 0 and it % recompute == 0)
        )
        recompute_due_to_increase = (
            recompute_on_increase and last_r is not None and r > last_r
        )

        if recompute_due_to_freq or recompute_due_to_increase:
            if A is None:
                A = assemble(J, keep_diagonal=True)
            else:
                assemble(J, tensor=A, keep_diagonal=True)

            if ident_zeros:
                # helps if zero diagonals appear (can happen with constraints / mixed forms)
                try:
                    A.ident_zeros()
                except Exception:
                    # ident_zeros exists for dolfin.Matrix; ignore if backend doesn't support
                    pass

            for bc in bcs:
                bc.apply(A)

            lin_solver.set_operator(A)

        # -------- linear solve and update --------
        du.vector().zero()
        lin_solver.solve(du.vector(), b)

        du_norm = du.vector().norm("l2")
        if (not np.isfinite(du_norm)) or du_norm > diverge_guard:
            raise RuntimeError(
                f"Newton diverged: ||du|| = {du_norm:.3e} at iteration {it}"
            )

        u.vector().axpy(lmbda, du.vector())

        for bc in bcs:
            bc.apply(u.vector())

        last_r = r
        it += 1

    return it, r, rel_r

def Nonlinear_Structure_Solver(group_comm, comm, group_rank, para):
    
    set_log_level(LogLevel.ERROR)

    parameters["form_compiler"]["quadrature_degree"] = 5
    # Physical Constants
    skip     = para['skip']
    rho_s    = para['rho_s']
    mu_s     = para['mu_s']
    lambda_s = para['lambda_s']
    L_2      = para['L_2']
    T        = para['T']
    dt       = para['dt']
    path_f   = para['path_f']
    path_s   = para['path_s']
    path_b   = para['boundary_s']
    recompute = para['recompute']
    kdt1 = dt
    kdt2 = 1e-3

    # Mesh Loading
    mesh_f = Mesh(MPI.COMM_SELF)
    with XDMFFile(MPI.COMM_SELF, path_f) as mshfile:
        mshfile.read(mesh_f)

    mesh_s = Mesh(group_comm)
    with XDMFFile(group_comm, path_s) as mshfile:
        mshfile.read(mesh_s)
    ref_s = mesh_s.coordinates().copy()

    if group_rank == 0:
        print("Structure Mesh Loading -- Done!")
        sys.stdout.flush()
    ########################### Boundary Mark ###########################
    mvc = MeshValueCollection("size_t", mesh_s, mesh_s.topology().dim() - 1)
    with XDMFFile(group_comm, path_b) as bndfile:
        bndfile.read(mvc, "marker")
    bmf_s = MeshFunction('size_t', mesh_s, mvc)
    ds_s  = Measure("ds", domain=mesh_s, subdomain_data=bmf_s)

    if group_rank == 0:
        print("Structure Boundary Marking -- Done!")
        sys.stdout.flush()
    del mvc

    ## Structure Function Space 
    eta_elem = VectorElement("CG", mesh_s.ufl_cell(), 2) 
    ksi_elem = VectorElement("CG", mesh_s.ufl_cell(), 2)
    elem_s   = MixedElement([eta_elem, ksi_elem])
    V_s      = FunctionSpace(mesh_s, elem_s)
    V_sig_s  = VectorFunctionSpace(mesh_s, "CG", 2)

    Ela_present        = Function(V_s)
    eta, ksi           = split(Ela_present)
    vp, zeta           = TestFunctions(V_s)

    Ela_old            = Function(V_s)
    eta_old, ksi_old   = Ela_old.split(True)

    eta_fem, ksi_fem   = Ela_present.split(True)

    sig_old_s          = Function(V_sig_s)
    u_old_s            = Function(V_sig_s)
    sig_old_f_s        = Function(V_sig_s)
    ## NS Function Space
    V_f     = VectorFunctionSpace(mesh_f, "CG", 2)

    u_old      = Function(V_f)
    sig_old_f  = Function(V_f)
     
    if group_rank == 0:
        print("Structure Function Space -- Done!")
        sys.stdout.flush()
    ##################### Boundary Conditions #####################
    u_in = Expression(("t < 1.0 ? (0.5 - 0.5*cos(pi*t))*63*x[1]*x[2]*(0.41 - x[1])*(0.41 - x[2])/0.41/0.41/0.41/0.41 : 63*x[1]*x[2]*(0.41 - x[1])*(0.41 - x[2])/0.41/0.41/0.41/0.41", "0", "0"), degree=6, t=0.0)
    eta_bottom_bc1 = DirichletBC(V_s.sub(0), Constant((0.0, 0.0, 0.0)), bmf_s, 1)
    eta_bottom_bc2 = DirichletBC(V_s.sub(1), Constant((0.0, 0.0, 0.0)), bmf_s, 1)
    bc_s           = [eta_bottom_bc1, eta_bottom_bc2]
    
    if group_rank == 0:
        print("Structure Boundary Condition -- Done!")
        sys.stdout.flush()
    
    dt       = Constant(dt)
    mu_s     = Constant(mu_s)
    rho_s    = Constant(rho_s)
    lambda_s = Constant(lambda_s)

    ## Weak Formulation
    def F(eta):
        """
        Deformation gradient
        """
        return Identity(3) + grad(eta)
    
    def E(eta):
        """
        Green-Lagrange strain tensor
        """
        return 0.5*(F(eta).T*F(eta) - Identity(3))
    
    def sigma_s(eta):
        """
        First Piola-Kirchhoff Stress
        """
        return F(eta)*(2*mu_s*E(eta) + lambda_s*tr(E(eta))*Identity(3))

    # Structure Equation 
    F_s = rho_s/dt*dot(ksi- ksi_old, zeta)*dx \
        + inner(sigma_s(eta), grad(zeta))*dx \
        - dot(ksi, vp)*dx \
        + 1.0/dt*dot(eta - eta_old, vp)*dx \
        + L_2*dot(ksi, zeta)*ds_s(5) \
        - 0.5*dot(sig_old_f_s, zeta)*ds_s(5) \
        - 0.5*L_2*dot(u_old_s, zeta)*ds_s(5)\
        - 0.5*L_2*dot(ksi_old, zeta)*ds_s(5) \
        - 0.5*dot(sig_old_s, zeta)*ds_s(5) \

    J = derivative(F_s, Ela_present)
    A_dummy_s = assemble(J, keep_diagonal=True)
    newton_solver = LUSolver(mesh_s.mpi_comm(), A_dummy_s, "mumps")
    newton_max_it = 25
    newton_atol = 1e-8
    newton_rtol = 1e-8

    if group_rank == 0:
        print("Structure Weak Form -- Done!")
        sys.stdout.flush()
    #################################################################
    xdmf_eta = XDMFFile(group_comm,'./benchmark/displacement.xdmf')
    # xdmf_ksi = XDMFFile(group_comm,'robin/ksi_p.xdmf')
    xdmf_eta.parameters["flush_output"] = True
    # xdmf_ksi.parameters["flush_output"] = True

    # Get coordinates of all dofs
    local_coordinates = V_sig_s.tabulate_dof_coordinates().reshape((-1, mesh_s.geometry().dim()))
    local_indices = V_sig_s.dofmap().dofs()

    gathered_coordinates = group_comm.gather(local_coordinates, root=0)
    gathered_indices = group_comm.gather(local_indices, root=0)
    indices_f = None
    num_dofs  = V_sig_s.dim()
    num_dof_f = V_f.dim()    
    temp = np.zeros(num_dof_f).reshape(-1, 3)

    if group_rank == 0:
        
        # Gather coordinates of all dofs for mesh_s
        all_coordinates = np.zeros((num_dofs, mesh_s.geometry().dim()))
        for coord, indices in zip(gathered_coordinates, gathered_indices):
            all_coordinates[indices] = coord
        structure_partition = all_coordinates[::3]

        fluid_partition = comm.recv(source=0) # Gathered fluid coordinates
        comm.send(structure_partition, dest=0)

        global_fluid = V_f.tabulate_dof_coordinates().reshape((-1, mesh_f.geometry().dim()))
        global_f_sub = global_fluid[::3]

        tree = KDTree(global_f_sub)
        _, indices_f = tree.query(fluid_partition)
     
        del tree

        ksi_values   = np.zeros(num_dofs)
        sig_s_values = np.zeros(num_dofs)
        eta_values   = np.zeros(num_dofs)

    indices_f = group_comm.bcast(indices_f, root=0)
    recv_data = None

    if group_rank == 0:
        print("Structure Iteration -- Start!")
        sys.stdout.flush()
    ######################### Time Iteration ########################
    t = 0.0
    p01 = Point(0.9, 0.2, 0.3)

    u_old.set_allow_extrapolation(True)
    sig_old_f.set_allow_extrapolation(True)

    B_u    = DirichletBC(V_sig_s, u_old, bmf_s, 5)
    B_sigf = DirichletBC(V_sig_s, sig_old_f, bmf_s, 5)

    n = int(0)

    while t <= T:
        if group_rank == 0:
            start = time.time()
        if t < 2.0:
            kdt = kdt2
        else:
            kdt = kdt1
        t += kdt
        u_in.t = t

        ksi_local    = ksi_old.vector().get_local()
        sig_s_local  = sig_old_s.vector().get_local()
        eta_local    = eta_old.vector().get_local()
        gathered_ksi = group_comm.gather(ksi_local, root=0)
        gathered_sig = group_comm.gather(sig_s_local, root=0)
        gathered_eta = group_comm.gather(eta_local, root=0)

        if group_rank == 0:
            for value_ksi, value_sig, value_eta, indices in zip(gathered_ksi, gathered_sig, gathered_eta, gathered_indices):
                ksi_values[indices] = value_ksi
                sig_s_values[indices] = value_sig
                eta_values[indices] = value_eta

            send_data = [ksi_values, sig_s_values, eta_values]
            recv_data = comm.recv(source=0)
            comm.send(send_data, dest=0)
            
        recv_data = group_comm.bcast(recv_data, root=0)

        temp[indices_f] = recv_data[0].reshape(-1, 3)
        u_old.vector()[:] = temp.reshape(-1)
        u_old.vector().apply("insert")

        temp[indices_f] = recv_data[1].reshape(-1, 3)
        sig_old_f.vector()[:] = -temp.reshape(-1)
        sig_old_f.vector().apply("insert")

        # storage NS function in Structure domain
        B_u.apply(u_old_s.vector())
        B_sigf.apply(sig_old_f_s.vector())

        Newton_Solver_Recompute_Robust( group_rank,
                                        F_s, Ela_present, bc_s, J, recompute, newton_solver,
                                        max_it=newton_max_it, atol=newton_atol, rtol=newton_rtol,
                                        recompute_on_increase=True,
                                        ident_zeros=True,
                                        lmbda=1.0,
                                        verbose=True
                                    )
        # solver.solve()
        eta_fem, ksi_fem = Ela_present.split(True)
        
        if n % skip == 0:
            # ALE.move(mesh_s, eta_fem)
            xdmf_eta.write(eta_fem, t)
            # xdmf_ksi.write(ksi_fem, t)
            mesh_s.coordinates()[:] = ref_s
        n += 1

        sig_old_s.vector()[:] = 0.5*L_2*u_old_s.vector() + 0.5*L_2*ksi_old.vector() \
                              + 0.5*sig_old_f_s.vector() + 0.5*sig_old_s.vector() - L_2*ksi_fem.vector()
        sig_old_s.vector().apply("insert")
        
        assign(eta_old, eta_fem)
        assign(ksi_old, ksi_fem)

        if group_rank == 0:
            end = time.time()
            print(f"---Time: {t}---Structure time-{end-start}-----")
            sys.stdout.flush()

        eta_fem.set_allow_extrapolation(True)

        if abs(eta_fem(p01)[1]) >= 10.0:
            print(f"Deformation is too large!")
            print("Stop Iteration!")
            sys.stdout.flush()
            break

        if t >= T and group_rank == 0:
            print("Computation Complete!")