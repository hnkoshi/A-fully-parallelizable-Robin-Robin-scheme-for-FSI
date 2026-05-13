import sys
import time
import numpy as np
from numpy import isnan
from scipy.spatial import KDTree
from dolfin import *
from mpi4py import MPI


def Newton_Solver_Recompute_Robust(group_rank,
    F, u, bcs, J, recompute, lin_solver,
    max_it=20, atol=1e-8, rtol=1e-8,
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

def Fluid_Solver(group_comm, comm, group_rank, midd, para):

    set_log_level(LogLevel.ERROR)

    parameters["form_compiler"]["quadrature_degree"] = 5
    # Physical Constants
    skip      = para['skip']
    rho_f     = para['rho_f']
    mu_f      = para['mu_f']
    L_1       = para['L_1']
    T         = para['T']
    dt        = para['dt']
    path_f    = para['path_f']
    path_s    = para['path_s']
    path_b    = para['boundary_f']
    recompute = para['recompute']
    kdt1 = dt
    kdt2 = 1e-3

    # Mesh Loading
    mesh_f = Mesh(group_comm)
    with XDMFFile(group_comm, path_f) as mshfile:
        mshfile.read(mesh_f)
    ref_f = mesh_f.coordinates().copy()

    mesh_s = Mesh(MPI.COMM_SELF)
    with XDMFFile(MPI.COMM_SELF, path_s) as mshfile:
        mshfile.read(mesh_s)

    if group_rank == 0:
        print("Fluid Mesh Loading -- Done!")
        sys.stdout.flush()
    ############## Boundary Mark ##############
    mvc = MeshValueCollection("size_t", mesh_f, mesh_f.topology().dim() - 1)
    with XDMFFile(group_comm, path_b) as bndfile:
        bndfile.read(mvc, "marker")
    bmf_f = MeshFunction('size_t', mesh_f, mvc)
    ds_f  = Measure("ds", domain=mesh_f, subdomain_data=bmf_f)

    if group_rank == 0:
        print("Structure Boundary Marking -- Done!")
        sys.stdout.flush()
    del mvc

    if group_rank == 0:
        print("Fluid Boundary Marking -- Done!")
        sys.stdout.flush()

    ## ALE Function Space
    V_ale = VectorFunctionSpace(mesh_f, "CG", 2)

    eta_ale          = TrialFunction(V_ale)
    tau_ale          = TestFunction(V_ale)
    eta_ale_fem      = Function(V_ale)
    eta_ale_old      = Function(V_ale)
    omega_ale        = Function(V_ale)

    ## NS Function Space
    u_elem  = VectorElement("CG", mesh_f.ufl_cell(), 2)
    p_elem  = FiniteElement("CG", mesh_f.ufl_cell(), 1)
    elem_f  = MixedElement([u_elem, p_elem])
    V_f     = FunctionSpace(mesh_f, elem_f)
    V_sig_f = VectorFunctionSpace(mesh_f, "CG", 2)

    NS_present   = Function(V_f)
    u, p         = split(NS_present)
    v, q         = TestFunctions(V_f)
    
    NS_old       = Function(V_f)
    u_old, _     = NS_old.split(True)

    u_fem, p_fem = NS_present.split(True)

    sig_old_f    = Function(V_sig_f)
    sig_old_s_f  = Function(V_sig_f)
    ksi_old_f    = Function(V_sig_f)

    ## Structure Function Space
    V_s      = VectorFunctionSpace(mesh_s, "CG", 2)
    
    eta_old    = Function(V_s)
    ksi_old    = Function(V_s)
    sig_old_s  = Function(V_s)

    if group_rank == 0:
        print("Fluid Function Space -- Done!")
        sys.stdout.flush()
    
    eta_old.set_allow_extrapolation(True)
    ksi_old.set_allow_extrapolation(True)
    sig_old_s.set_allow_extrapolation(True)
    ##################### Boundary Conditions #####################
    # ALE (BC on interface will be assigned during time iteration)
    eta_ale_out_bc = DirichletBC(V_ale, Constant((0.0, 0.0, 0.0)), bmf_f, 2)
    eta_ale_in_bc  = DirichletBC(V_ale, Constant((0.0, 0.0, 0.0)), bmf_f, 1)
    eta_ale_wall_bc = DirichletBC(V_ale, Constant((0.0, 0.0, 0.0)), bmf_f, 3)
    eta_ale_cyl_bc = DirichletBC(V_ale, Constant((0.0, 0.0, 0.0)), bmf_f, 4)

    # NS equation
    u_in = Expression(("t < 1.0 ? (0.5 - 0.5*cos(pi*t))*63*x[1]*x[2]*(0.41 - x[1])*(0.41 - x[2])/0.41/0.41/0.41/0.41 : 63*x[1]*x[2]*(0.41 - x[1])*(0.41 - x[2])/0.41/0.41/0.41/0.41", "0.0", "0.0"), degree=6, t=0.0)
    u_in_bc   = DirichletBC(V_f.sub(0), u_in, bmf_f, 1)
    u_wall_bc = DirichletBC(V_f.sub(0), Constant((0.0, 0.0, 0.0)), bmf_f, 3)
    u_cyl_bc  = DirichletBC(V_f.sub(0), Constant((0.0, 0.0, 0.0)), bmf_f, 4)
    bc_f     = [u_in_bc, u_wall_bc, u_cyl_bc]

    if group_rank == 0:
        print("Fluid Boundary Condition -- Done!")
        sys.stdout.flush()
    
    #################################################################

    dt       = Constant(dt)
    rho_f    = Constant(rho_f)
    mu_f     = Constant(mu_f)

    # ALE Mapping
    a01 = CellVolume(mesh_f)
    a02 = mesh_f.hmax()
    a03 = mesh_f.hmin()
    alpha = (1 - a03/a02)/(a01/a02) + 1.0
    a_ale = (alpha*inner(grad(eta_ale), grad(tau_ale)))*dx
    L_ale = dot(Constant((0, 0, 0)), tau_ale)*dx   

    eta_ale_interface_bc = DirichletBC(V_ale, eta_old, bmf_f, 5)
    bc_ale = [eta_ale_out_bc, eta_ale_in_bc, eta_ale_wall_bc, eta_ale_cyl_bc, eta_ale_interface_bc]

    A_ale = assemble(a_ale, keep_diagonal=True)
    for bc in bc_ale:
        bc.apply(A_ale)

    ale_solver = LUSolver(mesh_f.mpi_comm(), A_ale, "mumps")

    b_ale = None
    
    I = Identity(3)
    F_expr  = I + grad(eta_ale_fem)
    J_expr  = det(F_expr)
    Finv    = inv(F_expr)
    FinvT   = Finv.T

    # NS
    def sigma_f_u(u, Finv):
        return mu_f*(grad(u)*Finv + Finv.T*grad(u).T)

    F_nonlin = rho_f*inner(grad(u)*Finv*J_expr*(u - omega_ale), v)*dx

    F_lin = rho_f/dt*dot(J_expr*(u - u_old), v)*dx \
            + inner(J_expr*sigma_f_u(u, Finv)*FinvT, grad(v))*dx \
            - inner(J_expr*p*I*FinvT, grad(v))*dx \
            + div(J_expr*Finv*u)*q*dx \
            + L_1*dot(u, v)*ds_f(5) \
            - 0.5*dot(sig_old_f, v)*ds_f(5) \
            - 0.5*L_1*dot(ksi_old_f, v)*ds_f(5) \
            - 0.5*L_1*dot(u_old, v)*ds_f(5) \
            - 0.5*dot(sig_old_s_f, v)*ds_f(5)

    F_f = F_lin + F_nonlin

    J = derivative(F_f, NS_present)
    A_dummy_f = assemble(J, keep_diagonal=True)
    newton_solver = LUSolver(mesh_f.mpi_comm(), A_dummy_f, "mumps")
    newton_max_it = 25
    newton_atol = 1e-8
    newton_rtol = 1e-8

    if group_rank == 0:
        print("Fluid Weak Form -- Done!")
        sys.stdout.flush()
    #################################################################
    # xdmf_u = XDMFFile(group_comm, './robin/velocity_f.xdmf')
    # xdmf_p = XDMFFile(group_comm, './robin/pressure_f.xdmf')
    # xdmf_u.parameters["flush_output"] = True
    # xdmf_p.parameters["flush_output"] = True

    # Get coordinates of all dofs
    local_coordinates = V_sig_f.tabulate_dof_coordinates().reshape((-1, mesh_f.geometry().dim()))
    local_indices = V_sig_f.dofmap().dofs()

    gathered_coordinates = group_comm.gather(local_coordinates, root=0)
    gathered_indices = group_comm.gather(local_indices, root=0)
    indices_s = None
    num_dofs = V_sig_f.dim()
    num_dof_s = V_s.dim()

    temp = np.zeros(num_dof_s).reshape(-1, 3)

    if group_rank == 0:
        
        ## Indices for P2 Space
        # Gather coordinates of all dofs for mesh_f
        all_coordinates = np.zeros((num_dofs, mesh_f.geometry().dim()))

        for coord, indices in zip(gathered_coordinates, gathered_indices):
            all_coordinates[indices] = coord
        fluid_partition = all_coordinates[::3]

        comm.send(fluid_partition, dest=midd)
        structure_partition = comm.recv(source=midd) # Gathered strucutre coordinates
        
        global_structure = V_s.tabulate_dof_coordinates().reshape((-1, mesh_s.geometry().dim()))
        global_s_sub = global_structure[::3]

        tree = KDTree(global_s_sub)
        _, indices_s = tree.query(structure_partition)
        del tree

        u_values   = np.zeros(num_dofs)
        sig_f_values = np.zeros(num_dofs)
    
    indices_s = group_comm.bcast(indices_s, root=0)
    recv_data = None

    if group_rank == 0:
        print("Fluid Iteration -- Start!")
        sys.stdout.flush()
    ######################### Time Iteration ########################
    t = 0.0

    B_ksi  = DirichletBC(V_sig_f, ksi_old, bmf_f, 5)
    B_sigs = DirichletBC(V_sig_f, sig_old_s, bmf_f, 5)

    while t <= T:
        if group_rank == 0:
            start = time.time()
        if t < 2.0:
            kdt = kdt2
        else:
            kdt = kdt1
        t += kdt
        u_in.t = t

        # MPI Communication
        u_local      = u_old.vector().get_local()
        sig_f_local  = sig_old_f.vector().get_local()
        gathered_u   = group_comm.gather(u_local, root=0)
        gathered_sig = group_comm.gather(sig_f_local, root=0)

        if group_rank == 0:
            for value_u, value_sig, indices in zip(gathered_u, gathered_sig, gathered_indices):
                u_values[indices] = value_u
                sig_f_values[indices] = value_sig

            send_data = [u_values, sig_f_values]
            comm.send(send_data, dest=midd)
            recv_data = comm.recv(source=midd)

        recv_data = group_comm.bcast(recv_data, root=0)

        temp[indices_s] = recv_data[0].reshape(-1, 3)
        ksi_old.vector()[:] = temp.reshape(-1)
        ksi_old.vector().apply("insert")

        temp[indices_s] = recv_data[1].reshape(-1, 3)
        sig_old_s.vector()[:] = -temp.reshape(-1)
        sig_old_s.vector().apply("insert")

        temp[indices_s] = recv_data[2].reshape(-1, 3)
        eta_old.vector()[:] = temp.reshape(-1)
        eta_old.vector().apply("insert")

        B_ksi.apply(ksi_old_f.vector())
        B_sigs.apply(sig_old_s_f.vector())

        if b_ale is None:
            b_ale = assemble(L_ale)
        else:
            assemble(L_ale, tensor=b_ale)
            
        for bc in bc_ale:
            bc.apply(b_ale)
            
        ale_solver.solve(eta_ale_fem.vector(), b_ale)
        
        # Obtain the velocity of mesh displacement
        omega_ale.vector()[:] = (eta_ale_fem.vector() - eta_ale_old.vector())/kdt
        omega_ale.vector().apply("insert")

        Newton_Solver_Recompute_Robust( group_rank,
                                        F_f, NS_present, bc_f, J, recompute, newton_solver,
                                        max_it=newton_max_it, atol=newton_atol, rtol=newton_rtol,
                                        recompute_on_increase=True,
                                        ident_zeros=True,
                                        lmbda=1.0,
                                        verbose=True
                                    )
        # solver.solve()
        u_fem, p_fem = NS_present.split(True)

        # if n % skip == 0:
        #     ALE.move(mesh_f, eta_ale_fem)
        #     xdmf_u.write(u_fem, t)
        #     # xdmf_p.write(p_fem, t)
        #     mesh_f.coordinates()[:] = ref_f


        sig_old_f.vector()[:] = L_1*0.5*u_old.vector() + L_1*0.5*ksi_old_f.vector() \
                        + 0.5*sig_old_s_f.vector() + 0.5*sig_old_f.vector() - L_1*u_fem.vector()
        sig_old_f.vector().apply("insert")

        assign(u_old, u_fem)
        eta_ale_old.assign(eta_ale_fem)
        
        if group_rank == 0:
            end = time.time()
            print(f"---Time: {t}---Fluid time-{end-start}-----")
            sys.stdout.flush()