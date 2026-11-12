import numpy as np
from mpi4py import MPI
import dolfinx.mesh as mesh
import dolfinx.fem as fem
import dolfinx.fem.petsc
import ufl
import basix.ufl
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx.common import Timer
from petsc4py import PETSc
from scipy.linalg import cholesky
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from dolfinx import cpp as _cpp

from dolfinx import fem as _fem
from dolfinx import la as _la
from dolfinx_mpc.assemble_matrix import assemble_matrix, create_sparsity_pattern
from dolfinx_mpc.assemble_vector import apply_lifting, assemble_vector
from dolfinx_mpc.multipointconstraint import MultiPointConstraint

dt     = 5
T      = 10000.0
theta  = 0.5
mu     = 10.0
Temp   = 1.2

La, Lb = 0.0, 600.0
Ya, Yb = 0.0, 600.0
Za, Zb = 0.0, 600.0
line   = 0.6

dimension = 1
if dimension == 1:
    dt, T, line, Lb = 1.0, 10000.0, 5.0, 40000.0

if dimension == 1:
    nx = int(4*4000/2)
    domain = mesh.create_interval(MPI.COMM_WORLD, nx, [La, Lb])
elif dimension == 2:
    nx, ny = 20, 20
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([La, Ya]), np.array([Lb, Yb])],
        [nx, ny], cell_type=mesh.CellType.quadrilateral
    )
elif dimension == 3:
    nx, ny, nz = 5, 5, 5
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([La, Ya, Za]), np.array([Lb, Yb, Zb])],
        [nx, ny, nz], cell_type=mesh.CellType.hexahedron
    )
else:
    raise ValueError("dimension must be 1, 2 or 3")

print(f"Created {dimension}D mesh")
print(f"Number of cells: {domain.topology.index_map(domain.topology.dim).size_local}")
print(f"Number of vertices: {domain.topology.index_map(0).size_local}")
print(f"Geometric dimension: {domain.geometry.dim}")

if domain.comm.size != 1:
    raise NotImplementedError("Dense Cholesky noise generation below is serial-only.")

V_rho = fem.functionspace(domain, ("CG", 1))
V_vx  = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))

rho_element = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
u_element   = basix.ufl.element(
    "Lagrange", domain.basix_cell(), 1,
    shape=(domain.geometry.dim,)
)
W = fem.functionspace(domain, basix.ufl.mixed_element([rho_element, u_element]))

limits_a = np.array([La, Ya, Za][:dimension])
limits_b = np.array([Lb, Yb, Zb][:dimension])
periods = limits_b - limits_a
periodic_tol = 1e-8 * np.max(periods)

def periodic_boundary(mask):
    def boundary(x):
        selected = np.ones(x.shape[1], dtype=bool)
        for i in range(dimension):
            on_upper = np.isclose(x[i], limits_b[i], atol=periodic_tol, rtol=0)
            selected &= on_upper if mask & (1 << i) else ~on_upper
        return selected
    return boundary

def periodic_relation(mask):
    def relation(x):
        y = x.copy()
        for i in range(dimension):
            if mask & (1 << i):
                y[i] -= periods[i]
        return y
    return relation

with Timer("~PERIODIC: Initialize MPC W"):
    mpc_W = MultiPointConstraint(W)
    for space in [W.sub(0)] + [W.sub(1).sub(i) for i in range(dimension)]:
        for mask in range(1, 1 << dimension):
            mpc_W.create_periodic_constraint_geometrical(
                space, periodic_boundary(mask), periodic_relation(mask), []
            )
    mpc_W.finalize()

print(f"Full {dimension}D periodic MPC: {2**dimension-1} slave groups per scalar component")
rho, ux   = ufl.TrialFunctions(W)
phi, psix = ufl.TestFunctions(W)

rho_prev = fem.Function(V_rho, name="rho_prev")
ux_prev  = fem.Function(V_vx, name="ux_prev")
W_rand_x = fem.Function(V_vx, name="W_rand_x")

def init_rho(x):
    return line + 0*x[0]

rho_prev.interpolate(init_rho)
rho_mid = theta * rho + (1-theta) * rho_prev
ux_mid  = theta * ux  + (1-theta) * ux_prev

dxL = ufl.dx(scheme="vertex", degree=2,
             metadata={"representation": "quadrature", "degree": 2})

dim = domain.geometry.dim
F_rho = (rho - rho_prev) * phi * dxL \
        - dt * rho_prev * ufl.inner(ux_mid, ufl.grad(phi)) * ufl.dx

if dim == 1:
    viscosity = mu * ufl.inner(ufl.grad(ux_mid), ufl.grad(psix)) * ufl.dx
else:
    viscosity = mu * ufl.inner(
        ufl.grad(ux_mid) + ufl.grad(ux_mid).T
        - (2/dim)*ufl.div(ux_mid)*ufl.Identity(dim),
        ufl.grad(psix)) * ufl.dx

F_ux = ufl.inner(rho_prev * (ux - ux_prev), psix) * dxL \
       - dt * ufl.inner(ufl.outer(rho_prev * ux_mid, ux_prev),
                        ufl.grad(psix)) * ufl.dx \
       + dt * viscosity \
       + Temp*dt * ufl.inner(ufl.grad(rho_mid), psix) * ufl.dx \
       - ufl.inner(W_rand_x, psix) * dxL

F = F_rho + F_ux
lhs_coupled = ufl.lhs(F)
rhs_coupled = ufl.rhs(F)

u_noise = ufl.TrialFunction(V_vx)
v_noise = ufl.TestFunction(V_vx)

if dim == 1:
    viscous_form = ufl.inner(ufl.grad(u_noise), ufl.grad(v_noise)) * ufl.dx
else:
    viscous_form = ufl.inner(
        ufl.grad(u_noise) + ufl.grad(u_noise).T
        - (2/dim)*ufl.div(u_noise)*ufl.Identity(dim),
        ufl.grad(v_noise)
    ) * ufl.dx

K_noise = fem.petsc.assemble_matrix(fem.form(viscous_form), bcs=[])
K_noise.assemble()
ai, aj, av = K_noise.getValuesCSR()
PSI = csr_matrix((av, aj, ai), shape=K_noise.getSize()).toarray()

eps = 1e-12 * np.mean(np.diag(PSI))
Lchol = cholesky(PSI + eps * np.eye(PSI.shape[0]), lower=True)

M_noise = fem.petsc.assemble_matrix(
    fem.form(ufl.inner(u_noise, v_noise) * dxL), bcs=[]
)
M_noise.assemble()
mass_diag = M_noise.getDiagonal().array.copy()
if np.any(mass_diag <= 0):
    raise RuntimeError("Velocity lumped mass has nonpositive diagonal entries.")

class LinearProblem2(dolfinx.fem.petsc.LinearProblem):
    def __init__(
        self,
        a,
        L,
        mpc,
        bcs=None,
        u=None,
        petsc_options=None,
        form_compiler_options=None,
        jit_options=None,
    ):

        form_compiler_options = {} if form_compiler_options is None else form_compiler_options
        jit_options = {} if jit_options is None else jit_options

        self._a = _fem.form(a, jit_options=jit_options,
                            form_compiler_options=form_compiler_options)
        self._L = _fem.form(L, jit_options=jit_options,
                            form_compiler_options=form_compiler_options)

        if not mpc.finalized:
            raise RuntimeError("MPC must be finalized")

        self._mpc = mpc
        if u is None:
            self.u = _fem.Function(mpc.function_space)
        else:
            self.u = u

        self._x = self.u.x.petsc_vec
        pattern = create_sparsity_pattern(self._a, self._mpc)
        pattern.finalize()

        self._A = _cpp.la.petsc.create_matrix(
            self._mpc.function_space.mesh.comm, pattern
        )

        self._b = _la.create_petsc_vector(
            self._mpc.function_space.dofmap.index_map,
            self._mpc.function_space.dofmap.index_map_bs
        )

        self.bcs = [] if bcs is None else bcs
        self._solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self._solver.setOperators(self._A)

        solver_prefix = f"dolfinx_mpc_solve_{id(self)}"
        self._solver.setOptionsPrefix(solver_prefix)

        opts = PETSc.Options()
        opts.prefixPush(solver_prefix)

        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        opts.prefixPop()
        self._solver.setFromOptions()

    def solve(self):
        xi_x = np.random.randn(ux_prev.x.array.shape[0])
        dWx = np.sqrt(2 * mu * Temp * dt) * xi_x

        W_rand_x.x.array[:] = (Lchol @ dWx) / mass_diag
        W_rand_x.x.scatter_forward()

        self._A.zeroEntries()
        assemble_matrix(self._a, self._mpc, bcs=self.bcs, A=self._A)
        self._A.assemble()

        with self._b.localForm() as b_loc:
            b_loc.set(0)

        assemble_vector(self._L, self._mpc, b=self._b)
        apply_lifting(self._b, [self._a], [self.bcs], self._mpc)

        self._b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE
        )

        _fem.petsc.set_bc(self._b, self.bcs)
        self._solver.solve(self._b, self._x)

        if self._solver.getConvergedReason() <= 0:
            raise RuntimeError("PETSc solver did not converge")

        self.u.x.scatter_forward()
        self._mpc.backsubstitution(self.u)

        return self.u
solver_options = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-10,
    "ksp_atol": 1e-12,
    "ksp_max_it": 1000,
    "ksp_gmres_restart": 100,
    "pc_type": "ilu",
    "pc_factor_levels": 0,
    "ksp_error_if_not_converged": True,
}

coupled_problem = LinearProblem2(
    lhs_coupled,
    rhs_coupled,
    mpc_W,
    bcs=[],
    petsc_options=solver_options
)
w_tilda = coupled_problem.u

one = fem.Function(V_rho)
one.x.array[:] = 1.0
domain_volume = domain.comm.allreduce(
    fem.assemble_scalar(fem.form(one * ufl.dx)), op=MPI.SUM
)
mass_0 = domain.comm.allreduce(
    fem.assemble_scalar(fem.form(rho_prev * dxL)), op=MPI.SUM
)

ai, aj, av = M_noise.getValuesCSR()
dd = csr_matrix((av, aj, ai), shape=M_noise.getSize())

rho_coords = V_rho.tabulate_dof_coordinates()[:, :dimension]
u_coords = V_vx.tabulate_dof_coordinates()[:, :dimension]
inside_rho = np.all((rho_coords > limits_a + periodic_tol) &
                    (rho_coords < limits_b - periodic_tol), axis=1)
inside_u = np.all((u_coords > limits_a + periodic_tol) &
                  (u_coords < limits_b - periodic_tol), axis=1)

unique_rho = np.all(rho_coords < limits_b - periodic_tol, axis=1)
unique_u = np.all(u_coords < limits_b - periodic_tol, axis=1)

n_cells = np.array([nx] if dimension == 1 else [nx, ny] if dimension == 2 else [nx, ny, nz])
expected_inside = int(np.prod(n_cells - 1))
expected_unique = int(np.prod(n_cells))
if (inside_rho.sum() != expected_inside or inside_u.sum() != expected_inside or
        unique_rho.sum() != expected_unique or unique_u.sum() != expected_unique):
    raise RuntimeError("Wrong interior/periodic node count")

if u_coords.shape[0] * dimension != ux_prev.x.array.size:
    raise RuntimeError("Unexpected vector DOF layout")

print(f"Interior nodes: {inside_rho.sum()} rho, {inside_u.sum()*dimension} u entries")
print(f"Unique periodic nodes: {unique_rho.sum()} rho, {unique_u.sum()*dimension} u entries")

def periodic_jump(values, coords):
    indices = np.rint((coords - limits_a) * n_cells / periods).astype(int)
    seen = {}
    jump = 0.0
    for node, value in zip(indices, values):
        key = tuple(node % n_cells)
        if key in seen:
            jump = max(jump, float(np.max(np.abs(value - seen[key]))))
        else:
            seen[key] = np.array(value, copy=True)
    return jump

var_u, var_rho, var_u2 = [], [], []
var_u_interior, var_rho_interior = [], []
var_u_unique, var_rho_unique = [], []
rhosolution, uxsolution, uysolution = [], [], []

t = 0.0
step = 0
while t < T:

    coupled_problem.solve()
    rho_prev.interpolate(w_tilda.sub(0))
    ux_prev.interpolate(w_tilda.sub(1))
    rho_prev.x.scatter_forward()
    ux_prev.x.scatter_forward()

    mass = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(rho_prev * ufl.dx)), op=MPI.SUM
    )
    print(mass)

    var_u.append(np.var(ux_prev.x.array.copy()))
    mean_value = np.sum(dd @ ux_prev.x.array) / (dim * domain_volume)
    expr = (ux_prev.x.array.copy() - mean_value)**2
    variance = np.sum(dd @ expr) / (dim * domain_volume)
    var_u2.append(variance)
    var_rho.append(np.var(rho_prev.x.array.copy()))

    u_values = ux_prev.x.array.reshape(-1, dimension)
    var_rho_interior.append(np.var(rho_prev.x.array[inside_rho]))
    var_u_interior.append(np.var(u_values[inside_u]))
    var_rho_unique.append(np.var(rho_prev.x.array[unique_rho]))
    var_u_unique.append(np.var(u_values[unique_u]))

    step += 1
    if step <= 3:
        jump_rho = periodic_jump(rho_prev.x.array, rho_coords)
        jump_u = periodic_jump(u_values, u_coords)
        print(f"periodic jumps: rho={jump_rho:.3e}, u={jump_u:.3e}")
        if max(jump_rho, jump_u) > 1e-7:
            raise RuntimeError("Periodic constraints failed")

    t += dt
    if domain.comm.rank == 0:
        print(f"t={t:.2f}")

times = np.arange(1, len(var_rho)+1) * dt
h = np.prod(periods / n_cells)
var_rho_theory = line / h
var_u_theory = Temp / (line * h)

plt.figure(figsize=(8, 5))
plt.plot(times, np.array(var_rho_interior)/var_rho_theory,
         label="FEM (interior)", linewidth=2)
plt.plot(times, np.ones_like(var_rho), label="Theory", linewidth=2)
plt.ylabel("Variance, $(\\rho)$", fontsize=14, fontweight="bold")
plt.xlabel("Time, $t$", fontsize=14, fontweight="bold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(f"var_rho_plot_styled_dt={dt}_{line}_{h}_{T}.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(times, np.array(var_u_interior)/var_u_theory,
         label="FEM (interior)", linewidth=2)
plt.plot(times, np.ones_like(var_u), label="Theory", linewidth=2)
plt.xlabel("Time, $t$", fontsize=14, fontweight="bold")
plt.ylabel("Variance, $(u)$", fontsize=14, fontweight="bold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(f"var_u_plot_styled_dt={dt}_{line}_{h}_{T}.png", dpi=300)
plt.show()

start = min(500, max(len(var_rho_interior) - 1, 0))
print("Density interior / theory:", np.mean(var_rho_interior[start:]) / var_rho_theory)
print("Velocity interior / theory:", np.mean(var_u_interior[start:]) / var_u_theory)
print("Density unique / theory:", np.mean(var_rho_unique[start:]) / var_rho_theory)
print("Velocity unique / theory:", np.mean(var_u_unique[start:]) / var_u_theory)
print("Interior nodal mass:", h)
print("dt:", dt)
