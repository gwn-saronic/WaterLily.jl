"""
Adjoint sensitivity test for WaterLily.jl via ImplicitAD.jl.

Design variable: cx — x-coordinate of a static circular body centre.
Objective:       I  — mean kinetic energy of interior velocity at final step.

Test sequence:
  1. Smoke test  : forward run produces a trajectory of the correct size.
  2. Residual check : R evaluated at the forward solution is ≈0 for all DOFs.
  3. Gradient via ImplicitAD + ForwardDiff.
  4. Finite-difference verification.
"""

using WaterLily, ImplicitAD, ForwardDiff, LinearAlgebra, Test
include("../src/Adjoint.jl")

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

dims   = (32, 16)
uBC    = (1.0, 0.0)
Δt     = 0.01
Nt     = 2
ν      = 0.01

cy     = Float64(dims[2]) / 2.0
R_body = Float64(dims[2]) / 4.0
ϵ      = 1.0
cx0    = Float64(dims[1]) / 4.0   # initial body centre x-position

Ng = dims .+ 2
D  = length(dims)
Np = prod(Ng)
Nu = Np * D
ny = Nu + Np

t_vec = Float64[k * Δt for k in 0:Nt]
xd0   = Float64[cx0]
xc    = zeros(0, Nt + 1)   # no control variables

init, step!, res!, p = make_adjoint_callbacks(dims, uBC, Δt, ν, cy, R_body, ϵ)

# ---------------------------------------------------------------------------
# Objective: mean kinetic energy of interior velocity at the final time step
# ---------------------------------------------------------------------------

function objective(y_traj)
    u = reshape(y_traj[1:Nu, end], Ng..., D)
    s = zero(eltype(y_traj))
    n = 0
    for i in 1:D, I in WaterLily.inside_u(Ng, i)
        s += abs2(u[I, i])
        n += 1
    end
    return s / n
end

# ---------------------------------------------------------------------------
# 1. Smoke test: forward trajectory has correct shape
# ---------------------------------------------------------------------------

@testset "adjoint_test" begin

y_fwd = implicit_unsteady(init, step!, res!, t_vec, xd0, xc, p)
@test size(y_fwd) == (ny, Nt + 1)

# ---------------------------------------------------------------------------
# 2. Residual consistency check (all DOFs)
# ---------------------------------------------------------------------------

r_check = zeros(ny)
res!(r_check, y_fwd[:, 2], y_fwd[:, 1], t_vec[2], t_vec[1], xd0, Float64[], p)

# Collect index sets for diagnostics
lins = LinearIndices(Ng)
mom_idx = Int[]
cont_idx = Int[]
for i in 1:D
    r_range = CartesianIndices(ntuple(k -> k == i ? (3:Ng[k]-1) : (2:Ng[k]-1), D))
    for I in r_range
        push!(mom_idx, (i - 1) * Np + lins[I])
    end
end
for I in CartesianIndices(ntuple(k -> 2:Ng[k]-1, D))
    push!(cont_idx, Nu + lins[I])
end

@info "Residual norms:" momentum=norm(r_check[mom_idx]) continuity=norm(r_check[cont_idx]) total=norm(r_check)

@test norm(r_check[mom_idx])  < 1e-10   # momentum is algebraically exact
@test norm(r_check[cont_idx]) < 1e-6    # continuity limited by Poisson tol
@test norm(r_check)           < 1e-6    # total residual

# ---------------------------------------------------------------------------
# 3. Adjoint gradient via ImplicitAD + ForwardDiff (tangent / JVP mode)
# ---------------------------------------------------------------------------

function program(xd_in)
    y = implicit_unsteady(init, step!, res!, t_vec, xd_in, xc, p)
    return objective(y)
end

g = ForwardDiff.gradient(program, xd0)
dI_dcx_adj = g[1]

# ---------------------------------------------------------------------------
# 4. Central-difference finite-difference verification
# ---------------------------------------------------------------------------

h = 1e-5
I_p = objective(ImplicitAD.odesolve(init, step!, t_vec, [cx0 + h], xc, p))
I_m = objective(ImplicitAD.odesolve(init, step!, t_vec, [cx0 - h], xc, p))
dI_dcx_fd = (I_p - I_m) / (2h)

rel_err = abs(dI_dcx_adj - dI_dcx_fd) / (abs(dI_dcx_fd) + eps())

@info "Adjoint dI/dcx = $dI_dcx_adj   FD = $dI_dcx_fd   rel_err = $(round(100rel_err; digits=3))%"
@test rel_err < 0.05

end  # @testset
