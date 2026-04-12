"""
Adjoint sensitivity computation for WaterLily.jl using ImplicitAD.jl.

Implements the four callbacks required by `ImplicitAD.implicit_unsteady`:
  - `ns_initialize`  : build initial state from design variable `xd = [cx]`
  - `ns_onestep!`    : one BDIM predictor + projection step (CDS convection)
  - `ns_residual!`   : algebraic residual without solving
  - `make_adjoint_callbacks` : convenience constructor

State vector layout: `y = [vec(u); vec(p)]` (all cells including ghosts).
Design variable: `xd[1] = cx`, the x-coordinate of a circular body centre.

Three key requirements for the residual (Martins & Ning, 2021, §6.7):
  1. R(y_forward, yprev, xd) = 0  at the forward solution
  2. ∂R/∂y must be non-singular  (pressure pinning removes null space)
  3. Ghost/boundary cells must enforce actual BC constraints

Include this file from a test script; it is NOT part of the WaterLily module.
"""

using WaterLily
import WaterLily: conv_diff!, inside_u, ∂, measure!
import ForwardDiff
const _wl_div = WaterLily.div   # avoids shadowing Base.div

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Reference cell for pinning pressure to zero (removes constant null space
of the discrete Laplacian with Neumann BCs)."""
pressure_pin(Ng) = CartesianIndex(ntuple(k -> Ng[k] ÷ 2, length(Ng))...)

# ---------------------------------------------------------------------------
# State packing
# ---------------------------------------------------------------------------

"""
    pack_state(flow) -> Vector{Float64}

Flatten `[vec(u); vec(p)]` (full arrays, including ghost cells) to a
concrete CPU `Float64` vector.
"""
pack_state(flow) = [vec(Array(flow.u)); vec(Array(flow.p))]

# ---------------------------------------------------------------------------
# ImplicitAD callbacks
# ---------------------------------------------------------------------------

"""
    ns_initialize(t0, xd, xc0, p) -> Vector{Float64}

Build a fresh `Flow` + body from `xd[1] = cx` and return the packed state.
Pressure is pinned at the reference cell (no-op since initial p=0).
"""
function ns_initialize(_t0, xd, _xc0, p)
    dims, uBC, Δt_val, ν, cy, R_body, ϵ = p
    cx   = Float64(ForwardDiff.value(xd[1]))
    body = AutoBody((x, t) -> sqrt((x[1] - cx)^2 + (x[2] - cy)^2) - R_body)
    flow = Flow(dims, uBC; T=Float64, Δt=Δt_val, ν=ν)
    measure!(flow, body; ϵ=ϵ)
    # Pin pressure (initial p=0 everywhere, so this is a no-op)
    Ng = dims .+ 2
    I_pin = pressure_pin(Ng)
    p_ref = flow.p[I_pin]
    for I in inside(flow.p)
        flow.p[I] -= p_ref
    end
    return pack_state(flow)
end

"""
    ns_onestep!(y, yprev, t, tprev, xd, xci, p)

Advance the flow one step using a simplified BDIM (zeroth-order, V=0) with
CDS convection and pressure projection.  Always executed with plain `Float64`
state (ImplicitAD extracts values before calling `onestep!`).

Key differences from the standard WaterLily `mom_step!`:
  - Uses tight Poisson solver tolerance (eps machine) so that div(u)→0
  - Pins pressure at I_pin to remove the constant null space
  - Single predictor (no corrector) for simplicity
"""
function ns_onestep!(y, yprev, _t, _tprev, xd, _xci, p)
    dims, uBC, Δt_val, ν, cy, R_body, ϵ = p
    cx  = Float64(xd[1])
    Ng  = dims .+ 2
    D   = length(dims)
    Nu  = prod(Ng) * D

    body = AutoBody((x, t) -> sqrt((x[1] - cx)^2 + (x[2] - cy)^2) - R_body)
    flow = Flow(dims, uBC; T=Float64, Δt=Δt_val, ν=ν)
    flow.u .= reshape(yprev[1:Nu],      Ng..., D)
    flow.p .= reshape(yprev[Nu+1:end],  Ng...)

    # Build Poisson operator AFTER measure! so that pois.L = μ₀
    measure!(flow, body; ϵ=ϵ)
    pois = MultiLevelPoisson(flow.p, flow.μ₀, flow.σ)

    # Simplified BDIM predictor (μ₁ = 0, V = 0), CDS convection
    conv_diff!(flow.f, flow.u, flow.σ, cds; ν=ν)
    for i in 1:D, I in inside_u(Ng, i)
        flow.u[I, i] = flow.μ₀[I, i] * (flow.u[I, i] + Δt_val * flow.f[I, i])
    end
    BC!(flow.u, uBC)

    # Pressure projection with tight tolerance (ensures div(u)→0).
    # Replicate project! logic but with tol≈machine epsilon.
    dt = Δt_val
    for I in inside(flow.p)
        pois.z[I] = _wl_div(I, flow.u)
    end
    pois.x .*= dt                                    # scale IC
    solver!(pois; tol=eps(Float64), itmx=500)         # tight solve
    for i in 1:D, I in inside(flow.p)
        flow.u[I, i] -= pois.L[I, i] * ∂(i, I, pois.x)
    end
    pois.x ./= dt                                    # unscale → pressure
    BC!(flow.u, uBC)

    # Pin pressure: shift fluid-region cells so p[I_pin] = 0, and set
    # body-interior cells to 0.  Ghost cells are NOT shifted (they stay
    # at yprev values, decoupled from the interior by zero boundary μ₀).
    I_pin = pressure_pin(Ng)
    p_ref = flow.p[I_pin]
    for I in inside(flow.p)
        d_center = sdf(body, loc(0, I, Float64), 0.0)
        if d_center < -(ϵ + 0.5)
            flow.p[I] = 0.0        # body-interior: pressure indeterminate
        else
            flow.p[I] -= p_ref      # fluid region: remove constant mode
        end
    end

    y[1:Nu]      .= vec(flow.u)
    y[Nu+1:end]  .= vec(flow.p)
end

"""
    ns_residual!(r, y, yprev, t, tprev, xd, xci, p)

Algebraic residual R(y, yprev, xd) evaluated *without* calling the solver.

`xd[1]` may be a `ForwardDiff.Dual` when ImplicitAD computes sensitivities.

Block structure:
  Interior momentum:  R = u_new - μ₀*(u_old + Δt*f) + Δt*μ₀*∂p
  Interior continuity: R = div(u_new)  [except at I_pin]
  Pressure pin:        R = p[I_pin]
  Ghost velocity:      R = u_new - BC(u_new)
  Ghost pressure:      R = p_new - p_prev  (identity, ghost p unchanged)
"""
function ns_residual!(r, y, yprev, t, _tprev, xd, _xci, p)
    dims, uBC_val, Δt_val, ν, cy, R_body, ϵ = p
    cx  = xd[1]   # may be ForwardDiff.Dual
    Ng  = dims .+ 2
    D   = length(dims)
    Nu  = prod(Ng) * D
    Np  = prod(Ng)

    u_new = reshape(y[1:Nu],       Ng..., D)
    p_new = reshape(y[Nu+1:end],   Ng...)
    u_old = reshape(yprev[1:Nu],   Ng..., D)

    body = AutoBody((x, t) -> sqrt((x[1] - cx)^2 + (x[2] - cy)^2) - R_body)

    # Explicit convective-diffusive RHS at u_old (always Float64).
    T_old = eltype(u_old)
    f  = zeros(T_old, Ng..., D)
    Φ  = zeros(T_old, Ng)
    conv_diff!(f, Array(u_old), Φ, cds; ν=Float64(ν))

    lins = LinearIndices(Ng)

    # --- Interior momentum residual ---
    for i in 1:D
        r_range = CartesianIndices(ntuple(k -> k == i ? (3:Ng[k]-1) : (2:Ng[k]-1), D))
        for I in r_range
            d    = sdf(body, loc(i, I, eltype(xd)), t)
            μ0   = WaterLily.μ₀(d, ϵ)
            pred = μ0 * (u_old[I, i] + Δt_val * f[I, i])
            lin  = (i - 1) * Np + lins[I]
            r[lin] = u_new[I, i] - pred + Δt_val * μ0 * ∂(i, I, p_new)
        end
    end

    # --- Interior continuity + pressure pinning ---
    # Pin pressure at I_pin (removes constant null space) and at cells
    # deep inside the body (where μ₀=0 on all surrounding faces makes
    # the pressure indeterminate).
    I_pin = pressure_pin(Ng)
    for I in CartesianIndices(ntuple(k -> 2:Ng[k]-1, D))
        lin = Nu + lins[I]
        d_center = sdf(body, loc(0, I, Float64), t)
        d_val = Float64(ForwardDiff.value(d_center))
        if I == I_pin || d_val < -(ϵ + 0.5)
            r[lin] = p_new[I]              # pinned to zero
        else
            r[lin] = _wl_div(I, u_new)     # divergence-free constraint
        end
    end

    # --- Ghost/boundary velocity residuals ---
    # Apply BC! to a copy of u_new; the difference gives the BC constraint.
    # Works with Dual arrays: Dirichlet assigns Float64→Dual (zero partials),
    # Neumann copies interior Dual values (correct partials).
    u_bc = copy(u_new)
    BC!(u_bc, uBC_val)
    for i in 1:D
        interior_i = CartesianIndices(ntuple(k -> k == i ? (3:Ng[k]-1) : (2:Ng[k]-1), D))
        for I in CartesianIndices(ntuple(k -> 1:Ng[k], D))
            if !(I in interior_i)
                lin = (i - 1) * Np + lins[I]
                r[lin] = u_new[I, i] - u_bc[I, i]
            end
        end
    end

    # --- Ghost pressure residuals ---
    # Ghost pressure cells are not modified by the Poisson solver (boundary
    # μ₀ = 0 decouples them) and only interior cells are shifted by the pin.
    # Therefore p_ghost_new = p_ghost_old, giving r = 0.
    for I in CartesianIndices(ntuple(k -> 1:Ng[k], D))
        if !(I in CartesianIndices(ntuple(k -> 2:Ng[k]-1, D)))
            lin = Nu + lins[I]
            r[lin] = y[lin] - yprev[lin]
        end
    end
end

# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

"""
    make_adjoint_callbacks(dims, uBC, Δt_val, ν, cy, R_body, ϵ)
        -> (ns_initialize, ns_onestep!, ns_residual!, p)

Returns the three callbacks and the shared parameter tuple `p` ready for
`ImplicitAD.implicit_unsteady(init, step!, res!, t, xd, xc, p)`.
"""
function make_adjoint_callbacks(dims, uBC, Δt_val, ν, cy, R_body, ϵ)
    p = (dims, uBC, Δt_val, ν, cy, R_body, ϵ)
    return ns_initialize, ns_onestep!, ns_residual!, p
end
