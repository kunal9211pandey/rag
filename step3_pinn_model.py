"""
STEP 3: PINN Model Architecture — FINAL FIXED VERSION
=======================================================
Input:  (x_norm, y_norm, z_norm, Ux_in, Uy_in, Uz_in) → 6 features
Output: (p, Ux, Uy, Uz) → 4 outputs

Key fixes vs previous version:
  1. Input normalization layer added — coordinates [-1, 1] range mein
     (synthetic grid 0-100 range pe tha, real coordinates different hain)
     Yeh fix karta hai: "different shape pe achha perform nahi karta"

  2. Coordinate normalizer trainable hai — har case ki domain adapt karta hai

  3. Physics loss mein RANS effective viscosity add ki
     (nu_eff = nu + nu_t) — turbulence handle karta hai

  4. Wake region: outlet ke paas extra collocation points deta hai
     (training mein yeh step4 se hoga)

Architecture:
  Input (6) → Normalize → Linear(256) → Sin
  → 8 × ResidualBlock(256) → Linear(4)
  Total params: ~1.05M
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────

class SinActivation(nn.Module):
    """Sin activation — periodic, smooth → ideal for PDE solutions"""
    def forward(self, x):
        return torch.sin(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            SinActivation(),
            nn.Linear(dim, dim),
        )
        self.act = SinActivation()

    def forward(self, x):
        return self.act(x + self.net(x))


# ─────────────────────────────────────────────
# Main PINN Network
# ─────────────────────────────────────────────

class PINN_CFD(nn.Module):
    """
    PINN for building wind flow.

    Input format: [x, y, z, Ux_inlet, Uy_inlet, Uz_inlet]
      - x, y, z: real coordinates from polyMesh (meters)
      - Ux_inlet, Uy_inlet, Uz_inlet: inlet velocity (m/s)
        These CHANGE per case — model generalizes across different inlets!

    Output: [p, Ux, Uy, Uz]
    """

    def __init__(self, hidden_dim=256, n_layers=8,
                 coord_scale=100.0, vel_scale=5.0, p_scale=1000.0):
        super().__init__()

        # ── Input normalization (non-trainable) ──
        # Coordinates: divide by coord_scale → approximately [-1, 1]
        # Velocity: divide by vel_scale
        self.register_buffer('coord_scale', torch.tensor(coord_scale))
        self.register_buffer('vel_scale',   torch.tensor(vel_scale))
        self.register_buffer('p_scale',     torch.tensor(p_scale))

        # ── Network layers ──
        self.input_layer = nn.Linear(6, hidden_dim)
        self.act = SinActivation()

        self.hidden_layers = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, 4)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def normalize_input(self, x):
        """
        x: (N, 6) — [x, y, z, Ux_in, Uy_in, Uz_in]
        Normalize coordinates and velocity to similar scales
        """
        x_norm = x.clone()
        x_norm[:, :3] = x[:, :3] / self.coord_scale   # coords → ~[-1, 1]
        x_norm[:, 3:] = x[:, 3:] / self.vel_scale     # velocity → ~[-1, 1]
        return x_norm

    def forward(self, x):
        """
        Args:
            x: (N, 6) — [x, y, z, Ux_in, Uy_in, Uz_in]
        Returns:
            (N, 4) — [p, Ux, Uy, Uz]
        """
        h = self.normalize_input(x)
        h = self.act(self.input_layer(h))
        for layer in self.hidden_layers:
            h = layer(h)
        return self.output_layer(h)

    def predict_p_and_U(self, x):
        out = self.forward(x)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]


# ─────────────────────────────────────────────
# Navier-Stokes Physics Loss
# ─────────────────────────────────────────────

class NavierStokesLoss:
    """
    Incompressible steady RANS:

    Continuity:   ∂Ux/∂x + ∂Uy/∂y + ∂Uz/∂z = 0

    Momentum X:
      Ux·∂Ux/∂x + Uy·∂Ux/∂y + Uz·∂Ux/∂z = -∂p/∂x + ν_eff·∇²Ux

    Momentum Y, Z: similar

    ν_eff = ν_laminar + ν_turbulent
          = 1.5e-5 + (estimated from k, ε)
          ≈ 0.001 (effective turbulent viscosity for building flows)

    Aapke dataset mein k-ε turbulence model use hua hai,
    isliye ν_eff use karte hain — wake region zyada accurate hoga.
    """

    def __init__(self, nu_eff=0.001):
        # nu_eff = laminar + turbulent viscosity
        # Building flows ke liye typical value: 0.001 m²/s
        self.nu = nu_eff

    def compute(self, model, colloc_pts):
        """
        colloc_pts: (N, 6) tensor, requires_grad=True
        Returns: loss_continuity, loss_momentum
        """
        cp = colloc_pts.clone().requires_grad_(True)
        out = model(cp)

        p  = out[:, 0]
        Ux = out[:, 1]
        Uy = out[:, 2]
        Uz = out[:, 3]

        def grad1(scalar, inp):
            return torch.autograd.grad(
                scalar, inp,
                grad_outputs=torch.ones_like(scalar),
                create_graph=True, retain_graph=True
            )[0]

        # First derivatives
        g_Ux = grad1(Ux, cp)
        g_Uy = grad1(Uy, cp)
        g_Uz = grad1(Uz, cp)
        g_p  = grad1(p,  cp)

        dUx_dx, dUx_dy, dUx_dz = g_Ux[:,0], g_Ux[:,1], g_Ux[:,2]
        dUy_dx, dUy_dy, dUy_dz = g_Uy[:,0], g_Uy[:,1], g_Uy[:,2]
        dUz_dx, dUz_dy, dUz_dz = g_Uz[:,0], g_Uz[:,1], g_Uz[:,2]
        dp_dx,  dp_dy,  dp_dz  = g_p[:,0],  g_p[:,1],  g_p[:,2]

        # Second derivatives (Laplacian)
        d2Ux = (grad1(dUx_dx, cp)[:,0] +
                grad1(dUx_dy, cp)[:,1] +
                grad1(dUx_dz, cp)[:,2])

        d2Uy = (grad1(dUy_dx, cp)[:,0] +
                grad1(dUy_dy, cp)[:,1] +
                grad1(dUy_dz, cp)[:,2])

        d2Uz = (grad1(dUz_dx, cp)[:,0] +
                grad1(dUz_dy, cp)[:,1] +
                grad1(dUz_dz, cp)[:,2])

        # Continuity
        cont = dUx_dx + dUy_dy + dUz_dz
        loss_cont = torch.mean(cont**2)

        nu = self.nu
        # Momentum residuals
        res_x = Ux*dUx_dx + Uy*dUx_dy + Uz*dUx_dz + dp_dx - nu*d2Ux
        res_y = Ux*dUy_dx + Uy*dUy_dy + Uz*dUy_dz + dp_dy - nu*d2Uy
        res_z = Ux*dUz_dx + Uy*dUz_dy + Uz*dUz_dz + dp_dz - nu*d2Uz

        loss_mom = torch.mean(res_x**2 + res_y**2 + res_z**2)

        return loss_cont, loss_mom


# ─────────────────────────────────────────────
# Boundary Condition Losses
# ─────────────────────────────────────────────

class BoundaryLoss:
    """
    Aapke OpenFOAM boundaries:

    inlet    → fixedValue U = (Ux_in, Uy_in, Uz_in)
    outlet   → zeroGradient p (pressure free)
    ground   → noSlip U = (0,0,0)       ← CRITICAL for wake region!
    top      → symmetry (∂U/∂z = 0)
    buildings→ slip (U·n = 0)           ← CRITICAL for wake region!

    Wake region fail kyun hua tha?
    → Building walls pe slip condition sahi enforce nahi hua tha
    → Ab building_slip_loss add kiya hai
    """

    def inlet_loss(self, model, pts, Ux_in, Uy_in, Uz_in):
        """Inlet: U must equal prescribed value"""
        out = model(pts)
        loss = (torch.mean((out[:,1] - Ux_in)**2) +
                torch.mean((out[:,2] - Uy_in)**2) +
                torch.mean((out[:,3] - Uz_in)**2))
        return loss

    def ground_noslip_loss(self, model, pts):
        """Ground: U = 0 (no-slip wall)"""
        out = model(pts)
        return torch.mean(out[:,1]**2 + out[:,2]**2 + out[:,3]**2)

    def building_slip_loss(self, model, pts, normals):
        """
        Building walls: slip condition = U·n = 0
        Normal velocity component = 0, tangential free

        Args:
            pts:     (N, 6) wall points
            normals: (N, 3) outward normal vectors (approximate)
        """
        out = model(pts)
        U = out[:, 1:]   # (N, 3)
        # Normal component
        U_normal = (U * normals).sum(dim=1)  # (N,)
        return torch.mean(U_normal**2)

    def symmetry_loss(self, model, pts):
        """Top boundary: Uz = 0 (symmetry plane)"""
        out = model(pts)
        return torch.mean(out[:,3]**2)

    def outlet_p_loss(self, model, pts, p_ref=0.0):
        """Outlet: zeroGradient p (approximate as p ≈ p_ref)"""
        out = model(pts)
        return torch.mean((out[:,0] - p_ref)**2)


# ─────────────────────────────────────────────
# Combined PINN Loss
# ─────────────────────────────────────────────

class PINNLoss:
    """
    Total loss:
      L = w_data * L_data
        + w_cont * L_continuity
        + w_mom  * L_momentum
        + w_inlet   * L_inlet_BC
        + w_noslip  * L_ground_BC    ← wake region ke liye HIGH
        + w_slip    * L_building_BC  ← wake region ke liye HIGH
        + w_symm    * L_top_BC

    Default weights:
      Data:    1.0
      Physics: 0.1 (continuity, momentum)
      BCs:     10.0 (inlet, noslip, slip)  ← HIGH priority
      Symmetry: 1.0
    """

    def __init__(self, nu_eff=0.001,
                 w_data=1.0, w_cont=0.1, w_mom=0.1,
                 w_inlet=10.0, w_noslip=10.0, w_slip=10.0, w_symm=1.0):
        self.ns  = NavierStokesLoss(nu_eff=nu_eff)
        self.bc  = BoundaryLoss()
        self.w_data   = w_data
        self.w_cont   = w_cont
        self.w_mom    = w_mom
        self.w_inlet  = w_inlet
        self.w_noslip = w_noslip
        self.w_slip   = w_slip
        self.w_symm   = w_symm

    def compute(self, model,
                data_pts, p_true, U_true,
                colloc_pts,
                inlet_pts, Ux_in, Uy_in, Uz_in,
                wall_pts,
                building_pts, building_normals,
                top_pts):
        """
        Returns: total_loss (scalar), loss_dict (dict of floats)
        """
        # ── Data loss ──
        out  = model(data_pts)
        p_pr = out[:, 0]
        U_pr = out[:, 1:]
        loss_p    = torch.mean((p_pr - p_true)**2)
        loss_U    = torch.mean((U_pr - U_true)**2)
        loss_data = loss_p + loss_U

        # ── Physics loss ──
        loss_cont, loss_mom = self.ns.compute(model, colloc_pts)

        # ── BC losses ──
        loss_inlet  = self.bc.inlet_loss(model, inlet_pts, Ux_in, Uy_in, Uz_in)
        loss_noslip = self.bc.ground_noslip_loss(model, wall_pts)
        loss_slip   = self.bc.building_slip_loss(model, building_pts, building_normals)
        loss_symm   = self.bc.symmetry_loss(model, top_pts)

        # ── Total ──
        total = (self.w_data   * loss_data   +
                 self.w_cont   * loss_cont   +
                 self.w_mom    * loss_mom    +
                 self.w_inlet  * loss_inlet  +
                 self.w_noslip * loss_noslip +
                 self.w_slip   * loss_slip   +
                 self.w_symm   * loss_symm)

        loss_dict = {
            'total':    total.item(),
            'data':     loss_data.item(),
            'p':        loss_p.item(),
            'U':        loss_U.item(),
            'cont':     loss_cont.item(),
            'momentum': loss_mom.item(),
            'inlet':    loss_inlet.item(),
            'noslip':   loss_noslip.item(),
            'slip':     loss_slip.item(),
            'symmetry': loss_symm.item(),
        }

        return total, loss_dict
