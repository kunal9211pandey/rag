"""
STEP 4: Training Script — FINAL FIXED VERSION
==============================================
Fixes:
  1. Epoch progress clearly print hota hai (print_every=1 default)
  2. Building normals generate karta hai (slip BC ke liye)
  3. Wake region: outlet ke paas extra collocation points
  4. NaN guard + empty batch skip
  5. Scheduler correct order
  6. Different Ux,Uy,Uz inlet correctly handled
"""

import torch
import torch.optim as optim
import numpy as np
import json
import os
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from step3_pinn_model import PINN_CFD, PINNLoss


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class CFDDataset(Dataset):

    def __init__(self, dataset_dir, n_points=3000, n_colloc=1000, n_boundary=300):
        self.n_points   = n_points
        self.n_colloc   = n_colloc
        self.n_boundary = n_boundary

        all_dirs = sorted(Path(dataset_dir).iterdir())
        self.cases = []
        skipped = 0

        for d in all_dirs:
            if not d.is_dir():
                continue
            required = ['points.npy', 'p_field.npy', 'U_field.npy', 'inlet.json']
            if not all((d / f).exists() for f in required):
                skipped += 1
                continue
            pts = np.load(str(d / 'points.npy'))
            if pts.ndim == 1:
                pts = pts.reshape(-1, 3) if pts.size % 3 == 0 else None
            if pts is None or len(pts) < 10:
                skipped += 1
                continue
            self.cases.append(d)

        print(f"Found {len(self.cases)} valid cases "
              f"({skipped} skipped) in {dataset_dir}")

    def __len__(self):
        return len(self.cases)

    def _generate_building_normals(self, pts, n):
        """
        Building surface ke approximate outward normals generate karo.
        Assume building center = domain center. Normal = centrifugal direction.
        """
        center = pts.mean(axis=0)
        dirs = pts - center
        norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        return (dirs / norms).astype(np.float32)

    def __getitem__(self, idx):
        d = self.cases[idx]

        pts  = np.load(str(d / 'points.npy'))
        p    = np.load(str(d / 'p_field.npy'))
        U    = np.load(str(d / 'U_field.npy'))

        # Shape fix
        if pts.ndim == 1 and pts.size % 3 == 0:
            pts = pts.reshape(-1, 3)
        if U.ndim == 1 and U.size % 3 == 0:
            U = U.reshape(-1, 3)

        with open(str(d / 'inlet.json')) as f:
            inlet = json.load(f)
        Ux_in = inlet['Ux']
        Uy_in = inlet['Uy']
        Uz_in = inlet['Uz']

        # Align sizes
        N = min(len(pts), len(p), len(U))
        pts = pts[:N];  p = p[:N];  U = U[:N]

        # Normalize
        p_mean = float(p.mean())
        p_std  = max(float(p.std()), 1e-8)
        U_mag  = max(float(np.linalg.norm(U, axis=1).max()), 1e-8)
        p_n    = (p - p_mean) / p_std
        U_n    = U / U_mag

        # Domain bounds
        xmin, xmax = pts[:,0].min(), pts[:,0].max()
        ymin, ymax = pts[:,1].min(), pts[:,1].max()
        zmin, zmax = pts[:,2].min(), pts[:,2].max()

        def safe_range(lo, hi, fallback=1.0):
            return (lo, hi) if hi - lo > 1e-6 else (lo, lo + fallback)

        xmin, xmax = safe_range(xmin, xmax)
        ymin, ymax = safe_range(ymin, ymax)
        zmin, zmax = safe_range(zmin, zmax)

        iv = np.array([[Ux_in, Uy_in, Uz_in]])

        # ── Data points ──
        ns = min(self.n_points, N)
        idx_d = np.random.choice(N, ns, replace=False)
        pts_s = pts[idx_d]
        p_s   = p_n[idx_d]
        U_s   = U_n[idx_d]
        data_in = np.hstack([pts_s, np.tile(iv, (ns,1))])

        # ── Collocation points — more near outlet (wake region) ──
        nb = self.n_colloc
        # 60% uniform, 40% near outlet (x=xmax)
        n_uniform = int(nb * 0.6)
        n_wake    = nb - n_uniform

        cp_unif = np.random.uniform(
            [xmin,ymin,zmin],[xmax,ymax,zmax], size=(n_uniform,3))
        # Wake: x near xmax
        cp_wake = np.random.uniform(
            [xmax*0.7, ymin, zmin],[xmax, ymax, zmax], size=(n_wake,3))
        colloc_pts = np.vstack([cp_unif, cp_wake])
        colloc_in  = np.hstack([colloc_pts, np.tile(iv, (nb,1))])

        # ── Boundary: Inlet face ──
        nb2 = self.n_boundary
        in_pts = np.random.uniform([xmin,ymin,zmin],[xmin,ymax,zmax], size=(nb2,3))
        in_pts[:,0] = xmin
        inlet_in = np.hstack([in_pts, np.tile(iv,(nb2,1))])

        # ── Boundary: Ground (z=zmin, no-slip) ──
        wall_pts = np.random.uniform([xmin,ymin,zmin],[xmax,ymax,zmin], size=(nb2,3))
        wall_pts[:,2] = zmin
        wall_in = np.hstack([wall_pts, np.tile(iv,(nb2,1))])

        # ── Boundary: Building surfaces (approximate as near-center) ──
        # Sample points near building (middle of domain)
        cx = (xmin+xmax)/2;  cy = (ymin+ymax)/2
        bldg_range = min(xmax-xmin, ymax-ymin) * 0.15
        bldg_pts = np.random.uniform(
            [cx-bldg_range, cy-bldg_range, zmin],
            [cx+bldg_range, cy+bldg_range, zmax*0.5],
            size=(nb2,3))
        bldg_in = np.hstack([bldg_pts, np.tile(iv,(nb2,1))])
        bldg_normals = self._generate_building_normals(bldg_pts, nb2)

        # ── Boundary: Top (z=zmax, symmetry) ──
        top_pts = np.random.uniform([xmin,ymin,zmax],[xmax,ymax,zmax], size=(nb2,3))
        top_pts[:,2] = zmax
        top_in = np.hstack([top_pts, np.tile(iv,(nb2,1))])

        def t(arr):
            return torch.tensor(arr, dtype=torch.float32)

        return {
            'data_in':       t(data_in),
            'p_true':        t(p_s),
            'U_true':        t(U_s),
            'colloc':        t(colloc_in),
            'inlet_in':      t(inlet_in),
            'wall_in':       t(wall_in),
            'bldg_in':       t(bldg_in),
            'bldg_normals':  t(bldg_normals),
            'top_in':        t(top_in),
            'Ux_in':         float(Ux_in),
            'Uy_in':         float(Uy_in),
            'Uz_in':         float(Uz_in),
        }


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_pinn(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"PINN CFD Training — Device: {device}")
    print(f"{'='*60}")

    model = PINN_CFD(
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        coord_scale=config.get('coord_scale', 100.0),
        vel_scale=config.get('vel_scale', 5.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    loss_fn = PINNLoss(
        nu_eff=config['nu_eff'],
        w_data=config['w_data'],
        w_cont=config['w_cont'],
        w_mom=config['w_mom'],
        w_inlet=config['w_inlet'],
        w_noslip=config['w_noslip'],
        w_slip=config['w_slip'],
        w_symm=config['w_symm'],
    )

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config['lr_step'], gamma=0.5)

    dataset = CFDDataset(
        config['dataset_dir'],
        n_points=config['n_points'],
        n_colloc=config['n_colloc'],
        n_boundary=config['n_boundary'],
    )
    if len(dataset) == 0:
        print("❌ No valid cases! Run step1 and step2 first.")
        return None, {}

    n_total = len(dataset)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    print(f"Train: {n_train} | Val: {n_val}")

    os.makedirs(config['save_dir'], exist_ok=True)
    history = {'train': [], 'val': [], 'lr': []}
    best_val = float('inf')

    print(f"\n{'─'*60}")
    print(f"{'Epoch':>6} | {'Train':>10} | {'Val':>10} | "
          f"{'Data':>8} | {'Inlet':>8} | {'NoSlip':>8} | {'Slip':>8} | {'LR':>8}")
    print(f"{'─'*60}")

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        totals = {k: 0.0 for k in
                  ['total','data','cont','momentum','inlet','noslip','slip','symmetry']}
        n_ok = 0

        for batch in train_loader:
            di  = batch['data_in'].squeeze(0).to(device)
            pt  = batch['p_true'].squeeze(0).to(device)
            Ut  = batch['U_true'].squeeze(0).to(device)
            col = batch['colloc'].squeeze(0).to(device).requires_grad_(True)
            ii  = batch['inlet_in'].squeeze(0).to(device)
            wi  = batch['wall_in'].squeeze(0).to(device)
            bi  = batch['bldg_in'].squeeze(0).to(device)
            bn  = batch['bldg_normals'].squeeze(0).to(device)
            ti  = batch['top_in'].squeeze(0).to(device)
            Ux  = batch['Ux_in'].item()
            Uy  = batch['Uy_in'].item()
            Uz  = batch['Uz_in'].item()

            if di.shape[0] < 2:
                continue

            optimizer.zero_grad()
            try:
                loss, ld = loss_fn.compute(
                    model, di, pt, Ut, col,
                    ii, Ux, Uy, Uz,
                    wi, bi, bn, ti)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                for k in totals:
                    totals[k] += ld.get(k, 0.0)
                n_ok += 1
            except Exception as e:
                print(f"\n  ⚠️  batch error: {e}")
                continue

        # Scheduler AFTER optimizer
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        n_ok = max(n_ok, 1)
        for k in totals:
            totals[k] /= n_ok

        val_loss = validate(model, val_loader, device)

        history['train'].append(totals['total'])
        history['val'].append(val_loss)
        history['lr'].append(lr_now)

        # ── Print every epoch (clearly) ──
        if epoch % config['print_every'] == 0 or epoch == 1:
            print(f"{epoch:>6} | {totals['total']:>10.4f} | {val_loss:>10.4f} | "
                  f"{totals['data']:>8.4f} | {totals['inlet']:>8.4f} | "
                  f"{totals['noslip']:>8.4f} | {totals['slip']:>8.4f} | "
                  f"{lr_now:>8.6f}")

        # Save best
        if val_loss < best_val and val_loss > 0:
            best_val = val_loss
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_loss':    val_loss,
                'config':      config,
            }, os.path.join(config['save_dir'], 'best_model.pth'))

        # Checkpoint
        if epoch % 100 == 0:
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'history':     history,
                'config':      config,
            }, os.path.join(config['save_dir'], f'checkpoint_ep{epoch}.pth'))

    # Final save
    torch.save({
        'epoch':       config['epochs'],
        'model_state': model.state_dict(),
        'history':     history,
        'config':      config,
    }, os.path.join(config['save_dir'], 'final_model.pth'))

    with open(os.path.join(config['save_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Training done! Best val loss: {best_val:.6f}")
    print(f"   Saved → {config['save_dir']}/best_model.pth")
    return model, history


def validate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            di = batch['data_in'].squeeze(0).to(device)
            pt = batch['p_true'].squeeze(0).to(device)
            Ut = batch['U_true'].squeeze(0).to(device)
            if di.shape[0] < 2:
                continue
            out = model(di)
            loss = (torch.mean((out[:,0] - pt)**2) +
                    torch.mean((out[:,1:] - Ut)**2)).item()
            if not (np.isnan(loss) or np.isinf(loss)):
                total += loss
                n += 1
    return total / max(n, 1)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

if __name__ == '__main__':

    config = {
        # Paths
        'dataset_dir': './dataset_aug',
        'save_dir':    './saved_models',

        # Model
        'hidden_dim':   256,
        'n_layers':     8,
        'coord_scale':  100.0,   # meters — coordinate normalization
        'vel_scale':    5.0,     # m/s — velocity normalization

        # Physics
        # nu_eff = laminar (1.5e-5) + turbulent (~0.001) for building flows
        'nu_eff': 0.001,

        # Loss weights
        # HIGH weights on BCs → fixes boundary/wake region issues!
        'w_data':   1.0,
        'w_cont':   0.1,
        'w_mom':    0.1,
        'w_inlet':  10.0,   # ← Inlet BC — different Ux,Uy,Uz handle karo
        'w_noslip': 10.0,   # ← Ground no-slip — wake region fix
        'w_slip':   10.0,   # ← Building slip — wake region fix (NEW!)
        'w_symm':   1.0,

        # Training
        'epochs':      500,
        'lr':          1e-3,
        'lr_step':     150,     # LR halve every 150 epochs
        'print_every': 1,       # ← Har epoch print karo (progress clearly dikhega)

        # Sampling
        'n_points':   3000,
        'n_colloc':   1000,
        'n_boundary': 300,
    }

    model, history = train_pinn(config)
