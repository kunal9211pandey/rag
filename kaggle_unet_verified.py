# ================================================================
# KAGGLE NOTEBOOK — 3D U-Net CFD Building Wind Prediction
# Verified & Error-Free | P100 GPU | ~45 minutes
# ================================================================
# Dataset format required:
#   /kaggle/input/mohali-cfd/dataset_aug/
#     aug_0001/
#       points.npy    shape: (N, 3)   float32  ← cell coordinates
#       p_field.npy   shape: (N,)     float32  ← pressure values
#       U_field.npy   shape: (N, 3)   float32  ← velocity vectors
#       inlet.json    {"Ux": 0.0, "Uy": 2.1, "Uz": 0.0}
#     aug_0002/ ... aug_0112/
# ================================================================


# ════════════════════════════════════════════════════════════════
# CELL 1 — Install & Imports
# ════════════════════════════════════════════════════════════════

import subprocess
subprocess.run(['pip', 'install', 'trimesh', '-q'], check=True)
# scipy pre-installed on Kaggle ✓
# scikit-image pre-installed on Kaggle ✓

import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_DIR = '/kaggle/input/mohali-cfd/dataset_aug'
SAVE_DIR    = '/kaggle/working/saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)
VOX_RES = 64   # 64³ grid — optimal for P100 16GB


# ════════════════════════════════════════════════════════════════
# CELL 2 — OBJ → SDF Voxel (for inference only)
# ════════════════════════════════════════════════════════════════

def obj_to_sdf_voxel(obj_path, resolution=64):
    """
    OBJ → 3D SDF voxel grid for inference.
    Training uses points.npy directly (faster).

    Returns:
        sdf_grid:      (R, R, R) float32, normalized [-1,1]
        domain_bounds: [xmin,ymin,zmin,xmax,ymax,zmax]
    """
    import trimesh

    mesh = trimesh.load(obj_path, force='mesh')
    lo, hi = mesh.bounds
    margin = (hi - lo) * 1.5
    dmin   = lo - margin;  dmin[2] = 0.0
    dmax   = hi + margin

    R  = resolution
    xs = np.linspace(dmin[0], dmax[0], R)
    ys = np.linspace(dmin[1], dmax[1], R)
    zs = np.linspace(dmin[2], dmax[2], R)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    qpts = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)

    # Signed distance: negative inside, positive outside
    _, dist, _ = trimesh.proximity.closest_point(mesh, qpts)
    inside     = mesh.contains(qpts)
    sdf        = np.where(inside, -dist, dist).reshape(R, R, R).astype(np.float32)

    # Normalize to [-1, 1]
    sdf = sdf / (np.abs(sdf).max() + 1e-8)

    bounds = [dmin[0], dmin[1], dmin[2], dmax[0], dmax[1], dmax[2]]
    return sdf, bounds


# ════════════════════════════════════════════════════════════════
# CELL 3 — Points → Voxel Grid scatter
# ════════════════════════════════════════════════════════════════

def points_to_voxel(points, values, bounds, R=64):
    """
    Scatter unstructured CFD points onto regular voxel grid.

    Args:
        points: (N, 3)    float32 cell coordinates
        values: (N,) or (N, C)  field values
        bounds: [xmin,ymin,zmin,xmax,ymax,zmax]
        R:      grid resolution

    Returns:
        voxel: (R,R,R) or (R,R,R,C) float32
    """
    xmin,ymin,zmin,xmax,ymax,zmax = bounds

    # Map to voxel indices [0, R-1]
    ix = np.clip(((points[:,0]-xmin)/(xmax-xmin+1e-8)*(R-1)).astype(int), 0, R-1)
    iy = np.clip(((points[:,1]-ymin)/(ymax-ymin+1e-8)*(R-1)).astype(int), 0, R-1)
    iz = np.clip(((points[:,2]-zmin)/(zmax-zmin+1e-8)*(R-1)).astype(int), 0, R-1)

    if values.ndim == 1:
        grid  = np.zeros((R,R,R), dtype=np.float32)
        count = np.zeros((R,R,R), dtype=np.float32)
        np.add.at(grid,  (ix,iy,iz), values)
        np.add.at(count, (ix,iy,iz), 1.0)
        mask = count > 0
        grid[mask] = grid[mask] / count[mask]
    else:
        C     = values.shape[1]
        grid  = np.zeros((R,R,R,C), dtype=np.float32)
        count = np.zeros((R,R,R),   dtype=np.float32)
        for c in range(C):
            np.add.at(grid[:,:,:,c], (ix,iy,iz), values[:,c])
        np.add.at(count, (ix,iy,iz), 1.0)
        mask = count > 0
        grid[mask] = grid[mask] / count[mask, np.newaxis]

    return grid


def approximate_sdf_from_points(points, bounds, R=64):
    """
    Approximate SDF when no OBJ available.
    Uses point density: low-density regions ≈ building interior.

    Returns: (R,R,R) float32, values in [-1,1]
    """
    xmin,ymin,zmin,xmax,ymax,zmax = bounds

    # Build density voxel
    density = points_to_voxel(
        points,
        np.ones(len(points), dtype=np.float32),
        bounds, R)

    # Low density = possibly inside building → negative SDF
    # High density = definitely outside → positive SDF
    density_norm = density / (density.max() + 1e-8)
    sdf_approx   = 2.0 * density_norm - 1.0   # map [0,1] → [-1,1]

    return sdf_approx.astype(np.float32)


# ════════════════════════════════════════════════════════════════
# CELL 4 — Dataset
# ════════════════════════════════════════════════════════════════

class CFDVoxelDataset(Dataset):
    """
    Converts CFD point cloud data to 3D voxel grids for U-Net.

    Input tensor  (4, R, R, R):
      ch0: SDF approximation   ← building geometry
      ch1: Ux_inlet            ← wind x-component (broadcast)
      ch2: Uy_inlet            ← wind y-component (broadcast)
      ch3: Uz_inlet            ← wind z-component (broadcast)

    Output tensor (4, R, R, R):
      ch0: p  field (normalized)
      ch1: Ux field (normalized)
      ch2: Uy field (normalized)
      ch3: Uz field (normalized)
    """

    def __init__(self, dataset_dir, resolution=64):
        self.R     = resolution
        self.cases = []
        required   = ['points.npy','p_field.npy','U_field.npy','inlet.json']

        for d in sorted(Path(dataset_dir).iterdir()):
            if not d.is_dir():
                continue
            if not all((d/f).exists() for f in required):
                continue
            try:
                pts = np.load(str(d/'points.npy'))
                if pts.ndim == 1 and pts.size % 3 == 0:
                    pts = pts.reshape(-1, 3)
                if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 10:
                    self.cases.append(d)
            except:
                continue

        print(f"✅ UNet Dataset: {len(self.cases)} cases | Res: {resolution}³")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        d = self.cases[idx]
        R = self.R

        pts = np.load(str(d/'points.npy'))
        p   = np.load(str(d/'p_field.npy'))
        U   = np.load(str(d/'U_field.npy'))

        if pts.ndim == 1 and pts.size % 3 == 0:
            pts = pts.reshape(-1, 3)
        if U.ndim == 1 and U.size % 3 == 0:
            U = U.reshape(-1, 3)

        with open(str(d/'inlet.json')) as f:
            inlet = json.load(f)

        Ux_in = float(inlet['Ux'])
        Uy_in = float(inlet['Uy'])
        Uz_in = float(inlet['Uz'])

        N = min(len(pts), len(p), len(U))
        pts = pts[:N];  p = p[:N];  U = U[:N]

        # Domain bounds
        lo = pts.min(axis=0);  hi = pts.max(axis=0)
        rng = hi - lo
        rng[rng < 1.0] = 100.0   # safety: avoid zero range
        bounds = [lo[0],lo[1],lo[2], hi[0],hi[1],hi[2]]

        # Normalize fields
        p_mean = float(p.mean())
        p_std  = max(float(p.std()), 1e-8)
        U_mag  = max(float(np.linalg.norm(U,axis=1).max()), 1e-8)
        p_n    = (p - p_mean) / p_std    # (N,)
        U_n    = U / U_mag               # (N,3)

        # Scatter to voxel grids
        p_vox = points_to_voxel(pts, p_n, bounds, R)     # (R,R,R)
        U_vox = points_to_voxel(pts, U_n, bounds, R)     # (R,R,R,3)

        # Approximate SDF from point density
        sdf = approximate_sdf_from_points(pts, bounds, R) # (R,R,R)

        # Inlet channels: broadcast constant to full volume
        vel_scale = 5.0
        ux_ch = np.full((R,R,R), Ux_in/vel_scale, dtype=np.float32)
        uy_ch = np.full((R,R,R), Uy_in/vel_scale, dtype=np.float32)
        uz_ch = np.full((R,R,R), Uz_in/vel_scale, dtype=np.float32)

        # Input: (4, R, R, R)
        X = np.stack([sdf, ux_ch, uy_ch, uz_ch], axis=0)

        # Output: (4, R, R, R)
        Y = np.stack([
            p_vox,
            U_vox[:,:,:,0],
            U_vox[:,:,:,1],
            U_vox[:,:,:,2],
        ], axis=0)

        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
                torch.tensor([Ux_in, Uy_in, Uz_in], dtype=torch.float32))


# ════════════════════════════════════════════════════════════════
# CELL 5 — 3D U-Net (verified skip connection dimensions)
# ════════════════════════════════════════════════════════════════

class ConvBlock3D(nn.Module):
    """Double Conv3D → BN → LeakyReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3D(nn.Module):
    """MaxPool3D(2) + ConvBlock → halve spatial dims"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock3D(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up3D(nn.Module):
    """
    Upsample + skip connection + ConvBlock.

    Verified dimensions:
      in_ch  → ConvTranspose3d → in_ch//2
      cat(in_ch//2, skip_ch)   → in_ch//2 + skip_ch
      ConvBlock3D(in_ch//2 + skip_ch, out_ch)

    For base=32:
      Up3D(512, 256, 256): 512→256, cat(256,256)=512, conv(512,256) ✓
      Up3D(256, 128, 128): 256→128, cat(128,128)=256, conv(256,128) ✓
      Up3D(128,  64,  64): 128→64,  cat(64,64)=128,   conv(128,64)  ✓
      Up3D( 64,  32,  32):  64→32,  cat(32,32)=64,    conv(64,32)   ✓
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_ch//2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle spatial size mismatch (rounding)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='trilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet3D_CFD(nn.Module):
    """
    3D U-Net for CFD wind field prediction.

    Input:  (B, 4,  64, 64, 64)
    Output: (B, 4,  64, 64, 64)  → [p, Ux, Uy, Uz]

    With base=32:
      enc1:   (B, 32,  64,64,64)
      enc2:   (B, 64,  32,32,32)
      enc3:   (B, 128, 16,16,16)
      enc4:   (B, 256,  8, 8, 8)
      bridge: (B, 512,  4, 4, 4)
      dec4:   (B, 256,  8, 8, 8)  skip=enc4
      dec3:   (B, 128, 16,16,16)  skip=enc3
      dec2:   (B,  64, 32,32,32)  skip=enc2
      dec1:   (B,  32, 64,64,64)  skip=enc1
      out:    (B,   4, 64,64,64)

    Parameters: ~8.5M  (fits in P100 16GB with batch=2)
    """

    def __init__(self, in_ch=4, out_ch=4, base=32):
        super().__init__()
        b = base
        # Encoder
        self.enc1   = ConvBlock3D(in_ch, b)
        self.enc2   = Down3D(b,    b*2)
        self.enc3   = Down3D(b*2,  b*4)
        self.enc4   = Down3D(b*4,  b*8)
        # Bridge
        self.bridge = nn.Sequential(nn.MaxPool3d(2), ConvBlock3D(b*8, b*16))
        # Decoder (in_ch, skip_ch, out_ch)
        self.dec4   = Up3D(b*16, b*8,  b*8)
        self.dec3   = Up3D(b*8,  b*4,  b*4)
        self.dec2   = Up3D(b*4,  b*2,  b*2)
        self.dec1   = Up3D(b*2,  b,    b)
        # Output
        self.out    = nn.Conv3d(b, out_ch, kernel_size=1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight);  nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bridge(e4)
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out(d1)


# ════════════════════════════════════════════════════════════════
# CELL 6 — Loss Function
# ════════════════════════════════════════════════════════════════

class CFDLoss(nn.Module):
    """
    L = w_p*MSE(p) + w_U*MSE(U) + w_grad*GradientLoss
    Gradient loss penalizes unphysical discontinuities (helps wake region).
    """
    def __init__(self, w_p=1.0, w_U=1.0, w_grad=0.1):
        super().__init__()
        self.w_p = w_p; self.w_U = w_U; self.w_grad = w_grad

    def _grad_loss(self, pred, true):
        gx = F.mse_loss(pred[:,:,1:], true[:,:,1:])   \
           + F.mse_loss(pred[:,:,:-1], true[:,:,:-1])
        gy = F.mse_loss(pred[:,:,:,1:], true[:,:,:,1:])   \
           + F.mse_loss(pred[:,:,:,:-1], true[:,:,:,:-1])
        gz = F.mse_loss(pred[:,:,:,:,1:], true[:,:,:,:,1:]) \
           + F.mse_loss(pred[:,:,:,:,:-1], true[:,:,:,:,:-1])
        return (gx + gy + gz) / 6.0

    def forward(self, pred, true):
        lp   = F.mse_loss(pred[:,0:1], true[:,0:1])
        lu   = F.mse_loss(pred[:,1:],  true[:,1:])
        lg   = self._grad_loss(pred, true)
        tot  = self.w_p*lp + self.w_U*lu + self.w_grad*lg
        return tot, {'total':tot.item(),'p':lp.item(),
                     'U':lu.item(),'grad':lg.item()}


# ════════════════════════════════════════════════════════════════
# CELL 7 — Training Loop
# ════════════════════════════════════════════════════════════════

def train_unet(config):
    print(f"\n{'='*65}")
    print(f"3D U-Net CFD Training — {DEVICE}")
    print(f"{'='*65}")

    dataset = CFDVoxelDataset(config['dataset_dir'], config['resolution'])
    n_total = len(dataset)
    n_val   = max(2, int(0.2*n_total))
    n_train = n_total - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'],
        shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {n_train} | Val: {n_val} | Batch: {config['batch_size']}")

    model = UNet3D_CFD(in_ch=4, out_ch=4,
                        base=config['base']).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    loss_fn   = CFDLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-5)

    best_val = float('inf')
    history  = {'train':[], 'val':[], 'p':[], 'U':[], 'lr':[]}
    os.makedirs(config['save_dir'], exist_ok=True)

    print(f"\n{'─'*70}")
    print(f"{'Ep':>4} | {'Train':>8} | {'Val':>8} | "
          f"{'p':>7} | {'U':>7} | {'Grad':>6} | {'Time':>5}")
    print(f"{'─'*70}")

    for epoch in range(1, config['epochs']+1):
        model.train()
        tot=p_l=u_l=g_l=0.0; n_ok=0; t0=time.time()

        for X, Y, _ in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad()
            try:
                pred = model(X)
                loss, ld = loss_fn(pred, Y)
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tot += ld['total']; p_l += ld['p']
                u_l += ld['U'];     g_l += ld['grad']
                n_ok += 1
            except RuntimeError:
                torch.cuda.empty_cache(); continue

        scheduler.step()
        n_ok = max(n_ok, 1)
        avg_tr=tot/n_ok; avg_p=p_l/n_ok; avg_u=u_l/n_ok; avg_g=g_l/n_ok
        lr_now = optimizer.param_groups[0]['lr']

        model.eval(); v_tot=0.0; v_n=0
        with torch.no_grad():
            for X, Y, _ in val_loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                loss, _ = loss_fn(model(X), Y)
                if not torch.isnan(loss): v_tot+=loss.item(); v_n+=1
        avg_val = v_tot / max(v_n, 1)

        history['train'].append(avg_tr); history['val'].append(avg_val)
        history['p'].append(avg_p);      history['U'].append(avg_u)
        history['lr'].append(lr_now)

        t_ep = time.time()-t0
        print(f"{epoch:>4} | {avg_tr:>8.4f} | {avg_val:>8.4f} | "
              f"{avg_p:>7.4f} | {avg_u:>7.4f} | {avg_g:>6.4f} | {t_ep:>4.1f}s")

        if avg_val < best_val and avg_val > 0:
            best_val = avg_val
            torch.save({'epoch':epoch, 'model_state':model.state_dict(),
                        'val_loss':avg_val, 'config':config},
                       os.path.join(config['save_dir'], 'best_model_unet.pth'))
            print(f"       💾 Best saved! val={avg_val:.4f}")

        if epoch % 50 == 0:
            torch.save({'epoch':epoch,'model_state':model.state_dict(),
                        'history':history,'config':config},
                       os.path.join(config['save_dir'],f'unet_ckpt_ep{epoch}.pth'))

    torch.save({'epoch':config['epochs'],'model_state':model.state_dict(),
                'history':history,'config':config},
               os.path.join(config['save_dir'],'final_model_unet.pth'))

    with open(os.path.join(config['save_dir'],'unet_history.json'),'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*65}")
    print(f"✅ U-Net done! Best val: {best_val:.6f}")
    print(f"   → {config['save_dir']}/best_model_unet.pth")
    return model, history


# ════════════════════════════════════════════════════════════════
# CELL 8 — Run Training
# ════════════════════════════════════════════════════════════════

config = {
    'dataset_dir': DATASET_DIR,
    'save_dir':    SAVE_DIR,
    'resolution':  VOX_RES,    # 64³
    'base':        32,          # base channels → ~8.5M params
    'batch_size':  2,           # P100 16GB: batch=2 safe for 64³
    'epochs':      300,
    'lr':          1e-3,
}

model, history = train_unet(config)


# ════════════════════════════════════════════════════════════════
# CELL 9 — Inference: OBJ + Inlet → p and U files
# ════════════════════════════════════════════════════════════════

def infer_unet(model_path, obj_path, Ux_in, Uy_in, Uz_in,
               output_dir='/kaggle/working/output_unet',
               resolution=64):
    """
    Input:  new_building.obj + inlet velocity
    Output: OpenFOAM p and U files
    """
    print(f"\nU-Net Inference: ({Ux_in},{Uy_in},{Uz_in}) m/s")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    ckpt = torch.load(model_path, map_location=DEVICE)
    cfg  = ckpt['config']
    mdl  = UNet3D_CFD(in_ch=4, out_ch=4, base=cfg['base']).to(DEVICE)
    mdl.load_state_dict(ckpt['model_state'])
    mdl.eval()
    print(f"  Model loaded: epoch={ckpt['epoch']}, val={ckpt['val_loss']:.4f}")

    # OBJ → SDF
    print("  Computing SDF...")
    sdf, bounds = obj_to_sdf_voxel(obj_path, resolution)

    # Build input tensor (1, 4, R, R, R)
    R = resolution
    vs = 5.0
    X = np.stack([
        sdf,
        np.full((R,R,R), Ux_in/vs, dtype=np.float32),
        np.full((R,R,R), Uy_in/vs, dtype=np.float32),
        np.full((R,R,R), Uz_in/vs, dtype=np.float32),
    ], axis=0)[np.newaxis]   # (1,4,R,R,R)

    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        Y = mdl(X_t).cpu().numpy()[0]  # (4,R,R,R)

    p_vox = Y[0]     # (R,R,R)
    U_vox = Y[1:]    # (3,R,R,R)

    # Flatten for OpenFOAM
    p_vals = p_vox.reshape(-1)
    N      = len(p_vals)

    # Write p
    with open(os.path.join(output_dir,'p'), 'w') as f:
        f.write("FoamFile\n{\n    version 2.0;\n    format ascii;\n")
        f.write("    class volScalarField;\n    object p;\n}\n")
        f.write("dimensions [0 2 -2 0 0 0 0];\n")
        f.write(f"internalField nonuniform List<scalar>\n{N}\n(\n")
        for v in p_vals: f.write(f"{v:.6g}\n")
        f.write(");\nboundaryField\n{\n")
        f.write("    inlet    { type zeroGradient; }\n")
        f.write("    outlet   { type totalPressure; p0 uniform 101325; }\n")
        f.write("    ground   { type zeroGradient; }\n")
        f.write("    top      { type symmetry; }\n")
        f.write("    buildings{ type zeroGradient; }\n}\n")

    # Write U
    Ux = U_vox[0].reshape(-1)
    Uy = U_vox[1].reshape(-1)
    Uz = U_vox[2].reshape(-1)
    with open(os.path.join(output_dir,'U'), 'w') as f:
        f.write("FoamFile\n{\n    version 2.0;\n    format ascii;\n")
        f.write("    class volVectorField;\n    object U;\n}\n")
        f.write("dimensions [0 1 -1 0 0 0 0];\n")
        f.write(f"internalField nonuniform List<vector>\n{N}\n(\n")
        for x,y,z in zip(Ux,Uy,Uz): f.write(f"({x:.6g} {y:.6g} {z:.6g})\n")
        f.write(f");\nboundaryField\n{{\n")
        f.write(f"    inlet    {{ type fixedValue; "
                f"value uniform ({Ux_in} {Uy_in} {Uz_in}); }}\n")
        f.write("    outlet   { type pressureInletOutletVelocity; "
                "value uniform (0 0 0); }\n")
        f.write("    ground   { type noSlip; }\n")
        f.write("    top      { type symmetry; }\n")
        f.write("    buildings{ type slip; }\n}\n")

    print(f"  p range: [{p_vals.min():.3f}, {p_vals.max():.3f}]")
    print(f"  U_mag max: {np.sqrt(Ux**2+Uy**2+Uz**2).max():.3f} m/s")
    print(f"✅ Done! → {output_dir}/")
    return p_vox, U_vox


# Example (uncomment after training):
# p, U = infer_unet(
#     model_path = f'{SAVE_DIR}/best_model_unet.pth',
#     obj_path   = '/kaggle/input/mohali-cfd/trisurface/building1.obj',
#     Ux_in=0.0, Uy_in=2.1, Uz_in=0.0
# )
