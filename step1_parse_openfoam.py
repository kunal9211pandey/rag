"""
STEP 1: OpenFOAM Data Parser — FINAL FIXED VERSION
====================================================
Fixes:
  - Points vs cells mismatch: ab actual cell centers use karta hai
  - OpenFOAM ke 'C' file se real cell centers padha jaata hai
  - Synthetic grid sirf last resort hai
  - Winter building1,2,3,4,5 aur summer building1 ke liye
    4000/ folder missing hai — woh skip honge gracefully
"""

import numpy as np
import os
import json
import re
from pathlib import Path


def parse_scalar_field(filepath):
    """OpenFOAM scalar field (p, k, epsilon) → numpy (N,)"""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    # nonuniform list
    match = re.search(
        r'internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\n?\s*\((.*?)\)\s*;',
        content, re.DOTALL)
    if not match:
        # uniform
        m2 = re.search(r'internalField\s+uniform\s+([\d\.\-eE]+)', content)
        if m2:
            return np.array([float(m2.group(1))], dtype=np.float32)
        raise ValueError(f"Scalar parse failed: {filepath}")

    n_declared = int(match.group(1))
    raw = match.group(2).strip()
    values = [float(l.strip()) for l in raw.split('\n')
              if l.strip() and not l.strip().startswith('//')]
    arr = np.array(values, dtype=np.float32)
    print(f"    scalar: declared={n_declared}, parsed={len(arr)}")
    return arr


def parse_vector_field(filepath):
    """OpenFOAM vector field (U) → numpy (N, 3)"""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    match = re.search(
        r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\n?\s*\((.*?)\)\s*;',
        content, re.DOTALL)
    if not match:
        m2 = re.search(r'internalField\s+uniform\s+\(([\d\.\-eE\s]+)\)', content)
        if m2:
            vals = [float(v) for v in m2.group(1).split()]
            return np.array([vals], dtype=np.float32)
        raise ValueError(f"Vector parse failed: {filepath}")

    n_declared = int(match.group(1))
    raw = match.group(2).strip()
    vectors = []
    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            vals = line[1:-1].split()
            if len(vals) == 3:
                try:
                    vectors.append([float(v) for v in vals])
                except:
                    pass
    arr = np.array(vectors, dtype=np.float32)
    print(f"    vector: declared={n_declared}, parsed={len(arr)}")
    return arr


def get_cell_centers(case_path, n_cells):
    """
    Real cell centers get karo — 3 strategies:
    1. 4000/C file (OpenFOAM cell centers field)
    2. postProcessing se
    3. Synthetic grid (last resort)
    """
    case_path = Path(case_path)

    # Strategy 1: 4000/C
    c_file = case_path / '4000' / 'C'
    if c_file.exists():
        print(f"    Using real cell centers: 4000/C")
        try:
            pts = parse_vector_field(str(c_file))
            if len(pts) == n_cells:
                return pts
        except:
            pass

    # Strategy 2: postProcessing cellCentres
    for candidate in [
        case_path / 'constant' / 'cellCentres',
        case_path / '0' / 'C',
    ]:
        if candidate.exists():
            try:
                pts = parse_vector_field(str(candidate))
                if len(pts) == n_cells:
                    return pts
            except:
                pass

    # Strategy 3: Synthetic grid — uniform box covering domain
    # Note: yeh approximate hai, real mesh nahi hai
    # Isliye step1 mein warning deta hai
    print(f"    ⚠️  No C file found — synthetic grid ({n_cells} cells)")
    print(f"    💡 TIP: OpenFOAM mein 'writeCellCentres' run karo real coordinates ke liye")
    cbrt = int(np.ceil(n_cells ** (1/3)))
    xs = np.linspace(-100, 100, cbrt)
    ys = np.linspace(-100, 100, cbrt)
    zs = np.linspace(0, 80, cbrt)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    if len(pts) > n_cells:
        pts = pts[:n_cells]
    elif len(pts) < n_cells:
        pad = np.random.uniform([-100,-100,0],[100,100,80],
                                size=(n_cells-len(pts), 3)).astype(np.float32)
        pts = np.vstack([pts, pad])
    return pts


def extract_inlet_velocity(U_filepath):
    """0/U se inlet velocity"""
    with open(U_filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    # Uinlet variable
    m = re.search(r'Uinlet\s+\(([\d\.\-eE\s]+)\)', content)
    if m:
        vals = [float(v) for v in m.group(1).split()]
        return {'Ux': vals[0], 'Uy': vals[1], 'Uz': vals[2]}

    # inlet fixedValue
    m2 = re.search(
        r'inlet\s*\{[^}]*fixedValue[^}]*uniform\s+\(([\d\.\-eE\s]+)\)',
        content, re.DOTALL)
    if m2:
        vals = [float(v) for v in m2.group(1).split()]
        return {'Ux': vals[0], 'Uy': vals[1], 'Uz': vals[2]}

    # internalField uniform
    m3 = re.search(r'internalField\s+uniform\s+\(([\d\.\-eE\s]+)\)', content)
    if m3:
        vals = [float(v) for v in m3.group(1).split()]
        return {'Ux': vals[0], 'Uy': vals[1], 'Uz': vals[2]}

    raise ValueError(f"Inlet velocity nahi mili: {U_filepath}")


def extract_case(case_path, output_dir, case_id):
    case_path = Path(case_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Processing: {case_path}")

    # ── 1. Inlet (0/U) ──
    U0_file = case_path / '0' / 'U'
    inlet = extract_inlet_velocity(str(U0_file))
    with open(str(output_dir / 'inlet.json'), 'w') as f:
        json.dump(inlet, f, indent=2)
    print(f"  ✅ inlet.json → {inlet}")

    # ── 2. Pressure field (4000/p) ──
    p_file = case_path / '4000' / 'p'
    p_field = parse_scalar_field(str(p_file))
    np.save(str(output_dir / 'p_field.npy'), p_field)
    n_cells = len(p_field)
    print(f"  ✅ p_field.npy → shape {p_field.shape}")

    # ── 3. Velocity field (4000/U) ──
    U_file = case_path / '4000' / 'U'
    U_field = parse_vector_field(str(U_file))
    np.save(str(output_dir / 'U_field.npy'), U_field)
    print(f"  ✅ U_field.npy → shape {U_field.shape}")

    # ── 4. Cell centers (REAL coordinates) ──
    print(f"  Getting cell centers ({n_cells} cells)...")
    points = get_cell_centers(case_path, n_cells)
    np.save(str(output_dir / 'points.npy'), points)
    print(f"  ✅ points.npy → shape {points.shape}")

    # ── 5. Metadata ──
    meta = {
        'case_id':    case_id,
        'case_path':  str(case_path),
        'n_cells':    int(n_cells),
        'n_points':   int(len(points)),
        'inlet':      inlet,
        'p_min':      float(p_field.min()),
        'p_max':      float(p_field.max()),
        'U_max_mag':  float(np.linalg.norm(U_field, axis=1).max()),
        'using_synthetic_grid': not (case_path / '4000' / 'C').exists(),
    }
    with open(str(output_dir / 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ meta.json saved")
    return meta


if __name__ == '__main__':
    DATASET_ROOT = './mohali_dataset'   # ← apna path set karo
    OUTPUT_ROOT  = './dataset'

    seasons   = ['mohali_autumn', 'mohali_spring', 'mohali_summer', 'mohali_winter']
    buildings = ['building1', 'building2', 'building3', 'building4', 'building5']

    all_meta = []
    case_id  = 1

    for season in seasons:
        for building in buildings:
            case_path  = os.path.join(DATASET_ROOT, season, building)
            output_dir = os.path.join(OUTPUT_ROOT, f'case_{case_id:02d}')

            if not os.path.exists(case_path):
                print(f"⚠️  Skip (no folder): {case_path}")
                case_id += 1
                continue

            # Check 4000/p exists
            p_path = os.path.join(case_path, '4000', 'p')
            if not os.path.exists(p_path):
                print(f"⚠️  Skip (no 4000/p): {case_path}")
                case_id += 1
                continue

            try:
                meta = extract_case(case_path, output_dir, case_id)
                meta['season']   = season
                meta['building'] = building
                all_meta.append(meta)
            except Exception as e:
                print(f"❌ Error: {case_path}: {e}")

            case_id += 1

    with open(os.path.join(OUTPUT_ROOT, 'all_cases_meta.json'), 'w') as f:
        json.dump(all_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Done! {len(all_meta)} cases extracted → {OUTPUT_ROOT}/")
    print(f"\n💡 IMPORTANT: Agar 'using_synthetic_grid: true' dikh raha hai,")
    print(f"   OpenFOAM mein 'writeCellCentres' run karo aur dobara step1 chalao.")
    print(f"   Real coordinates se model ki accuracy bahut improve hogi!")
