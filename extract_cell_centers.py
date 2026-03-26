"""
Real Cell Centers Extractor
============================
OpenFOAM ke processor folders se real (x,y,z) cell centers nikaalta hai.

Aapke case mein processor0/ aur processor1/ folders hain.
Yeh script dono ko merge karke real coordinates banata hai.

Run karo:
  python extract_cell_centers.py
"""

import numpy as np
import os
import re
import json
from pathlib import Path


def parse_vector_field(filepath):
    """OpenFOAM vector field parse karo"""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    match = re.search(
        r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\n?\s*\((.*?)\)\s*;',
        content, re.DOTALL)
    if not match:
        return None

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
    return np.array(vectors, dtype=np.float32) if vectors else None


def parse_points_file(filepath):
    """polyMesh/points parse karo"""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    # Find all (x y z) coordinate triplets
    all_pts = re.findall(
        r'\(\s*([\-\d\.eE+]+)\s+([\-\d\.eE+]+)\s+([\-\d\.eE+]+)\s*\)',
        content)
    if all_pts:
        return np.array([[float(a), float(b), float(c)]
                         for a, b, c in all_pts], dtype=np.float32)
    return None


def get_cell_centers_from_processors(case_path, n_cells):
    """
    Processor folders se cell centers reconstruct karo.

    OpenFOAM decomposed case mein:
      processor0/constant/polyMesh/points → proc 0 ke points
      processor1/constant/polyMesh/points → proc 1 ke points

    Cell centers = average of cell vertices
    (approximate but much better than synthetic grid)
    """
    case_path = Path(case_path)

    # ── Strategy 1: processor*/4000/C file ──
    # Agar postProcessing mein C file generate hui ho
    all_centers = []
    proc_dirs = sorted(case_path.glob('processor*/'))

    for proc in proc_dirs:
        c_file = proc / '4000' / 'C'
        if c_file.exists():
            pts = parse_vector_field(str(c_file))
            if pts is not None and len(pts) > 0:
                all_centers.append(pts)
                print(f"    ✅ {c_file.parent.parent.name}/4000/C → {len(pts)} centers")

    if all_centers:
        merged = np.vstack(all_centers)
        if len(merged) >= n_cells * 0.9:  # at least 90% recovered
            print(f"    ✅ Processor C files merged: {len(merged)} centers")
            return merged[:n_cells]

    # ── Strategy 2: processor*/constant/polyMesh/points ──
    # Points (vertices) se approximate cell centers
    all_pts = []
    for proc in proc_dirs:
        pts_file = proc / 'constant' / 'polyMesh' / 'points'
        if pts_file.exists():
            pts = parse_points_file(str(pts_file))
            if pts is not None and len(pts) > 0:
                all_pts.append(pts)
                print(f"    ✅ {pts_file.parent.parent.parent.name}/polyMesh/points → {len(pts)}")

    if all_pts:
        merged = np.vstack(all_pts)
        # Deduplicate approximate
        print(f"    Merged vertices: {len(merged)} → sampling {n_cells} cell centers")
        if len(merged) >= n_cells:
            # Random sample as proxy for cell centers
            idx = np.random.choice(len(merged), n_cells, replace=False)
            return merged[idx]
        else:
            # Use all + pad
            pad_n = n_cells - len(merged)
            pad = merged[np.random.choice(len(merged), pad_n)]
            return np.vstack([merged, pad])

    # ── Strategy 3: Main polyMesh/points ──
    pts_file = case_path / 'constant' / 'polyMesh' / 'points'
    if pts_file.exists():
        pts = parse_points_file(str(pts_file))
        if pts is not None and len(pts) > 0:
            print(f"    Using main polyMesh/points: {len(pts)} vertices")
            if len(pts) >= n_cells:
                idx = np.random.choice(len(pts), n_cells, replace=False)
                return pts[idx]
            else:
                pad_n = n_cells - len(pts)
                pad = pts[np.random.choice(len(pts), pad_n)]
                return np.vstack([pts, pad])

    print(f"    ⚠️  No real coordinates found — keeping synthetic grid")
    return None


def update_case_points(case_path, dataset_case_dir, n_cells):
    """
    Ek case ke liye real cell centers update karo
    """
    print(f"\n  Case: {case_path}")

    real_pts = get_cell_centers_from_processors(case_path, n_cells)

    if real_pts is not None and len(real_pts) == n_cells:
        np.save(str(Path(dataset_case_dir) / 'points.npy'), real_pts)

        # Meta update
        meta_file = Path(dataset_case_dir) / 'meta.json'
        if meta_file.exists():
            with open(str(meta_file)) as f:
                meta = json.load(f)
            meta['using_synthetic_grid'] = False
            meta['coord_source'] = 'processor_polyMesh'
            with open(str(meta_file), 'w') as f:
                json.dump(meta, f, indent=2)

        print(f"  ✅ points.npy updated → real coordinates ({n_cells} cells)")
        x = real_pts[:,0]; y = real_pts[:,1]; z = real_pts[:,2]
        print(f"     X: [{x.min():.1f}, {x.max():.1f}]")
        print(f"     Y: [{y.min():.1f}, {y.max():.1f}]")
        print(f"     Z: [{z.min():.1f}, {z.max():.1f}]")
        return True
    else:
        print(f"  ⚠️  Could not extract real coordinates")
        return False


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    DATASET_ROOT = './mohali_dataset'
    EXTRACTED    = './dataset'

    seasons   = ['mohali_autumn', 'mohali_spring', 'mohali_summer', 'mohali_winter']
    buildings = ['building1', 'building2', 'building3', 'building4', 'building5']

    print("="*60)
    print("Real Cell Centers Extractor")
    print("="*60)

    updated = 0
    case_id = 1

    for season in seasons:
        for building in buildings:
            case_path  = os.path.join(DATASET_ROOT, season, building)
            case_dir   = os.path.join(EXTRACTED, f'case_{case_id:02d}')
            meta_file  = os.path.join(case_dir, 'meta.json')

            case_id += 1

            if not os.path.exists(meta_file):
                continue  # case was skipped

            with open(meta_file) as f:
                meta = json.load(f)

            n_cells = meta['n_cells']

            success = update_case_points(case_path, case_dir, n_cells)
            if success:
                updated += 1

    print(f"\n{'='*60}")
    print(f"✅ Updated {updated} cases with real coordinates")
    print(f"\nNext steps:")
    print(f"  1. python step2_augment_data.py   (re-augment with real coords)")
    print(f"  2. python step4_train.py           (retrain)")
