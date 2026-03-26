"""
STEP 2: Data Augmentation — FINAL FIXED VERSION
================================================
14 valid cases × 8 angles = 112 augmented samples

Key fixes:
  - Shape check before reshape
  - Inlet direction consistency ensured
  - meta.json missing check added
"""

import numpy as np
import json
import os
from pathlib import Path


def rotate_points_2d(points, angle_deg):
    """XY plane mein rotate (Z unchanged)"""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    rot = points.copy()
    rot[:, 0] = c * points[:, 0] - s * points[:, 1]
    rot[:, 1] = s * points[:, 0] + c * points[:, 1]
    return rot


def rotate_velocity_2d(U_field, angle_deg):
    """Velocity vectors rotate karo"""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    rot = U_field.copy()
    rot[:, 0] = c * U_field[:, 0] - s * U_field[:, 1]
    rot[:, 1] = s * U_field[:, 0] + c * U_field[:, 1]
    return rot


def rotate_inlet(inlet, angle_deg):
    """Inlet velocity rotate karo"""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return {
        'Ux': float(c * inlet['Ux'] - s * inlet['Uy']),
        'Uy': float(s * inlet['Ux'] + c * inlet['Uy']),
        'Uz': float(inlet['Uz'])
    }


def augment_case(source_dir, output_base_dir, base_case_id, angles):
    source_dir = Path(source_dir)

    points  = np.load(str(source_dir / 'points.npy'))
    p_field = np.load(str(source_dir / 'p_field.npy'))
    U_field = np.load(str(source_dir / 'U_field.npy'))

    # Shape fix
    if points.ndim == 1 and points.size % 3 == 0:
        points = points.reshape(-1, 3)
    if U_field.ndim == 1 and U_field.size % 3 == 0:
        U_field = U_field.reshape(-1, 3)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape wrong: {points.shape}")
    if U_field.ndim != 2 or U_field.shape[1] != 3:
        raise ValueError(f"U_field shape wrong: {U_field.shape}")

    with open(str(source_dir / 'inlet.json')) as f:
        inlet = json.load(f)
    with open(str(source_dir / 'meta.json')) as f:
        orig_meta = json.load(f)

    saved = []
    for i, angle in enumerate(angles):
        aug_id  = (base_case_id - 1) * len(angles) + i + 1
        out_dir = Path(output_base_dir) / f'aug_{aug_id:04d}'
        out_dir.mkdir(parents=True, exist_ok=True)

        pts_rot   = rotate_points_2d(points, angle)
        U_rot     = rotate_velocity_2d(U_field, angle)
        inlet_rot = rotate_inlet(inlet, angle)

        np.save(str(out_dir / 'points.npy'),  pts_rot)
        np.save(str(out_dir / 'p_field.npy'), p_field.copy())
        np.save(str(out_dir / 'U_field.npy'), U_rot)

        with open(str(out_dir / 'inlet.json'), 'w') as f:
            json.dump(inlet_rot, f, indent=2)

        meta = {
            'aug_id':         aug_id,
            'source_case_id': base_case_id,
            'rotation_deg':   angle,
            'season':         orig_meta.get('season', ''),
            'building':       orig_meta.get('building', ''),
            'n_cells':        orig_meta.get('n_cells', 0),
            'inlet':          inlet_rot,
        }
        with open(str(out_dir / 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        saved.append(meta)
        print(f"  ✅ aug_{aug_id:04d} | {angle}° | "
              f"inlet=({inlet_rot['Ux']:.2f}, {inlet_rot['Uy']:.2f}, {inlet_rot['Uz']:.2f})")

    return saved


if __name__ == '__main__':
    DATASET_DIR   = './dataset'
    AUGMENTED_DIR = './dataset_aug'

    # 8 angles → 14 cases × 8 = 112 samples
    ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

    os.makedirs(AUGMENTED_DIR, exist_ok=True)
    all_aug_meta = []
    required = ['points.npy', 'p_field.npy', 'U_field.npy', 'inlet.json', 'meta.json']

    for case_id in range(1, 21):
        src = os.path.join(DATASET_DIR, f'case_{case_id:02d}')
        if not os.path.exists(src):
            continue

        missing = [f for f in required if not os.path.exists(os.path.join(src, f))]
        if missing:
            print(f"⚠️  case_{case_id:02d} skip (missing: {missing})")
            continue

        print(f"\nAugmenting case_{case_id:02d}...")
        try:
            metas = augment_case(src, AUGMENTED_DIR, case_id, ANGLES)
            all_aug_meta.extend(metas)
        except Exception as e:
            print(f"❌ case_{case_id:02d} error: {e}")

    with open(os.path.join(AUGMENTED_DIR, 'augmented_manifest.json'), 'w') as f:
        json.dump(all_aug_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Augmentation done! Total samples: {len(all_aug_meta)}")
