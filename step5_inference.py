"""
STEP 5: Inference Script
=========================
Input:  new_building.obj + inlet velocity
Output: p file + U file (OpenFOAM format)

Usage:
  python step5_inference.py \
    --obj    path/to/building.obj \
    --Ux     0.0 \
    --Uy     2.1 \
    --Uz     0.0 \
    --model  saved_models/best_model.pth \
    --output output/
"""

import torch
import numpy as np
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

from step3_pinn_model import PINN_CFD


# ─────────────────────────────────────────────
# OBJ → Points extractor
# ─────────────────────────────────────────────

def load_obj_points(obj_path):
    """
    OBJ file se vertices (points) extract karo
    Plus domain ke andar extra points sample karo

    Returns:
        all_points: (N, 3) array — building surface + domain interior
        surface_pts: (M, 3) — building surface only
    """
    vertices = []

    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])

    surface_pts = np.array(vertices, dtype=np.float32)
    print(f"  OBJ vertices loaded: {len(surface_pts)}")

    # Domain bounding box compute karo
    x_min, y_min, z_min = surface_pts.min(axis=0)
    x_max, y_max, z_max = surface_pts.max(axis=0)

    # Domain extend karo (building ke around flow domain)
    margin_x = (x_max - x_min) * 2.0
    margin_y = (y_max - y_min) * 2.0
    margin_z = (z_max - z_min) * 2.0

    domain_min = [x_min - margin_x, y_min - margin_y, 0.0]
    domain_max = [x_max + margin_x, y_max + margin_y, z_max + margin_z]

    # Interior domain points sample karo
    n_interior = 50000
    interior_pts = np.random.uniform(
        low=domain_min, high=domain_max, size=(n_interior, 3)
    ).astype(np.float32)

    # Building ke andar ke points remove karo (rough filter by bbox)
    # A point is outside building if not inside surface bbox
    inside_mask = (
        (interior_pts[:, 0] >= x_min) & (interior_pts[:, 0] <= x_max) &
        (interior_pts[:, 1] >= y_min) & (interior_pts[:, 1] <= y_max) &
        (interior_pts[:, 2] >= z_min) & (interior_pts[:, 2] <= z_max)
    )
    # Keep outside points only (rough approximation)
    exterior_pts = interior_pts[~inside_mask]

    # Combine surface + exterior
    all_points = np.vstack([surface_pts, exterior_pts])
    print(f"  Domain points: {len(all_points)} (surface={len(surface_pts)}, exterior={len(exterior_pts)})")

    return all_points, surface_pts, [domain_min, domain_max]


# ─────────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────────

def load_model(model_path):
    """Saved model load karo"""
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']

    model = PINN_CFD(
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"  Model loaded from: {model_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")

    return model


# ─────────────────────────────────────────────
# Predict P and U
# ─────────────────────────────────────────────

def predict(model, points, Ux_in, Uy_in, Uz_in, batch_size=10000):
    """
    PINN se P and U predict karo

    Args:
        model:     trained PINN_CFD
        points:    (N, 3) numpy array
        Ux_in, Uy_in, Uz_in: inlet velocity
        batch_size: process in batches (memory efficient)
    Returns:
        p_pred: (N,) pressure
        U_pred: (N, 3) velocity
    """
    N = len(points)
    p_all = np.zeros(N, dtype=np.float32)
    U_all = np.zeros((N, 3), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            pts_batch = points[start:end]

            inlet_vec = np.array([[Ux_in, Uy_in, Uz_in]] * len(pts_batch))
            x_input = np.hstack([pts_batch, inlet_vec])
            x_tensor = torch.tensor(x_input, dtype=torch.float32)

            out = model(x_tensor)
            p_all[start:end] = out[:, 0].numpy()
            U_all[start:end] = out[:, 1:].numpy()

            if start % 50000 == 0:
                print(f"  Predicted {end}/{N} points...")

    return p_all, U_all


# ─────────────────────────────────────────────
# Write OpenFOAM files
# ─────────────────────────────────────────────

def write_openfoam_header(f, field_class, location, obj_name, dimensions):
    f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
    f.write("  =========                 |\n")
    f.write("  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n")
    f.write("   \\\\    /   O peration     | PINN Prediction\n")
    f.write("    \\\\  /    A nd           |\n")
    f.write("     \\\\/     M anipulation  |\n")
    f.write("\\*---------------------------------------------------------------------------*/\n")
    f.write("FoamFile\n{\n")
    f.write(f"    version     2.0;\n")
    f.write(f"    format      ascii;\n")
    f.write(f"    class       {field_class};\n")
    f.write(f"    location    \"{location}\";\n")
    f.write(f"    object      {obj_name};\n")
    f.write("}\n")
    f.write("// " + "* " * 36 + "//\n\n")
    f.write(f"dimensions      {dimensions};\n\n")


def write_p_file(output_path, p_values):
    """OpenFOAM p file likhna"""
    with open(output_path, 'w') as f:
        write_openfoam_header(
            f,
            field_class="volScalarField",
            location="predicted",
            obj_name="p",
            dimensions="[0 2 -2 0 0 0 0]"
        )
        f.write(f"internalField   nonuniform List<scalar>\n")
        f.write(f"{len(p_values)}\n(\n")
        for v in p_values:
            f.write(f"{v:.6g}\n")
        f.write(");\n\n")
        f.write("boundaryField\n{\n")
        f.write("    inlet { type zeroGradient; }\n")
        f.write("    outlet { type totalPressure; p0 uniform 101325; }\n")
        f.write("    ground { type zeroGradient; }\n")
        f.write("    top { type symmetry; }\n")
        f.write("    buildings { type zeroGradient; }\n")
        f.write("}\n\n// " + "* " * 36 + "//\n")

    print(f"  ✅ p file written: {output_path}")


def write_U_file(output_path, U_values):
    """OpenFOAM U file likhna"""
    with open(output_path, 'w') as f:
        write_openfoam_header(
            f,
            field_class="volVectorField",
            location="predicted",
            obj_name="U",
            dimensions="[0 1 -1 0 0 0 0]"
        )
        f.write(f"internalField   nonuniform List<vector>\n")
        f.write(f"{len(U_values)}\n(\n")
        for v in U_values:
            f.write(f"({v[0]:.6g} {v[1]:.6g} {v[2]:.6g})\n")
        f.write(");\n\n")
        f.write("boundaryField\n{\n")
        f.write("    inlet { type fixedValue; value uniform (0 2.1 0); }\n")
        f.write("    outlet { type pressureInletOutletVelocity; value uniform (0 0 0); }\n")
        f.write("    ground { type noSlip; }\n")
        f.write("    top { type symmetry; }\n")
        f.write("    buildings { type slip; }\n")
        f.write("}\n\n// " + "* " * 36 + "//\n")

    print(f"  ✅ U file written: {output_path}")


# ─────────────────────────────────────────────
# Main inference pipeline
# ─────────────────────────────────────────────

def run_inference(obj_path, Ux_in, Uy_in, Uz_in, model_path, output_dir):
    """
    End-to-end inference:
      OBJ + Inlet → P file + U file
    """
    print(f"\n{'='*60}")
    print(f"PINN CFD Inference")
    print(f"{'='*60}")
    print(f"OBJ:    {obj_path}")
    print(f"Inlet:  ({Ux_in}, {Uy_in}, {Uz_in}) m/s")
    print(f"Model:  {model_path}")
    print(f"Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load model
    print(f"\n[1/4] Loading model...")
    model = load_model(model_path)

    # Step 2: Extract points from OBJ
    print(f"\n[2/4] Processing OBJ geometry...")
    all_points, surface_pts, domain = load_obj_points(obj_path)

    # Step 3: Predict
    print(f"\n[3/4] Running PINN prediction...")
    p_pred, U_pred = predict(model, all_points, Ux_in, Uy_in, Uz_in)

    print(f"  p range: [{p_pred.min():.2f}, {p_pred.max():.2f}]")
    print(f"  U_mag range: [{np.linalg.norm(U_pred, axis=1).min():.2f}, "
          f"{np.linalg.norm(U_pred, axis=1).max():.2f}]")

    # Step 4: Write OpenFOAM files
    print(f"\n[4/4] Writing OpenFOAM files...")
    write_p_file(os.path.join(output_dir, 'p'), p_pred)
    write_U_file(os.path.join(output_dir, 'U'), U_pred)

    # Save points too (for post-processing)
    np.save(os.path.join(output_dir, 'points_predicted.npy'), all_points)

    # Save summary
    summary = {
        'timestamp':   datetime.now().isoformat(),
        'obj_file':    obj_path,
        'inlet':       {'Ux': Ux_in, 'Uy': Uy_in, 'Uz': Uz_in},
        'model':       model_path,
        'n_points':    len(all_points),
        'p_min':       float(p_pred.min()),
        'p_max':       float(p_pred.max()),
        'U_max_mag':   float(np.linalg.norm(U_pred, axis=1).max()),
        'domain':      domain,
    }
    with open(os.path.join(output_dir, 'prediction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Inference complete!")
    print(f"   Output files:")
    print(f"     {output_dir}/p")
    print(f"     {output_dir}/U")
    print(f"   These files can be directly used in OpenFOAM / ParaView!")

    return p_pred, U_pred


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINN CFD Inference')
    parser.add_argument('--obj',    required=True,  help='Building OBJ file path')
    parser.add_argument('--Ux',     type=float, default=0.0,  help='Inlet Ux (m/s)')
    parser.add_argument('--Uy',     type=float, default=2.1,  help='Inlet Uy (m/s)')
    parser.add_argument('--Uz',     type=float, default=0.0,  help='Inlet Uz (m/s)')
    parser.add_argument('--model',  default='saved_models/best_model.pth')
    parser.add_argument('--output', default='./predicted_output')

    args = parser.parse_args()

    p, U = run_inference(
        obj_path   = args.obj,
        Ux_in      = args.Ux,
        Uy_in      = args.Uy,
        Uz_in      = args.Uz,
        model_path = args.model,
        output_dir = args.output,
    )
