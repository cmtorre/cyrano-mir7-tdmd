# =============================================================================
# MRE_CYR_WT_ChX_rg_enddist.py
# HERRAMIENTA : UCSF ChimeraX — Python API interna
# VARIABLES   : RG, END_DIST
# SECUENCIA   : AACAACAAAAUCACCAAUGUCUUCCA (26 nt, 5'→3')
# DIRECCIÓN   : 5'→3' — numeración PDB desde extremo 5' del MRE (RNAComposer)
# MODO        : Solo cálculo + capas visuales interactivas (sin capturas automáticas)
# EJECUCIÓN   : open "ruta\MRE_CYR_WT_ChX_rg_enddist.py"
# AUTOR       : Pipeline OE2 — MRE_CYR_WT
# =============================================================================

import csv, math, os
from datetime import datetime
import numpy as np

MODEL_INDEX = 0
PDB_INPUT   = "MRE_Cyrano_WT.pdb"

DIR_DATA = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-2\Var_E-Ter-CYR\2. ChimeraX\1. MRE_CYR_WT\RG-END_DIST"

# -----------------------------------------------------------------------------
# Regiones — 26 nt, dirección 5'→3' (numeración PDB RNAComposer)
# nt 1 = extremo 5'  |  nt 26 = extremo 3'
# -----------------------------------------------------------------------------
REGIONS = {
    "ANCHOR": (26, 26),
    "P1":     (19, 25),
    "BULGE":  (15, 18),
    "P2":     (1,  14),
}

REGION_MAP = {}
REGION_MAP[26] = "ANCHOR"
for i in range(19, 26): REGION_MAP[i] = "P1"
for i in range(15, 19): REGION_MAP[i] = "BULGE"
for i in range(1,  15): REGION_MAP[i] = "P2"

COLORS = {
    "ANCHOR": "#E63946",
    "P1":     "#2196F3",
    "BULGE":  "#FF9800",
    "P2":     "#4CAF50",
    "GRAY":   "#C0C0C0",
}

ATOMIC_MASS = {"H":1.008,"C":12.011,"N":14.007,"O":15.999,"P":30.974,"S":32.06,"MG":24.305,"ZN":65.38,"FE":55.845}

# -----------------------------------------------------------------------------
# Funciones de cálculo
# -----------------------------------------------------------------------------
def get_mass(atom):
    elem = atom.element.name.upper() if hasattr(atom, "element") else atom.name[0].upper()
    return ATOMIC_MASS.get(elem, 12.0)

def calc_rg(atoms_list):
    if not atoms_list: return None
    coords = np.array([a.coord for a in atoms_list])
    masses = np.array([get_mass(a) for a in atoms_list])
    total_mass = masses.sum()
    if total_mass == 0: return None
    cm = np.average(coords, weights=masses, axis=0)
    dist_sq = np.sum((coords - cm)**2, axis=1)
    return round(math.sqrt(np.dot(masses, dist_sq) / total_mass), 3)

def get_c4p(res):
    for atom in res.atoms:
        if atom.name == "C4'": return atom
    return None

def calc_end_dist(res_first, res_last):
    a1 = get_c4p(res_first); a2 = get_c4p(res_last)
    if a1 is None or a2 is None: return None
    return round(float(np.linalg.norm(np.array(a1.coord) - np.array(a2.coord))), 3)

# -----------------------------------------------------------------------------
# Funciones ChimeraX
# -----------------------------------------------------------------------------
def cx(session, cmd):
    from chimerax.core.commands import run as cx_run
    cx_run(session, cmd)

def setup_base_view(session):
    cx(session, "set bgcolor white")
    cx(session, "lighting simple")
    cx(session, "graphics silhouettes false")

def apply_style(session):
    cx(session, "nucleotides #1 tube/slab")
    cx(session, "show #1 cartoons")

def color_all_regions(session):
    cx(session, f"color #1:1-14  {COLORS['P2']}")
    cx(session, f"color #1:15-18 {COLORS['BULGE']}")
    cx(session, f"color #1:19-25 {COLORS['P1']}")
    cx(session, f"color #1:26    {COLORS['ANCHOR']}")

def highlight_region(session, start, end, color):
    cx(session, f"color #1 {COLORS['GRAY']}")
    cx(session, f"color #1:{start}-{end} {color}")

# -----------------------------------------------------------------------------
# Capas visuales interactivas — END_DIST
# (sin capturas; el usuario posiciona y captura manualmente)
# -----------------------------------------------------------------------------
def setup_enddist_layers(session):
    print("\n--- CAPAS END_DIST ---")
    setup_base_view(session)
    apply_style(session)

    # Global
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "distance #1:1@C4' #1:26@C4'")
    print("  CAPA END_DIST_GRL    : nt1(5')→nt26(3') activa — posiciona y captura")

    print("\n  Para capas regionales, ejecuta en consola ChimeraX:")
    print("  --- P1 ---")
    print(f"  color #1 {COLORS['GRAY']}")
    print(f"  color #1:19-25 {COLORS['P1']}")
    print("  view #1:19-25 pad 0.4")
    print("  distance delete all")
    print("  distance #1:19@C4' #1:25@C4'")
    print("  --- BULGE ---")
    print(f"  color #1 {COLORS['GRAY']}")
    print(f"  color #1:15-18 {COLORS['BULGE']}")
    print("  view #1:15-18 pad 0.4")
    print("  distance delete all")
    print("  distance #1:15@C4' #1:18@C4'")
    print("  --- P2 ---")
    print(f"  color #1 {COLORS['GRAY']}")
    print(f"  color #1:1-14 {COLORS['P2']}")
    print("  view #1:1-14 pad 0.4")
    print("  distance delete all")
    print("  distance #1:1@C4' #1:14@C4'")
    print("  --- TODAS ---")
    color_all_regions(session)
    print("  view #1")
    print("  distance #1:1@C4' #1:26@C4'")
    print("  distance #1:19@C4' #1:25@C4'")
    print("  distance #1:15@C4' #1:18@C4'")
    print("  distance #1:1@C4' #1:14@C4'")

# -----------------------------------------------------------------------------
# Capas visuales interactivas — RG (elipsoides de inercia)
# -----------------------------------------------------------------------------
def setup_rg_layers(session):
    print("\n--- CAPAS RG ---")
    setup_base_view(session)
    apply_style(session)
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "measure inertia #1:19-25 showEllipsoid true")
    cx(session, "measure inertia #1:15-18 showEllipsoid true")
    cx(session, "measure inertia #1:1-14 showEllipsoid true")
    print("  CAPA RG_TODAS activa (P1 + BULGE + P2) — posiciona y captura")
    print("\n  Para capas regionales individuales, ejecuta en consola ChimeraX:")
    print("  --- RG_P1 ---")
    print(f"  color #1 {COLORS['GRAY']}")
    print(f"  color #1:19-25 {COLORS['P1']}")
    print("  view #1:19-25 pad 0.4")
    print("  measure inertia #1:19-25 showEllipsoid true")
    print("  --- RG_BULGE ---")
    print(f"  color #1 {COLORS['GRAY']}")
    print(f"  color #1:15-18 {COLORS['BULGE']}")
    print("  view #1:15-18 pad 0.4")
    print("  measure inertia #1:15-18 showEllipsoid true")
    print("  --- RG_P2 ---")
    print(f"  color #1 {COLORS['GRAY']}")
    print(f"  color #1:1-14 {COLORS['P2']}")
    print("  view #1:1-14 pad 0.4")
    print("  measure inertia #1:1-14 showEllipsoid true")

# -----------------------------------------------------------------------------
# run()
# -----------------------------------------------------------------------------
def run(session):
    os.makedirs(DIR_DATA, exist_ok=True)

    model    = session.models[MODEL_INDEX]
    residues = list(model.residues)
    res_by_num = {res.number: res for res in residues}

    # --- Cálculos globales ---
    all_atoms  = [a for res in residues for a in res.atoms]
    rg_global  = calc_rg(all_atoms)
    ed_global  = calc_end_dist(res_by_num[1], res_by_num[26])

    # --- Cálculos por región ---
    rows_region = []
    for region, (start, end) in REGIONS.items():
        res_reg   = [res_by_num[i] for i in range(start, end+1) if i in res_by_num]
        atoms_reg = [a for res in res_reg for a in res.atoms]
        rg_reg    = calc_rg(atoms_reg)
        ed_reg    = calc_end_dist(res_by_num[start], res_by_num[end]) if start != end else "N/A (n=1)"
        rows_region.append({
            "REGION":   region,
            "NT_START": start,
            "NT_END":   end,
            "N_NT":     end - start + 1,
            "N_ATOMS":  len(atoms_reg),
            "RG":       rg_reg if rg_reg is not None else "N/A",
            "END_DIST": ed_reg,
        })

    # --- Cálculos por nucleótido ---
    rows_nt = []
    for res in residues:
        atoms = list(res.atoms)
        rows_nt.append({
            "RESNUM":  res.number,
            "RESNAME": res.name.strip().upper(),
            "REGION":  REGION_MAP.get(res.number, "UNK"),
            "N_ATOMS": len(atoms),
            "RG_NT":   calc_rg(atoms) if atoms else "N/A",
        })

    # --- Export CSV ---
    csv_global = os.path.join(DIR_DATA, "MRE_CYR_WT_ChX_rg_enddist_global.csv")
    with open(csv_global, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["VARIABLE", "VALOR", "UNIDAD", "REFERENCIA"])
        w.writerow(["RG_GLOBAL",       rg_global, "Å", f"{len(all_atoms)} átomos"])
        w.writerow(["END_DIST_GLOBAL", ed_global,  "Å", "C4'(nt1-5')→C4'(nt26-3') MRE"])

    csv_reg = os.path.join(DIR_DATA, "MRE_CYR_WT_ChX_rg_enddist_region.csv")
    with open(csv_reg, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_region[0].keys())
        w.writeheader(); w.writerows(rows_region)

    csv_nt = os.path.join(DIR_DATA, "MRE_CYR_WT_ChX_rg_enddist_nt.csv")
    with open(csv_nt, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_nt[0].keys())
        w.writeheader(); w.writerows(rows_nt)

    # --- Resumen terminal ---
    print("\n" + "="*70)
    print("RG + END_DIST — MRE_CYR_WT  |  dirección 5'→3'  |  nt1=5'  nt26=3'")
    print("="*70)
    print(f"\nGLOBAL: RG={rg_global} Å  |  END_DIST={ed_global} Å")
    print(f"\n{'REGION':<8} {'NT':>8} {'N_NT':>5} {'N_ATOMS':>8} {'RG(Å)':>10} {'END_DIST(Å)':>13}")
    for r in rows_region:
        nt_range = f"{r['NT_START']}-{r['NT_END']}"
        print(f"{r['REGION']:<8} {nt_range:>8} {str(r['N_NT']):>5} {str(r['N_ATOMS']):>8} {str(r['RG']):>10} {str(r['END_DIST']):>13}")
    print(f"\n✅ CSV global  → {csv_global}")
    print(f"✅ CSV región  → {csv_reg}")
    print(f"✅ CSV por nt  → {csv_nt}")
    print("="*70)

    # --- Activar capas visuales ---
    setup_enddist_layers(session)
    setup_rg_layers(session)

    print("\n✅ Capas visuales activas. Posiciona, ajusta y captura manualmente.\n")

run(session)
