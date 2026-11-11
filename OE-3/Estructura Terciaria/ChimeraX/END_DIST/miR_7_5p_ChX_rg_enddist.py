# =============================================================================
# miR_7_5p_ChX_rg_enddist.py
# HERRAMIENTA : UCSF ChimeraX — Python API interna
# VARIABLES   : RG, END_DIST
# NIVELES     : Global | Por región (ANCHOR/P1/BULGE/P2) | Por nucleótido (RG)
# SECUENCIA   : UGGAAGACUAGUGAUUUUGUUGUU (24 nt, 5'→3')
# IMÁGENES    : PNG por elemento (END_DIST + RG) | MP4 manual
# EJECUCIÓN   : open "ruta\miR_7_5p_ChX_rg_enddist.py"
# AUTOR       : Pipeline OE3 — miR-7-5p
# =============================================================================

import csv
import math
import os
from datetime import datetime
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
MODEL_INDEX = 0
PDB_INPUT   = "miR_7_5p.pdb"

DIR_DATA = (
    "C:\\Users\\cmart\\OneDrive\\Documentos OneDrive\\1. circRNA_TESIS\\Tesis\\"
    "7.Metodo\\OE-3\\v-terc-MIR\\2.ChimeraX\\miR_7_5p\\RG_END_DIST"
)
DIR_VIZ = (
    "C:\\Users\\cmart\\OneDrive\\Documentos OneDrive\\1. circRNA_TESIS\\Tesis\\"
    "7.Metodo\\OE-3\\v-terc-MIR\\2.ChimeraX\\miR_7_5p\\VISUALIZACION"
)

IMG_W           = 1800
IMG_H           = 1800
IMG_SUPERSAMPLE = 3

VID_FRAMERATE  = 15
VID_FRAMES     = 120
VID_TURN_STEP  = 3

# Regiones — miR-7-5p: 24 nt (5'→3')
# ANCHOR: nt 1 | P1: nt 2-8 | BULGE: nt 9-10 | P2: nt 11-24
REGIONS = {
    "ANCHOR": (1,  1),
    "P1":     (2,  8),
    "BULGE":  (9,  10),
    "P2":     (11, 24),
}

REGION_MAP = {}
REGION_MAP[1] = "ANCHOR"
for i in range(2, 9):    REGION_MAP[i] = "P1"
for i in range(9, 11):   REGION_MAP[i] = "BULGE"
for i in range(11, 25):  REGION_MAP[i] = "P2"

COLORS = {
    "ANCHOR": "#E63946",
    "P1":     "#2196F3",
    "BULGE":  "#FF9800",
    "P2":     "#4CAF50",
    "GRAY":   "#C0C0C0",
}

# ---------------------------------------------------------------------------
# CÁLCULO RG — Svergun & Koch 2003, Rep Prog Phys 66:1735
# ---------------------------------------------------------------------------
ATOMIC_MASS = {
    "H": 1.008,  "C": 12.011, "N": 14.007,
    "O": 15.999, "P": 30.974, "S": 32.06,
    "MG": 24.305, "ZN": 65.38, "FE": 55.845,
}

def get_mass(atom):
    elem = atom.element.name.upper() if hasattr(atom, "element") else atom.name[0].upper()
    return ATOMIC_MASS.get(elem, 12.0)

def calc_rg(atoms_list):
    if not atoms_list:
        return None
    coords = np.array([a.coord for a in atoms_list])
    masses = np.array([get_mass(a) for a in atoms_list])
    total_mass = masses.sum()
    if total_mass == 0:
        return None
    cm = np.average(coords, weights=masses, axis=0)
    dist_sq = np.sum((coords - cm) ** 2, axis=1)
    return round(math.sqrt(np.dot(masses, dist_sq) / total_mass), 3)

# ---------------------------------------------------------------------------
# CÁLCULO END_DIST — C4'→C4'
# ---------------------------------------------------------------------------
def get_c4p(res):
    for atom in res.atoms:
        if atom.name == "C4'":
            return atom
    return None

def calc_end_dist(res_first, res_last):
    a1 = get_c4p(res_first)
    a2 = get_c4p(res_last)
    if a1 is None or a2 is None:
        return None
    return round(float(np.linalg.norm(np.array(a1.coord) - np.array(a2.coord))), 3)

# ---------------------------------------------------------------------------
# UTILIDADES ChimeraX
# ---------------------------------------------------------------------------
def cx(session, cmd):
    from chimerax.core.commands import run as cx_run
    cx_run(session, cmd)

def close_model(session, mid):
    try:
        cx(session, f"close #{mid}")
    except Exception:
        pass

def close_distances(session):
    try:
        from chimerax.core.commands import run as cx_run
        cx_run(session, "distance delete all")
    except Exception:
        pass
    for mid in [8, 7, 6, 5, 4, 3]:
        close_model(session, mid)

def close_ellipsoids(session):
    for mid in [7, 6, 5, 4, 3]:
        close_model(session, mid)

def save_img(session, filepath):
    cx(session, f'save "{filepath}" width {IMG_W} height {IMG_H} supersample {IMG_SUPERSAMPLE}')

def setup_base_view(session):
    cx(session, "set bgcolor white")
    cx(session, "lighting simple")
    cx(session, "graphics silhouettes false")

def apply_style(session):
    cx(session, "nucleotides #1 tube/slab")
    cx(session, "show #1 cartoons")

def color_all_regions(session):
    cx(session, f"color #1:1       {COLORS['ANCHOR']}")
    cx(session, f"color #1:2-8     {COLORS['P1']}")
    cx(session, f"color #1:9-10    {COLORS['BULGE']}")
    cx(session, f"color #1:11-24   {COLORS['P2']}")

def highlight_region(session, start, end, color):
    cx(session, f"color #1 {COLORS['GRAY']}")
    cx(session, f"color #1:{start}-{end} {color}")

# ---------------------------------------------------------------------------
# IMÁGENES END_DIST
# ---------------------------------------------------------------------------
def generate_enddist_images(session, res_by_num, dir_viz):
    imgs = {}
    setup_base_view(session)
    apply_style(session)

    # Global (nt 1 → nt 24)
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "distance #1:1@C4' #1:24@C4'")
    p = os.path.join(dir_viz, "miR_7_5p_ChX_END_DIST_GRL.png")
    save_img(session, p)
    close_distances(session)
    imgs["END_DIST_GRL"] = p
    print(f"✅ {os.path.basename(p)}")

    # ANCHOR (n=1, sin línea)
    highlight_region(session, 1, 1, COLORS["ANCHOR"])
    cx(session, "view #1:1 pad 0.8")
    p = os.path.join(dir_viz, "miR_7_5p_ChX_END_DIST_ANCHOR.png")
    save_img(session, p)
    imgs["END_DIST_ANCHOR"] = p
    print(f"✅ {os.path.basename(p)}  [n=1 — sin línea END_DIST]")

    # Por región
    for region, (start, end) in [("P1", (2, 8)), ("BULGE", (9, 10)), ("P2", (11, 24))]:
        highlight_region(session, start, end, COLORS[region])
        cx(session, f"view #1:{start}-{end} pad 0.4")
        cx(session, f"distance #1:{start}@C4' #1:{end}@C4'")
        tag = "B" if region == "BULGE" else region
        p = os.path.join(dir_viz, f"miR_7_5p_ChX_END_DIST_{tag}.png")
        save_img(session, p)
        close_distances(session)
        imgs[f"END_DIST_{tag}"] = p
        print(f"✅ {os.path.basename(p)}")

    # Todas simultáneas
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "distance #1:1@C4' #1:24@C4'")
    cx(session, "distance #1:2@C4' #1:8@C4'")
    cx(session, "distance #1:9@C4' #1:10@C4'")
    cx(session, "distance #1:11@C4' #1:24@C4'")
    p = os.path.join(dir_viz, "miR_7_5p_ChX_END_DIST_TODAS.png")
    save_img(session, p)
    imgs["END_DIST_TODAS"] = p
    print(f"✅ {os.path.basename(p)}")

    return imgs

# ---------------------------------------------------------------------------
# IMÁGENES RG
# ---------------------------------------------------------------------------
def generate_rg_images(session, dir_viz):
    imgs = {}
    setup_base_view(session)
    apply_style(session)
    close_ellipsoids(session)

    rg_regions = [("P1", (2, 8)), ("BULGE", (9, 10)), ("P2", (11, 24))]

    for region, (start, end) in rg_regions:
        close_ellipsoids(session)
        highlight_region(session, start, end, COLORS[region])
        cx(session, f"view #1:{start}-{end} pad 0.4")
        cx(session, f"measure inertia #1:{start}-{end} showEllipsoid true")
        tag = "B" if region == "BULGE" else region
        p = os.path.join(dir_viz, f"miR_7_5p_ChX_RG_{tag}.png")
        save_img(session, p)
        imgs[f"RG_{tag}"] = p
        print(f"✅ {os.path.basename(p)}")

    close_ellipsoids(session)
    color_all_regions(session)
    cx(session, "view #1")
    for region, (start, end) in rg_regions:
        cx(session, f"measure inertia #1:{start}-{end} showEllipsoid true")
    p = os.path.join(dir_viz, "miR_7_5p_ChX_RG_TODAS.png")
    save_img(session, p)
    imgs["RG_TODAS"] = p
    print(f"✅ {os.path.basename(p)}")

    return imgs

# ---------------------------------------------------------------------------
# VIDEOS — escenas para ejecución manual
# ---------------------------------------------------------------------------
def setup_video_scenes(session, dir_viz):
    mp4_paths = {}

    # Escena END_DIST_TODAS
    setup_base_view(session)
    apply_style(session)
    close_distances(session)
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "distance #1:1@C4' #1:24@C4'")
    cx(session, "distance #1:2@C4' #1:8@C4'")
    cx(session, "distance #1:9@C4' #1:10@C4'")
    cx(session, "distance #1:11@C4' #1:24@C4'")
    mp4_ed = os.path.join(dir_viz, "miR_7_5p_ChX_END_DIST_TODAS.mp4")
    print("\nVIDEO END_DIST_TODAS — ejecutar manualmente en consola ChimeraX:")
    print("   movie record")
    print(f"   turn y {VID_TURN_STEP} {VID_FRAMES}")
    print(f'   movie encode output "{mp4_ed}" format h264 framerate {VID_FRAMERATE}')
    mp4_paths["END_DIST_TODAS"] = mp4_ed

    # Escena RG_TODAS
    setup_base_view(session)
    apply_style(session)
    close_ellipsoids(session)
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "measure inertia #1:2-8   showEllipsoid true")
    cx(session, "measure inertia #1:9-10  showEllipsoid true")
    cx(session, "measure inertia #1:11-24 showEllipsoid true")
    mp4_rg = os.path.join(dir_viz, "miR_7_5p_ChX_RG_TODAS.mp4")
    print("\nVIDEO RG_TODAS — ejecutar manualmente en consola ChimeraX:")
    print("   movie record")
    print(f"   turn y {VID_TURN_STEP} {VID_FRAMES}")
    print(f'   movie encode output "{mp4_rg}" format h264 framerate {VID_FRAMERATE}')
    mp4_paths["RG_TODAS"] = mp4_rg

    return mp4_paths

# ---------------------------------------------------------------------------
# GLB
# ---------------------------------------------------------------------------
def export_glb(session, dir_viz):
    glb_path = os.path.join(dir_viz, "miR_7_5p_ChX.glb")
    try:
        cx(session, f'save "{glb_path}"')
        print(f"✅ GLB: {os.path.basename(glb_path)}")
        return glb_path
    except Exception as e:
        print(f"⚠️  GLB no generado: {e}")
        return None

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(session):
    today = datetime.now().strftime("%d/%m/%Y")

    for d in [DIR_DATA, DIR_VIZ]:
        print(f"DEBUG dir: {d}")
        print(f"DEBUG existe: {os.path.exists(d)}")
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print("DEBUG creada OK")
        else:
            print("DEBUG encontrada OK")

    model      = session.models[MODEL_INDEX]
    residues   = list(model.residues)
    res_by_num = {res.number: res for res in residues}

    all_atoms = [a for res in residues for a in res.atoms]
    rg_global = calc_rg(all_atoms)
    ed_global = calc_end_dist(res_by_num[1], res_by_num[24])

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

    csv_global = os.path.join(DIR_DATA, "miR_7_5p_ChX_rg_enddist_global.csv")
    with open(csv_global, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["VARIABLE", "VALOR", "UNIDAD", "REFERENCIA"])
        w.writerow(["RG_GLOBAL",       rg_global, "Å", f"{len(all_atoms)} átomos"])
        w.writerow(["END_DIST_GLOBAL", ed_global,  "Å", "C4'(nt1)→C4'(nt24) 5'→3'"])

    csv_reg = os.path.join(DIR_DATA, "miR_7_5p_ChX_rg_enddist_region.csv")
    with open(csv_reg, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_region[0].keys())
        w.writeheader()
        w.writerows(rows_region)

    csv_nt = os.path.join(DIR_DATA, "miR_7_5p_ChX_rg_enddist_nt.csv")
    with open(csv_nt, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_nt[0].keys())
        w.writeheader()
        w.writerows(rows_nt)

    print("\n--- Generando imágenes END_DIST ---")
    imgs_ed = generate_enddist_images(session, res_by_num, DIR_VIZ)

    print("\n--- Generando imágenes RG ---")
    imgs_rg = generate_rg_images(session, DIR_VIZ)

    print("\n--- Escenas de video (ejecución manual) ---")
    mp4s = setup_video_scenes(session, DIR_VIZ)

    print("\n--- Exportando GLB ---")
    setup_base_view(session)
    apply_style(session)
    color_all_regions(session)
    cx(session, "view #1")
    glb_path = export_glb(session, DIR_VIZ)

    report_path = os.path.join(DIR_DATA, "miR_7_5p_ChX_rg_enddist_reporte.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("REPORTE DE ANALISIS RG + END_DIST - miR-7-5p\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"HERRAMIENTA      : UCSF ChimeraX — Python API interna\n")
        f.write(f"VERSION          : verificar en ChimeraX → Help > About\n")
        f.write(f"FECHA            : {today}\n")
        f.write(f"INPUT            : {PDB_INPUT}\n")
        f.write(f"PARAMETROS       : default\n")
        f.write(f"                   RG: Rg = sqrt(Σmi·di² / Σmi); masas IUPAC\n")
        f.write(f"                   Ref: Svergun & Koch 2003, Rep Prog Phys 66:1735\n")
        f.write(f"                   END_DIST: C4'→C4' uniforme todas las regiones\n")
        f.write(f"                   END_DIST ANCHOR: N/A (n=1)\n")
        f.write(f"                   Imágenes: {IMG_W}x{IMG_H}px, supersample {IMG_SUPERSAMPLE}\n")
        f.write(f"                   Fondo blanco, iluminación simple\n")
        f.write(f"                   Estilo: tube/slab + cartoons\n\n")
        f.write(f"SECUENCIA        : 5'-UGGAAGACUAGUGAUUUUGUUGUU-3' (24 nt)\n")
        f.write(f"DIRECCIÓN        : 5'→3' (convención miRNA — inversa a variantes Cyrano)\n\n")
        f.write(f"ELEMENTOS ESTRUCTURALES (miR-7-5p — 24 nt)\n")
        f.write(f"                   ANCHOR : nt 1\n")
        f.write(f"                   P1     : nt 2–8\n")
        f.write(f"                   BULGE  : nt 9–10\n")
        f.write(f"                   P2     : nt 11–24\n\n")

        f.write("-" * 70 + "\n")
        f.write("ARCHIVOS DE SALIDA\n")
        f.write("-" * 70 + "\n")
        f.write(f"  CSV global      : miR_7_5p_ChX_rg_enddist_global.csv\n")
        f.write(f"  CSV por región  : miR_7_5p_ChX_rg_enddist_region.csv\n")
        f.write(f"  CSV por nt      : miR_7_5p_ChX_rg_enddist_nt.csv\n")
        f.write(f"  Script Python   : miR_7_5p_ChX_rg_enddist.py\n")
        for k, v in {**imgs_ed, **imgs_rg}.items():
            f.write(f"  PNG {k:<20}: {os.path.basename(v)}\n")
        for k, v in mp4s.items():
            f.write(f"  MP4 {k:<20}: {os.path.basename(v)} (manual)\n")
        if glb_path:
            f.write(f"  GLB             : {os.path.basename(glb_path)}\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("VARIABLES EXTRAIDAS - GLOBAL\n")
        f.write("-" * 70 + "\n")
        f.write(f"  RG_GLOBAL       : {rg_global} Å  ({len(all_atoms)} átomos)\n")
        f.write(f"  END_DIST_GLOBAL : {ed_global} Å  (C4' nt1 → C4' nt24, 5'→3')\n\n")

        f.write("-" * 70 + "\n")
        f.write("VARIABLES EXTRAIDAS - POR REGION\n")
        f.write("-" * 70 + "\n")
        f.write(f"\n  {'REGION':<8} {'NT':>8} {'N_NT':>5} {'N_ATOMS':>8} {'RG (Å)':>10} {'END_DIST (Å)':>14}\n")
        f.write(f"  {'-'*58}\n")
        for r in rows_region:
            nt_range = f"{r['NT_START']}-{r['NT_END']}"
            f.write(
                f"  {r['REGION']:<8} {nt_range:>8} {r['N_NT']:>5} "
                f"{r['N_ATOMS']:>8} {str(r['RG']):>10} {str(r['END_DIST']):>14}\n"
            )

        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("VARIABLES EXTRAIDAS - RG POR NUCLEOTIDO\n")
        f.write("-" * 70 + "\n")
        f.write(f"\n  {'NT':<5} {'RES':<5} {'REGION':<8} {'N_ATOMS':>8} {'RG_NT (Å)':>12}\n")
        f.write(f"  {'-'*42}\n")
        for r in rows_nt:
            f.write(
                f"  {r['RESNUM']:<5} {r['RESNAME']:<5} {r['REGION']:<8} "
                f"{r['N_ATOMS']:>8} {str(r['RG_NT']):>12}\n"
            )

        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("PARAMETROS DE IMAGEN\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Resolución      : {IMG_W} x {IMG_H} px\n")
        f.write(f"  Supersample     : {IMG_SUPERSAMPLE} (antialiasing)\n")
        f.write(f"  Fondo           : blanco (#FFFFFF)\n")
        f.write(f"  Iluminación     : simple\n")
        f.write(f"  Silhouettes     : desactivadas\n")
        f.write(f"  Estilo          : tube/slab + cartoons\n")
        f.write(f"  Colores región  : ANCHOR={COLORS['ANCHOR']} | P1={COLORS['P1']} | BULGE={COLORS['BULGE']} | P2={COLORS['P2']}\n")
        f.write(f"  Video           : {VID_FRAMES} frames, {VID_TURN_STEP}°/frame, {VID_FRAMERATE} fps (manual)\n\n")

        f.write("-" * 70 + "\n")
        f.write("NOTAS GENERALES\n")
        f.write("-" * 70 + "\n")
        f.write("  - Rg con masas atómicas reales IUPAC (no pseudoátomos).\n")
        f.write("  - END_DIST C4'→C4' uniforme en todas las regiones.\n")
        f.write("  - END_DIST ANCHOR: N/A (n=1, un solo nucleótido).\n")
        f.write("  - Numeración 5'→3': nt 1 = U (5' seed anchor), nt 24 = U (3' tail).\n")
        f.write("  - Convención inversa a variantes Cyrano (numeradas 3'→5').\n")
        f.write("  - Videos MP4 generados manualmente con comandos ChimeraX.\n")
        f.write("  - Verificar versión ChimeraX en Help > About.\n")
        f.write("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("RG + END_DIST — miR-7-5p")
    print("=" * 70)
    print(f"\nGLOBAL:")
    print(f"  RG_GLOBAL       : {rg_global} Å")
    print(f"  END_DIST_GLOBAL : {ed_global} Å")
    print(f"\n{'REGION':<8} {'NT':>8} {'N_NT':>5} {'N_ATOMS':>8} {'RG(Å)':>10} {'END_DIST(Å)':>13}")
    print("-" * 58)
    for r in rows_region:
        nt_range = f"{r['NT_START']}-{r['NT_END']}"
        print(
            f"{r['REGION']:<8} {nt_range:>8} {r['N_NT']:>5} "
            f"{r['N_ATOMS']:>8} {str(r['RG']):>10} {str(r['END_DIST']):>13}"
        )
    print(f"\n{'NT':<5} {'RES':<5} {'REGION':<8} {'N_ATOMS':>8} {'RG_NT(Å)':>12}")
    print("-" * 42)
    for r in rows_nt:
        print(
            f"{r['RESNUM']:<5} {r['RESNAME']:<5} {r['REGION']:<8} "
            f"{r['N_ATOMS']:>8} {str(r['RG_NT']):>12}"
        )
    print(f"\n✅ CSV global     → {csv_global}")
    print(f"✅ CSV región     → {csv_reg}")
    print(f"✅ CSV por nt     → {csv_nt}")
    print(f"✅ Reporte .txt   → {report_path}")
    print("=" * 70 + "\n")


run(session)
