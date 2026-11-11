# =============================================================================
# DUP_WT_miR7_ChX_rg_enddist.py
# HERRAMIENTA : UCSF ChimeraX — Python API interna
# VARIABLES   : RG, END_DIST
# NIVELES     : Global complejo | Por cadena | Por región | Por nucleótido
# DÚPLEX      : CYR_WT (Chain A, 26 nt) : miR-7-5p (Chain B, 24 nt)
# SECUENCIA A : 5'-AACAACAAAAUCACCAAUGUCUUCCA-3' (26 nt, 5'→3', nt 1=extremo 5')
# SECUENCIA B : 5'-UGGAAGACUAGUGAUUUUGUUGUU-3'  (24 nt, 5'→3', nt 1=extremo 5')
# END_DIST    : C4'(nt1) → C4'(último nt) — por hebra; C4'(A:nt1)→C4'(B:nt24) global
# EJECUCIÓN   : En consola ChimeraX → open "ruta\DUP_WT_miR7_ChX_rg_enddist.py"
# PIPELINE    : OE-4 — Tesis Maestría Bioquímica UNAM
# =============================================================================

import csv, math, os
from datetime import datetime
import numpy as np

MODEL_INDEX = 0
PDB_INPUT   = "WT_miR7.pdb"

OUTPUT_DIR_RG = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-4\6. ChimeraX\1.WT_MIR\RG_END_DIST"

# -----------------------------------------------------------------------------
# REGION MAPS
# Chain A = CYR_WT  (26 nt, 5'→3', nt 1 = extremo 5', nt 26 = extremo 3')
# Chain B = miR-7-5p (24 nt, 5'→3', nt 1 = extremo 5', nt 24 = extremo 3')
# -----------------------------------------------------------------------------

REGION_MAP_A = {}
REGION_MAP_A[26] = "ANCHOR_CYR"
for i in range(19, 26): REGION_MAP_A[i] = "P1_CYR"
for i in range(15, 19): REGION_MAP_A[i] = "B_CYR"
for i in range(1,  15): REGION_MAP_A[i] = "P2_CYR"

REGION_MAP_B = {}
REGION_MAP_B[1] = "ANCHOR_MIR"
for i in range(2,  9):  REGION_MAP_B[i] = "P1_MIR"
for i in range(9,  11): REGION_MAP_B[i] = "B_MIR"
for i in range(11, 25): REGION_MAP_B[i] = "P2_MIR"

# Orden fijo — siempre completo aunque sea NA
REGION_ORDER_A = ["ANCHOR_CYR", "P1_CYR", "B_CYR", "P2_CYR", "OVERHANG_CYR"]
REGION_ORDER_B = ["ANCHOR_MIR", "P1_MIR", "B_MIR", "P2_MIR", "OVERHANG_MIR"]

REGION_RANGES_A = {
    "ANCHOR_CYR":   (26, 26),
    "P1_CYR":       (19, 25),
    "B_CYR":        (15, 18),
    "P2_CYR":       (1,  14),
    "OVERHANG_CYR": None,
}
REGION_RANGES_B = {
    "ANCHOR_MIR":   (1,  1),
    "P1_MIR":       (2,  8),
    "B_MIR":        (9,  10),
    "P2_MIR":       (11, 24),
    "OVERHANG_MIR": None,
}

COLORS = {
    "ANCHOR_CYR": "#E63946",
    "P1_CYR":     "#2196F3",
    "B_CYR":      "#FF9800",
    "P2_CYR":     "#4CAF50",
    "ANCHOR_MIR": "#C62828",
    "P1_MIR":     "#1565C0",
    "B_MIR":      "#E65100",
    "P2_MIR":     "#2E7D32",
}

ATOMIC_MASS = {
    "H":1.008,"C":12.011,"N":14.007,"O":15.999,
    "P":30.974,"S":32.06,"MG":24.305,"ZN":65.38,"FE":55.845
}

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------

def get_mass(atom):
    elem = atom.element.name.upper() if hasattr(atom,"element") else atom.name[0].upper()
    return ATOMIC_MASS.get(elem, 12.0)

def calc_rg(atoms_list):
    if not atoms_list: return None
    coords = np.array([a.coord for a in atoms_list])
    masses = np.array([get_mass(a) for a in atoms_list])
    total_mass = masses.sum()
    if total_mass == 0: return None
    cm = np.average(coords, weights=masses, axis=0)
    dist_sq = np.sum((coords-cm)**2, axis=1)
    return round(math.sqrt(np.dot(masses, dist_sq)/total_mass), 3)

def get_c4p(res):
    for atom in res.atoms:
        if atom.name == "C4'": return atom
    return None

def calc_end_dist(res_first, res_last):
    a1 = get_c4p(res_first); a2 = get_c4p(res_last)
    if a1 is None or a2 is None: return None
    return round(float(np.linalg.norm(np.array(a1.coord)-np.array(a2.coord))), 3)

def cx(session, cmd):
    from chimerax.core.commands import run as cx_run
    cx_run(session, cmd)

def apply_colors(session):
    cx(session, f"color #1/A:26      {COLORS['ANCHOR_CYR']}")
    cx(session, f"color #1/A:19-25   {COLORS['P1_CYR']}")
    cx(session, f"color #1/A:15-18   {COLORS['B_CYR']}")
    cx(session, f"color #1/A:1-14    {COLORS['P2_CYR']}")
    cx(session, f"color #1/B:1       {COLORS['ANCHOR_MIR']}")
    cx(session, f"color #1/B:2-8     {COLORS['P1_MIR']}")
    cx(session, f"color #1/B:9-10    {COLORS['B_MIR']}")
    cx(session, f"color #1/B:11-24   {COLORS['P2_MIR']}")

# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

def run(session):
    os.makedirs(OUTPUT_DIR_RG, exist_ok=True)
    print(f"DEBUG output RG_END_DIST: {OUTPUT_DIR_RG}")

    model    = session.models[MODEL_INDEX]
    today    = datetime.now().strftime("%d/%m/%Y")

    residues_A  = [r for r in model.residues if r.chain_id == "A"]
    residues_B  = [r for r in model.residues if r.chain_id == "B"]
    res_by_numA = {r.number: r for r in residues_A}
    res_by_numB = {r.number: r for r in residues_B}

    all_atoms_A = [a for r in residues_A for a in r.atoms]
    all_atoms_B = [a for r in residues_B for a in r.atoms]
    all_atoms   = all_atoms_A + all_atoms_B

    # Global
    rg_duplex = calc_rg(all_atoms)
    ed_duplex = calc_end_dist(res_by_numA.get(1), res_by_numB.get(24))  # A:nt1 -> B:nt24
    rg_A      = calc_rg(all_atoms_A)
    rg_B      = calc_rg(all_atoms_B)
    ed_A      = calc_end_dist(res_by_numA.get(1),  res_by_numA.get(26)) # C4'(A:nt1)->C4'(A:nt26)
    ed_B      = calc_end_dist(res_by_numB.get(1),  res_by_numB.get(24)) # C4'(B:nt1)->C4'(B:nt24)

    # Por región — orden fijo, OVERHANG siempre presente
    def region_rows_fixed(region_order, region_ranges, res_by_num, chain_id):
        rows = []
        for region in region_order:
            rng = region_ranges[region]
            if rng is None:
                rows.append({
                    "CHAIN": chain_id, "REGION": region,
                    "NT_START": "NA", "NT_END": "NA", "N_NT": "NA",
                    "N_ATOMS": "NA", "RG": "NA", "END_DIST": "NA",
                })
            else:
                start, end = rng
                res_reg   = [res_by_num[i] for i in range(start, end+1) if i in res_by_num]
                atoms_reg = [a for r in res_reg for a in r.atoms]
                rg_reg    = calc_rg(atoms_reg)
                ed_reg    = calc_end_dist(res_by_num[start], res_by_num[end]) \
                            if start != end and start in res_by_num and end in res_by_num \
                            else "NA (n=1)"
                rows.append({
                    "CHAIN":    chain_id,
                    "REGION":   region,
                    "NT_START": start,
                    "NT_END":   end,
                    "N_NT":     end-start+1,
                    "N_ATOMS":  len(atoms_reg),
                    "RG":       rg_reg  if rg_reg  is not None else "NA",
                    "END_DIST": ed_reg,
                })
        return rows

    rows_reg_A = region_rows_fixed(REGION_ORDER_A, REGION_RANGES_A, res_by_numA, "A")
    rows_reg_B = region_rows_fixed(REGION_ORDER_B, REGION_RANGES_B, res_by_numB, "B")
    rows_reg   = rows_reg_A + rows_reg_B

    # Por nucleótido
    def nt_rows(residues, region_map, chain_id):
        rows = []
        for res in residues:
            atoms = list(res.atoms)
            rows.append({
                "CHAIN":   chain_id,
                "RESNUM":  res.number,
                "RESNAME": res.name.strip().upper(),
                "REGION":  region_map.get(res.number, "UNK"),
                "N_ATOMS": len(atoms),
                "RG_NT":   calc_rg(atoms) if atoms else "NA",
            })
        return rows

    rows_nt = nt_rows(residues_A, REGION_MAP_A, "A") + nt_rows(residues_B, REGION_MAP_B, "B")

    # CSV global
    csv_global = os.path.join(OUTPUT_DIR_RG, "DUP_WT_miR7_rg_enddist_global.csv")
    with open(csv_global, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["VARIABLE","CHAIN","VALOR","UNIDAD","REFERENCIA"])
        w.writerow(["RG_GLOBAL_DUPLEX", "A+B", rg_duplex, "A",  f"{len(all_atoms)} atomos"])
        w.writerow(["END_DIST_DUPLEX",  "A+B", ed_duplex,  "A",  "C4'(A:nt1)->C4'(B:nt24)"])
        w.writerow(["RG_GLOBAL_CYR",    "A",   rg_A,       "A",  f"{len(all_atoms_A)} atomos"])
        w.writerow(["END_DIST_CYR",     "A",   ed_A,       "A",  "C4'(A:nt1)->C4'(A:nt26)"])
        w.writerow(["RG_GLOBAL_MIR",    "B",   rg_B,       "A",  f"{len(all_atoms_B)} atomos"])
        w.writerow(["END_DIST_MIR",     "B",   ed_B,       "A",  "C4'(B:nt1)->C4'(B:nt24)"])

    # CSV región
    csv_reg = os.path.join(OUTPUT_DIR_RG, "DUP_WT_miR7_rg_enddist_region.csv")
    with open(csv_reg, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_reg[0].keys())
        w.writeheader(); w.writerows(rows_reg)

    # CSV por nucleótido
    csv_nt = os.path.join(OUTPUT_DIR_RG, "DUP_WT_miR7_rg_enddist_per_nt.csv")
    with open(csv_nt, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_nt[0].keys())
        w.writeheader(); w.writerows(rows_nt)

    # Reporte
    report_path = os.path.join(OUTPUT_DIR_RG, "DUP_WT_miR7_rg_enddist_reporte.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*70+"\n")
        f.write("REPORTE RG + END_DIST — DUPLEX WT_miR7\n")
        f.write("="*70+"\n\n")
        f.write(f"HERRAMIENTA      : UCSF ChimeraX — Python API interna\n")
        f.write(f"VERSION          : verificar en ChimeraX -> Help > About\n")
        f.write(f"FECHA            : {today}\n")
        f.write(f"INPUT            : {PDB_INPUT}\n")
        f.write(f"PARAMETROS\n")
        f.write(f"  RG             : radio de giro masa-pesado (todos los atomos)\n")
        f.write(f"  END_DIST       : distancia euclidiana C4'(nt_inicio)->C4'(nt_fin)\n")
        f.write(f"                   por hebra : C4'(nt1)->C4'(ultimo nt de la hebra)\n")
        f.write(f"                   duplex    : C4'(A:nt1)->C4'(B:nt24)\n\n")
        f.write(f"CADENAS\n")
        f.write(f"  Chain A — CYR_WT  : 5'-AACAACAAAAUCACCAAUGUCUUCCA-3' (26 nt)\n")
        f.write(f"                       numeracion 5'->3' (nt 1 = extremo 5', nt 26 = extremo 3')\n")
        f.write(f"  Chain B — miR-7-5p: 5'-UGGAAGACUAGUGAUUUUGUUGUU-3'  (24 nt)\n")
        f.write(f"                       numeracion 5'->3' (nt 1 = extremo 5', nt 24 = extremo 3')\n\n")
        f.write(f"ELEMENTOS ESTRUCTURALES\n")
        f.write(f"  Chain A (CYR_WT)\n")
        f.write(f"    ANCHOR_CYR   : A:26       (extremo 3')\n")
        f.write(f"    P1_CYR       : A:19-25\n")
        f.write(f"    B_CYR        : A:15-18\n")
        f.write(f"    P2_CYR       : A:1-14     (extremo 5')\n")
        f.write(f"    OVERHANG_CYR : NA\n")
        f.write(f"  Chain B (miR-7-5p)\n")
        f.write(f"    ANCHOR_MIR   : B:1        (extremo 5')\n")
        f.write(f"    P1_MIR       : B:2-8\n")
        f.write(f"    B_MIR        : B:9-10\n")
        f.write(f"    P2_MIR       : B:11-24    (extremo 3')\n")
        f.write(f"    OVERHANG_MIR : NA\n\n")
        f.write("-"*70+"\n")
        f.write("ARCHIVOS DE SALIDA\n")
        f.write("-"*70+"\n")
        f.write(f"  CSV global    : DUP_WT_miR7_rg_enddist_global.csv\n")
        f.write(f"  CSV region    : DUP_WT_miR7_rg_enddist_region.csv\n")
        f.write(f"  CSV por nt    : DUP_WT_miR7_rg_enddist_per_nt.csv\n\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES EXTRAIDAS — GLOBAL\n")
        f.write("-"*70+"\n")
        f.write(f"  {'VARIABLE':<25} {'CHAIN':<6} {'VALOR':>10} UNIDAD  REFERENCIA\n")
        f.write(f"  {'RG_GLOBAL_DUPLEX':<25} {'A+B':<6} {str(rg_duplex):>10} A\n")
        f.write(f"  {'END_DIST_DUPLEX':<25} {'A+B':<6} {str(ed_duplex):>10} A       C4'(A:nt1)->C4'(B:nt24)\n")
        f.write(f"  {'RG_GLOBAL_CYR':<25} {'A':<6} {str(rg_A):>10} A\n")
        f.write(f"  {'END_DIST_CYR':<25} {'A':<6} {str(ed_A):>10} A       C4'(A:nt1)->C4'(A:nt26)\n")
        f.write(f"  {'RG_GLOBAL_MIR':<25} {'B':<6} {str(rg_B):>10} A\n")
        f.write(f"  {'END_DIST_MIR':<25} {'B':<6} {str(ed_B):>10} A       C4'(B:nt1)->C4'(B:nt24)\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES EXTRAIDAS — POR REGION\n")
        f.write("-"*70+"\n")
        f.write(f"  {'CH':<3} {'REGION':<14} {'NT':>8} {'N_NT':>5} {'N_ATOMS':>8} {'RG(A)':>10} {'END_DIST(A)':>13}\n")
        for r in rows_reg:
            nt_range = f"{r['NT_START']}-{r['NT_END']}" if r["NT_START"] != "NA" else "NA"
            f.write(f"  {r['CHAIN']:<3} {r['REGION']:<14} {nt_range:>8} {str(r['N_NT']):>5} "
                    f"{str(r['N_ATOMS']):>8} {str(r['RG']):>10} {str(r['END_DIST']):>13}\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES EXTRAIDAS — POR NUCLEOTIDO\n")
        f.write("-"*70+"\n")
        f.write(f"  {'CH':<3} {'NT':<5} {'RES':<5} {'REGION':<14} {'N_ATOMS':>8} {'RG_NT(A)':>10}\n")
        for r in rows_nt:
            f.write(f"  {r['CHAIN']:<3} {r['RESNUM']:<5} {r['RESNAME']:<5} {r['REGION']:<14} "
                    f"{str(r['N_ATOMS']):>8} {str(r['RG_NT']):>10}\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("NOTAS GENERALES\n")
        f.write("-"*70+"\n")
        f.write("  - RG calculado con pesos de masa atomica (masa-pesado).\n")
        f.write("  - END_DIST: distancia C4'->C4' entre extremos de cada hebra.\n")
        f.write("  - END_DIST duplex: C4'(A:nt1)->C4'(B:nt24) — extremos opuestos del duplex.\n")
        f.write("  - OVERHANG = NA indica ausencia de region no apareada terminal.\n")
        f.write("  - Sin generacion de imagenes — capas coloreadas listas en ChimeraX.\n")
        f.write("  - Para videos: movie record / turn y N frames / movie encode output ruta.\n")
        f.write("  - Verificar VERSION en ChimeraX -> Help > About.\n")
        f.write("="*70+"\n")

    # -------------------------------------------------------------------------
    # CAPA VISUAL — estilo, colores, distancias y elipsoides
    # Sin guardar imágenes — quedan activos para video manual
    # -------------------------------------------------------------------------
    cx(session, "set bgcolor white")
    cx(session, "lighting simple")
    cx(session, "nucleotides #1 tube/slab")
    cx(session, "show #1 cartoons")
    apply_colors(session)
    cx(session, "view #1")

    # -- END_DIST: líneas de distancia C4'→C4' por hebra y global del dúplex --
    # Dúplex global: A:nt1 → B:nt24
    cx(session, "distance #1/A:1@C4' #1/B:24@C4'")
    # CYR_WT: A:nt1 → A:nt26
    cx(session, "distance #1/A:1@C4' #1/A:26@C4'")
    # miR-7-5p: B:nt1 → B:nt24
    cx(session, "distance #1/B:1@C4' #1/B:24@C4'")
    # Por región CYR (solo las que tienen >1 nt)
    cx(session, "distance #1/A:19@C4' #1/A:25@C4'")   # P1_CYR
    cx(session, "distance #1/A:15@C4' #1/A:18@C4'")   # B_CYR
    cx(session, "distance #1/A:1@C4'  #1/A:14@C4'")   # P2_CYR
    # Por región MIR (solo las que tienen >1 nt)
    cx(session, "distance #1/B:2@C4'  #1/B:8@C4'")    # P1_MIR
    cx(session, "distance #1/B:11@C4' #1/B:24@C4'")   # P2_MIR

    # -- RG: elipsoides de inercia por región (medida visual del radio de giro) --
    cx(session, "measure inertia #1/A:19-25 showEllipsoid true")  # P1_CYR
    cx(session, "measure inertia #1/A:15-18 showEllipsoid true")  # B_CYR
    cx(session, "measure inertia #1/A:1-14  showEllipsoid true")  # P2_CYR
    cx(session, "measure inertia #1/B:2-8   showEllipsoid true")  # P1_MIR
    cx(session, "measure inertia #1/B:9-10  showEllipsoid true")  # B_MIR
    cx(session, "measure inertia #1/B:11-24 showEllipsoid true")  # P2_MIR

    cx(session, "view #1")

    # -------------------------------------------------------------------------
    # INSTRUCCIONES PARA VIDEO MANUAL
    # Ejecutar en consola ChimeraX después de correr este script:
    #
    # -- Video END_DIST + RG (vista completa con todas las capas):
    #    movie record
    #    turn y 3 120
    #    movie encode output "C:\...\OE-4\6. ChimeraX\1.WT_MIR\RG_END_DIST\DUP_WT_miR7_rg_enddist_todas.mp4" format h264 framerate 15
    #
    # -- Para video solo de una región (ej. B_CYR):
    #    view #1/A:15-18 pad 0.4
    #    movie record
    #    turn y 3 120
    #    movie encode output "C:\...\DUP_WT_miR7_rg_enddist_B_CYR.mp4" format h264 framerate 15
    # -------------------------------------------------------------------------

    print("\n"+"="*70)
    print("RG + END_DIST — DUPLEX WT_miR7")
    print("="*70)
    print(f"\n  GLOBAL DUPLEX : RG={rg_duplex} A  |  END_DIST={ed_duplex} A  [C4'(A:1)->C4'(B:24)]")
    print(f"  GLOBAL CYR_WT : RG={rg_A} A  |  END_DIST={ed_A} A  [C4'(A:1)->C4'(A:26)]")
    print(f"  GLOBAL miR-7  : RG={rg_B} A  |  END_DIST={ed_B} A  [C4'(B:1)->C4'(B:24)]")
    print(f"\n  {'CH':<3} {'REGION':<14} {'NT':>8} {'N_NT':>5} {'N_ATOMS':>8} {'RG(A)':>10} {'END_DIST(A)':>13}")
    for r in rows_reg:
        nt_range = f"{r['NT_START']}-{r['NT_END']}" if r["NT_START"] != "NA" else "NA"
        print(f"  {r['CHAIN']:<3} {r['REGION']:<14} {nt_range:>8} {str(r['N_NT']):>5} "
              f"{str(r['N_ATOMS']):>8} {str(r['RG']):>10} {str(r['END_DIST']):>13}")
    print(f"\n  CSV global    -> {csv_global}")
    print(f"  CSV region    -> {csv_reg}")
    print(f"  CSV por nt    -> {csv_nt}")
    print(f"  Reporte .txt  -> {report_path}")
    print("\n  CAPAS ACTIVAS EN CHIMERA:")
    print("    - Colores por region (A y B)")
    print("    - Lineas END_DIST: duplex global + por hebra + por region")
    print("    - Elipsoides RG: P1_CYR, B_CYR, P2_CYR, P1_MIR, B_MIR, P2_MIR")
    print("\n  PARA VIDEO MANUAL:")
    print("    movie record")
    print("    turn y 3 120")
    print('    movie encode output "ruta\nombre.mp4" format h264 framerate 15')
    print("="*70+"\n")

run(session)
