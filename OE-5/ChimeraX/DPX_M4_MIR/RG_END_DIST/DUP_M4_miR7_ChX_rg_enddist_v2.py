# =====================================================================
# DUP_M4_miR7_ChX_rg_enddist_v2.py
# HERRAMIENTA : UCSF ChimeraX — Python API interna
# DÚPLEX      : M4 (Chain A, 20 nt) : miR-7-5p (Chain B, 24 nt)
# SEQ A       : 5'-AACAACAAAAUCAUCUUCCA-3'
# SEQ B       : 5'-UGGAAGACUAGUGAUUUUGUUGUU-3'
# PIPELINE    : OE-5 — Tesis Maestría Bioquímica UNAM
# =====================================================================

import csv, math, os
from datetime import datetime
import numpy as np

# ---------------------------------------------------------------------
# PARÁMETROS GLOBALES
# ---------------------------------------------------------------------
MODEL_INDEX  = 0
PDB_INPUT    = "M4_miR7.pdb"
DUPLEX_NAME  = "M4_miR7"
VARIANT_ID   = "M4"
CHAIN_A_LEN  = 20
CHAIN_B_LEN  = 24
DUPLEX_END_B = 24
OUTPUT_DIR   = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-5\ChimeraX\5.Dup_M4_MIR"

# ---------------------------------------------------------------------
# REGIONES — Chain A (M4)  ANCHOR_CYR=(20, 20) | P1_CYR=(14, 19) | BULGE_CYR=NA | P2_CYR=(1, 13) | OVERHANG_CYR=NA
# ---------------------------------------------------------------------
REGION_MAP_A = {}
REGION_MAP_A[20] = "ANCHOR_CYR"
for i in range(14, 20): REGION_MAP_A[i] = "P1_CYR"
for i in range(1, 14): REGION_MAP_A[i] = "P2_CYR"

REGION_RANGES_A = {
    "ANCHOR_CYR":   (20, 20),
    "P1_CYR":   (14, 19),
    "BULGE_CYR":   None,
    "P2_CYR":   (1, 13),
    "OVERHANG_CYR":   None,
}
REGION_ORDER_A = ["ANCHOR_CYR","P1_CYR","BULGE_CYR","P2_CYR","OVERHANG_CYR"]

# ---------------------------------------------------------------------
# REGIONES — Chain B (miR-7-5p)  ANCHOR_MIR=(1, 1) | P1_MIR=(2, 7) | BULGE_MIR=(8, 11) | P2_MIR=(12, 24) | OVERHANG_MIR=NA
# ---------------------------------------------------------------------
REGION_MAP_B = {}
REGION_MAP_B[1] = "ANCHOR_MIR"
for i in range(2, 8): REGION_MAP_B[i] = "P1_MIR"
for i in range(8, 12): REGION_MAP_B[i] = "BULGE_MIR"
for i in range(12, 25): REGION_MAP_B[i] = "P2_MIR"

REGION_RANGES_B = {
    "ANCHOR_MIR":   (1, 1),
    "P1_MIR":   (2, 7),
    "BULGE_MIR":   (8, 11),
    "P2_MIR":   (12, 24),
    "OVERHANG_MIR":   None,
}
REGION_ORDER_B = ["ANCHOR_MIR","P1_MIR","BULGE_MIR","P2_MIR","OVERHANG_MIR"]

REGION_PAIRS = [
    ("ANCHOR","ANCHOR_CYR","ANCHOR_MIR"),
    ("P1","P1_CYR","P1_MIR"),
    ("BULGE","BULGE_CYR","BULGE_MIR"),
    ("P2","P2_CYR","P2_MIR"),
    ("OVERHANG","OVERHANG_CYR","OVERHANG_MIR"),
]

# END_DIST_REGION_INTER: None = NA metodológico
END_DIST_REGION_INTER_DEF = {
    "ANCHOR":   None,
    "P1":       (14, 7),
    "BULGE":    None,
    "P2":       (1, 24),
    "OVERHANG": None,
}

COLORS = {
    "ANCHOR_CYR":   "#E63946",
    "P1_CYR":       "#2196F3",
    "BULGE_CYR":    "#FF9800",
    "P2_CYR":       "#4CAF50",
    "OVERHANG_CYR": "#9C27B0",
    "ANCHOR_MIR":   "#C62828",
    "P1_MIR":       "#1565C0",
    "BULGE_MIR":    "#E65100",
    "P2_MIR":       "#2E7D32",
    "OVERHANG_MIR": "#6A1B9A",
}

ATOMIC_MASS = {
    "H":1.008,"C":12.011,"N":14.007,"O":15.999,
    "P":30.974,"S":32.06,"MG":24.305,"ZN":65.38,"FE":55.845
}

# =====================================================================
# FUNCIONES AUXILIARES
# =====================================================================

def get_mass(atom):
    elem = atom.element.name.upper() if hasattr(atom, "element") else atom.name[0].upper()
    return ATOMIC_MASS.get(elem, 12.0)

def calc_rg(atoms_list):
    if not atoms_list: return None
    coords = np.array([a.coord for a in atoms_list])
    masses = np.array([get_mass(a) for a in atoms_list])
    total  = masses.sum()
    if total == 0: return None
    cm     = np.average(coords, weights=masses, axis=0)
    return round(math.sqrt(np.dot(masses, np.sum((coords-cm)**2,axis=1)) / total), 3)

def get_c4p(res):
    for atom in res.atoms:
        if atom.name == "C4'": return atom
    return None

def calc_end_dist(res_a, res_b):
    if res_a is None or res_b is None: return None
    a1 = get_c4p(res_a); a2 = get_c4p(res_b)
    if a1 is None or a2 is None: return None
    return round(float(np.linalg.norm(np.array(a1.coord)-np.array(a2.coord))), 3)

def fmt(val): return val if val is not None else "NA"

def cx(session, cmd):
    from chimerax.core.commands import run as cx_run
    cx_run(session, cmd)
def safe_distance(session, sel_a, sel_b, label=""):
    """
    Traza una línea de distancia entre sel_a y sel_b.
    Si ya existe esa distancia en ChimeraX, la omite silenciosamente.
    """
    from chimerax.core.errors import UserError
    from chimerax.core.commands import run as cx_run
    cmd = f"distance {sel_a} {sel_b}"
    try:
        cx_run(session, cmd)
    except UserError as e:
        if "Distance already exists" in str(e):
            print(f"  [INFO] Distancia ya existe (omitida): {sel_a} -- {sel_b}"
                  + (f"  [{label}]" if label else ""))
        else:
            raise



# =====================================================================
# CAPA VISUAL
# =====================================================================
def apply_colors(session):
    cx(session, "color #1/A:20  " + COLORS['ANCHOR_CYR'])
    cx(session, "color #1/A:14-19  " + COLORS['P1_CYR'])
    cx(session, "color #1/A:1-13  " + COLORS['P2_CYR'])
    cx(session, "color #1/B:1  " + COLORS['ANCHOR_MIR'])
    cx(session, "color #1/B:2-7  " + COLORS['P1_MIR'])
    cx(session, "color #1/B:8-11  " + COLORS['BULGE_MIR'])
    cx(session, "color #1/B:12-24  " + COLORS['P2_MIR'])

def inertia_named(session, selection, label):
    ids_before = {m.id[0] for m in session.models if len(m.id)==1}
    cx(session, f"measure inertia {selection} showEllipsoid true")
    ids_after  = {m.id[0] for m in session.models if len(m.id)==1}
    new_ids    = ids_after - ids_before
    if new_ids:
        new_id = max(new_ids)
        cx(session, f'rename #{new_id} id #{new_id} name "{label}"')

def add_distance_layers(session):
    # Capa 1: END_DIST globales
    safe_distance(session, "#1/A:1@C4'", "#1/A:20@C4'", "1/A:1@C4'  #1/A:20@C4'")   # END_DIST_CYR")
    safe_distance(session, "#1/B:1@C4'", "#1/B:24@C4'", "1/B:1@C4'  #1/B:24@C4'")   # END_DIST_MIR")
    safe_distance(session, "#1/A:1@C4'", "#1/B:24@C4'", "1/A:1@C4'  #1/B:24@C4'")   # END_DIST_INTERHEBRA")
    # END_DIST_DUPLEX coincide con INTERHEBRA (sin overhang) — documentado en reporte
    # Capa 2: END_DIST regionales interhebra
    safe_distance(session, "#1/A:14@C4'", "#1/B:7@C4'", "1/A:14@C4' #1/B:7@C4'")   # END_DIST_REGION_INTER_P1")
    # END_DIST_REGION_INTER_BULGE: NA
    safe_distance(session, "#1/A:1@C4'", "#1/B:24@C4'", "1/A:1@C4' #1/B:24@C4'")   # END_DIST_REGION_INTER_P2")
    # Capa 3: END_DIST regionales intra-hebra
    safe_distance(session, "#1/A:14@C4'", "#1/A:19@C4'", "1/A:14@C4' #1/A:19@C4'")   # P1_CYR")
    # END_DIST_REGION_BULGE_CYR: NA
    safe_distance(session, "#1/A:1@C4'", "#1/A:13@C4'", "1/A:1@C4' #1/A:13@C4'")   # P2_CYR")
    # END_DIST_REGION_OVERHANG_CYR: NA
    safe_distance(session, "#1/B:2@C4'", "#1/B:7@C4'", "1/B:2@C4' #1/B:7@C4'")   # P1_MIR")
    safe_distance(session, "#1/B:8@C4'", "#1/B:11@C4'", "1/B:8@C4' #1/B:11@C4'")   # BULGE_MIR")
    safe_distance(session, "#1/B:12@C4'", "#1/B:24@C4'", "1/B:12@C4' #1/B:24@C4'")   # P2_MIR")
    # END_DIST_REGION_OVERHANG_MIR: NA
    # Capa 4a: elipsoides RG intra-hebra
    inertia_named(session, "#1/A:14-19", "RG_HEBRA_A_P1")
    # RG_HEBRA_A_BULGE: NA
    inertia_named(session, "#1/A:1-13", "RG_HEBRA_A_P2")
    # RG_HEBRA_A_OVERHANG: NA
    inertia_named(session, "#1/B:2-7", "RG_HEBRA_B_P1")
    inertia_named(session, "#1/B:8-11", "RG_HEBRA_B_BULGE")
    inertia_named(session, "#1/B:12-24", "RG_HEBRA_B_P2")
    # RG_HEBRA_B_OVERHANG: NA
    # Capa 4b: elipsoides RG interhebra
    inertia_named(session, "#1/A:20 #1/B:1", "RG_INTERHEBRA_ANCHOR")
    inertia_named(session, "#1/A:14-19 #1/B:2-7", "RG_INTERHEBRA_P1")
    inertia_named(session, "#1/B:8-11", "RG_INTERHEBRA_BULGE")   # solo hebra B
    inertia_named(session, "#1/A:1-13 #1/B:12-24", "RG_INTERHEBRA_P2")
    # RG_INTERHEBRA_OVERHANG: NA (no existe en ninguna hebra)

# =====================================================================
# FUNCIÓN PRINCIPAL
# =====================================================================
def run(session):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model       = session.models[MODEL_INDEX]
    today       = datetime.now().strftime("%d/%m/%Y")

    residues_A  = sorted([r for r in model.residues if r.chain_id=="A"], key=lambda r:r.number)
    residues_B  = sorted([r for r in model.residues if r.chain_id=="B"], key=lambda r:r.number)
    res_by_numA = {r.number:r for r in residues_A}
    res_by_numB = {r.number:r for r in residues_B}
    all_atoms_A = [a for r in residues_A for a in r.atoms]
    all_atoms_B = [a for r in residues_B for a in r.atoms]
    all_atoms   = all_atoms_A + all_atoms_B

    # RG global
    rg_duplex = calc_rg(all_atoms)
    rg_A      = calc_rg(all_atoms_A)
    rg_B      = calc_rg(all_atoms_B)

    # END_DIST global
    ed_cyr        = calc_end_dist(res_by_numA.get(1), res_by_numA.get(CHAIN_A_LEN))
    ed_mir        = calc_end_dist(res_by_numB.get(1), res_by_numB.get(CHAIN_B_LEN))
    ed_interhebra = calc_end_dist(res_by_numA.get(1), res_by_numB.get(CHAIN_B_LEN))
    ed_duplex     = calc_end_dist(res_by_numA.get(1), res_by_numB.get(DUPLEX_END_B))

    def calc_region_rg(region_ranges, res_by_num):
        result = {}
        for region, rng in region_ranges.items():
            if rng is None:
                result[region] = (None, 0, 0)
            else:
                s, e      = rng
                res_reg   = [res_by_num[i] for i in range(s, e+1) if i in res_by_num]
                atoms_reg = [a for r in res_reg for a in r.atoms]
                result[region] = (calc_rg(atoms_reg), e-s+1, len(atoms_reg))
        return result

    rg_reg_A = calc_region_rg(REGION_RANGES_A, res_by_numA)
    rg_reg_B = calc_region_rg(REGION_RANGES_B, res_by_numB)

    rg_reg_inter = {}
    for label, reg_a, reg_b in REGION_PAIRS:
        rng_a = REGION_RANGES_A.get(reg_a)
        rng_b = REGION_RANGES_B.get(reg_b)
        atoms_combined = []; n_nt = 0
        if rng_a is not None:
            for i in range(rng_a[0], rng_a[1]+1):
                if i in res_by_numA: atoms_combined += list(res_by_numA[i].atoms); n_nt += 1
        if rng_b is not None:
            for i in range(rng_b[0], rng_b[1]+1):
                if i in res_by_numB: atoms_combined += list(res_by_numB[i].atoms); n_nt += 1
        rg_reg_inter[label] = (calc_rg(atoms_combined), n_nt, len(atoms_combined))

    def calc_region_end_dist(region_ranges, res_by_num):
        result = {}
        for region, rng in region_ranges.items():
            if rng is None: result[region] = None
            elif rng[0] == rng[1]: result[region] = "NA (n=1)"
            elif rng[0] in res_by_num and rng[1] in res_by_num:
                result[region] = calc_end_dist(res_by_num[rng[0]], res_by_num[rng[1]])
            else: result[region] = None
        return result

    ed_reg_A = calc_region_end_dist(REGION_RANGES_A, res_by_numA)
    ed_reg_B = calc_region_end_dist(REGION_RANGES_B, res_by_numB)

    ed_reg_inter = {}
    for label, def_pair in END_DIST_REGION_INTER_DEF.items():
        if def_pair is None: ed_reg_inter[label] = None
        else:
            a_s, b_e = def_pair
            ed_reg_inter[label] = calc_end_dist(res_by_numA.get(a_s), res_by_numB.get(b_e))

    def calc_nt_rg(residues, region_map, chain_id):
        rows = []
        for res in residues:
            atoms = list(res.atoms)
            rows.append({"CHAIN":chain_id,"RESNUM":res.number,"RESNAME":res.name.strip().upper(),
                         "REGION":region_map.get(res.number,"UNK"),"N_ATOMS":len(atoms),
                         "RG_NT":fmt(calc_rg(atoms)) if atoms else "NA"})
        return rows

    rows_nt = calc_nt_rg(residues_A, REGION_MAP_A, "A") + calc_nt_rg(residues_B, REGION_MAP_B, "B")

    # ---- CSV GLOBAL ----
    csv_global = os.path.join(OUTPUT_DIR, f"DUP_{DUPLEX_NAME}_rg_enddist_global.csv")
    with open(csv_global, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["VARIABLE","CHAIN","VALOR","UNIDAD","N_ATOMS","REFERENCIA"])
        w.writerow(["RG_GLOBAL_DUPLEX","A+B",fmt(rg_duplex),"A",len(all_atoms),"todos los atomos A+B"])
        w.writerow(["RG_GLOBAL_CYR","A",fmt(rg_A),"A",len(all_atoms_A),"todos los atomos Chain A"])
        w.writerow(["RG_GLOBAL_MIR","B",fmt(rg_B),"A",len(all_atoms_B),"todos los atomos Chain B"])
        w.writerow(["END_DIST_CYR","A",fmt(ed_cyr),"A","NA",f"C4'(A:nt1)->C4'(A:nt{CHAIN_A_LEN})"])
        w.writerow(["END_DIST_MIR","B",fmt(ed_mir),"A","NA",f"C4'(B:nt1)->C4'(B:nt{CHAIN_B_LEN})"])
        w.writerow(["END_DIST_INTERHEBRA","A+B",fmt(ed_interhebra),"A","NA",f"C4'(A:nt1)->C4'(B:nt{CHAIN_B_LEN})"])
        w.writerow(["END_DIST_DUPLEX","A+B",fmt(ed_duplex),"A","NA",
                    f"C4'(A:nt1)->C4'(B:nt{DUPLEX_END_B}) [fin duplex sin overhang]"])

    # ---- CSV REGIONAL ----
    csv_reg = os.path.join(OUTPUT_DIR, f"DUP_{DUPLEX_NAME}_rg_enddist_region.csv")
    with open(csv_reg, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["NIVEL","REGION","CHAIN","NT_START_A","NT_END_A","NT_START_B","NT_END_B",
                    "N_NT","N_ATOMS","RG","UNIDAD_RG","END_DIST","UNIDAD_ED","END_DIST_REF"])
        for label, reg_a, reg_b in REGION_PAIRS:
            rng_a = REGION_RANGES_A.get(reg_a)
            rng_b = REGION_RANGES_B.get(reg_b)
            rg_val, n_nt, n_atoms = rg_reg_inter[label]
            ed_val = ed_reg_inter.get(label)
            inter_def = END_DIST_REGION_INTER_DEF.get(label)
            a_s = rng_a[0] if rng_a else "NA"; a_e = rng_a[1] if rng_a else "NA"
            b_s = rng_b[0] if rng_b else "NA"; b_e = rng_b[1] if rng_b else "NA"
            if inter_def:
                ed_str = str(fmt(ed_val))
                ref    = f"C4'(A:nt{inter_def[0]})->C4'(B:nt{inter_def[1]})"
                ud     = "A"
            elif label == "ANCHOR": ed_str="NA (n=1 ambas)"; ref="NA"; ud="NA"
            elif label == "OVERHANG": ed_str="NA"; ref="NA"; ud="NA"
            else: ed_str="NA (region ausente en una hebra)"; ref="NA"; ud="NA"
            w.writerow(["INTERHEBRA",label,"A+B",a_s,a_e,b_s,b_e,n_nt,n_atoms,fmt(rg_val),"A",ed_str,ud,ref])
        for region in REGION_ORDER_A:
            label = region.replace("_CYR","")
            rng   = REGION_RANGES_A.get(region)
            rg_val, n_nt, n_atoms = rg_reg_A[region]
            ed_val = ed_reg_A[region]
            a_s = rng[0] if rng else "NA"; a_e = rng[1] if rng else "NA"
            if rng is None: ed_str="NA"; ref="NA"; ud="NA"
            elif ed_val=="NA (n=1)": ed_str="NA (n=1)"; ref=f"C4'(A:nt{a_s}) — un solo nt"; ud="NA"
            else: ed_str=str(fmt(ed_val)); ref=f"C4'(A:nt{a_s})->C4'(A:nt{a_e})"; ud="A"
            w.writerow(["REGIONAL_CYR",label,"A",a_s,a_e,"NA","NA",n_nt,n_atoms,fmt(rg_val),"A",ed_str,ud,ref])
        for region in REGION_ORDER_B:
            label = region.replace("_MIR","")
            rng   = REGION_RANGES_B.get(region)
            rg_val, n_nt, n_atoms = rg_reg_B[region]
            ed_val = ed_reg_B[region]
            b_s = rng[0] if rng else "NA"; b_e = rng[1] if rng else "NA"
            if rng is None: ed_str="NA"; ref="NA"; ud="NA"
            elif ed_val=="NA (n=1)": ed_str="NA (n=1)"; ref=f"C4'(B:nt{b_s}) — un solo nt"; ud="NA"
            else: ed_str=str(fmt(ed_val)); ref=f"C4'(B:nt{b_s})->C4'(B:nt{b_e})"; ud="A"
            w.writerow(["REGIONAL_MIR",label,"B","NA","NA",b_s,b_e,n_nt,n_atoms,fmt(rg_val),"A",ed_str,ud,ref])

    # ---- CSV POR NT ----
    csv_nt = os.path.join(OUTPUT_DIR, f"DUP_{DUPLEX_NAME}_rg_enddist_per_nt.csv")
    with open(csv_nt, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["CHAIN","RESNUM","RESNAME","REGION","N_ATOMS","RG_NT"])
        w.writeheader(); w.writerows(rows_nt)

    # ---- REPORTE TXT ----
    report_path = os.path.join(OUTPUT_DIR, f"DUP_{DUPLEX_NAME}_rg_enddist_reporte.txt")
    SEP="="*72; SEP2="-"*72
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(SEP+"\n")
        f.write(f"REPORTE RG + END_DIST — DUPLEX {DUPLEX_NAME}\n")
        f.write(SEP+"\n\n")
        f.write(f"HERRAMIENTA : UCSF ChimeraX — Python API interna\n")
        f.write(f"FECHA       : {today}\n")
        f.write(f"INPUT       : {PDB_INPUT}\n")
        f.write(f"VARIANTE    : {VARIANT_ID}\n\n")
        f.write("REGIONES\n")
        for reg,rng in list(REGION_RANGES_A.items())+list(REGION_RANGES_B.items()):
            f.write(f"  {reg:<16}: {str(rng)}\n")
        f.write("\nNOTA BULGE INTERHEBRA\n")
        f.write("  END_DIST_REGION_INTER[BULGE]=NA cuando BULGE no existe en ambas hebras.\n")
        f.write("  RG_INTERHEBRA[BULGE] se calcula con atomos de la(s) hebra(s) donde existe.\n\n")
        f.write(SEP2+"\nVARIABLES GLOBALES\n"+SEP2+"\n")
        f.write(f"  {'VARIABLE':<25} {'CHAIN':<5} {'VALOR':>10}  REFERENCIA\n")
        for row in [
            ("RG_GLOBAL_DUPLEX","A+B",fmt(rg_duplex),f"{len(all_atoms)} atomos"),
            ("RG_GLOBAL_CYR","A",fmt(rg_A),f"{len(all_atoms_A)} atomos"),
            ("RG_GLOBAL_MIR","B",fmt(rg_B),f"{len(all_atoms_B)} atomos"),
            ("END_DIST_CYR","A",fmt(ed_cyr),f"C4'(A:nt1)->C4'(A:nt{CHAIN_A_LEN})"),
            ("END_DIST_MIR","B",fmt(ed_mir),f"C4'(B:nt1)->C4'(B:nt{CHAIN_B_LEN})"),
            ("END_DIST_INTERHEBRA","A+B",fmt(ed_interhebra),f"C4'(A:nt1)->C4'(B:nt{CHAIN_B_LEN})"),
            ("END_DIST_DUPLEX","A+B",fmt(ed_duplex),f"C4'(A:nt1)->C4'(B:nt{DUPLEX_END_B})"),
        ]:
            f.write(f"  {row[0]:<25} {row[1]:<5} {str(row[2]):>10}  {row[3]}\n")
        f.write("\n"+SEP2+"\nREGIONALES INTERHEBRA\n"+SEP2+"\n")
        f.write(f"  {'REGION':<10} {'N_NT':>5} {'N_AT':>6} {'RG(A)':>10} {'END_DIST':>28}\n")
        for label, reg_a, reg_b in REGION_PAIRS:
            rg_val, n_nt, n_atoms = rg_reg_inter[label]
            ed_val = ed_reg_inter.get(label)
            inter_def = END_DIST_REGION_INTER_DEF.get(label)
            if inter_def: ed_str=str(fmt(ed_val))
            elif label=="ANCHOR": ed_str="NA (n=1 ambas)"
            elif label=="OVERHANG": ed_str="NA"
            else: ed_str="NA (region ausente en una hebra)"
            f.write(f"  {label:<10} {str(n_nt):>5} {str(n_atoms):>6} {str(fmt(rg_val)):>10} {ed_str:>28}\n")
        f.write("\n"+SEP2+"\nREGIONALES POR HEBRA\n"+SEP2+"\n")
        f.write(f"  {'CH':<3} {'REGION':<12} {'NT':>8} {'RG(A)':>10} {'END_DIST':>13}\n")
        for region in REGION_ORDER_A:
            label=region.replace("_CYR",""); rng=REGION_RANGES_A.get(region)
            rg_val,n_nt,n_atoms=rg_reg_A[region]; ed_val=ed_reg_A[region]
            nt_str=f"{rng[0]}-{rng[1]}" if rng else "NA"
            ed_str="NA" if rng is None else ("NA (n=1)" if ed_val=="NA (n=1)" else str(fmt(ed_val)))
            f.write(f"  A   {label:<12} {nt_str:>8} {str(fmt(rg_val)):>10} {ed_str:>13}\n")
        for region in REGION_ORDER_B:
            label=region.replace("_MIR",""); rng=REGION_RANGES_B.get(region)
            rg_val,n_nt,n_atoms=rg_reg_B[region]; ed_val=ed_reg_B[region]
            nt_str=f"{rng[0]}-{rng[1]}" if rng else "NA"
            ed_str="NA" if rng is None else ("NA (n=1)" if ed_val=="NA (n=1)" else str(fmt(ed_val)))
            f.write(f"  B   {label:<12} {nt_str:>8} {str(fmt(rg_val)):>10} {ed_str:>13}\n")
        f.write("\n"+SEP2+"\nPOR NUCLEOTIDO\n"+SEP2+"\n")
        f.write(f"  {'CH':<3} {'NT':<5} {'RES':<5} {'REGION':<16} {'N_AT':>6} {'RG_NT(A)':>10}\n")
        for r in rows_nt:
            f.write(f"  {r['CHAIN']:<3} {r['RESNUM']:<5} {r['RESNAME']:<5} "
                    f"{r['REGION']:<16} {str(r['N_ATOMS']):>6} {str(r['RG_NT']):>10}\n")
        f.write("\n"+SEP+"\n")

    # ---- CAPA VISUAL ----
    cx(session, "set bgcolor white")
    cx(session, "lighting simple")
    cx(session, "nucleotides #1 tube/slab")
    cx(session, "show #1 cartoons")
    apply_colors(session)
    cx(session, "view #1")
    add_distance_layers(session)
    cx(session, "view #1")

    # ---- CONSOLA ----
    SEP_P="="*72
    print("\n"+SEP_P)
    print(f"RG + END_DIST — DUPLEX {DUPLEX_NAME}")
    print(SEP_P)
    for row in [
        ("RG_GLOBAL_DUPLEX","A+B",fmt(rg_duplex)),
        ("RG_GLOBAL_CYR","A",fmt(rg_A)),
        ("RG_GLOBAL_MIR","B",fmt(rg_B)),
        ("END_DIST_CYR","A",fmt(ed_cyr)),
        ("END_DIST_MIR","B",fmt(ed_mir)),
        ("END_DIST_INTERHEBRA","A+B",fmt(ed_interhebra)),
        ("END_DIST_DUPLEX","A+B",fmt(ed_duplex)),
    ]:
        print(f"  {row[0]:<25} {row[1]:<5} {str(row[2]):>10} A")
    print(f"  OUTPUTS: {OUTPUT_DIR}")
    print(SEP_P+"\n")

run(session)
