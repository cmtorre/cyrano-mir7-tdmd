# =============================================================================
# DUP_WT_miR7_ChX_rg_enddist_v2.py
# HERRAMIENTA : UCSF ChimeraX — Python API interna
# VARIABLES   : RG, END_DIST
# NIVELES RG  :
#   - RG_GLOBAL_DUPLEX       : todos los átomos Chain A + Chain B
#   - RG_GLOBAL_CYR          : todos los átomos Chain A
#   - RG_GLOBAL_MIR          : todos los átomos Chain B
#   - RG_REGIONAL_DUPLEX     : por región interhebra (A+B combinados)
#   - RG_REGIONAL_CYR        : por región Chain A
#   - RG_REGIONAL_MIR        : por región Chain B
#   - RG_NT                  : por nucleótido (todos los átomos del residuo)
# NIVELES END_DIST:
#   - END_DIST_CYR            : C4'(A:nt1)  -> C4'(A:nt26)
#   - END_DIST_MIR            : C4'(B:nt1)  -> C4'(B:nt24)
#   - END_DIST_INTERHEBRA     : C4'(A:nt1)  -> C4'(B:nt24)
#   - END_DIST_DUPLEX         : C4'(A:nt1)  -> C4'(B:nt_duplex_end)  [sin overhang]
#   - END_DIST_REGION_INTER   : C4'(A:nt_start_region) -> C4'(B:nt_end_region)
#   - END_DIST_REGION_CYR     : C4'(A:nt_start) -> C4'(A:nt_end) por región
#   - END_DIST_REGION_MIR     : C4'(B:nt_start) -> C4'(B:nt_end) por región
# DÚPLEX      : CYR_WT (Chain A, 26 nt) : miR-7-5p (Chain B, 24 nt)
# SECUENCIA A : 5'-AACAACAAAAUCACCAAUGUCUUCCA-3'  (nt1=5', nt26=3')
# SECUENCIA B : 5'-UGGAAGACUAGUGAUUUUGUUGUU-3'    (nt1=5', nt24=3')
# PIPELINE    : OE-5 — Tesis Maestría Bioquímica UNAM
# EJECUCIÓN   : En consola ChimeraX → open "ruta\DUP_WT_miR7_ChX_rg_enddist_v2.py"
# =============================================================================

import csv, math, os
from datetime import datetime
import numpy as np

# -----------------------------------------------------------------------------
# PARÁMETROS GLOBALES
# -----------------------------------------------------------------------------
MODEL_INDEX  = 0
PDB_INPUT    = "WT_miR7.pdb"
DUPLEX_NAME  = "WT_miR7"
CHAIN_A_LEN  = 26   # nt totales Chain A
CHAIN_B_LEN  = 24   # nt totales Chain B

# nt final del dúplex en cada hebra (antes de overhang)
# WT no tiene overhang → coincide con el extremo de la cadena
DUPLEX_END_A = 26   # A:nt usado como inicio: nt1
DUPLEX_END_B = 24   # B:nt usado como fin en END_DIST_DUPLEX

OUTPUT_DIR = (
    r"C:\Users\cmart\OneDrive\Documentos OneDrive"
    r"\1. circRNA_TESIS\Tesis\7.Metodo\OE-5\ChimeraX\1.Dup_WT_MIR"
)

# -----------------------------------------------------------------------------
# MAPAS DE REGIÓN — Chain A (CYR_WT)
# ANCHOR=nt26 | P1=19-25 | BULGE=15-18 | P2=1-14 | OVERHANG=NA
# -----------------------------------------------------------------------------
REGION_MAP_A = {}
REGION_MAP_A[26] = "ANCHOR_CYR"
for i in range(19, 26): REGION_MAP_A[i] = "P1_CYR"
for i in range(15, 19): REGION_MAP_A[i] = "BULGE_CYR"
for i in range(1,  15): REGION_MAP_A[i] = "P2_CYR"

REGION_RANGES_A = {
    "ANCHOR_CYR":   (26, 26),
    "P1_CYR":       (19, 25),
    "BULGE_CYR":    (15, 18),
    "P2_CYR":       (1,  14),
    "OVERHANG_CYR": None,
}
REGION_ORDER_A = ["ANCHOR_CYR", "P1_CYR", "BULGE_CYR", "P2_CYR", "OVERHANG_CYR"]

# -----------------------------------------------------------------------------
# MAPAS DE REGIÓN — Chain B (miR-7-5p)
# ANCHOR=nt1 | P1=2-8 | BULGE=9-10 | P2=11-24 | OVERHANG=NA
# -----------------------------------------------------------------------------
REGION_MAP_B = {}
REGION_MAP_B[1] = "ANCHOR_MIR"
for i in range(2,  9):  REGION_MAP_B[i] = "P1_MIR"
for i in range(9,  11): REGION_MAP_B[i] = "BULGE_MIR"
for i in range(11, 25): REGION_MAP_B[i] = "P2_MIR"

REGION_RANGES_B = {
    "ANCHOR_MIR":   (1,  1),
    "P1_MIR":       (2,  8),
    "BULGE_MIR":    (9,  10),
    "P2_MIR":       (11, 24),
    "OVERHANG_MIR": None,
}
REGION_ORDER_B = ["ANCHOR_MIR", "P1_MIR", "BULGE_MIR", "P2_MIR", "OVERHANG_MIR"]

# Regiones canónicas para nivel interhebra (pares A↔B)
REGION_PAIRS = [
    ("ANCHOR",   "ANCHOR_CYR",   "ANCHOR_MIR"),
    ("P1",       "P1_CYR",       "P1_MIR"),
    ("BULGE",    "BULGE_CYR",    "BULGE_MIR"),
    ("P2",       "P2_CYR",       "P2_MIR"),
    ("OVERHANG", "OVERHANG_CYR", "OVERHANG_MIR"),
]

# -----------------------------------------------------------------------------
# END_DIST REGIÓN INTERHEBRA: C4'(A:nt_start_región) → C4'(B:nt_end_región)
# Antiparalelo: nt_start_A apareado con nt_end_B
# ANCHOR: n=1 en ambas → NA
# -----------------------------------------------------------------------------
END_DIST_REGION_INTER_DEF = {
    # region_label : (A_nt_start, B_nt_end)
    "ANCHOR": None,          # n=1 en ambas
    "P1":     (19, 8),       # A:nt19 → B:nt8
    "BULGE":  (15, 10),      # A:nt15 → B:nt10
    "P2":     (1,  24),      # A:nt1  → B:nt24
    "OVERHANG": None,        # no existe en WT
}

# -----------------------------------------------------------------------------
# COLORES POR REGIÓN (hex)
# -----------------------------------------------------------------------------
COLORS = {
    "ANCHOR_CYR":   "#E63946",   # rojo
    "P1_CYR":       "#2196F3",   # azul
    "BULGE_CYR":    "#FF9800",   # naranja
    "P2_CYR":       "#4CAF50",   # verde
    "OVERHANG_CYR": "#9C27B0",   # violeta (reservado)
    "ANCHOR_MIR":   "#C62828",   # rojo oscuro
    "P1_MIR":       "#1565C0",   # azul oscuro
    "BULGE_MIR":    "#E65100",   # naranja oscuro
    "P2_MIR":       "#2E7D32",   # verde oscuro
    "OVERHANG_MIR": "#6A1B9A",   # violeta oscuro (reservado)
}

# -----------------------------------------------------------------------------
# MASAS ATÓMICAS
# -----------------------------------------------------------------------------
ATOMIC_MASS = {
    "H":  1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "P": 30.974, "S": 32.06,  "MG": 24.305, "ZN": 65.38, "FE": 55.845
}

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_mass(atom):
    elem = atom.element.name.upper() if hasattr(atom, "element") else atom.name[0].upper()
    return ATOMIC_MASS.get(elem, 12.0)

def calc_rg(atoms_list):
    """Radio de giro masa-pesado sobre todos los átomos de la lista."""
    if not atoms_list:
        return None
    coords    = np.array([a.coord for a in atoms_list])
    masses    = np.array([get_mass(a) for a in atoms_list])
    total_mass = masses.sum()
    if total_mass == 0:
        return None
    cm     = np.average(coords, weights=masses, axis=0)
    dist_sq = np.sum((coords - cm) ** 2, axis=1)
    return round(math.sqrt(np.dot(masses, dist_sq) / total_mass), 3)

def get_c4p(res):
    """Devuelve el átomo C4' del residuo, o None si no existe."""
    for atom in res.atoms:
        if atom.name == "C4'":
            return atom
    return None

def calc_end_dist(res_a, res_b):
    """Distancia euclidiana C4'(res_a) → C4'(res_b). Devuelve float o None."""
    if res_a is None or res_b is None:
        return None
    a1 = get_c4p(res_a)
    a2 = get_c4p(res_b)
    if a1 is None or a2 is None:
        return None
    return round(float(np.linalg.norm(np.array(a1.coord) - np.array(a2.coord))), 3)

def fmt(val):
    """Formatea valor para CSV/reporte: None → 'NA'."""
    return val if val is not None else "NA"

def cx(session, cmd):
    from chimerax.core.commands import run as cx_run
    cx_run(session, cmd)

# =============================================================================
# CAPA VISUAL
# =============================================================================

def apply_colors(session):
    cx(session, f"color #1/A:26      {COLORS['ANCHOR_CYR']}")
    cx(session, f"color #1/A:19-25   {COLORS['P1_CYR']}")
    cx(session, f"color #1/A:15-18   {COLORS['BULGE_CYR']}")
    cx(session, f"color #1/A:1-14    {COLORS['P2_CYR']}")
    cx(session, f"color #1/B:1       {COLORS['ANCHOR_MIR']}")
    cx(session, f"color #1/B:2-8     {COLORS['P1_MIR']}")
    cx(session, f"color #1/B:9-10    {COLORS['BULGE_MIR']}")
    cx(session, f"color #1/B:11-24   {COLORS['P2_MIR']}")

def add_distance_layers(session):
    """
    Capa 1 — END_DIST globales (dúplex, interhebra, por hebra)
    Capa 2 — END_DIST regionales interhebra
    Capa 3 — END_DIST regionales por hebra
    Capa 4 — Elipsoides RG por región (excluye n=1 y NA)
    """
    # --- Capa 1: globales ---
    # END_DIST_CYR
    cx(session, "distance #1/A:1@C4'  #1/A:26@C4'")
    # END_DIST_MIR
    cx(session, "distance #1/B:1@C4'  #1/B:24@C4'")
    # END_DIST_INTERHEBRA
    cx(session, "distance #1/A:1@C4'  #1/B:24@C4'")
    # END_DIST_DUPLEX (WT sin overhang → mismo que INTERHEBRA; se dibuja igual)
    # ChimeraX no permite duplicar línea idéntica, se documenta en reporte

    # --- Capa 2: regionales interhebra (solo regiones con >1 nt en ambas) ---
    # P1:    A:nt19 → B:nt8
    cx(session, "distance #1/A:19@C4' #1/B:8@C4'")
    # BULGE: A:nt15 → B:nt10
    cx(session, "distance #1/A:15@C4' #1/B:10@C4'")
    # P2:    A:nt1  → B:nt24  (coincide con INTERHEBRA numéricamente; línea ya trazada)

    # --- Capa 3: regionales intra-hebra ---
    # CYR (A)
    cx(session, "distance #1/A:19@C4' #1/A:25@C4'")   # P1_CYR
    cx(session, "distance #1/A:15@C4' #1/A:18@C4'")   # BULGE_CYR
    cx(session, "distance #1/A:1@C4'  #1/A:14@C4'")   # P2_CYR
    # MIR (B)
    cx(session, "distance #1/B:2@C4'  #1/B:8@C4'")    # P1_MIR
    cx(session, "distance #1/B:9@C4'  #1/B:10@C4'")   # BULGE_MIR  (n=2, se dibuja)
    cx(session, "distance #1/B:11@C4' #1/B:24@C4'")   # P2_MIR

    # --- Capas 4a/4b: elipsoides RG con nombre ---
    # measure inertia crea sub-modelos numerados secuencialmente.
    # Se captura el ID máximo antes y después de cada llamada para renombrar.

    def inertia_named(session, selection, label):
        """
        Ejecuta measure inertia sobre 'selection', luego renombra
        el modelo recién creado con 'label'.
        """
        ids_before = {m.id[0] for m in session.models if len(m.id) == 1}
        cx(session, f"measure inertia {selection} showEllipsoid true")
        ids_after  = {m.id[0] for m in session.models if len(m.id) == 1}
        new_ids    = ids_after - ids_before
        if new_ids:
            new_id = max(new_ids)
            cx(session, f'rename #{new_id} id #{new_id} name "{label}"')

    # Capa 4a — intra-hebra (excluye ANCHOR n=1 y OVERHANG NA)
    inertia_named(session, "#1/A:19-25",           "RG_HEBRA_A_P1")
    inertia_named(session, "#1/A:15-18",           "RG_HEBRA_A_BULGE")
    inertia_named(session, "#1/A:1-14",            "RG_HEBRA_A_P2")
    inertia_named(session, "#1/B:2-8",             "RG_HEBRA_B_P1")
    inertia_named(session, "#1/B:9-10",            "RG_HEBRA_B_BULGE")
    inertia_named(session, "#1/B:11-24",           "RG_HEBRA_B_P2")

    # Capa 4b — interhebra (A+B combinados por región)
    inertia_named(session, "#1/A:26 #1/B:1",       "RG_INTERHEBRA_ANCHOR")
    inertia_named(session, "#1/A:19-25 #1/B:2-8",  "RG_INTERHEBRA_P1")
    inertia_named(session, "#1/A:15-18 #1/B:9-10", "RG_INTERHEBRA_BULGE")
    inertia_named(session, "#1/A:1-14 #1/B:11-24", "RG_INTERHEBRA_P2")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def run(session):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"DEBUG output dir: {OUTPUT_DIR}")

    model       = session.models[MODEL_INDEX]
    today       = datetime.now().strftime("%d/%m/%Y")

    # -------------------------------------------------------------------------
    # 1. RECOLECCIÓN DE RESIDUOS Y ÁTOMOS
    # -------------------------------------------------------------------------
    residues_A  = sorted([r for r in model.residues if r.chain_id == "A"], key=lambda r: r.number)
    residues_B  = sorted([r for r in model.residues if r.chain_id == "B"], key=lambda r: r.number)
    res_by_numA = {r.number: r for r in residues_A}
    res_by_numB = {r.number: r for r in residues_B}

    all_atoms_A = [a for r in residues_A for a in r.atoms]
    all_atoms_B = [a for r in residues_B for a in r.atoms]
    all_atoms   = all_atoms_A + all_atoms_B

    # -------------------------------------------------------------------------
    # 2. RG — NIVEL GLOBAL
    # -------------------------------------------------------------------------
    rg_duplex = calc_rg(all_atoms)
    rg_A      = calc_rg(all_atoms_A)
    rg_B      = calc_rg(all_atoms_B)

    # -------------------------------------------------------------------------
    # 3. END_DIST — NIVEL GLOBAL
    # -------------------------------------------------------------------------
    ed_cyr         = calc_end_dist(res_by_numA.get(1),          res_by_numA.get(CHAIN_A_LEN))
    ed_mir         = calc_end_dist(res_by_numB.get(1),          res_by_numB.get(CHAIN_B_LEN))
    ed_interhebra  = calc_end_dist(res_by_numA.get(1),          res_by_numB.get(CHAIN_B_LEN))
    ed_duplex      = calc_end_dist(res_by_numA.get(1),          res_by_numB.get(DUPLEX_END_B))
    # WT: DUPLEX_END_B == CHAIN_B_LEN == 24 → ed_duplex == ed_interhebra numéricamente

    # -------------------------------------------------------------------------
    # 4. RG — NIVEL REGIONAL (por hebra)
    # -------------------------------------------------------------------------
    def calc_region_rg(region_ranges, res_by_num):
        """Devuelve dict {region_label: (rg, n_nt, n_atoms)} para una hebra."""
        result = {}
        for region, rng in region_ranges.items():
            if rng is None:
                result[region] = (None, 0, 0)
            else:
                start, end = rng
                res_reg    = [res_by_num[i] for i in range(start, end + 1) if i in res_by_num]
                atoms_reg  = [a for r in res_reg for a in r.atoms]
                result[region] = (calc_rg(atoms_reg), end - start + 1, len(atoms_reg))
        return result

    rg_reg_A = calc_region_rg(REGION_RANGES_A, res_by_numA)
    rg_reg_B = calc_region_rg(REGION_RANGES_B, res_by_numB)

    # -------------------------------------------------------------------------
    # 5. RG — NIVEL REGIONAL INTERHEBRA (A+B combinados por región canónica)
    # -------------------------------------------------------------------------
    rg_reg_inter = {}
    for label, reg_a, reg_b in REGION_PAIRS:
        rng_a = REGION_RANGES_A.get(reg_a)
        rng_b = REGION_RANGES_B.get(reg_b)
        atoms_combined = []
        n_nt = 0
        if rng_a is not None:
            for i in range(rng_a[0], rng_a[1] + 1):
                if i in res_by_numA:
                    atoms_combined += list(res_by_numA[i].atoms)
                    n_nt += 1
        if rng_b is not None:
            for i in range(rng_b[0], rng_b[1] + 1):
                if i in res_by_numB:
                    atoms_combined += list(res_by_numB[i].atoms)
                    n_nt += 1
        rg_reg_inter[label] = (calc_rg(atoms_combined), n_nt, len(atoms_combined))

    # -------------------------------------------------------------------------
    # 6. END_DIST — NIVEL REGIONAL
    # -------------------------------------------------------------------------
    def calc_region_end_dist(region_ranges, res_by_num):
        """END_DIST intra-hebra por región. n=1 → 'NA (n=1)'."""
        result = {}
        for region, rng in region_ranges.items():
            if rng is None:
                result[region] = None
            else:
                start, end = rng
                if start == end:
                    result[region] = "NA (n=1)"
                elif start in res_by_num and end in res_by_num:
                    result[region] = calc_end_dist(res_by_num[start], res_by_num[end])
                else:
                    result[region] = None
        return result

    ed_reg_A = calc_region_end_dist(REGION_RANGES_A, res_by_numA)
    ed_reg_B = calc_region_end_dist(REGION_RANGES_B, res_by_numB)

    # END_DIST regional interhebra: C4'(A:nt_start_region) → C4'(B:nt_end_region)
    ed_reg_inter = {}
    for label, def_pair in END_DIST_REGION_INTER_DEF.items():
        if def_pair is None:
            ed_reg_inter[label] = None
        else:
            a_start, b_end = def_pair
            ed_reg_inter[label] = calc_end_dist(
                res_by_numA.get(a_start),
                res_by_numB.get(b_end)
            )

    # -------------------------------------------------------------------------
    # 7. RG — NIVEL POR NUCLEÓTIDO
    # -------------------------------------------------------------------------
    def calc_nt_rg(residues, region_map, chain_id):
        rows = []
        for res in residues:
            atoms = list(res.atoms)
            rows.append({
                "CHAIN":   chain_id,
                "RESNUM":  res.number,
                "RESNAME": res.name.strip().upper(),
                "REGION":  region_map.get(res.number, "UNK"),
                "N_ATOMS": len(atoms),
                "RG_NT":   fmt(calc_rg(atoms)) if atoms else "NA",
            })
        return rows

    rows_nt = calc_nt_rg(residues_A, REGION_MAP_A, "A") + \
              calc_nt_rg(residues_B, REGION_MAP_B, "B")

    # =========================================================================
    # 8. SALIDA — CSV GLOBAL
    # =========================================================================
    csv_global = os.path.join(OUTPUT_DIR, f"DUP_{DUPLEX_NAME}_rg_enddist_global.csv")
    with open(csv_global, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["VARIABLE", "CHAIN", "VALOR", "UNIDAD", "N_ATOMS", "REFERENCIA"])
        # RG global
        w.writerow(["RG_GLOBAL_DUPLEX",   "A+B", fmt(rg_duplex), "A", len(all_atoms),   "todos los atomos A+B"])
        w.writerow(["RG_GLOBAL_CYR",      "A",   fmt(rg_A),      "A", len(all_atoms_A), "todos los atomos Chain A"])
        w.writerow(["RG_GLOBAL_MIR",      "B",   fmt(rg_B),      "A", len(all_atoms_B), "todos los atomos Chain B"])
        # END_DIST global
        w.writerow(["END_DIST_CYR",       "A",   fmt(ed_cyr),        "A", "NA", "C4'(A:nt1)->C4'(A:nt26)"])
        w.writerow(["END_DIST_MIR",       "B",   fmt(ed_mir),        "A", "NA", "C4'(B:nt1)->C4'(B:nt24)"])
        w.writerow(["END_DIST_INTERHEBRA","A+B", fmt(ed_interhebra), "A", "NA", "C4'(A:nt1)->C4'(B:nt24)"])
        w.writerow(["END_DIST_DUPLEX",    "A+B", fmt(ed_duplex),     "A", "NA",
                    f"C4'(A:nt1)->C4'(B:nt{DUPLEX_END_B}) [fin duplex sin overhang]"])

    # =========================================================================
    # 9. SALIDA — CSV REGIONAL
    # =========================================================================
    csv_reg = os.path.join(OUTPUT_DIR, f"DUP_{DUPLEX_NAME}_rg_enddist_region.csv")
    region_labels = ["ANCHOR", "P1", "BULGE", "P2", "OVERHANG"]

    with open(csv_reg, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "NIVEL", "REGION", "CHAIN",
            "NT_START_A", "NT_END_A", "NT_START_B", "NT_END_B",
            "N_NT", "N_ATOMS",
            "RG", "UNIDAD_RG",
            "END_DIST", "UNIDAD_ED", "END_DIST_REF"
        ])

        # --- Nivel interhebra (A+B combinados) ---
        for label, reg_a, reg_b in REGION_PAIRS:
            rng_a = REGION_RANGES_A.get(reg_a)
            rng_b = REGION_RANGES_B.get(reg_b)
            rg_val, n_nt, n_atoms = rg_reg_inter[label]
            ed_val = ed_reg_inter.get(label)
            if ed_val is None and label == "ANCHOR":
                ed_str = "NA (n=1 ambas)"
            elif ed_val is None and label == "OVERHANG":
                ed_str = "NA"
            else:
                ed_str = fmt(ed_val)
            a_start = rng_a[0] if rng_a else "NA"
            a_end   = rng_a[1] if rng_a else "NA"
            b_start = rng_b[0] if rng_b else "NA"
            b_end   = rng_b[1] if rng_b else "NA"
            inter_def = END_DIST_REGION_INTER_DEF.get(label)
            if inter_def:
                ref = f"C4'(A:nt{inter_def[0]})->C4'(B:nt{inter_def[1]})"
            else:
                ref = "NA"
            w.writerow([
                "INTERHEBRA", label, "A+B",
                a_start, a_end, b_start, b_end,
                n_nt, n_atoms,
                fmt(rg_val), "A",
                ed_str, "A", ref
            ])

        # --- Nivel por hebra CYR (A) ---
        for region in REGION_ORDER_A:
            label = region.replace("_CYR", "")
            rng   = REGION_RANGES_A.get(region)
            rg_val, n_nt, n_atoms = rg_reg_A[region]
            ed_val = ed_reg_A[region]
            a_start = rng[0] if rng else "NA"
            a_end   = rng[1] if rng else "NA"
            if ed_val is None and rng is None:
                ed_str = "NA"
                ref    = "NA"
            elif ed_val == "NA (n=1)":
                ed_str = "NA (n=1)"
                ref    = f"C4'(A:nt{a_start}) — un solo nt"
            else:
                ed_str = fmt(ed_val)
                ref    = f"C4'(A:nt{a_start})->C4'(A:nt{a_end})"
            w.writerow([
                "REGIONAL_CYR", label, "A",
                a_start, a_end, "NA", "NA",
                n_nt, n_atoms,
                fmt(rg_val), "A",
                ed_str, "A" if ed_str not in ("NA", "NA (n=1)") else "NA", ref
            ])

        # --- Nivel por hebra MIR (B) ---
        for region in REGION_ORDER_B:
            label = region.replace("_MIR", "")
            rng   = REGION_RANGES_B.get(region)
            rg_val, n_nt, n_atoms = rg_reg_B[region]
            ed_val = ed_reg_B[region]
            b_start = rng[0] if rng else "NA"
            b_end   = rng[1] if rng else "NA"
            if ed_val is None and rng is None:
                ed_str = "NA"
                ref    = "NA"
            elif ed_val == "NA (n=1)":
                ed_str = "NA (n=1)"
                ref    = f"C4'(B:nt{b_start}) — un solo nt"
            else:
                ed_str = fmt(ed_val)
                ref    = f"C4'(B:nt{b_start})->C4'(B:nt{b_end})"
            w.writerow([
                "REGIONAL_MIR", label, "B",
                "NA", "NA", b_start, b_end,
                n_nt, n_atoms,
                fmt(rg_val), "A",
                ed_str, "A" if ed_str not in ("NA", "NA (n=1)") else "NA", ref
            ])

    # =========================================================================
    # 10. SALIDA — CSV POR NUCLEÓTIDO
    # =========================================================================
    csv_nt = os.path.join(OUTPUT_DIR, f"DUP_{DUPLEX_NAME}_rg_enddist_per_nt.csv")
    with open(csv_nt, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["CHAIN","RESNUM","RESNAME","REGION","N_ATOMS","RG_NT"])
        w.writeheader()
        w.writerows(rows_nt)

    # =========================================================================
    # 11. SALIDA — REPORTE .TXT
    # =========================================================================
    report_path = os.path.join(OUTPUT_DIR, f"DUP_{DUPLEX_NAME}_rg_enddist_reporte.txt")
    SEP  = "=" * 72
    SEP2 = "-" * 72

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(SEP + "\n")
        f.write(f"REPORTE RG + END_DIST — DUPLEX {DUPLEX_NAME}\n")
        f.write(SEP + "\n\n")
        f.write(f"HERRAMIENTA      : UCSF ChimeraX — Python API interna\n")
        f.write(f"VERSION          : verificar en ChimeraX -> Help > About\n")
        f.write(f"FECHA            : {today}\n")
        f.write(f"INPUT            : {PDB_INPUT}\n\n")
        f.write(f"DEFINICIONES\n")
        f.write(f"  RG             : radio de giro masa-pesado (todos los atomos)\n")
        f.write(f"                   Rg = sqrt( sum(mi*|ri-rcm|^2) / sum(mi) )\n")
        f.write(f"  END_DIST       : distancia euclidiana C4'(nt_inicio)->C4'(nt_fin)\n\n")
        f.write(f"CADENAS\n")
        f.write(f"  Chain A — CYR_WT   : 5'-AACAACAAAAUCACCAAUGUCUUCCA-3' ({CHAIN_A_LEN} nt)\n")
        f.write(f"                        nt1=5', nt{CHAIN_A_LEN}=3'\n")
        f.write(f"  Chain B — miR-7-5p : 5'-UGGAAGACUAGUGAUUUUGUUGUU-3'  ({CHAIN_B_LEN} nt)\n")
        f.write(f"                        nt1=5', nt{CHAIN_B_LEN}=3'\n\n")
        f.write(f"REGIONES ESTRUCTURALES\n")
        f.write(f"  Chain A (CYR_WT)\n")
        f.write(f"    ANCHOR_CYR   : A:26\n")
        f.write(f"    P1_CYR       : A:19-25\n")
        f.write(f"    BULGE_CYR    : A:15-18\n")
        f.write(f"    P2_CYR       : A:1-14\n")
        f.write(f"    OVERHANG_CYR : NA\n")
        f.write(f"  Chain B (miR-7-5p)\n")
        f.write(f"    ANCHOR_MIR   : B:1\n")
        f.write(f"    P1_MIR       : B:2-8\n")
        f.write(f"    BULGE_MIR    : B:9-10\n")
        f.write(f"    P2_MIR       : B:11-24\n")
        f.write(f"    OVERHANG_MIR : NA\n\n")
        f.write(f"NOTA END_DIST_DUPLEX vs END_DIST_INTERHEBRA\n")
        f.write(f"  WT no tiene overhang en ninguna hebra.\n")
        f.write(f"  DUPLEX_END_B = nt{DUPLEX_END_B} = extremo 3' de miR-7-5p.\n")
        f.write(f"  Por lo tanto END_DIST_DUPLEX == END_DIST_INTERHEBRA numericamente.\n")
        f.write(f"  Ambos valores se registran explicitamente para coherencia con variantes.\n\n")
        f.write(SEP2 + "\n")
        f.write("ARCHIVOS DE SALIDA\n")
        f.write(SEP2 + "\n")
        f.write(f"  CSV global    : DUP_{DUPLEX_NAME}_rg_enddist_global.csv\n")
        f.write(f"  CSV region    : DUP_{DUPLEX_NAME}_rg_enddist_region.csv\n")
        f.write(f"  CSV por nt    : DUP_{DUPLEX_NAME}_rg_enddist_per_nt.csv\n\n")
        f.write(SEP2 + "\n")
        f.write("VARIABLES — NIVEL GLOBAL\n")
        f.write(SEP2 + "\n")
        f.write(f"  {'VARIABLE':<25} {'CHAIN':<5} {'VALOR':>10} {'UNIDAD':<6}  REFERENCIA\n")
        f.write(f"  {'RG_GLOBAL_DUPLEX':<25} {'A+B':<5} {str(fmt(rg_duplex)):>10} {'A':<6}  {len(all_atoms)} atomos\n")
        f.write(f"  {'RG_GLOBAL_CYR':<25} {'A':<5} {str(fmt(rg_A)):>10} {'A':<6}  {len(all_atoms_A)} atomos\n")
        f.write(f"  {'RG_GLOBAL_MIR':<25} {'B':<5} {str(fmt(rg_B)):>10} {'A':<6}  {len(all_atoms_B)} atomos\n")
        f.write(f"  {'END_DIST_CYR':<25} {'A':<5} {str(fmt(ed_cyr)):>10} {'A':<6}  C4'(A:nt1)->C4'(A:nt{CHAIN_A_LEN})\n")
        f.write(f"  {'END_DIST_MIR':<25} {'B':<5} {str(fmt(ed_mir)):>10} {'A':<6}  C4'(B:nt1)->C4'(B:nt{CHAIN_B_LEN})\n")
        f.write(f"  {'END_DIST_INTERHEBRA':<25} {'A+B':<5} {str(fmt(ed_interhebra)):>10} {'A':<6}  C4'(A:nt1)->C4'(B:nt{CHAIN_B_LEN})\n")
        f.write(f"  {'END_DIST_DUPLEX':<25} {'A+B':<5} {str(fmt(ed_duplex)):>10} {'A':<6}  C4'(A:nt1)->C4'(B:nt{DUPLEX_END_B})\n")
        f.write("\n")
        f.write(SEP2 + "\n")
        f.write("VARIABLES — NIVEL REGIONAL INTERHEBRA\n")
        f.write(SEP2 + "\n")
        f.write(f"  {'REGION':<10} {'NT_A':>8} {'NT_B':>8} {'N_NT':>5} {'N_ATOMS':>8} {'RG(A)':>10} {'END_DIST(A)':>13}  REFERENCIA_ED\n")
        for label, reg_a, reg_b in REGION_PAIRS:
            rng_a = REGION_RANGES_A.get(reg_a)
            rng_b = REGION_RANGES_B.get(reg_b)
            rg_val, n_nt, n_atoms = rg_reg_inter[label]
            ed_val = ed_reg_inter.get(label)
            a_range = f"{rng_a[0]}-{rng_a[1]}" if rng_a else "NA"
            b_range = f"{rng_b[0]}-{rng_b[1]}" if rng_b else "NA"
            inter_def = END_DIST_REGION_INTER_DEF.get(label)
            if inter_def:
                ed_str = str(fmt(ed_val))
                ref = f"C4'(A:nt{inter_def[0]})->C4'(B:nt{inter_def[1]})"
            elif label == "ANCHOR":
                ed_str = "NA (n=1)"
                ref = "n=1 en ambas hebras"
            else:
                ed_str = "NA"
                ref = "NA"
            f.write(f"  {label:<10} {a_range:>8} {b_range:>8} {str(n_nt):>5} {str(n_atoms):>8} "
                    f"{str(fmt(rg_val)):>10} {ed_str:>13}  {ref}\n")
        f.write("\n")
        f.write(SEP2 + "\n")
        f.write("VARIABLES — NIVEL REGIONAL POR HEBRA\n")
        f.write(SEP2 + "\n")
        f.write(f"  {'CH':<3} {'REGION':<12} {'NT':>8} {'N_NT':>5} {'N_ATOMS':>8} {'RG(A)':>10} {'END_DIST(A)':>13}  REFERENCIA_ED\n")
        for region in REGION_ORDER_A:
            label  = region.replace("_CYR","")
            rng    = REGION_RANGES_A.get(region)
            rg_val, n_nt, n_atoms = rg_reg_A[region]
            ed_val = ed_reg_A[region]
            nt_str = f"{rng[0]}-{rng[1]}" if rng else "NA"
            if rng is None:
                ed_str = "NA"; ref = "NA"
            elif ed_val == "NA (n=1)":
                ed_str = "NA (n=1)"; ref = f"C4'(A:nt{rng[0]}) — un solo nt"
            else:
                ed_str = str(fmt(ed_val)); ref = f"C4'(A:nt{rng[0]})->C4'(A:nt{rng[1]})"
            f.write(f"  {'A':<3} {label:<12} {nt_str:>8} {str(n_nt):>5} {str(n_atoms):>8} "
                    f"{str(fmt(rg_val)):>10} {ed_str:>13}  {ref}\n")
        for region in REGION_ORDER_B:
            label  = region.replace("_MIR","")
            rng    = REGION_RANGES_B.get(region)
            rg_val, n_nt, n_atoms = rg_reg_B[region]
            ed_val = ed_reg_B[region]
            nt_str = f"{rng[0]}-{rng[1]}" if rng else "NA"
            if rng is None:
                ed_str = "NA"; ref = "NA"
            elif ed_val == "NA (n=1)":
                ed_str = "NA (n=1)"; ref = f"C4'(B:nt{rng[0]}) — un solo nt"
            else:
                ed_str = str(fmt(ed_val)); ref = f"C4'(B:nt{rng[0]})->C4'(B:nt{rng[1]})"
            f.write(f"  {'B':<3} {label:<12} {nt_str:>8} {str(n_nt):>5} {str(n_atoms):>8} "
                    f"{str(fmt(rg_val)):>10} {ed_str:>13}  {ref}\n")
        f.write("\n")
        f.write(SEP2 + "\n")
        f.write("VARIABLES — NIVEL POR NUCLEOTIDO\n")
        f.write(SEP2 + "\n")
        f.write(f"  {'CH':<3} {'NT':<5} {'RES':<5} {'REGION':<14} {'N_ATOMS':>8} {'RG_NT(A)':>10}\n")
        for r in rows_nt:
            f.write(f"  {r['CHAIN']:<3} {r['RESNUM']:<5} {r['RESNAME']:<5} {r['REGION']:<14} "
                    f"{str(r['N_ATOMS']):>8} {str(r['RG_NT']):>10}\n")
        f.write("\n")
        f.write(SEP2 + "\n")
        f.write("NOTAS\n")
        f.write(SEP2 + "\n")
        f.write("  - RG: masa-pesado con masas atomicas reales (tabla ATOMIC_MASS).\n")
        f.write("  - END_DIST: distancia euclidiana C4'->C4'; C4' = pseudoatomo canonico backbone RNA.\n")
        f.write("  - END_DIST_REGION_INTER: C4'(A:nt_start_region)->C4'(B:nt_end_region) por antiparalelismo.\n")
        f.write("  - ANCHOR (n=1): RG calculado; END_DIST = NA (n=1).\n")
        f.write("  - OVERHANG: NA en ambas hebras para WT.\n")
        f.write("  - Elipsoides RG intra-hebra: P1_CYR, BULGE_CYR, P2_CYR, P1_MIR, BULGE_MIR, P2_MIR.\n")
        f.write("  - Elipsoides RG interhebra: ANCHOR(A:26+B:1), P1(A:19-25+B:2-8), BULGE(A:15-18+B:9-10), P2(A:1-14+B:11-24).\n")
        f.write("  - Lineas END_DIST activas: globales (CYR, MIR, INTERHEBRA, DUPLEX) + regionales inter + regionales intra.\n")
        f.write("  - Para video: movie record / turn y 3 120 / movie encode output ruta format h264 framerate 15\n")
        f.write(SEP + "\n")

    # =========================================================================
    # 12. CAPA VISUAL CHIMERAX
    # =========================================================================
    cx(session, "set bgcolor white")
    cx(session, "lighting simple")
    cx(session, "nucleotides #1 tube/slab")
    cx(session, "show #1 cartoons")
    apply_colors(session)
    cx(session, "view #1")
    add_distance_layers(session)
    cx(session, "view #1")

    # =========================================================================
    # 13. PRINT CONSOLA
    # =========================================================================
    SEP_P  = "=" * 72
    SEP2_P = "-" * 72
    print("\n" + SEP_P)
    print(f"RG + END_DIST — DUPLEX {DUPLEX_NAME}")
    print(SEP_P)
    print(f"\n  {'VARIABLE':<25} {'CHAIN':<5} {'VALOR':>10}  UNIDAD  REFERENCIA")
    print(f"  {'RG_GLOBAL_DUPLEX':<25} {'A+B':<5} {str(fmt(rg_duplex)):>10}  A")
    print(f"  {'RG_GLOBAL_CYR':<25} {'A':<5} {str(fmt(rg_A)):>10}  A")
    print(f"  {'RG_GLOBAL_MIR':<25} {'B':<5} {str(fmt(rg_B)):>10}  A")
    print(f"  {'END_DIST_CYR':<25} {'A':<5} {str(fmt(ed_cyr)):>10}  A       C4'(A:nt1)->C4'(A:nt{CHAIN_A_LEN})")
    print(f"  {'END_DIST_MIR':<25} {'B':<5} {str(fmt(ed_mir)):>10}  A       C4'(B:nt1)->C4'(B:nt{CHAIN_B_LEN})")
    print(f"  {'END_DIST_INTERHEBRA':<25} {'A+B':<5} {str(fmt(ed_interhebra)):>10}  A       C4'(A:nt1)->C4'(B:nt{CHAIN_B_LEN})")
    print(f"  {'END_DIST_DUPLEX':<25} {'A+B':<5} {str(fmt(ed_duplex)):>10}  A       C4'(A:nt1)->C4'(B:nt{DUPLEX_END_B})")
    print(f"\n  REGIONAL INTERHEBRA")
    print(f"  {'REGION':<10} {'NT_A':>8} {'NT_B':>8} {'N_NT':>5} {'N_ATOMS':>8} {'RG(A)':>10} {'END_DIST(A)':>13}")
    for label, reg_a, reg_b in REGION_PAIRS:
        rng_a = REGION_RANGES_A.get(reg_a)
        rng_b = REGION_RANGES_B.get(reg_b)
        rg_val, n_nt, n_atoms = rg_reg_inter[label]
        ed_val = ed_reg_inter.get(label)
        a_range = f"{rng_a[0]}-{rng_a[1]}" if rng_a else "NA"
        b_range = f"{rng_b[0]}-{rng_b[1]}" if rng_b else "NA"
        inter_def = END_DIST_REGION_INTER_DEF.get(label)
        ed_str = str(fmt(ed_val)) if inter_def else ("NA (n=1)" if label == "ANCHOR" else "NA")
        print(f"  {label:<10} {a_range:>8} {b_range:>8} {str(n_nt):>5} {str(n_atoms):>8} "
              f"{str(fmt(rg_val)):>10} {ed_str:>13}")
    print(f"\n  CAPAS ACTIVAS:")
    print(f"    - Colores por region (A y B)")
    print(f"    - Lineas END_DIST: globales (CYR, MIR, INTERHEBRA) + regionales inter + regionales intra")
    print(f"    - Elipsoides RG intra-hebra: P1_CYR, BULGE_CYR, P2_CYR, P1_MIR, BULGE_MIR, P2_MIR")
    print(f"    - Elipsoides RG interhebra: ANCHOR, P1, BULGE, P2 (seleccion A+B combinada)")
    print(f"\n  OUTPUTS:")
    print(f"    CSV global  -> {csv_global}")
    print(f"    CSV region  -> {csv_reg}")
    print(f"    CSV por nt  -> {csv_nt}")
    print(f"    Reporte     -> {report_path}")
    print(f"\n  PARA VIDEO MANUAL:")
    print(f"    movie record")
    print(f"    turn y 3 120")
    print(f'    movie encode output "{OUTPUT_DIR}\\DUP_{DUPLEX_NAME}_rg_enddist.mp4" format h264 framerate 15')
    print(SEP_P + "\n")

run(session)
