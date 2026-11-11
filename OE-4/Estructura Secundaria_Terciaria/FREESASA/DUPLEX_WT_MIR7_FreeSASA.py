"""
DUPLEX_WT_MIR7_FreeSASA.py
==========================
Script de análisis SASA para el dúplex RNA WT_miR7
Herramienta: FreeSASA (https://freesasa.github.io/)
Autor: cmart
Fecha: ver reporte de salida

DÚPLEX: WT_miR7
  Hebra A — CYR/WT (26 nt, 5'→3')
    Secuencia : AACAACAAAAUCACCAAUGUCUUCCA
    ANCHOR_CYR    : nt 26
    P1_CYR        : nt 19–25
    B_CYR         : nt 15–18
    P2_CYR        : nt 1–14
    OVERHANG_CYR  : NA

  Hebra B — MIR/miR-7-5p (24 nt, 5'→3')
    Secuencia : UGGAAGACUAGUGAUUUUGUUGUU
    ANCHOR_MIR    : nt 1
    P1_MIR        : nt 2–8
    B_MIR         : nt 9–10
    P2_MIR        : nt 11–24
    OVERHANG_MIR  : NA

FLUJO:
  1. Configuración de rutas, parámetros y mapas de regiones
  2. Carga del PDB con FreeSASA
  3. Clasificación de átomos por cadena/residuo (BB vs BASE)
  4. Cálculo de niveles globales: GLOBAL_DUPLEX, GLOBAL_CYR, GLOBAL_MIR
  5. Tablas por nucleótido por cadena
  6. Tablas por elemento estructural por cadena (estructura fija, OVERHANG → NA)
  7. Exportación a 3 CSVs
  8. Generación de reporte .txt con los tres niveles
  9. Copia del script al directorio de salida

NOTA SOBRE OVERHANG:
  Las regiones OVERHANG_CYR y OVERHANG_MIR están definidas en el esquema
  para mantener estructura fija entre dúplex. En WT_miR7 ambas son NA
  (no hay nucleótidos de overhang). Se reportan explícitamente como NA.
"""

import freesasa
import pandas as pd
import os
import sys
import shutil
from datetime import datetime

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

PDB_PATH = os.path.expanduser(
    "~/tesis_rnas/OE4/1.FreeSASA/1.WT_MIR/WT_miR7.pdb"
)

OUTPUT_DIR = os.path.expanduser(
    "~/tesis_rnas/OE4/1.FreeSASA/1.WT_MIR"
)

PREFIX = "DUPLEX_WT_MIR7_FreeSASA"

# ── Mapas de regiones por cadena ──────────────────────────────────────
# Clave: nombre de región (sufijo _CYR / _MIR indica cadena)
# Valor: lista de índices de residuo PDB (base 1, dirección 5'→3')
#        None → región definida en esquema pero ausente en este dúplex (OVERHANG)

REGION_MAP_CYR = {
    "ANCHOR_CYR":   [26],              # nt 26 (extremo 3', t1)
    "P1_CYR":       list(range(19, 26)),  # nt 19–25
    "B_CYR":        list(range(15, 19)),  # nt 15–18
    "P2_CYR":       list(range(1,  15)),  # nt 1–14
    "OVERHANG_CYR": None,              # NA en este dúplex
}

REGION_MAP_MIR = {
    "ANCHOR_MIR":   [1],               # nt 1 (extremo 5', opuesto a ANCHOR_CYR)
    "P1_MIR":       list(range(2,  9)),   # nt 2–8
    "B_MIR":        list(range(9,  11)),  # nt 9–10
    "P2_MIR":       list(range(11, 25)),  # nt 11–24
    "OVERHANG_MIR": None,              # NA en este dúplex
}

# Átomos de backbone RNA (convención PDB estándar)
BACKBONE_ATOMS = {
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'",
    "C2'", "O2'", "C1'", "O4'"
}

# Metadatos de secuencias
META = {
    "CYR": {"n_nt": 26, "seq": "AACAACAAAAUCACCAAUGUCUUCCA", "chain": "A"},
    "MIR": {"n_nt": 24, "seq": "UGGAAGACUAGUGAUUUUGUUGUU",   "chain": "B"},
}

# ─────────────────────────────────────────────
# 2. SETUP
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
today = datetime.now().strftime("%d/%m/%Y")

# ─────────────────────────────────────────────
# 3. CARGA DEL PDB Y CÁLCULO SASA
# ─────────────────────────────────────────────

print(f"[INFO] Cargando PDB: {PDB_PATH}")
if not os.path.isfile(PDB_PATH):
    sys.exit(f"[ERROR] Archivo PDB no encontrado: {PDB_PATH}")

# FreeSASA: Lee & Richards, probe_radius=1.4 Å, n_points=100 (default)
structure = freesasa.Structure(PDB_PATH)
result    = freesasa.calc(structure)

# Versión de FreeSASA (compatible con bioconda)
try:
    import importlib.metadata
    version_str = importlib.metadata.version("freesasa")
except Exception:
    try:
        import subprocess
        version_str = subprocess.check_output(
            ["freesasa", "--version"], stderr=subprocess.STDOUT
        ).decode().strip().split("\n")[0]
    except Exception:
        version_str = "2.2.1"

print(f"[INFO] FreeSASA v{version_str} — cálculo completado")

# ─────────────────────────────────────────────
# 4. CLASIFICACIÓN DE ÁTOMOS POR CADENA/RESIDUO
# ─────────────────────────────────────────────
# Estructura de residue_data:
#   { chain_id: { resnum_int: {chain, resname, SASA_NT, SASA_BB, SASA_BASE} } }
#
# La clave primaria es chain_id para permitir residuos con el mismo
# número de residuo en cadenas distintas (frecuente en PDBs de dúplex).

def classify_atoms(structure, result):
    """
    Clasifica átomos por cadena y residuo.
    Retorna dict anidado: chain → resnum → {chain, resname, SASA_*}.
    """
    data = {}
    n_atoms = structure.nAtoms()

    for i in range(n_atoms):
        chain   = structure.chainLabel(i)        # "A" o "B"
        resnum  = structure.residueNumber(i)
        resname = structure.residueName(i).strip()
        atom    = structure.atomName(i).strip()
        sasa_i  = result.atomArea(i)

        try:
            resnum_int = int(resnum.strip())
        except ValueError:
            continue

        if chain not in data:
            data[chain] = {}

        if resnum_int not in data[chain]:
            data[chain][resnum_int] = {
                "chain":     chain,
                "resname":   resname,
                "SASA_NT":   0.0,
                "SASA_BB":   0.0,
                "SASA_BASE": 0.0,
            }

        data[chain][resnum_int]["SASA_NT"] += sasa_i
        if atom in BACKBONE_ATOMS:
            data[chain][resnum_int]["SASA_BB"]   += sasa_i
        else:
            data[chain][resnum_int]["SASA_BASE"] += sasa_i

    return data

residue_data = classify_atoms(structure, result)

# Verificación de cadenas detectadas
detected_chains = sorted(residue_data.keys())
print(f"[INFO] Cadenas detectadas en PDB: {detected_chains}")
for ch in ["A", "B"]:
    if ch not in residue_data:
        sys.exit(f"[ERROR] Cadena {ch} no encontrada en el PDB. "
                 f"Cadenas presentes: {detected_chains}")

chain_cyr = residue_data["A"]   # Hebra CYR
chain_mir = residue_data["B"]   # Hebra MIR

# ─────────────────────────────────────────────
# 5. VARIABLES GLOBALES — TRES NIVELES
# ─────────────────────────────────────────────

def global_sasa(chain_dict):
    """Suma SASA_TOTAL, SASA_BB, SASA_BASES de todos los residuos de una cadena."""
    return {
        "SASA_TOTAL": sum(v["SASA_NT"]   for v in chain_dict.values()),
        "SASA_BB":    sum(v["SASA_BB"]   for v in chain_dict.values()),
        "SASA_BASES": sum(v["SASA_BASE"] for v in chain_dict.values()),
    }

glob_cyr    = global_sasa(chain_cyr)
glob_mir    = global_sasa(chain_mir)
glob_duplex = {
    "SASA_TOTAL": result.totalArea(),   # FreeSASA calcula sobre el dúplex completo
    "SASA_BB":    glob_cyr["SASA_BB"]  + glob_mir["SASA_BB"],
    "SASA_BASES": glob_cyr["SASA_BASES"] + glob_mir["SASA_BASES"],
}

print(f"\n[GLOBAL_DUPLEX]")
print(f"  SASA_TOTAL : {glob_duplex['SASA_TOTAL']:.4f} A2")
print(f"  SASA_BB    : {glob_duplex['SASA_BB']:.4f} A2")
print(f"  SASA_BASES : {glob_duplex['SASA_BASES']:.4f} A2")

print(f"\n[GLOBAL_CYR  — Chain A, {META['CYR']['n_nt']} nt]")
print(f"  SASA_TOTAL : {glob_cyr['SASA_TOTAL']:.4f} A2")
print(f"  SASA_BB    : {glob_cyr['SASA_BB']:.4f} A2")
print(f"  SASA_BASES : {glob_cyr['SASA_BASES']:.4f} A2")

print(f"\n[GLOBAL_MIR  — Chain B, {META['MIR']['n_nt']} nt]")
print(f"  SASA_TOTAL : {glob_mir['SASA_TOTAL']:.4f} A2")
print(f"  SASA_BB    : {glob_mir['SASA_BB']:.4f} A2")
print(f"  SASA_BASES : {glob_mir['SASA_BASES']:.4f} A2")

# ─────────────────────────────────────────────
# 6. TABLA POR NUCLEÓTIDO
# ─────────────────────────────────────────────

# Lookup inverso: resnum → region_name, por cadena
# Construido desde REGION_MAP_* para asignación O(1) en build_nt_table
def build_region_lookup(region_map):
    """
    Invierte region_map → {resnum: region_name}.
    Regiones con nt_list=None (OVERHANG ausente) se omiten del lookup.
    """
    lookup = {}
    for region_name, nt_list in region_map.items():
        if nt_list is None:
            continue
        for nt in nt_list:
            lookup[nt] = region_name
    return lookup

region_lookup_cyr = build_region_lookup(REGION_MAP_CYR)
region_lookup_mir = build_region_lookup(REGION_MAP_MIR)

def build_nt_table(chain_dict, chain_label, region_lookup):
    """
    Construye DataFrame con una fila por residuo.
    chain_label  : "CYR" o "MIR" (para columna STRAND).
    region_lookup: dict resnum → region_name para asignar columna REGION.
    Nucleótidos sin región asignada → "UNASSIGNED".
    """
    rows = []
    for resnum in sorted(chain_dict.keys()):
        v = chain_dict[resnum]
        rows.append({
            "STRAND":    chain_label,
            "CHAIN":     v["chain"],
            "NT_INDEX":  resnum,
            "RESNAME":   v["resname"],
            "REGION":    region_lookup.get(resnum, "UNASSIGNED"),
            "SASA_NT":   round(v["SASA_NT"],   4),
            "SASA_BB":   round(v["SASA_BB"],   4),
            "SASA_BASE": round(v["SASA_BASE"], 4),
        })
    return pd.DataFrame(rows)

df_nt_cyr = build_nt_table(chain_cyr, "CYR", region_lookup_cyr)
df_nt_mir = build_nt_table(chain_mir, "MIR", region_lookup_mir)
df_nt_all = pd.concat([df_nt_cyr, df_nt_mir], ignore_index=True)

print(f"\n[NT TABLE — CYR, {len(df_nt_cyr)} residuos]")
print(df_nt_cyr.to_string(index=False))
print(f"\n[NT TABLE — MIR, {len(df_nt_mir)} residuos]")
print(df_nt_mir.to_string(index=False))

# ─────────────────────────────────────────────
# 7. TABLA POR ELEMENTO ESTRUCTURAL
# ─────────────────────────────────────────────
# Estructura fija: todas las regiones aparecen siempre.
# OVERHANG_CYR / OVERHANG_MIR → SASA_NT/BB/BASE = NA, MISSING_NT = "NA_by_design"

def build_element_table(region_map, chain_dict, chain_label):
    """
    Agrega SASA por elemento estructural.
    Regiones con nt_list=None → fila NA (OVERHANG ausente por diseño).
    """
    rows = []
    for region_name, nt_list in region_map.items():

        # Caso OVERHANG ausente (NA by design)
        if nt_list is None:
            rows.append({
                "STRAND":      chain_label,
                "ELEMENT":     region_name,
                "NT_RANGE":    "NA",
                "N_NT":        0,
                "SASA_NT":     "NA",
                "SASA_BB":     "NA",
                "SASA_BASE":   "NA",
                "MISSING_NT":  "NA_by_design",
            })
            continue

        nt_in_pdb = [n for n in nt_list if n in chain_dict]
        missing   = [n for n in nt_list if n not in chain_dict]

        sasa_nt   = sum(chain_dict[n]["SASA_NT"]   for n in nt_in_pdb)
        sasa_bb   = sum(chain_dict[n]["SASA_BB"]   for n in nt_in_pdb)
        sasa_base = sum(chain_dict[n]["SASA_BASE"] for n in nt_in_pdb)

        if missing:
            print(f"[WARN] {region_name}: nt no encontrados en PDB: {missing}")

        rows.append({
            "STRAND":      chain_label,
            "ELEMENT":     region_name,
            "NT_RANGE":    f"{nt_list[0]}-{nt_list[-1]}",
            "N_NT":        len(nt_in_pdb),
            "SASA_NT":     round(sasa_nt,   4),
            "SASA_BB":     round(sasa_bb,   4),
            "SASA_BASE":   round(sasa_base, 4),
            "MISSING_NT":  str(missing) if missing else "none",
        })

    return pd.DataFrame(rows)

df_elem_cyr = build_element_table(REGION_MAP_CYR, chain_cyr, "CYR")
df_elem_mir = build_element_table(REGION_MAP_MIR, chain_mir, "MIR")
df_elem_all = pd.concat([df_elem_cyr, df_elem_mir], ignore_index=True)

print(f"\n[ELEMENT TABLE — CYR]")
print(df_elem_cyr.to_string(index=False))
print(f"\n[ELEMENT TABLE — MIR]")
print(df_elem_mir.to_string(index=False))

# ─────────────────────────────────────────────
# 8. TABLA GLOBAL (3 filas: DUPLEX, CYR, MIR)
# ─────────────────────────────────────────────

df_global = pd.DataFrame([
    {
        "LEVEL":      "GLOBAL_DUPLEX",
        "MOLECULE":   "WT_miR7",
        "SASA_TOTAL": round(glob_duplex["SASA_TOTAL"], 4),
        "SASA_BB":    round(glob_duplex["SASA_BB"],    4),
        "SASA_BASES": round(glob_duplex["SASA_BASES"], 4),
    },
    {
        "LEVEL":      "GLOBAL_CYR",
        "MOLECULE":   "WT",
        "SASA_TOTAL": round(glob_cyr["SASA_TOTAL"], 4),
        "SASA_BB":    round(glob_cyr["SASA_BB"],    4),
        "SASA_BASES": round(glob_cyr["SASA_BASES"], 4),
    },
    {
        "LEVEL":      "GLOBAL_MIR",
        "MOLECULE":   "miR-7-5p",
        "SASA_TOTAL": round(glob_mir["SASA_TOTAL"], 4),
        "SASA_BB":    round(glob_mir["SASA_BB"],    4),
        "SASA_BASES": round(glob_mir["SASA_BASES"], 4),
    },
])

# ─────────────────────────────────────────────
# 9. EXPORTACIÓN CSV
# ─────────────────────────────────────────────

csv_global = os.path.join(OUTPUT_DIR, f"{PREFIX}_global.csv")
csv_nt     = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_nt.csv")
csv_elem   = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_element.csv")

df_global.to_csv(csv_global, index=False)
df_nt_all.to_csv(csv_nt,     index=False)
df_elem_all.to_csv(csv_elem, index=False)

print(f"\n[OUTPUT] CSV global    -> {csv_global}")
print(f"[OUTPUT] CSV por nt    -> {csv_nt}")
print(f"[OUTPUT] CSV por elem  -> {csv_elem}")

# ─────────────────────────────────────────────
# 10. REPORTE .TXT
# ─────────────────────────────────────────────

report_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte.txt")

def fmt_sasa_block(label, glob_dict):
    """Genera bloque de 3 variables SASA para el reporte."""
    return [
        f"  VARIABLE_EXTRAIDA : SASA_TOTAL",
        f"  VALOR             : {glob_dict['SASA_TOTAL']:.4f} A2",
        f"  NOTAS             : Area superficial total accesible al solvente",
        "",
        f"  VARIABLE_EXTRAIDA : SASA_BB",
        f"  VALOR             : {glob_dict['SASA_BB']:.4f} A2",
        f"  NOTAS             : Contribucion de atomos de backbone (P, OP1, OP2, azucares)",
        "",
        f"  VARIABLE_EXTRAIDA : SASA_BASES",
        f"  VALOR             : {glob_dict['SASA_BASES']:.4f} A2",
        f"  NOTAS             : Contribucion de atomos de bases nitrogenadas",
        "",
    ]

report_lines = [
    "=" * 70,
    "REPORTE DE ANALISIS SASA - DUPLEX WT_miR7",
    "=" * 70,
    "",
    f"HERRAMIENTA      : FreeSASA",
    f"VERSION          : {version_str}",
    f"FECHA            : {today}",
    f"INPUT            : {PDB_PATH}",
    "PARAMETROS       : default (algoritmo Lee & Richards, probe_radius=1.4 A, n_points=100)",
    "",
    f"HEBRA A (CYR/WT,       {META['CYR']['n_nt']} nt, 5'->3'): {META['CYR']['seq']}",
    f"  ANCHOR_CYR   : nt 26",
    f"  P1_CYR       : nt 19-25",
    f"  B_CYR        : nt 15-18",
    f"  P2_CYR       : nt 1-14",
    f"  OVERHANG_CYR : NA",
    "",
    f"HEBRA B (MIR/miR-7-5p, {META['MIR']['n_nt']} nt, 5'->3'): {META['MIR']['seq']}",
    f"  ANCHOR_MIR   : nt 1",
    f"  P1_MIR       : nt 2-8",
    f"  B_MIR        : nt 9-10",
    f"  P2_MIR       : nt 11-24",
    f"  OVERHANG_MIR : NA",
    "",
    "-" * 70,
    "ARCHIVOS DE SALIDA",
    "-" * 70,
    f"  CSV global     : {os.path.basename(csv_global)}",
    f"  CSV por nt     : {os.path.basename(csv_nt)}",
    f"  CSV por elem   : {os.path.basename(csv_elem)}",
    f"  Script Python  : {PREFIX}.py",
    "",
    "-" * 70,
    "VARIABLES GLOBALES — GLOBAL_DUPLEX (hebras A + B)",
    "-" * 70,
] + fmt_sasa_block("DUPLEX", glob_duplex) + [
    "-" * 70,
    f"VARIABLES GLOBALES — GLOBAL_CYR (Chain A, WT, {META['CYR']['n_nt']} nt)",
    "-" * 70,
] + fmt_sasa_block("CYR", glob_cyr) + [
    "-" * 70,
    f"VARIABLES GLOBALES — GLOBAL_MIR (Chain B, miR-7-5p, {META['MIR']['n_nt']} nt)",
    "-" * 70,
] + fmt_sasa_block("MIR", glob_mir) + [
    "-" * 70,
    f"VARIABLES POR NUCLEOTIDO — CYR / Chain A ({len(df_nt_cyr)} nt)",
    "-" * 70,
]

for _, row in df_nt_cyr.iterrows():
    report_lines.append(
        f"  NT {int(row['NT_INDEX']):>3} ({row['RESNAME']}) | "
        f"SASA_NT={row['SASA_NT']:.4f} | "
        f"SASA_BB={row['SASA_BB']:.4f} | "
        f"SASA_BASE={row['SASA_BASE']:.4f}  A2"
    )

report_lines += [
    "",
    "-" * 70,
    f"VARIABLES POR NUCLEOTIDO — MIR / Chain B ({len(df_nt_mir)} nt)",
    "-" * 70,
]

for _, row in df_nt_mir.iterrows():
    report_lines.append(
        f"  NT {int(row['NT_INDEX']):>3} ({row['RESNAME']}) | "
        f"SASA_NT={row['SASA_NT']:.4f} | "
        f"SASA_BB={row['SASA_BB']:.4f} | "
        f"SASA_BASE={row['SASA_BASE']:.4f}  A2"
    )

report_lines += [
    "",
    "-" * 70,
    "VARIABLES POR ELEMENTO ESTRUCTURAL — CYR",
    "-" * 70,
]

for _, row in df_elem_cyr.iterrows():
    if row["SASA_NT"] == "NA":
        report_lines += [
            f"  ELEMENTO      : {row['ELEMENT']}  (NT_RANGE=NA, n=0)",
            f"  SASA_NT       : NA",
            f"  SASA_BB       : NA",
            f"  SASA_BASE     : NA",
            f"  MISSING_NT    : NA_by_design",
            "",
        ]
    else:
        report_lines += [
            f"  ELEMENTO      : {row['ELEMENT']}  (nt {row['NT_RANGE']}, n={row['N_NT']})",
            f"  SASA_NT       : {row['SASA_NT']:.4f} A2",
            f"  SASA_BB       : {row['SASA_BB']:.4f} A2",
            f"  SASA_BASE     : {row['SASA_BASE']:.4f} A2",
            f"  MISSING_NT    : {row['MISSING_NT']}",
            "",
        ]

report_lines += [
    "-" * 70,
    "VARIABLES POR ELEMENTO ESTRUCTURAL — MIR",
    "-" * 70,
]

for _, row in df_elem_mir.iterrows():
    if row["SASA_NT"] == "NA":
        report_lines += [
            f"  ELEMENTO      : {row['ELEMENT']}  (NT_RANGE=NA, n=0)",
            f"  SASA_NT       : NA",
            f"  SASA_BB       : NA",
            f"  SASA_BASE     : NA",
            f"  MISSING_NT    : NA_by_design",
            "",
        ]
    else:
        report_lines += [
            f"  ELEMENTO      : {row['ELEMENT']}  (nt {row['NT_RANGE']}, n={row['N_NT']})",
            f"  SASA_NT       : {row['SASA_NT']:.4f} A2",
            f"  SASA_BB       : {row['SASA_BB']:.4f} A2",
            f"  SASA_BASE     : {row['SASA_BASE']:.4f} A2",
            f"  MISSING_NT    : {row['MISSING_NT']}",
            "",
        ]

report_lines += [
    "-" * 70,
    "NOTAS GENERALES",
    "-" * 70,
    "  - Clasificacion BB/BASE basada en nomenclatura PDB estandar para RNA.",
    "  - Atomos BB: P, OP1, OP2, O5', C5', C4', C3', O3', C2', O2', C1', O4'.",
    "  - Todos los valores en A2 (angstroms cuadrados).",
    "  - GLOBAL_DUPLEX: SASA_TOTAL calculado por FreeSASA sobre el duplex completo.",
    "  - GLOBAL_CYR / GLOBAL_MIR: suma de SASA_NT por residuo de cada cadena.",
    "  - OVERHANG_CYR y OVERHANG_MIR reportados como NA (ausentes en WT_miR7).",
    "  - Si MISSING_NT != none, revisar numeracion de residuos en el PDB.",
    "  - PDB generado por AlphaFold3 (AF3) a partir de archivo CIF.",
    "=" * 70,
]

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"[OUTPUT] Reporte .txt  -> {report_path}")

# ─────────────────────────────────────────────
# 11. COPIA DEL SCRIPT EN OUTPUT_DIR
# ─────────────────────────────────────────────

script_src = os.path.abspath(__file__)
script_dst = os.path.join(OUTPUT_DIR, f"{PREFIX}.py")
try:
    shutil.copy2(script_src, script_dst)
    print(f"[OUTPUT] Script .py    -> {script_dst}")
except Exception as e:
    print(f"[WARN] No se pudo copiar el script: {e}")

print("\n[DONE] Todos los archivos generados correctamente.")
