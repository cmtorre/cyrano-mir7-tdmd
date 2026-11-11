"""
CYR_MRE_M6_FreeSASA.py
======================
Script de análisis SASA para la hebra MRE_Cyrano_M6
Herramienta: FreeSASA (https://freesasa.github.io/)
Autor: cmart
Fecha: ver reporte de salida

SECUENCIA (16 nt): AACAACAAAAUCACCA
ELEMENTOS:
  ANCHOR    : t1
  P1/SEED   : t2–3
  BULGE     : N/A (ausente en esta variante)
  P2/3'COMP : t4–16

FLUJO:
  1. Configuración de rutas y parámetros
  2. Carga del PDB con FreeSASA
  3. Extracción de variables globales (SASA_TOTAL, SASA_BB, SASA_BASES)
  4. Extracción por nucleótido (16 nt): SASA_NT, SASA_BB, SASA_BASE
  5. Agregación por elemento estructural (ANCHOR, P1/SEED, P2/3'COMP)
  6. Exportación a CSV
  7. Generación de reporte .txt
"""

import freesasa
import pandas as pd
import os
import sys
from datetime import datetime

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

PDB_PATH = os.path.expanduser(
    "~/tesis_rnas/OE-2/variables-est-terciarias/"
    "FreeSASA/7.MRE_CYR_M6/MRE_CYR_M6.pdb"
)

OUTPUT_DIR = os.path.expanduser(
    "~/tesis_rnas/OE-2/variables-est-terciarias/"
    "FreeSASA/7.MRE_CYR_M6"
)

PREFIX = "CYR_MRE_M6_FreeSASA"

# Definición de elementos estructurales (índices de residuo, base 1)
# BULGE ausente en M6: no se define entrada para este elemento
STRUCTURAL_ELEMENTS = {
    "ANCHOR":         [1],                  # t1
    "P1_SEED":        list(range(2, 4)),    # t2–3
    "P2_3PRIME_COMP": list(range(4, 17)),  # t4–16
}

# Átomos de backbone en RNA (convención PDB estándar)
BACKBONE_ATOMS = {
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'",
    "C2'", "O2'", "C1'", "O4'"
}

# ─────────────────────────────────────────────
# 2. SETUP DE SALIDA
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
today = datetime.now().strftime("%d/%m/%Y")

# ─────────────────────────────────────────────
# 3. CARGA DEL PDB Y CÁLCULO SASA
# ─────────────────────────────────────────────

print(f"[INFO] Cargando PDB: {PDB_PATH}")
if not os.path.isfile(PDB_PATH):
    sys.exit(f"[ERROR] Archivo PDB no encontrado: {PDB_PATH}")

# FreeSASA con parámetros default (Lee & Richards, probe_radius=1.4 Å, n_points=100)
structure = freesasa.Structure(PDB_PATH)
result    = freesasa.calc(structure)

# Obtención de versión compatible con instalación vía bioconda
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
        version_str = "2.2.1"  # versión confirmada en este entorno

print(f"[INFO] FreeSASA v{version_str} — cálculo completado")

# ─────────────────────────────────────────────
# 4. CLASIFICACIÓN DE ÁTOMOS POR RESIDUO
# ─────────────────────────────────────────────

def classify_atoms(structure, result):
    residues = {}
    n_atoms = structure.nAtoms()
    for i in range(n_atoms):
        chain   = structure.chainLabel(i)
        resnum  = structure.residueNumber(i)
        resname = structure.residueName(i)
        atom    = structure.atomName(i).strip()
        sasa_i  = result.atomArea(i)

        try:
            resnum_int = int(resnum.strip())
        except ValueError:
            continue

        if resnum_int not in residues:
            residues[resnum_int] = {
                "chain":     chain,
                "resname":   resname.strip(),
                "SASA_NT":   0.0,
                "SASA_BB":   0.0,
                "SASA_BASE": 0.0,
            }

        residues[resnum_int]["SASA_NT"] += sasa_i
        if atom in BACKBONE_ATOMS:
            residues[resnum_int]["SASA_BB"]   += sasa_i
        else:
            residues[resnum_int]["SASA_BASE"] += sasa_i

    return residues

residue_data = classify_atoms(structure, result)

# ─────────────────────────────────────────────
# 5. VARIABLES GLOBALES
# ─────────────────────────────────────────────

SASA_TOTAL = result.totalArea()
SASA_BB    = sum(v["SASA_BB"]   for v in residue_data.values())
SASA_BASES = sum(v["SASA_BASE"] for v in residue_data.values())

print(f"\n[GLOBAL]")
print(f"  SASA_TOTAL : {SASA_TOTAL:.4f} A2")
print(f"  SASA_BB    : {SASA_BB:.4f} A2")
print(f"  SASA_BASES : {SASA_BASES:.4f} A2")

# ─────────────────────────────────────────────
# 6. TABLA POR NUCLEÓTIDO (16 nt)
# ─────────────────────────────────────────────

rows_nt = []
for resnum in sorted(residue_data.keys()):
    v = residue_data[resnum]
    rows_nt.append({
        "NT_INDEX":  resnum,
        "CHAIN":     v["chain"],
        "RESNAME":   v["resname"],
        "SASA_NT":   round(v["SASA_NT"],   4),
        "SASA_BB":   round(v["SASA_BB"],   4),
        "SASA_BASE": round(v["SASA_BASE"], 4),
    })

df_nt = pd.DataFrame(rows_nt)
print(f"\n[NT TABLE] {len(df_nt)} residuos encontrados")
print(df_nt.to_string(index=False))

# ─────────────────────────────────────────────
# 7. TABLA POR ELEMENTO ESTRUCTURAL
# ─────────────────────────────────────────────

rows_elem = []
for elem_name, nt_list in STRUCTURAL_ELEMENTS.items():
    nt_in_pdb = [n for n in nt_list if n in residue_data]
    missing   = [n for n in nt_list if n not in residue_data]

    sasa_nt   = sum(residue_data[n]["SASA_NT"]   for n in nt_in_pdb)
    sasa_bb   = sum(residue_data[n]["SASA_BB"]   for n in nt_in_pdb)
    sasa_base = sum(residue_data[n]["SASA_BASE"] for n in nt_in_pdb)

    rows_elem.append({
        "ELEMENT":     elem_name,
        "NT_RANGE":    f"{nt_list[0]}-{nt_list[-1]}",
        "N_NT":        len(nt_in_pdb),
        "SASA_NT":     round(sasa_nt,   4),
        "SASA_BB":     round(sasa_bb,   4),
        "SASA_BASE":   round(sasa_base, 4),
        "MISSING_NT":  str(missing) if missing else "none",
    })
    if missing:
        print(f"[WARN] {elem_name}: nt no encontrados en PDB: {missing}")

df_elem = pd.DataFrame(rows_elem)
print(f"\n[ELEMENT TABLE]")
print(df_elem.to_string(index=False))

# ─────────────────────────────────────────────
# 8. TABLA GLOBAL (una fila)
# ─────────────────────────────────────────────

df_global = pd.DataFrame([{
    "MOLECULE":    "MRE_CYR_M6",
    "SASA_TOTAL":  round(SASA_TOTAL, 4),
    "SASA_BB":     round(SASA_BB,    4),
    "SASA_BASES":  round(SASA_BASES, 4),
}])

# ─────────────────────────────────────────────
# 9. EXPORTACIÓN CSV
# ─────────────────────────────────────────────

csv_global = os.path.join(OUTPUT_DIR, f"{PREFIX}_global.csv")
csv_nt     = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_nt.csv")
csv_elem   = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_element.csv")

df_global.to_csv(csv_global, index=False)
df_nt.to_csv(csv_nt,         index=False)
df_elem.to_csv(csv_elem,     index=False)

print(f"\n[OUTPUT] CSV global    -> {csv_global}")
print(f"[OUTPUT] CSV por nt    -> {csv_nt}")
print(f"[OUTPUT] CSV por elem  -> {csv_elem}")

# ─────────────────────────────────────────────
# 10. REPORTE .TXT
# ─────────────────────────────────────────────

report_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte.txt")

report_lines = [
    "=" * 70,
    "REPORTE DE ANALISIS SASA - MRE_CYR_M6",
    "=" * 70,
    "",
    f"HERRAMIENTA      : FreeSASA",
    f"VERSION          : {version_str}",
    f"FECHA            : {today}",
    f"INPUT            : {PDB_PATH}",
    "PARAMETROS       : default (algoritmo Lee & Richards, probe_radius=1.4 A, n_points=100)",
    "",
    "SECUENCIA (16 nt): AACAACAAAAUCACCA",
    "ELEMENTOS        : ANCHOR(t1) | P1_SEED(t2-3) | BULGE(N/A) | P2_3PRIME_COMP(t4-16)",
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
    "VARIABLES EXTRAIDAS - GLOBAL",
    "-" * 70,
    f"  VARIABLE_EXTRAIDA : SASA_TOTAL",
    f"  VALOR             : {SASA_TOTAL:.4f} A2",
    f"  NOTAS             : Area superficial total accesible al solvente",
    "",
    f"  VARIABLE_EXTRAIDA : SASA_BB",
    f"  VALOR             : {SASA_BB:.4f} A2",
    f"  NOTAS             : Contribucion de atomos de backbone (P, OP1, OP2, azucares)",
    "",
    f"  VARIABLE_EXTRAIDA : SASA_BASES",
    f"  VALOR             : {SASA_BASES:.4f} A2",
    f"  NOTAS             : Contribucion de atomos de bases nitrogenadas",
    "",
    "-" * 70,
    "VARIABLES EXTRAIDAS - POR NUCLEOTIDO (16 nt)",
    "-" * 70,
]

for _, row in df_nt.iterrows():
    report_lines += [
        f"  NT {int(row['NT_INDEX']):>3} ({row['RESNAME']}) | "
        f"SASA_NT={row['SASA_NT']:.4f} | "
        f"SASA_BB={row['SASA_BB']:.4f} | "
        f"SASA_BASE={row['SASA_BASE']:.4f}  A2"
    ]

report_lines += [
    "",
    "-" * 70,
    "VARIABLES EXTRAIDAS - POR ELEMENTO ESTRUCTURAL",
    "-" * 70,
    "  NOTA: Elemento BULGE ausente en variante M6.",
    "",
]

for _, row in df_elem.iterrows():
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
    "  - Si MISSING_NT != none, revisar numeracion de residuos en el PDB.",
    "  - BULGE no presente en M6: elemento excluido del analisis por diseno experimental.",
    "=" * 70,
]

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"[OUTPUT] Reporte .txt  -> {report_path}")

# ─────────────────────────────────────────────
# 11. GUARDAR COPIA DEL SCRIPT EN OUTPUT_DIR
# ─────────────────────────────────────────────

import shutil
script_src = os.path.abspath(__file__)
script_dst = os.path.join(OUTPUT_DIR, f"{PREFIX}.py")
try:
    shutil.copy2(script_src, script_dst)
    print(f"[OUTPUT] Script .py    -> {script_dst}")
except Exception as e:
    print(f"[WARN] No se pudo copiar el script: {e}")

print("\n[DONE] Todos los archivos generados correctamente.")
