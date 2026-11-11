"""
CYR_MRE_WT_FR3D.py
==================
Script de análisis de interacciones nt-nt para la hebra MRE_Cyrano_WT
Herramienta: FR3D (fr3d-python v0.0.2)
Autor: cmart
Fecha: ver reporte de salida

FLUJO:
  1. Configuracion de rutas y parametros
  2. Carga del PDB con load_structure() de fr3d
  3. Anotacion de interacciones nt-nt con annotate_nt_nt_in_structure()
  4. Clasificacion de interacciones en NC_PAIRS y STACKING
  5. Extraccion de variables globales, por nt y por elemento estructural
  6. Exportacion a CSV
  7. Generacion de reporte .txt
  8. Copia del script a OUTPUT_DIR

CLASIFICACION FR3D:
  NC_PAIRS : interacciones de tipo basepair no-canonico
             categorias FR3D: cWW, tWW, cWH, tWH, cWS, tWS,
                              cHH, tHH, cHS, tHS, cSS, tSS
             SE EXCLUYE: cWW Watson-Crick canonico (A-U, G-C, G-U wobble)
  STACKING : interacciones de apilamiento
             categorias FR3D: s35, s53, s33, s55
"""

import os
import sys
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. CONFIGURACION
# ─────────────────────────────────────────────

PDB_PATH = os.path.expanduser(
    "~/tesis_rnas/OE-2-variables-estructurales/variables-est-terciarias/"
    "ChimeraX/MRE_CYR_WT/MRE_Cyrano_WT.pdb"
)

OUTPUT_DIR = os.path.expanduser(
    "~/tesis_rnas/OE-2-variables-estructurales/variables-est-terciarias/"
    "FR3D/MRE_CYR_WT"
)

PREFIX = "CYR_MRE_WT_FR3D"

# Elementos estructurales (numeracion de residuos en el PDB, base 1)
STRUCTURAL_ELEMENTS = {
    "P1_SEED":        list(range(2, 9)),    # nt 2-8
    "BULGE":          list(range(9, 13)),   # nt 9-12
    "P2_3PRIME_COMP": list(range(13, 27)),  # nt 13-26
}

# Prefijos de interaccion FR3D que se clasifican como STACKING
STACKING_PREFIXES = ("s35", "s53", "s33", "s55", "s3S", "s5S")

# Pares Watson-Crick canonicos que se EXCLUYEN de NC_PAIRS
# cWW entre A-U, G-C, G-U (wobble) son canonicos
CANONICAL_PAIRS = {frozenset(["A", "U"]), frozenset(["G", "C"]), frozenset(["G", "U"])}

# ─────────────────────────────────────────────
# 2. SETUP
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
today = datetime.now().strftime("%d/%m/%Y")

# Version
try:
    import importlib.metadata
    version_str = importlib.metadata.version("fr3d")
except Exception:
    version_str = "0.0.2"

# ─────────────────────────────────────────────
# 3. CARGA DE ESTRUCTURA
# ─────────────────────────────────────────────

from fr3d.classifiers.NA_pairwise_interactions import (
    load_structure,
    annotate_nt_nt_in_structure,
    nt_nt_cutoffs,
    focus_basepair_cutoffs,
    load_ideal_basepair_hydrogen_bonds,
)

print(f"[INFO] Cargando PDB: {PDB_PATH}")
if not os.path.isfile(PDB_PATH):
    sys.exit(f"[ERROR] Archivo PDB no encontrado: {PDB_PATH}")

structure, messages = load_structure(PDB_PATH)
if structure is None:
    sys.exit(f"[ERROR] load_structure fallo: {messages}")
for m in messages:
    print(f"[FR3D MSG] {m}")
print(f"[INFO] Estructura cargada correctamente")

# ─────────────────────────────────────────────
# 4. ANOTACION DE INTERACCIONES nt-nt
# ─────────────────────────────────────────────

# categories define que tipos de interacciones buscar
# 'basepair' activa pares de bases (canonicos + no-canonicos)
# 'stacking' activa apilamiento
# 'sO' activa interacciones base-oxigeno (no requeridas aqui)
categories = {
    "basepair": "basepair",
    "stacking": "stacking",
    "sO": "",
    "bph": "",
    "br": "",
    "coplanar": "",
}

print(f"[INFO] Anotando interacciones nt-nt (FR3D v{version_str})...")

interaction_to_list_of_tuples, category_to_interactions, timerData, pair_to_data = \
    annotate_nt_nt_in_structure(structure, categories)

print(f"[INFO] Anotacion completada")
print(f"[INFO] Tipos de interaccion encontrados: {list(interaction_to_list_of_tuples.keys())}")

# ─────────────────────────────────────────────
# 5. EXTRACCION Y CLASIFICACION
# ─────────────────────────────────────────────

def parse_unit_id(unit_id):
    """
    Extrae el numero de residuo de un unit_id de FR3D.
    Formato tipico: 'filename|chain|resnum|resname' o variantes.
    Retorna int o None.
    """
    parts = str(unit_id).split("|")
    for part in parts:
        try:
            return int(part.strip())
        except ValueError:
            continue
    return None

def is_canonical_cWW(interaction, resname_i, resname_j):
    """
    Retorna True si el par es cWW Watson-Crick canonico (excluir de NC_PAIRS).
    """
    if not str(interaction).startswith("cWW"):
        return False
    pair = frozenset([resname_i.strip().upper(), resname_j.strip().upper()])
    return pair in CANONICAL_PAIRS

def get_resname(structure, resnum):
    """Obtiene resname de un residuo por numero."""
    try:
        for res in structure.residues():
            rn = res.unit_id().split("|")[2] if "|" in res.unit_id() else None
            if rn and int(rn) == resnum:
                return res.sequence
    except Exception:
        pass
    return "X"

# Acumuladores por residuo
nc_pairs_per_nt   = defaultdict(set)   # resnum -> set de (partner, interaction)
stacking_per_nt   = defaultdict(set)   # resnum -> set de (partner, interaction)

# Listas de todas las interacciones clasificadas
all_nc_pairs   = []
all_stacking   = []

for interaction_type, pairs in interaction_to_list_of_tuples.items():
    itype = str(interaction_type).strip()

    for pair in pairs:
        # pair es una tupla (unit_id_i, unit_id_j) o similar
        try:
            if hasattr(pair, '__iter__') and not isinstance(pair, str):
                pair_list = list(pair)
                uid_i = str(pair_list[0])
                uid_j = str(pair_list[1])
            else:
                continue
        except Exception:
            continue

        resnum_i = parse_unit_id(uid_i)
        resnum_j = parse_unit_id(uid_j)

        if resnum_i is None or resnum_j is None:
            continue

        # Clasificar STACKING
        if any(itype.startswith(p) for p in STACKING_PREFIXES):
            stacking_per_nt[resnum_i].add((resnum_j, itype))
            stacking_per_nt[resnum_j].add((resnum_i, itype))
            all_stacking.append({
                "NT_I":        resnum_i,
                "UID_I":       uid_i,
                "NT_J":        resnum_j,
                "UID_J":       uid_j,
                "INTERACTION": itype,
            })

        # Clasificar NC_PAIRS (basepairs no-canonicos)
        elif "basepair" in category_to_interactions.get(itype, itype).lower() \
             or itype[0] in ("c", "t") and len(itype) >= 3:
            # Verificar si es canonico para excluirlo
            # Obtenemos resnames de los unit_ids
            parts_i = uid_i.split("|")
            parts_j = uid_j.split("|")
            resname_i = parts_i[-1] if parts_i else "X"
            resname_j = parts_j[-1] if parts_j else "X"

            if is_canonical_cWW(itype, resname_i, resname_j):
                continue  # excluir Watson-Crick canonico

            nc_pairs_per_nt[resnum_i].add((resnum_j, itype))
            nc_pairs_per_nt[resnum_j].add((resnum_i, itype))
            all_nc_pairs.append({
                "NT_I":        resnum_i,
                "UID_I":       uid_i,
                "NT_J":        resnum_j,
                "UID_J":       uid_j,
                "INTERACTION": itype,
            })

# ─────────────────────────────────────────────
# 6. VARIABLES GLOBALES
# ─────────────────────────────────────────────

NC_PAIRS_TOTAL = len(all_nc_pairs)
STACKING_TOTAL = len(all_stacking)

print(f"\n[GLOBAL]")
print(f"  NC_PAIRS_TOTAL : {NC_PAIRS_TOTAL}")
print(f"  STACKING_TOTAL : {STACKING_TOTAL}")

# ─────────────────────────────────────────────
# 7. TABLA POR NUCLEOTIDO
# ─────────────────────────────────────────────

# Obtener todos los residuos de la estructura
all_residues = sorted(set(
    list(nc_pairs_per_nt.keys()) +
    list(stacking_per_nt.keys()) +
    list(range(1, 27))  # garantizar los 26 nt aunque tengan 0 interacciones
))

rows_nt = []
for resnum in sorted(all_residues):
    rows_nt.append({
        "NT_INDEX":      resnum,
        "NC_PAIRS_NT":   len(nc_pairs_per_nt.get(resnum, set())),
        "STACKING_NT":   len(stacking_per_nt.get(resnum, set())),
        "NC_PARTNERS":   ";".join(str(p[0]) for p in nc_pairs_per_nt.get(resnum, set())),
        "STACK_PARTNERS": ";".join(str(p[0]) for p in stacking_per_nt.get(resnum, set())),
    })

df_nt = pd.DataFrame(rows_nt)
print(f"\n[NT TABLE] {len(df_nt)} residuos")
print(df_nt.to_string(index=False))

# ─────────────────────────────────────────────
# 8. TABLA POR ELEMENTO ESTRUCTURAL
# ─────────────────────────────────────────────

rows_elem = []
for elem_name, nt_list in STRUCTURAL_ELEMENTS.items():
    # Pares que involucran AL MENOS UN nt del elemento
    nc_elem = [p for p in all_nc_pairs
               if p["NT_I"] in nt_list or p["NT_J"] in nt_list]
    st_elem = [p for p in all_stacking
               if p["NT_I"] in nt_list or p["NT_J"] in nt_list]

    rows_elem.append({
        "ELEMENT":   elem_name,
        "NT_RANGE":  f"{nt_list[0]}-{nt_list[-1]}",
        "N_NT":      len(nt_list),
        "NC_PAIRS":  len(nc_elem),
        "STACKING":  len(st_elem),
    })

df_elem = pd.DataFrame(rows_elem)
print(f"\n[ELEMENT TABLE]")
print(df_elem.to_string(index=False))

# ─────────────────────────────────────────────
# 9. TABLA DETALLE DE INTERACCIONES
# ─────────────────────────────────────────────

df_nc_detail = pd.DataFrame(all_nc_pairs) if all_nc_pairs else \
    pd.DataFrame(columns=["NT_I","UID_I","NT_J","UID_J","INTERACTION"])

df_st_detail = pd.DataFrame(all_stacking) if all_stacking else \
    pd.DataFrame(columns=["NT_I","UID_I","NT_J","UID_J","INTERACTION"])

df_global = pd.DataFrame([{
    "MOLECULE":       "MRE_CYR_WT",
    "NC_PAIRS_TOTAL": NC_PAIRS_TOTAL,
    "STACKING_TOTAL": STACKING_TOTAL,
}])

# ─────────────────────────────────────────────
# 10. EXPORTACION CSV
# ─────────────────────────────────────────────

csv_global    = os.path.join(OUTPUT_DIR, f"{PREFIX}_global.csv")
csv_nt        = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_nt.csv")
csv_elem      = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_element.csv")
csv_nc_detail = os.path.join(OUTPUT_DIR, f"{PREFIX}_nc_pairs_detail.csv")
csv_st_detail = os.path.join(OUTPUT_DIR, f"{PREFIX}_stacking_detail.csv")

df_global.to_csv(csv_global,       index=False)
df_nt.to_csv(csv_nt,               index=False)
df_elem.to_csv(csv_elem,           index=False)
df_nc_detail.to_csv(csv_nc_detail, index=False)
df_st_detail.to_csv(csv_st_detail, index=False)

print(f"\n[OUTPUT] CSV global         -> {csv_global}")
print(f"[OUTPUT] CSV por nt         -> {csv_nt}")
print(f"[OUTPUT] CSV por elem       -> {csv_elem}")
print(f"[OUTPUT] CSV NC detail      -> {csv_nc_detail}")
print(f"[OUTPUT] CSV stacking detail-> {csv_st_detail}")

# ─────────────────────────────────────────────
# 11. REPORTE .TXT
# ─────────────────────────────────────────────

report_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte.txt")

report_lines = [
    "=" * 70,
    "REPORTE DE ANALISIS INTERACCIONES nt-nt - MRE_CYR_WT",
    "=" * 70,
    "",
    f"HERRAMIENTA      : FR3D (Find RNA 3D) - fr3d-python",
    f"VERSION          : {version_str}",
    f"FECHA            : {today}",
    f"INPUT            : {PDB_PATH}",
    "PARAMETROS       : default (categories: basepair + stacking; class_limits_2023)",
    "                   NC_PAIRS excluye cWW Watson-Crick canonico (A-U, G-C, G-U)",
    "",
    "-" * 70,
    "ARCHIVOS DE SALIDA",
    "-" * 70,
    f"  CSV global          : {os.path.basename(csv_global)}",
    f"  CSV por nt          : {os.path.basename(csv_nt)}",
    f"  CSV por elem        : {os.path.basename(csv_elem)}",
    f"  CSV NC detail       : {os.path.basename(csv_nc_detail)}",
    f"  CSV stacking detail : {os.path.basename(csv_st_detail)}",
    f"  Script Python       : {PREFIX}.py",
    "",
    "-" * 70,
    "VARIABLES EXTRAIDAS - GLOBAL",
    "-" * 70,
    f"  VARIABLE_EXTRAIDA : NC_PAIRS_TOTAL",
    f"  VALOR             : {NC_PAIRS_TOTAL}",
    f"  NOTAS             : Pares no-canonicos intramoleculares (Leontis-Westhof)",
    "",
    f"  VARIABLE_EXTRAIDA : STACKING_TOTAL",
    f"  VALOR             : {STACKING_TOTAL}",
    f"  NOTAS             : Interacciones de apilamiento (s35/s53/s33/s55)",
    "",
    "-" * 70,
    "VARIABLES EXTRAIDAS - POR NUCLEOTIDO",
    "-" * 70,
]

for _, row in df_nt.iterrows():
    report_lines.append(
        f"  NT {int(row['NT_INDEX']):>3} | "
        f"NC_PAIRS={int(row['NC_PAIRS_NT'])} | "
        f"STACKING={int(row['STACKING_NT'])} | "
        f"NC_partners=[{row['NC_PARTNERS']}] | "
        f"stack_partners=[{row['STACK_PARTNERS']}]"
    )

report_lines += [
    "",
    "-" * 70,
    "VARIABLES EXTRAIDAS - POR ELEMENTO ESTRUCTURAL",
    "-" * 70,
]

for _, row in df_elem.iterrows():
    report_lines += [
        f"  ELEMENTO  : {row['ELEMENT']}  (nt {row['NT_RANGE']}, n={row['N_NT']})",
        f"  NC_PAIRS  : {row['NC_PAIRS']}",
        f"  STACKING  : {row['STACKING']}",
        "",
    ]

report_lines += [
    "-" * 70,
    "NOTAS GENERALES",
    "-" * 70,
    "  - Interacciones clasificadas segun geometria Leontis-Westhof.",
    "  - NC_PAIRS incluye todos los pares no-WC (cWH, tWH, cWS, tWS, etc.).",
    "  - STACKING incluye s35, s53, s33, s55.",
    "  - Counts por elemento incluyen pares donde AL MENOS un nt pertenece al elemento.",
    "  - Analisis intramolecular: solo interacciones dentro de la hebra MRE aislada.",
    "  - Para interacciones con miR-7-5p usar el duplex completo como INPUT.",
    "=" * 70,
]

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"[OUTPUT] Reporte .txt       -> {report_path}")

# ─────────────────────────────────────────────
# 12. COPIA DEL SCRIPT
# ─────────────────────────────────────────────

import shutil
script_src = os.path.abspath(__file__)
script_dst = os.path.join(OUTPUT_DIR, f"{PREFIX}.py")
try:
    shutil.copy2(script_src, script_dst)
    print(f"[OUTPUT] Script .py         -> {script_dst}")
except Exception as e:
    print(f"[WARN] No se pudo copiar el script: {e}")

print("\n[DONE] Todos los archivos generados correctamente.")
