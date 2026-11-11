"""
miR_7_5p_FR3D.py
================
Script de análisis de interacciones nt-nt para la hebra miR-7-5p
Herramienta: FR3D (fr3d-python v0.0.2)
Autor: cmart
Fecha: ver reporte de salida

SECUENCIA miR-7-5p (24 nt):
  5'-UGGAAGACUAGUGAUUUUGUUGUU-3'
     t1  t2-8  t9-10    t11-24
     ANCHOR P1_SEED BULGE  P2

FLUJO:
  1. Configuracion de rutas y parametros
  2. Carga del PDB con load_structure() de fr3d
  3. Anotacion de interacciones nt-nt con annotate_nt_nt_in_structure()
  4. Clasificacion y deduplicacion de interacciones en NC_PAIRS y STACKING
  5. Extraccion de variables globales, por nt y por elemento estructural
  6. Exportacion a CSV (5 archivos)
  7. Generacion de reporte .txt
  8. Copia del script a OUTPUT_DIR

FORMATO UNIT_ID (confirmado en corpus MRE M1-M6):
  'ID|model|chain|resname|resnum'
  campo[4] = numero de residuo (resnum)
  campo[3] = resname (A, C, G, U)

CLASIFICACION FR3D:
  STACKING  : s35, s53, ns35, ns53 (deduplicados: s53->s35, ns53->ns35)
  NC_PAIRS  : basepairs no-canonicos (Leontis-Westhof)
              SE EXCLUYE: cWW Watson-Crick canonico (A-U, G-C, G-U)
  EXCLUIDO  : p_1 (conectividad fosfodiester backbone)

NOTA DEDUPLICACION:
  s35(i,j) y s53(j,i) son la misma interaccion en direcciones opuestas.
  Se registran como un unico par en all_stacking (clave: min,max,itype_base).
  stacking_per_nt mantiene vista bilateral para STACKING_NT por residuo.
  STACKING_TOTAL = pares unicos = 'Found N interactions' de FR3D.

NOTA ASIGNACION POR ELEMENTO:
  Criterio estricto: cada par se asigna al elemento de NT_I (indice menor).
  Garantiza particion sin solapamiento: suma elementos = STACKING_TOTAL.
"""

import os
import sys
import shutil
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. CONFIGURACION
# ─────────────────────────────────────────────

PDB_PATH = os.path.expanduser(
    "~/tesis_rnas/OE3/FreeSASA/MIR/miR_7_5p.pdb"
)

OUTPUT_DIR = os.path.expanduser(
    "~/tesis_rnas/OE3/FR3D/MIR"
)

PREFIX = "miR_7_5p_FR3D"

# Elementos estructurales miR-7-5p (numeracion de residuos en el PDB, base 1)
# t1=ANCHOR | t2-8=P1_SEED | t9-10=BULGE | t11-24=P2_3PRIME_COMP
STRUCTURAL_ELEMENTS = {
    "ANCHOR":         list(range(1, 2)),    # nt 1       (1 nt)
    "P1_SEED":        list(range(2, 9)),    # nt 2-8     (7 nt)
    "BULGE":          list(range(9, 11)),   # nt 9-10    (2 nt)
    "P2_3PRIME_COMP": list(range(11, 25)), # nt 11-24   (14 nt)
}

N_NT_TOTAL = 24

STACKING_PREFIXES = ("s35", "s53", "ns35", "ns53", "s33", "s55", "s3S", "s5S")
EXCLUDED_TYPES    = {"p_1", "p1", "p_2", "p2"}
CANONICAL_PAIRS   = {frozenset(["A", "U"]), frozenset(["G", "C"]), frozenset(["G", "U"])}
RESNUM_FIELD      = 4
RESNAME_FIELD     = 3

# ─────────────────────────────────────────────
# 2. SETUP
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
today = datetime.now().strftime("%d/%m/%Y")

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
print("[INFO] Estructura cargada correctamente")

# ─────────────────────────────────────────────
# 4. ANOTACION DE INTERACCIONES nt-nt
# ─────────────────────────────────────────────

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

print("[INFO] Anotacion completada")
print(f"[INFO] Tipos de interaccion encontrados: {list(interaction_to_list_of_tuples.keys())}")

# ─────────────────────────────────────────────
# 5. FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def parse_unit_id(unit_id):
    """
    Extrae resnum y resname de un unit_id FR3D.
    Formato: 'ID|model|chain|resname|resnum'
    Retorna (int, str) o (None, None).
    """
    parts = str(unit_id).split("|")
    try:
        resnum  = int(parts[RESNUM_FIELD].strip())
        resname = parts[RESNAME_FIELD].strip().upper()
        return resnum, resname
    except (IndexError, ValueError):
        return None, None

def is_canonical_cWW(interaction, resname_i, resname_j):
    if not str(interaction).startswith("cWW"):
        return False
    return frozenset([resname_i, resname_j]) in CANONICAL_PAIRS

# ─────────────────────────────────────────────
# 6. CLASIFICACION Y DEDUPLICACION
# ─────────────────────────────────────────────

seen_stacking   = set()
seen_nc         = set()
nc_pairs_per_nt = defaultdict(set)
stacking_per_nt = defaultdict(set)
all_nc_pairs    = []
all_stacking    = []

for interaction_type, pairs in interaction_to_list_of_tuples.items():
    itype = str(interaction_type).strip()

    if itype in EXCLUDED_TYPES:
        continue

    for pair in pairs:
        try:
            pl = list(pair)
            uid_i, uid_j = str(pl[0]), str(pl[1])
        except Exception:
            continue

        ri, rni = parse_unit_id(uid_i)
        rj, rnj = parse_unit_id(uid_j)

        if ri is None or rj is None:
            continue

        # ── STACKING ──────────────────────────────────────
        if any(itype.startswith(p) for p in STACKING_PREFIXES):
            stacking_per_nt[ri].add((rj, itype))
            stacking_per_nt[rj].add((ri, itype))
            itype_base = itype.replace("s53", "s35").replace("ns53", "ns35")
            key = (min(ri, rj), max(ri, rj), itype_base)
            if key not in seen_stacking:
                seen_stacking.add(key)
                lo, hi = (ri, rj) if ri <= rj else (rj, ri)
                uid_lo = uid_i    if ri <= rj else uid_j
                uid_hi = uid_j    if ri <= rj else uid_i
                all_stacking.append({
                    "NT_I": lo, "UID_I": uid_lo,
                    "NT_J": hi, "UID_J": uid_hi,
                    "INTERACTION": itype_base,
                })

        # ── NC_PAIRS ──────────────────────────────────────
        elif "basepair" in category_to_interactions.get(itype, itype).lower() \
             or (itype[0] in ("c", "t") and len(itype) >= 3):
            if is_canonical_cWW(itype, rni, rnj):
                continue
            nc_pairs_per_nt[ri].add((rj, itype))
            nc_pairs_per_nt[rj].add((ri, itype))
            key = (min(ri, rj), max(ri, rj), itype)
            if key not in seen_nc:
                seen_nc.add(key)
                lo, hi = (ri, rj) if ri <= rj else (rj, ri)
                uid_lo = uid_i    if ri <= rj else uid_j
                uid_hi = uid_j    if ri <= rj else uid_i
                all_nc_pairs.append({
                    "NT_I": lo, "UID_I": uid_lo,
                    "NT_J": hi, "UID_J": uid_hi,
                    "INTERACTION": itype,
                })

# ─────────────────────────────────────────────
# 7. VARIABLES GLOBALES
# ─────────────────────────────────────────────

NC_PAIRS_TOTAL = len(all_nc_pairs)
STACKING_TOTAL = len(all_stacking)

print(f"\n[GLOBAL]")
print(f"  NC_PAIRS_TOTAL : {NC_PAIRS_TOTAL}")
print(f"  STACKING_TOTAL : {STACKING_TOTAL}")

# ─────────────────────────────────────────────
# 8. TABLA POR NUCLEOTIDO
# ─────────────────────────────────────────────

all_residues = sorted(set(
    list(nc_pairs_per_nt.keys()) +
    list(stacking_per_nt.keys()) +
    list(range(1, N_NT_TOTAL + 1))
))

rows_nt = []
for r in sorted(all_residues):
    rows_nt.append({
        "NT_INDEX":       r,
        "NC_PAIRS_NT":    len(nc_pairs_per_nt.get(r, set())),
        "STACKING_NT":    len(stacking_per_nt.get(r, set())),
        "NC_PARTNERS":    ";".join(str(x) for x in sorted(set(
                              p[0] for p in nc_pairs_per_nt.get(r, set())))),
        "STACK_PARTNERS": ";".join(str(x) for x in sorted(set(
                              p[0] for p in stacking_per_nt.get(r, set())))),
    })

df_nt = pd.DataFrame(rows_nt)
print(f"\n[NT TABLE] {len(df_nt)} residuos")
print(df_nt.to_string(index=False))

# ─────────────────────────────────────────────
# 9. TABLA POR ELEMENTO ESTRUCTURAL
# ─────────────────────────────────────────────

rows_elem = []
for en, el in STRUCTURAL_ELEMENTS.items():
    nc_e = [p for p in all_nc_pairs if p["NT_I"] in el]
    st_e = [p for p in all_stacking  if p["NT_I"] in el]
    rows_elem.append({
        "ELEMENT":  en,
        "NT_RANGE": f"nt{el[0]}_to_{el[-1]}",
        "N_NT":     len(el),
        "NC_PAIRS": len(nc_e),
        "STACKING": len(st_e),
    })

df_elem = pd.DataFrame(rows_elem)
print(f"\n[ELEMENT TABLE]")
print(df_elem.to_string(index=False))

# ─────────────────────────────────────────────
# 10. DATAFRAMES DE DETALLE
# ─────────────────────────────────────────────

df_nc = pd.DataFrame(all_nc_pairs) if all_nc_pairs else \
    pd.DataFrame(columns=["NT_I", "UID_I", "NT_J", "UID_J", "INTERACTION"])

df_st = pd.DataFrame(all_stacking) if all_stacking else \
    pd.DataFrame(columns=["NT_I", "UID_I", "NT_J", "UID_J", "INTERACTION"])

df_g = pd.DataFrame([{
    "MOLECULE":       "miR_7_5p",
    "NC_PAIRS_TOTAL": NC_PAIRS_TOTAL,
    "STACKING_TOTAL": STACKING_TOTAL,
}])

# ─────────────────────────────────────────────
# 11. EXPORTACION CSV
# ─────────────────────────────────────────────

csv_global    = os.path.join(OUTPUT_DIR, f"{PREFIX}_global.csv")
csv_nt        = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_nt.csv")
csv_elem      = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_element.csv")
csv_nc_detail = os.path.join(OUTPUT_DIR, f"{PREFIX}_nc_pairs_detail.csv")
csv_st_detail = os.path.join(OUTPUT_DIR, f"{PREFIX}_stacking_detail.csv")

df_g.to_csv(csv_global,     index=False)
df_nt.to_csv(csv_nt,        index=False)
df_elem.to_csv(csv_elem,    index=False)
df_nc.to_csv(csv_nc_detail, index=False)
df_st.to_csv(csv_st_detail, index=False)

print(f"\n[OUTPUT] CSV global          -> {csv_global}")
print(f"[OUTPUT] CSV por nt          -> {csv_nt}")
print(f"[OUTPUT] CSV por elem        -> {csv_elem}")
print(f"[OUTPUT] CSV NC detail       -> {csv_nc_detail}")
print(f"[OUTPUT] CSV stacking detail -> {csv_st_detail}")

# ─────────────────────────────────────────────
# 12. REPORTE .TXT
# ─────────────────────────────────────────────

report_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte.txt")

report_lines = [
    "=" * 70,
    "REPORTE DE ANALISIS INTERACCIONES nt-nt - miR-7-5p",
    "=" * 70,
    "",
    f"HERRAMIENTA      : FR3D (Find RNA 3D) - fr3d-python",
    f"VERSION          : {version_str}",
    f"FECHA            : {today}",
    f"INPUT            : {PDB_PATH}",
    "PARAMETROS       : categories: basepair + stacking (class_limits_2023)",
    "                   NC_PAIRS excluye cWW Watson-Crick canonico (A-U, G-C, G-U)",
    "                   STACKING incluye s35, s53, ns35, ns53",
    "                   EXCLUIDO: p_1 (conectividad fosfodiester backbone)",
    "",
    "SECUENCIA miR-7-5p (24 nt):",
    "  5'-UGGAAGACUAGUGAUUUUGUUGUU-3'",
    "  t1=ANCHOR | t2-8=P1_SEED | t9-10=BULGE | t11-24=P2_3PRIME_COMP",
    "",
    "FORMATO UNIT_ID  : ID|model|chain|resname|resnum",
    "                   resnum = campo[4], resname = campo[3]",
    "",
    "NOTA DEDUPLICACION:",
    "  s35(i,j) y s53(j,i) son la misma interaccion en direcciones opuestas.",
    "  Se registran como un unico par en all_stacking (clave: min,max,itype_base).",
    "  stacking_per_nt mantiene vista bilateral para STACKING_NT por residuo.",
    "  STACKING_TOTAL = pares unicos = 'Found N interactions' de FR3D.",
    "",
    "NOTA ASIGNACION POR ELEMENTO:",
    "  Criterio estricto: cada par se asigna al elemento de NT_I (indice menor).",
    "  Garantiza particion sin solapamiento: suma elementos = STACKING_TOTAL.",
    "",
    "-" * 70,
    "ARCHIVOS DE SALIDA",
    "-" * 70,
    f"  {os.path.basename(csv_global)}",
    f"  {os.path.basename(csv_nt)}",
    f"  {os.path.basename(csv_elem)}",
    f"  {os.path.basename(csv_nc_detail)}",
    f"  {os.path.basename(csv_st_detail)}",
    f"  {PREFIX}_reporte.txt",
    f"  {PREFIX}.py",
    "",
    "-" * 70,
    "VARIABLES EXTRAIDAS - GLOBAL",
    "-" * 70,
    f"  VARIABLE          : NC_PAIRS_TOTAL",
    f"  VALOR             : {NC_PAIRS_TOTAL}",
    f"  NOTAS             : Pares no-canonicos intramoleculares (Leontis-Westhof)",
    "",
    f"  VARIABLE          : STACKING_TOTAL",
    f"  VALOR             : {STACKING_TOTAL}",
    f"  NOTAS             : Pares unicos de apilamiento (s35 + ns35, deduplicados)",
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
    "  - STACKING_NT por nt incluye s35 + s53 + ns35 + ns53 (vista bilateral).",
    "  - Asignacion por elemento: criterio estricto NT_I (donador/menor indice).",
    "    Cada par se asigna a un unico elemento -> suma por elemento = GLOBAL.",
    "  - Analisis intramolecular: interacciones dentro de la hebra miR-7-5p aislada.",
    "  - Para interacciones con MRE usar el duplex completo como INPUT.",
    "=" * 70,
]

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"[OUTPUT] Reporte .txt        -> {report_path}")

# ─────────────────────────────────────────────
# 13. COPIA DEL SCRIPT
# ─────────────────────────────────────────────

script_src = os.path.abspath(__file__)
script_dst = os.path.join(OUTPUT_DIR, f"{PREFIX}.py")

if os.path.abspath(script_src) != os.path.abspath(script_dst):
    try:
        shutil.copy2(script_src, script_dst)
        print(f"[OUTPUT] Script .py          -> {script_dst}")
    except Exception as e:
        print(f"[WARN] No se pudo copiar el script: {e}")
else:
    print(f"[INFO] Script ya en OUTPUT_DIR, copia omitida.")

print("\n[DONE] Todos los archivos generados correctamente.")
