"""
WT_MIR7_FR3D_STACKING.py
===
Script de extracción de STACKING para el dúplex WT_miR7
Herramienta: FR3D (fr3d-python v0.0.2)
Autor: cmart

NOTA: annotate_nt_nt_in_structure requiere 'basepair' en categories
      aunque no se use — se incluye para evitar KeyError.
      Los basepairs detectados se ignoran en el procesamiento.

ESTRUCTURA:
  Hebra A — CYR_WT (26 nt): 5'-AACAACAAAAUCACCAAUGUCUUCCA-3'
    ANCHOR_CYR: nt26 | P1_CYR: nt19-25 | B_CYR: nt15-18 | P2_CYR: nt1-14
  Hebra B — miR-7-5p (24 nt): 5'-UGGAAGACUAGUGAUUUUGUUGUU-3'
    ANCHOR_MIR: nt1 | P1_MIR: nt2-8 | B_MIR: nt9-10 | P2_MIR: nt11-24

TIPOS DE STACKING FR3D (verificados en debug):
  s35, s53 → normalizado a s35
  s55, ns55, ns33
  Deduplicacion: par único por clave (min,max) + tipo normalizado

OUTPUTS:
  _per_nt.csv      : una fila por nucleotido
  _per_region.csv  : una fila por region estructural
  _global.csv      : GLOBAL_DUPLEX, GLOBAL_CYR, GLOBAL_MIR, INTER_AB
  _reporte.txt
  _stacking_pairs.csv : todos los pares únicos con ORIGIN (detalle)
"""

import os, sys, shutil, csv
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. CONFIGURACION
# ─────────────────────────────────────────────

PDB_PATH   = os.path.expanduser(
    "~/tesis_rnas/OE4/1.FreeSASA/1.WT_MIR/WT_miR7_HBonds.pdb")
OUTPUT_DIR = os.path.expanduser(
    "~/tesis_rnas/OE4/2.FR3D/1.WT_MIR")
PREFIX     = "WT_MIR7_FR3D_STACKING"

SEQ = {
    "A": {i+1: b for i, b in enumerate("AACAACAAAAUCACCAAUGUCUUCCA")},
    "B": {i+1: b for i, b in enumerate("UGGAAGACUAGUGAUUUUGUUGUU")},
}

STRAND_MAP = {"A": "CYR", "B": "MIR"}

REGION_MAP = {
    "A": {
        "ANCHOR_CYR"   : list(range(26, 27)),
        "P1_CYR"       : list(range(19, 26)),
        "B_CYR"        : list(range(15, 19)),
        "P2_CYR"       : list(range(1,  15)),
        "OVERHANG_CYR" : None,
    },
    "B": {
        "ANCHOR_MIR"   : list(range(1,  2)),
        "P1_MIR"       : list(range(2,  9)),
        "B_MIR"        : list(range(9,  11)),
        "P2_MIR"       : list(range(11, 25)),
        "OVERHANG_MIR" : None,
    },
}

N_NT = {"A": 26, "B": 24}

# Tipos de stacking — s53 normalizado a s35
STACK_NORM = {
    "s35" : "s35",
    "s53" : "s35",
    "s55" : "s55",
    "ns55": "ns55",
    "ns33": "ns33",
}
STACK_TYPES_OUT = ["s35", "s55", "ns55", "ns33"]
EXCLUDED        = {"p_1","p1","p_2","p2"}

RESNUM_FIELD  = 4
RESNAME_FIELD = 3
CHAIN_FIELD   = 2

# ─────────────────────────────────────────────
# 2. SETUP
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
today = datetime.now().strftime("%d/%m/%Y")

# ─────────────────────────────────────────────
# 3. CARGA
# ─────────────────────────────────────────────

from fr3d.classifiers.NA_pairwise_interactions import (
    load_structure, annotate_nt_nt_in_structure)

print(f"[INFO] PDB: {PDB_PATH}")
if not os.path.isfile(PDB_PATH):
    sys.exit(f"[ERROR] PDB no encontrado: {PDB_PATH}")

h_count = sum(
    1 for line in open(PDB_PATH)
    if line.startswith("ATOM") and line[12:16].strip().startswith("H")
)
print(f"[INFO] Atomos H: {h_count}")

structure, messages = load_structure(PDB_PATH)
if structure is None:
    sys.exit(f"[ERROR] load_structure: {messages}")
for m in messages:
    print(f"[FR3D] {m}")
print("[OK] Estructura cargada")

# ─────────────────────────────────────────────
# 4. ANOTACION
# FR3D requiere 'basepair' en categories — se incluye pero se ignora
# ─────────────────────────────────────────────

categories = {"basepair": "basepair", "stacking": "stacking"}
print("[INFO] Anotando...")
i2l, cat2i, timer, p2d = annotate_nt_nt_in_structure(structure, categories)
print(f"[OK] Tipos encontrados: {list(i2l.keys())}")

# Filtrar solo tipos de stacking conocidos
stack_types_found = [t for t in i2l.keys() if str(t).strip() in STACK_NORM]
print(f"[OK] Tipos stacking: {stack_types_found}")

# ─────────────────────────────────────────────
# 5. FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def parse_unit_id(uid):
    parts = str(uid).split("|")
    try:
        resnum  = int(parts[RESNUM_FIELD].strip())
        resname = parts[RESNAME_FIELD].strip().upper()
        chain   = parts[CHAIN_FIELD].strip().upper()
        return resnum, resname, chain
    except (IndexError, ValueError):
        return None, None, None

def get_region(chain, resnum):
    for region, residues in REGION_MAP.get(chain, {}).items():
        if residues and resnum in residues:
            return region
    return "UNASSIGNED"

def origin_tag(ci, cj):
    return f"intra_{ci}" if ci == cj else "inter_AB"

# ─────────────────────────────────────────────
# 6. EXTRACCION Y DEDUPLICACION
# ─────────────────────────────────────────────

stacking_per_nt = defaultdict(lambda: defaultdict(list))
seen_pairs      = set()
all_stacking    = []

for itype, pairs in i2l.items():
    raw  = str(itype).strip()
    norm = STACK_NORM.get(raw)
    if norm is None:
        continue

    for pair in pairs:
        try:
            pl    = list(pair)
            uid_i = str(pl[0])
            uid_j = str(pl[1])
        except Exception:
            continue

        ri, rni, ci = parse_unit_id(uid_i)
        rj, rnj, cj = parse_unit_id(uid_j)
        if ri is None or rj is None:
            continue

        key_i = (ci, ri)
        key_j = (cj, rj)

        lo, hi = sorted([key_i, key_j])
        dedup  = (lo, hi, norm)
        if dedup in seen_pairs:
            continue
        seen_pairs.add(dedup)

        origin = origin_tag(ci, cj)

        all_stacking.append({
            "CHAIN_I" : lo[0], "NT_I": lo[1],
            "CHAIN_J" : hi[0], "NT_J": hi[1],
            "STACK_TYPE": norm,
            "ORIGIN"  : origin,
        })

        # Bilateral para conteos por nt
        stacking_per_nt[key_i][key_j].append(norm)
        stacking_per_nt[key_j][key_i].append(norm)

print(f"[OK] Pares únicos de stacking: {len(all_stacking)}")

# ─────────────────────────────────────────────
# 7. TABLA POR NUCLEOTIDO
# ─────────────────────────────────────────────

nt_rows = []
for chain in ["A", "B"]:
    strand = STRAND_MAP[chain]
    for resnum in range(1, N_NT[chain] + 1):
        key     = (chain, resnum)
        resname = SEQ[chain].get(resnum, "?")
        region  = get_region(chain, resnum)
        partners = stacking_per_nt.get(key, {})

        st_intra = sum(
            len(types) for pk, types in partners.items()
            if pk[0] == chain)
        st_inter = sum(
            len(types) for pk, types in partners.items()
            if pk[0] != chain)

        type_counts = defaultdict(int)
        for types in partners.values():
            for t in types:
                type_counts[t] += 1

        partner_str = ";".join(
            f"{pk[0]}{pk[1]}" for pk in sorted(partners.keys())
        ) if partners else ""

        nt_rows.append({
            "STRAND"        : strand,
            "CHAIN"         : chain,
            "NT_INDEX"      : resnum,
            "RESNAME"       : resname,
            "REGION"        : region,
            "ST_TOTAL"      : st_intra + st_inter,
            "ST_INTRA"      : st_intra,
            "ST_INTER"      : st_inter,
            **{t: type_counts.get(t, 0) for t in STACK_TYPES_OUT},
            "STACK_PARTNERS": partner_str,
        })

df_nt = pd.DataFrame(nt_rows)

# ─────────────────────────────────────────────
# 8. TABLA POR REGION
# ─────────────────────────────────────────────

region_rows = []
for chain in ["A", "B"]:
    strand = STRAND_MAP[chain]
    for region, residues in REGION_MAP[chain].items():
        if residues is None:
            region_rows.append({
                "STRAND"  : strand, "CHAIN": chain,
                "REGION"  : region, "NT_RANGE": "NA", "N_NT": 0,
                "ST_TOTAL": "NA", "ST_INTRA": "NA", "ST_INTER": "NA",
                **{t: "NA" for t in STACK_TYPES_OUT},
            })
            continue

        sub      = [p for p in all_stacking
                    if p["CHAIN_I"] == chain and p["NT_I"] in residues]
        st_intra = sum(1 for p in sub if "intra" in p["ORIGIN"])
        st_inter = sum(1 for p in sub if p["ORIGIN"] == "inter_AB")
        tc       = defaultdict(int)
        for p in sub:
            tc[p["STACK_TYPE"]] += 1

        region_rows.append({
            "STRAND"  : strand, "CHAIN": chain,
            "REGION"  : region,
            "NT_RANGE": f"nt{residues[0]}_to_{residues[-1]}",
            "N_NT"    : len(residues),
            "ST_TOTAL": st_intra + st_inter,
            "ST_INTRA": st_intra,
            "ST_INTER": st_inter,
            **{t: tc.get(t, 0) for t in STACK_TYPES_OUT},
        })

df_region = pd.DataFrame(region_rows)

# ─────────────────────────────────────────────
# 9. TABLA GLOBAL
# ─────────────────────────────────────────────

def global_stack(scope, chain_f=None, origin_f=None):
    sub = all_stacking
    if chain_f:
        sub = [p for p in sub if p["CHAIN_I"] == chain_f]
    if origin_f:
        sub = [p for p in sub if p["ORIGIN"] == origin_f]
    st_intra = sum(1 for p in sub if "intra" in p["ORIGIN"])
    st_inter = sum(1 for p in sub if p["ORIGIN"] == "inter_AB")
    tc = defaultdict(int)
    for p in sub:
        tc[p["STACK_TYPE"]] += 1
    return {
        "SCOPE"   : scope,
        "ST_TOTAL": st_intra + st_inter,
        "ST_INTRA": st_intra,
        "ST_INTER": st_inter,
        **{t: tc.get(t, 0) for t in STACK_TYPES_OUT},
    }

df_global = pd.DataFrame([
    global_stack("GLOBAL_DUPLEX"),
    global_stack("GLOBAL_CYR",  chain_f="A", origin_f="intra_A"),
    global_stack("GLOBAL_MIR",  chain_f="B", origin_f="intra_B"),
    global_stack("INTER_AB",    origin_f="inter_AB"),
])

# ─────────────────────────────────────────────
# 10. PRINT RESUMEN
# ─────────────────────────────────────────────

print(f"\n[GLOBAL]\n{df_global.to_string(index=False)}")
print(f"\n[POR REGION]\n{df_region.to_string(index=False)}")
print(f"\n[POR NT — CYR]\n{df_nt[df_nt['CHAIN']=='A'].to_string(index=False)}")
print(f"\n[POR NT — MIR]\n{df_nt[df_nt['CHAIN']=='B'].to_string(index=False)}")

# ─────────────────────────────────────────────
# 11. EXPORTACION CSV
# ─────────────────────────────────────────────

NT_COLS  = ["STRAND","CHAIN","NT_INDEX","RESNAME","REGION",
            "ST_TOTAL","ST_INTRA","ST_INTER"] + STACK_TYPES_OUT + ["STACK_PARTNERS"]
REG_COLS = ["STRAND","CHAIN","REGION","NT_RANGE","N_NT",
            "ST_TOTAL","ST_INTRA","ST_INTER"] + STACK_TYPES_OUT
GLB_COLS = ["SCOPE","ST_TOTAL","ST_INTRA","ST_INTER"] + STACK_TYPES_OUT
PAR_COLS = ["CHAIN_I","NT_I","CHAIN_J","NT_J","STACK_TYPE","ORIGIN"]

df_pairs = pd.DataFrame(all_stacking, columns=PAR_COLS)

outputs = {
    f"{PREFIX}_per_nt.csv"     : (df_nt,     NT_COLS),
    f"{PREFIX}_per_region.csv" : (df_region, REG_COLS),
    f"{PREFIX}_global.csv"     : (df_global, GLB_COLS),
    f"{PREFIX}_pairs.csv"      : (df_pairs,  PAR_COLS),
}

for fname, (df, cols) in outputs.items():
    path = os.path.join(OUTPUT_DIR, fname)
    df[cols].to_csv(path, index=False)
    print(f"[OUTPUT] {fname}")

# ─────────────────────────────────────────────
# 12. REPORTE .TXT
# ─────────────────────────────────────────────

report_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte.txt")
lines = [
    "="*70,
    "REPORTE STACKING — DUPLEX WT_miR7",
    "="*70, "",
    f"HERRAMIENTA : FR3D (fr3d-python v0.0.2)",
    f"FECHA       : {today}",
    f"INPUT       : {PDB_PATH}",
    f"ATOMOS H    : {h_count}",
    "TIPOS       : s35, s53→s35, s55, ns55, ns33",
    "              s53 normalizado a s35 (par idéntico, dirección inversa)",
    "              basepairs ignorados (requerido por FR3D pero no procesado)",
    "              p_1, p_2 excluidos", "",
    "ESTRUCTURA:",
    "  Hebra A — CYR_WT (26 nt): ANCHOR(26)|P1(19-25)|B(15-18)|P2(1-14)",
    "  Hebra B — miR-7-5p (24 nt): ANCHOR(1)|P1(2-8)|B(9-10)|P2(11-24)",
    "", "-"*70, "GLOBAL", "-"*70,
]
for _, row in df_global.iterrows():
    lines.append(
        f"  {row['SCOPE']:<20} ST_TOTAL={row['ST_TOTAL']}  "
        f"INTRA={row['ST_INTRA']}  INTER={row['ST_INTER']}  "
        f"| s35={row['s35']} s55={row['s55']} "
        f"ns55={row['ns55']} ns33={row['ns33']}")

lines += ["", "-"*70, "POR REGION", "-"*70]
for _, row in df_region.iterrows():
    if row["ST_TOTAL"] == "NA":
        lines.append(f"  {row['STRAND']} {row['REGION']} : NA")
        continue
    lines.append(
        f"  {row['STRAND']:<4} {row['REGION']:<20} "
        f"nt={row['NT_RANGE']}  n={row['N_NT']}  "
        f"ST={row['ST_TOTAL']} INTRA={row['ST_INTRA']} INTER={row['ST_INTER']}  "
        f"s35={row['s35']} s55={row['s55']} "
        f"ns55={row['ns55']} ns33={row['ns33']}")

lines += [
    "", "-"*70, "NOTAS", "-"*70,
    "  - Pares únicos: s35(i,j) y s53(j,i) colapsados en s35.",
    "  - ST_INTER=0 en MIR por construcción (pares inter asignados a NT_I min = CYR).",
    "  - INTER_AB = vista completa de pares entre hebras.",
    "  - _pairs.csv contiene todos los pares crudos con ORIGIN.",
    "="*70,
]

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[OUTPUT] {PREFIX}_reporte.txt")

# ─────────────────────────────────────────────
# 13. COPIA DEL SCRIPT
# ─────────────────────────────────────────────

script_src = os.path.abspath(__file__)
script_dst = os.path.join(OUTPUT_DIR, f"{PREFIX}.py")
if os.path.abspath(script_src) != os.path.abspath(script_dst):
    try:
        shutil.copy2(script_src, script_dst)
        print(f"[OUTPUT] {PREFIX}.py")
    except Exception as e:
        print(f"[WARN] copia: {e}")

print(f"\n[DONE] Stacking extraído.")
