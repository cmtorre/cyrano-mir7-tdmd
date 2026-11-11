"""
DUPLEX_M6_MIR7_FreeSASA.py
Script de analisis SASA para el duplex RNA M6_miR7
Herramienta: FreeSASA (https://freesasa.github.io/)
Autor: cmart — Fecha: ver reporte de salida

DUPLEX: M6_miR7
  Hebra A — CYR/M6 (16 nt, 5'->3'):  AACAACAAAAUCACCA
    ANCHOR_CYR   : nt 16
    P1_CYR       : NA
    B_CYR        : NA
    P2_CYR       : nt 1-14
    OVERHANG_CYR : nt 15

  Hebra B — MIR/miR-7-5p (24 nt, 5'->3'): UGGAAGACUAGUGAUUUUGUUGUU
    ANCHOR_MIR   : nt 1
    P1_MIR       : NA
    B_MIR        : NA
    P2_MIR       : nt 11-24
    OVERHANG_MIR : nt 2-10
"""

import freesasa
import pandas as pd
import os, sys, shutil
from datetime import datetime

# ─────────────────────────────────────────────
# 1. CONFIGURACION
# ─────────────────────────────────────────────

PDB_PATH   = os.path.expanduser("~/tesis_rnas/OE4/1.FreeSASA/7.M6_MIR/M6_miR7.pdb")
OUTPUT_DIR = os.path.expanduser("~/tesis_rnas/OE4/1.FreeSASA/7.M6_MIR")
PREFIX     = "DUPLEX_M6_MIR7_FreeSASA"

REGION_MAP_CYR = {
    "ANCHOR_CYR": [16],
    "P1_CYR": None,
    "B_CYR": None,
    "P2_CYR": list(range(1, 15)),
    "OVERHANG_CYR": [15],
}

REGION_MAP_MIR = {
    "ANCHOR_MIR": [1],
    "P1_MIR": None,
    "B_MIR": None,
    "P2_MIR": list(range(11, 25)),
    "OVERHANG_MIR": list(range(2, 11)),
}

BACKBONE_ATOMS = {
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'",
    "C2'", "O2'", "C1'", "O4'"
}

META = {
    "CYR": {"n_nt": 16, "seq": "AACAACAAAAUCACCA", "chain": "A"},
    "MIR": {"n_nt": 24, "seq": "UGGAAGACUAGUGAUUUUGUUGUU", "chain": "B"},
}

# ─────────────────────────────────────────────
# 2. SETUP
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
today = datetime.now().strftime("%d/%m/%Y")

# ─────────────────────────────────────────────
# 3. CARGA PDB Y CALCULO SASA
# ─────────────────────────────────────────────

print(f"[INFO] Cargando PDB: {PDB_PATH}")
if not os.path.isfile(PDB_PATH):
    sys.exit(f"[ERROR] PDB no encontrado: {PDB_PATH}")

structure = freesasa.Structure(PDB_PATH)
result    = freesasa.calc(structure)

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

print(f"[INFO] FreeSASA v{version_str} — calculo completado")

# ─────────────────────────────────────────────
# 4. CLASIFICACION DE ATOMOS
# ─────────────────────────────────────────────

def classify_atoms(structure, result):
    data = {}
    for i in range(structure.nAtoms()):
        chain   = structure.chainLabel(i)
        resname = structure.residueName(i).strip()
        atom    = structure.atomName(i).strip()
        sasa_i  = result.atomArea(i)
        try:
            resnum = int(structure.residueNumber(i).strip())
        except ValueError:
            continue
        data.setdefault(chain, {})
        data[chain].setdefault(resnum, {
            "chain": chain, "resname": resname,
            "SASA_NT": 0.0, "SASA_BB": 0.0, "SASA_BASE": 0.0,
        })
        data[chain][resnum]["SASA_NT"] += sasa_i
        if atom in BACKBONE_ATOMS:
            data[chain][resnum]["SASA_BB"]   += sasa_i
        else:
            data[chain][resnum]["SASA_BASE"] += sasa_i
    return data

residue_data    = classify_atoms(structure, result)
detected_chains = sorted(residue_data.keys())
print(f"[INFO] Cadenas detectadas: {detected_chains}")
for ch in ["A", "B"]:
    if ch not in residue_data:
        sys.exit(f"[ERROR] Cadena {ch} no encontrada. Presentes: {detected_chains}")

chain_cyr = residue_data["A"]
chain_mir = residue_data["B"]

# ─────────────────────────────────────────────
# 5. VARIABLES GLOBALES — TRES NIVELES
# ─────────────────────────────────────────────

def global_sasa(cd):
    return {
        "SASA_TOTAL": sum(v["SASA_NT"]   for v in cd.values()),
        "SASA_BB":    sum(v["SASA_BB"]   for v in cd.values()),
        "SASA_BASES": sum(v["SASA_BASE"] for v in cd.values()),
    }

glob_cyr    = global_sasa(chain_cyr)
glob_mir    = global_sasa(chain_mir)
glob_duplex = {
    "SASA_TOTAL": result.totalArea(),
    "SASA_BB":    glob_cyr["SASA_BB"]    + glob_mir["SASA_BB"],
    "SASA_BASES": glob_cyr["SASA_BASES"] + glob_mir["SASA_BASES"],
}

for lvl, g in [("GLOBAL_DUPLEX", glob_duplex), ("GLOBAL_CYR", glob_cyr), ("GLOBAL_MIR", glob_mir)]:
    print(f"\n[{lvl}]")
    for k in ("SASA_TOTAL", "SASA_BB", "SASA_BASES"):
        print(f"  {k:<12}: {g[k]:.4f} A2")

# ─────────────────────────────────────────────
# 6. TABLA POR NUCLEOTIDO
# ─────────────────────────────────────────────

def build_region_lookup(region_map):
    lk = {}
    for rname, nt_list in region_map.items():
        if nt_list is None:
            continue
        for nt in nt_list:
            lk[nt] = rname
    return lk

rl_cyr = build_region_lookup(REGION_MAP_CYR)
rl_mir = build_region_lookup(REGION_MAP_MIR)

def build_nt_table(chain_dict, strand_label, region_lookup):
    rows = []
    for resnum in sorted(chain_dict.keys()):
        v = chain_dict[resnum]
        rows.append({
            "STRAND":    strand_label,
            "CHAIN":     v["chain"],
            "NT_INDEX":  resnum,
            "RESNAME":   v["resname"],
            "REGION":    region_lookup.get(resnum, "UNASSIGNED"),
            "SASA_NT":   round(v["SASA_NT"],   4),
            "SASA_BB":   round(v["SASA_BB"],   4),
            "SASA_BASE": round(v["SASA_BASE"], 4),
        })
    return pd.DataFrame(rows)

df_nt_cyr = build_nt_table(chain_cyr, "CYR", rl_cyr)
df_nt_mir = build_nt_table(chain_mir, "MIR", rl_mir)
df_nt_all = pd.concat([df_nt_cyr, df_nt_mir], ignore_index=True)

print(f"\n[NT TABLE — CYR, {len(df_nt_cyr)} nt]")
print(df_nt_cyr.to_string(index=False))
print(f"\n[NT TABLE — MIR, {len(df_nt_mir)} nt]")
print(df_nt_mir.to_string(index=False))

# ─────────────────────────────────────────────
# 7. TABLA POR ELEMENTO ESTRUCTURAL
# ─────────────────────────────────────────────

def build_element_table(region_map, chain_dict, strand_label):
    rows = []
    for rname, nt_list in region_map.items():
        if nt_list is None:
            rows.append({
                "STRAND": strand_label, "ELEMENT": rname,
                "NT_RANGE": "NA", "N_NT": 0,
                "SASA_NT": "NA", "SASA_BB": "NA", "SASA_BASE": "NA",
                "MISSING_NT": "NA_by_design",
            })
            continue
        found   = [n for n in nt_list if n in chain_dict]
        missing = [n for n in nt_list if n not in chain_dict]
        if missing:
            print(f"[WARN] {rname}: nt ausentes en PDB: {missing}")
        rows.append({
            "STRAND":    strand_label,
            "ELEMENT":   rname,
            "NT_RANGE":  f"{nt_list[0]}-{nt_list[-1]}",
            "N_NT":      len(found),
            "SASA_NT":   round(sum(chain_dict[n]["SASA_NT"]   for n in found), 4),
            "SASA_BB":   round(sum(chain_dict[n]["SASA_BB"]   for n in found), 4),
            "SASA_BASE": round(sum(chain_dict[n]["SASA_BASE"] for n in found), 4),
            "MISSING_NT": str(missing) if missing else "none",
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
# 8. TABLA GLOBAL
# ─────────────────────────────────────────────

df_global = pd.DataFrame([
    {"LEVEL": "GLOBAL_DUPLEX", "MOLECULE": "M6_miR7",
      "SASA_TOTAL": round(glob_duplex["SASA_TOTAL"], 4),
      "SASA_BB":    round(glob_duplex["SASA_BB"],    4),
      "SASA_BASES": round(glob_duplex["SASA_BASES"], 4)},
    {"LEVEL": "GLOBAL_CYR", "MOLECULE": "M6",
      "SASA_TOTAL": round(glob_cyr["SASA_TOTAL"], 4),
      "SASA_BB":    round(glob_cyr["SASA_BB"],    4),
      "SASA_BASES": round(glob_cyr["SASA_BASES"], 4)},
    {"LEVEL": "GLOBAL_MIR", "MOLECULE": "miR-7-5p",
      "SASA_TOTAL": round(glob_mir["SASA_TOTAL"], 4),
      "SASA_BB":    round(glob_mir["SASA_BB"],    4),
      "SASA_BASES": round(glob_mir["SASA_BASES"], 4)},
])

# ─────────────────────────────────────────────
# 9. EXPORTACION CSV
# ─────────────────────────────────────────────

csv_global = os.path.join(OUTPUT_DIR, f"{PREFIX}_global.csv")
csv_nt     = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_nt.csv")
csv_elem   = os.path.join(OUTPUT_DIR, f"{PREFIX}_per_element.csv")

df_global.to_csv(csv_global, index=False)
df_nt_all.to_csv(csv_nt,     index=False)
df_elem_all.to_csv(csv_elem, index=False)

print(f"\n[OUTPUT] CSV global -> {csv_global}")
print(f"[OUTPUT] CSV por nt -> {csv_nt}")
print(f"[OUTPUT] CSV elem   -> {csv_elem}")

# ─────────────────────────────────────────────
# 10. REPORTE .TXT
# ─────────────────────────────────────────────

report_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte.txt")

def sasa_block(g):
    return [
        "  VARIABLE_EXTRAIDA : SASA_TOTAL",
        f"  VALOR             : {g['SASA_TOTAL']:.4f} A2",
        "  NOTAS             : Area superficial total accesible al solvente",
        "",
        "  VARIABLE_EXTRAIDA : SASA_BB",
        f"  VALOR             : {g['SASA_BB']:.4f} A2",
        "  NOTAS             : Contribucion de atomos de backbone (P, OP1, OP2, azucares)",
        "",
        "  VARIABLE_EXTRAIDA : SASA_BASES",
        f"  VALOR             : {g['SASA_BASES']:.4f} A2",
        "  NOTAS             : Contribucion de atomos de bases nitrogenadas",
        "",
    ]

def elem_block(df):
    lines = []
    for _, row in df.iterrows():
        if row["SASA_NT"] == "NA":
            lines += [
                f"  ELEMENTO   : {row['ELEMENT']}  (NT_RANGE=NA, n=0)",
                "  SASA_NT    : NA",
                "  SASA_BB    : NA",
                "  SASA_BASE  : NA",
                "  MISSING_NT : NA_by_design",
                "",
            ]
        else:
            lines += [
                f"  ELEMENTO   : {row['ELEMENT']}  (nt {row['NT_RANGE']}, n={row['N_NT']})",
                f"  SASA_NT    : {row['SASA_NT']:.4f} A2",
                f"  SASA_BB    : {row['SASA_BB']:.4f} A2",
                f"  SASA_BASE  : {row['SASA_BASE']:.4f} A2",
                f"  MISSING_NT : {row['MISSING_NT']}",
                "",
            ]
    return lines

RL = [
    "=" * 70,
    f"REPORTE DE ANALISIS SASA - DUPLEX M6_miR7",
    "=" * 70,
    "",
    "HERRAMIENTA : FreeSASA",
    f"VERSION     : {version_str}",
    f"FECHA       : {today}",
    f"INPUT       : {PDB_PATH}",
    "PARAMETROS  : default (Lee & Richards, probe_radius=1.4 A, n_points=100)",
    "",
    f"HEBRA A (CYR/M6, 16 nt, 5'->3'): AACAACAAAAUCACCA",
    f"  ANCHOR_CYR   : nt 16",
    f"  P1_CYR       : NA",
    f"  B_CYR        : NA",
    f"  P2_CYR       : nt 1-14",
    f"  OVERHANG_CYR : nt 15",
    "",
    f"HEBRA B (MIR/miR-7-5p, 24 nt, 5'->3'): UGGAAGACUAGUGAUUUUGUUGUU",
    f"  ANCHOR_MIR   : nt 1",
    f"  P1_MIR       : NA",
    f"  B_MIR        : NA",
    f"  P2_MIR       : nt 11-24",
    f"  OVERHANG_MIR : nt 2-10",
    "",
    "-" * 70,
    "ARCHIVOS DE SALIDA",
    "-" * 70,
    f"  CSV global  : {os.path.basename(csv_global)}",
    f"  CSV por nt  : {os.path.basename(csv_nt)}",
    f"  CSV elem    : {os.path.basename(csv_elem)}",
    f"  Script      : {PREFIX}.py",
    "",
    "-" * 70,
    "VARIABLES GLOBALES — GLOBAL_DUPLEX (hebras A+B)",
    "-" * 70,
] + sasa_block(glob_duplex) + [
    "-" * 70,
    f"VARIABLES GLOBALES — GLOBAL_CYR (Chain A, M6, {META['CYR']['n_nt']} nt)",
    "-" * 70,
] + sasa_block(glob_cyr) + [
    "-" * 70,
    f"VARIABLES GLOBALES — GLOBAL_MIR (Chain B, miR-7-5p, {META['MIR']['n_nt']} nt)",
    "-" * 70,
] + sasa_block(glob_mir) + [
    "-" * 70,
    f"VARIABLES POR NUCLEOTIDO — CYR / Chain A ({len(df_nt_cyr)} nt)",
    "-" * 70,
]

for _, row in df_nt_cyr.iterrows():
    RL.append(
        f"  NT {int(row['NT_INDEX']):>3} ({row['RESNAME']}) | "
        f"REGION={row['REGION']:<15} | "
        f"SASA_NT={row['SASA_NT']:.4f} | "
        f"SASA_BB={row['SASA_BB']:.4f} | "
        f"SASA_BASE={row['SASA_BASE']:.4f}  A2"
    )

RL += [
    "",
    "-" * 70,
    f"VARIABLES POR NUCLEOTIDO — MIR / Chain B ({len(df_nt_mir)} nt)",
    "-" * 70,
]

for _, row in df_nt_mir.iterrows():
    RL.append(
        f"  NT {int(row['NT_INDEX']):>3} ({row['RESNAME']}) | "
        f"REGION={row['REGION']:<15} | "
        f"SASA_NT={row['SASA_NT']:.4f} | "
        f"SASA_BB={row['SASA_BB']:.4f} | "
        f"SASA_BASE={row['SASA_BASE']:.4f}  A2"
    )

RL += [
    "",
    "-" * 70,
    "VARIABLES POR ELEMENTO ESTRUCTURAL — CYR",
    "-" * 70,
] + elem_block(df_elem_cyr) + [
    "-" * 70,
    "VARIABLES POR ELEMENTO ESTRUCTURAL — MIR",
    "-" * 70,
] + elem_block(df_elem_mir) + [
    "-" * 70,
    "NOTAS GENERALES",
    "-" * 70,
    "  - Clasificacion BB/BASE: nomenclatura PDB estandar para RNA.",
    "  - Atomos BB: P, OP1, OP2, O5', C5', C4', C3', O3', C2', O2', C1', O4'.",
    "  - Valores en A2 (angstroms cuadrados).",
    "  - GLOBAL_DUPLEX SASA_TOTAL: calculado por FreeSASA sobre el duplex completo.",
    "  - GLOBAL_CYR/MIR SASA_TOTAL: suma de SASA_NT por residuo de cada cadena.",
    "  - NA_by_design: region ausente por diseno experimental en este duplex.",
    "  - PDB generado por AlphaFold3 (AF3) a partir de archivo CIF.",
    "=" * 70,
]

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(RL))

print(f"[OUTPUT] Reporte .txt -> {report_path}")

# ─────────────────────────────────────────────
# 11. COPIA DEL SCRIPT
# ─────────────────────────────────────────────

script_dst = os.path.join(OUTPUT_DIR, f"{PREFIX}.py")
try:
    shutil.copy2(os.path.abspath(__file__), script_dst)
    print(f"[OUTPUT] Script .py   -> {script_dst}")
except Exception as e:
    print(f"[WARN] No se pudo copiar el script: {e}")

print("\n[DONE] Todos los archivos generados correctamente.")
