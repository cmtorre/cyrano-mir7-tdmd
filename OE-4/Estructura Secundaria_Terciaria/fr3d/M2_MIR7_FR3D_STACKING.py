"""
M2_MIR7_FR3D_STACKING.py
Herramienta: FR3D (fr3d-python v0.0.2) — Solo STACKING
Autor: cmart

M2 (24 nt) - Hebra A: 5'-AACAACAAAAUCACCAAUGCUCCA-3'
  ANCHOR_CYR: nt24 | P1_CYR: nt21-23 | B_CYR: nt15-20 | P2_CYR: nt1-14 | OVERHANG: NA
miR7 (24 nt) - Hebra B: 5'-UGGAAGACUAGUGAUUUUGUUGUU-3'
  ANCHOR_MIR: nt1 | P1_MIR: nt2-4 | B_MIR: nt5-10 | P2_MIR: nt11-24 | OVERHANG: NA
"""

import os, sys, shutil
import pandas as pd
from datetime import datetime
from collections import defaultdict

PDB_PATH   = os.path.expanduser("~/tesis_rnas/OE4/1.FreeSASA/3.M2_MIR/M2_miR7_HBonds.pdb")
OUTPUT_DIR = os.path.expanduser("~/tesis_rnas/OE4/2.FR3D/3.M2_MIR")
PREFIX     = "M2_MIR7_FR3D_STACKING"

SEQ = {
    "A": {i+1: b for i, b in enumerate("AACAACAAAAUCACCAAUGCUCCA")},
    "B": {i+1: b for i, b in enumerate("UGGAAGACUAGUGAUUUUGUUGUU")},
}
STRAND_MAP = {"A": "CYR", "B": "MIR"}
REGION_MAP = {
    "A": {
        "ANCHOR_CYR"   : list(range(24, 25)),
        "P1_CYR"       : list(range(21, 24)),
        "B_CYR"        : list(range(15, 21)),
        "P2_CYR"       : list(range(1,  15)),
        "OVERHANG_CYR" : None,
    },
    "B": {
        "ANCHOR_MIR"   : list(range(1,  2)),
        "P1_MIR"       : list(range(2,  5)),
        "B_MIR"        : list(range(5,  11)),
        "P2_MIR"       : list(range(11, 25)),
        "OVERHANG_MIR" : None,
    },
}
N_NT = {"A": 24, "B": 24}

STACK_NORM      = {"s35":"s35","s53":"s35","s55":"s55","ns55":"ns55","ns33":"ns33"}
STACK_TYPES_OUT = ["s35","s55","ns55","ns33"]
RESNUM_FIELD, RESNAME_FIELD, CHAIN_FIELD = 4, 3, 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
today = datetime.now().strftime("%d/%m/%Y")

from fr3d.classifiers.NA_pairwise_interactions import load_structure, annotate_nt_nt_in_structure

if not os.path.isfile(PDB_PATH): sys.exit(f"[ERROR] PDB no encontrado: {PDB_PATH}")
h_count = sum(1 for l in open(PDB_PATH) if l.startswith("ATOM") and l[12:16].strip().startswith("H"))
print(f"[INFO] {PREFIX} | H={h_count}")
structure, messages = load_structure(PDB_PATH)
if structure is None: sys.exit(f"[ERROR] {messages}")
print("[OK] Estructura cargada")

i2l, cat2i, timer, p2d = annotate_nt_nt_in_structure(
    structure, {"basepair":"basepair","stacking":"stacking"})
print(f"[OK] Tipos: {list(i2l.keys())}")

def parse_uid(uid):
    p = str(uid).split("|")
    try: return int(p[RESNUM_FIELD]), p[RESNAME_FIELD].upper(), p[CHAIN_FIELD].upper()
    except: return None, None, None

def get_region(chain, rn):
    for reg, res in REGION_MAP.get(chain, {}).items():
        if res and rn in res: return reg
    return "UNASSIGNED"

stacking_per_nt = defaultdict(lambda: defaultdict(list))
seen_pairs = set()
all_stacking = []

for itype, pairs in i2l.items():
    norm = STACK_NORM.get(str(itype).strip())
    if norm is None: continue
    for pair in pairs:
        try: pl = list(pair); uid_i, uid_j = str(pl[0]), str(pl[1])
        except: continue
        ri, rni, ci = parse_uid(uid_i)
        rj, rnj, cj = parse_uid(uid_j)
        if ri is None: continue
        ki, kj = (ci,ri), (cj,rj)
        lo, hi = sorted([ki, kj])
        dedup = (lo, hi, norm)
        if dedup in seen_pairs: continue
        seen_pairs.add(dedup)
        origin = f"intra_{ci}" if ci == cj else "inter_AB"
        all_stacking.append({"CHAIN_I":lo[0],"NT_I":lo[1],"CHAIN_J":hi[0],"NT_J":hi[1],"STACK_TYPE":norm,"ORIGIN":origin})
        stacking_per_nt[ki][kj].append(norm)
        stacking_per_nt[kj][ki].append(norm)

print(f"[OK] Pares únicos: {len(all_stacking)}")

nt_rows = []
for chain in ["A","B"]:
    for rn in range(1, N_NT[chain]+1):
        key = (chain, rn)
        partners = stacking_per_nt.get(key, {})
        st_intra = sum(len(t) for pk,t in partners.items() if pk[0]==chain)
        st_inter = sum(len(t) for pk,t in partners.items() if pk[0]!=chain)
        tc = defaultdict(int)
        for t in partners.values():
            for x in t: tc[x] += 1
        nt_rows.append({
            "STRAND":STRAND_MAP[chain],"CHAIN":chain,"NT_INDEX":rn,
            "RESNAME":SEQ[chain].get(rn,"?"),"REGION":get_region(chain,rn),
            "ST_TOTAL":st_intra+st_inter,"ST_INTRA":st_intra,"ST_INTER":st_inter,
            **{t:tc.get(t,0) for t in STACK_TYPES_OUT},
            "STACK_PARTNERS":";".join(f"{pk[0]}{pk[1]}" for pk in sorted(partners.keys())) if partners else ""
        })
df_nt = pd.DataFrame(nt_rows)

region_rows = []
for chain in ["A","B"]:
    for region, residues in REGION_MAP[chain].items():
        if residues is None:
            region_rows.append({"STRAND":STRAND_MAP[chain],"CHAIN":chain,"REGION":region,
                "NT_RANGE":"NA","N_NT":0,"ST_TOTAL":"NA","ST_INTRA":"NA","ST_INTER":"NA",
                **{t:"NA" for t in STACK_TYPES_OUT}})
            continue
        sub = [p for p in all_stacking if p["CHAIN_I"]==chain and p["NT_I"] in residues]
        st_intra = sum(1 for p in sub if "intra" in p["ORIGIN"])
        st_inter = sum(1 for p in sub if p["ORIGIN"]=="inter_AB")
        tc = defaultdict(int)
        for p in sub: tc[p["STACK_TYPE"]] += 1
        region_rows.append({"STRAND":STRAND_MAP[chain],"CHAIN":chain,"REGION":region,
            "NT_RANGE":f"nt{residues[0]}_to_{residues[-1]}","N_NT":len(residues),
            "ST_TOTAL":st_intra+st_inter,"ST_INTRA":st_intra,"ST_INTER":st_inter,
            **{t:tc.get(t,0) for t in STACK_TYPES_OUT}})
df_region = pd.DataFrame(region_rows)

def gstack(scope, chain_f=None, origin_f=None):
    sub = all_stacking
    if chain_f:  sub = [p for p in sub if p["CHAIN_I"]==chain_f]
    if origin_f: sub = [p for p in sub if p["ORIGIN"]==origin_f]
    st_intra = sum(1 for p in sub if "intra" in p["ORIGIN"])
    st_inter = sum(1 for p in sub if p["ORIGIN"]=="inter_AB")
    tc = defaultdict(int)
    for p in sub: tc[p["STACK_TYPE"]] += 1
    return {"SCOPE":scope,"ST_TOTAL":st_intra+st_inter,"ST_INTRA":st_intra,"ST_INTER":st_inter,
            **{t:tc.get(t,0) for t in STACK_TYPES_OUT}}

df_global = pd.DataFrame([
    gstack("GLOBAL_DUPLEX"),
    gstack("GLOBAL_CYR",  chain_f="A", origin_f="intra_A"),
    gstack("GLOBAL_MIR",  chain_f="B", origin_f="intra_B"),
    gstack("INTER_AB",    origin_f="inter_AB"),
])
print(f"\n[GLOBAL]\n{df_global.to_string(index=False)}")

NT_COLS  = ["STRAND","CHAIN","NT_INDEX","RESNAME","REGION","ST_TOTAL","ST_INTRA","ST_INTER"]+STACK_TYPES_OUT+["STACK_PARTNERS"]
REG_COLS = ["STRAND","CHAIN","REGION","NT_RANGE","N_NT","ST_TOTAL","ST_INTRA","ST_INTER"]+STACK_TYPES_OUT
GLB_COLS = ["SCOPE","ST_TOTAL","ST_INTRA","ST_INTER"]+STACK_TYPES_OUT
PAR_COLS = ["CHAIN_I","NT_I","CHAIN_J","NT_J","STACK_TYPE","ORIGIN"]

for fname, df, cols in [
    (f"{PREFIX}_per_nt.csv",     df_nt,     NT_COLS),
    (f"{PREFIX}_per_region.csv", df_region, REG_COLS),
    (f"{PREFIX}_global.csv",     df_global, GLB_COLS),
    (f"{PREFIX}_pairs.csv",      pd.DataFrame(all_stacking,columns=PAR_COLS), PAR_COLS),
]:
    df[cols].to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"[OUTPUT] {fname}")

rlines = ["="*60, f"STACKING — M2_miR7", "="*60, "",
    f"FECHA: {today}  |  INPUT: {PDB_PATH}  |  H: {h_count}", "", "GLOBAL:"]
for _,row in df_global.iterrows():
    rlines.append(f"  {row['SCOPE']:<20} ST={row['ST_TOTAL']} INTRA={row['ST_INTRA']} INTER={row['ST_INTER']} | s35={row['s35']} s55={row['s55']} ns55={row['ns55']} ns33={row['ns33']}")
rlines += ["","POR REGION:"]
for _,row in df_region.iterrows():
    if row["ST_TOTAL"]=="NA": rlines.append(f"  {row['STRAND']} {row['REGION']}: NA"); continue
    rlines.append(f"  {row['STRAND']:<4} {row['REGION']:<20} n={row['N_NT']} ST={row['ST_TOTAL']} INTRA={row['ST_INTRA']} INTER={row['ST_INTER']} s35={row['s35']} s55={row['s55']} ns55={row['ns55']} ns33={row['ns33']}")
rlines.append("="*60)
with open(os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte.txt"), "w") as f:
    f.write("\n".join(rlines))
print(f"[OUTPUT] {PREFIX}_reporte.txt")
src = os.path.abspath(__file__)
dst = os.path.join(OUTPUT_DIR, f"{PREFIX}.py")
if src != dst:
    try: shutil.copy2(src, dst)
    except: pass
print("[DONE]")
