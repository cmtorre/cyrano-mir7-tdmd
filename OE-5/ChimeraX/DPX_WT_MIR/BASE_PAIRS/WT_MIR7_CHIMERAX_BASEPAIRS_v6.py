"""
WT_MIR7_CHIMERAX_BASEPAIRS.py
===
Script de extracción de BASE_PAIRS para el dúplex WT_miR7
Herramienta : UCSF ChimeraX 1.9
Ejecución   : Consola ChimeraX → open "ruta/WT_MIR7_CHIMERAX_BASEPAIRS.py"
Input PDB   : WT_miR7_HBonds.pdb (con H, generado con addh)
Autor       : cmart

ESTRATEGIA v6:
  1. Ejecuta 'hbonds relax true log true saveFile' → H-bonds
  2. Parsea el archivo con parser robusto (split por '/', tokens por spec)
     Maneja: O2', números 1-2 dígitos, espaciado variable
  3. Clasifica cada par LW por átomos donor/acceptor + orientación glicosídica
  4. Excluye contactos ribosa-ribosa (O2'→O4', etc.)
  5. Registra TODOS los nucleótidos de ambas hebras, con o sin interacciones
  6. Agrega por nucleótido (intra/inter), por región y globalmente

CLASIFICACION LW:
  W edge : N1,N3,N6,N4,O6,O4,O2,N2
  H edge : N7,C8
  S edge : O2'
  P      : OP1,OP2,O5',O3',P → BPH si contacta base
  Ribosa : O4',C1'-C5' → excluir
  Orientación cis/trans: vectores glicosídicos C1'→N1/N9

OUTPUTS CSV:
  _per_nt_detail.csv     — una fila por interacción detectada (bilateral)
  _per_nt_summary.csv    — una fila por nucleótido (todos los nt de ambas hebras)
  _per_region.csv        — agregado por región y hebra (intra/inter/total)
  _global.csv            — conteos globales del dúplex
  _reporte.txt           — resumen legible
"""

import os
import re
import csv
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

# ─────────────────────────────────────────────
# 1. CONFIGURACION
# ─────────────────────────────────────────────

PDB_PATH = (
    r"C:\Users\cmart\OneDrive\Documentos OneDrive"
    r"\1. circRNA_TESIS\Tesis\7.Metodo\OE-4\6. ChimeraX"
    r"\1.WT_MIR\WT_miR7_HBonds.pdb"
)
OUTPUT_DIR = (
    r"C:\Users\cmart\OneDrive\Documentos OneDrive"
    r"\1. circRNA_TESIS\Tesis\7.Metodo\OE-4\9.ChimeraX"
    r"\1.WT_MIR"
)
PREFIX   = "WT_MIR7_CHIMERAX_BP"
LOG_PATH = os.path.join(OUTPUT_DIR, "_hbonds_raw.txt")

# ─────────────────────────────────────────────
# 2. SECUENCIAS Y REGIONES
# ─────────────────────────────────────────────

SEQ = {
    "A": {i+1: b for i, b in enumerate("AACAACAAAAUCACCAAUGUCUUCCA")},
    "B": {i+1: b for i, b in enumerate("UGGAAGACUAGUGAUUUUGUUGUU")},
}

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

# ─────────────────────────────────────────────
# 3. DEFINICION DE ATOMOS
# ─────────────────────────────────────────────

WC_ATOMS        = {"N1","N3","N6","N4","O6","O4","O2","N2"}
HOOG_ATOMS      = {"N7","C8"}
BASE_ATOMS      = WC_ATOMS | HOOG_ATOMS
SUGAR_O2P       = {"O2'"}
PHOSPHATE_ATOMS = {"OP1","OP2","O5'","O3'","P"}
RIBOSE_ATOMS    = {"O4'","C1'","C2'","C3'","C4'","C5'"}
NON_BASE_ATOMS  = SUGAR_O2P | PHOSPHATE_ATOMS | RIBOSE_ATOMS

CANONICAL_PAIRS = {frozenset(["A","U"]), frozenset(["G","C"]), frozenset(["G","U"])}
VALID_EDGES     = {"WW","WS","SW","WH","HW","HS","SH","HH","SS"}

LW_MAP = {
    "cWW":"cWW_NC","tWW":"tWW",
    "cWS":"cWS",   "cSW":"cSW",
    "tWS":"tWS",   "tSW":"tSW",
    "cWH":"cHW",   "cHW":"cHW",
    "tWH":"tHW",   "tHW":"tHW",
    "cHS":"cHS",   "cSH":"cHS",
    "tHS":"tHS",   "tSH":"tHS",
    "cHH":"cHH",   "tHH":"tHH",
    "cSS":"cSS",   "tSS":"tSS",
}

# ─────────────────────────────────────────────
# 4. PARSER DEL LOG
# ─────────────────────────────────────────────

def parse_hbond_line(line):
    """
    Parsea una línea del log hbonds de ChimeraX.
    Formato: /CHAIN RESNAME NUM ATOM  /CHAIN RESNAME NUM ATOM  /CHAIN RESNAME NUM ATOM  D..A  D-H..A
    Retorna dict o None si no es línea de hbond.
    """
    line = line.strip()
    if not line.startswith('/'):
        return None
    # Split por '/' — cada spec es un átomo
    parts = [p.strip() for p in line.split('/') if p.strip()]
    if len(parts) < 3:
        return None
    try:
        def parse_spec(s):
            tokens = s.split()
            # tokens[0]=chain, tokens[1]=resname, tokens[2]=resnum, tokens[3]=atom
            return tokens[0], tokens[1], int(tokens[2]), tokens[3]

        cd, bd, nd, ad = parse_spec(parts[0])   # donor
        ca, ba, na, aa = parse_spec(parts[1])   # acceptor
        # Distancias: últimos dos floats del tercer spec
        nums = re.findall(r'[\d]+\.[\d]+', parts[2])
        if len(nums) < 2:
            return None
        dist_da  = float(nums[-2])
        dist_dha = float(nums[-1])
        return {
            "chain_d": cd, "resname_d": bd, "num_d": nd, "atom_d": ad,
            "chain_a": ca, "resname_a": ba, "num_a": na, "atom_a": aa,
            "dist_da": dist_da, "dist_dha": dist_dha,
        }
    except Exception:
        return None

# ─────────────────────────────────────────────
# 5. CLASIFICACION LW
# ─────────────────────────────────────────────

def get_edge(atom_name):
    a = atom_name.strip()
    if a in SUGAR_O2P:       return "S"
    if a in PHOSPHATE_ATOMS: return "P"
    if a in HOOG_ATOMS:      return "H"
    if a in WC_ATOMS:        return "W"
    return None

def get_orientation(chain_d, num_d, chain_a, num_a, model):
    """Cis (<90°) o trans (>=90°) por vectores C1'→N1/N9."""
    try:
        def gvec(chain, num):
            res = [r for r in model.residues
                   if r.chain_id == chain and r.number == num]
            if not res:
                return None
            atoms = {a.name: np.array(a.coord) for a in res[0].atoms}
            c1 = atoms.get("C1'")
            n  = atoms.get("N9") or atoms.get("N1")
            if c1 is None or n is None:
                return None
            return n - c1
        vi = gvec(chain_d, num_d)
        vj = gvec(chain_a, num_a)
        if vi is None or vj is None:
            return "c"
        cos_a = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
        return "c" if np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))) < 90.0 else "t"
    except Exception:
        return "c"

def classify_lw(don_atom, acc_atom, chain_d, num_d, chain_a, num_a,
                base_d, base_a, model):
    """
    Retorna (LEONTIS_CLASS, IS_CANONICAL) o (None, None) si excluir.
    Jerarquía:
      1. Ambos no-base (ribosa-ribosa, etc.) → excluir
      2. Fosfato + base → BPH
      3. O2' + base → SUGAR_sO real
      4. O2' o ribosa sin base → excluir
      5. Base + base → clasificar 12 familias LW
    """
    dn = don_atom.strip()
    an = acc_atom.strip()

    # 1. Ambos no-base
    if dn in NON_BASE_ATOMS and an in NON_BASE_ATOMS:
        return None, None

    # 2. Fosfato + base
    if dn in PHOSPHATE_ATOMS or an in PHOSPHATE_ATOMS:
        if dn in BASE_ATOMS or an in BASE_ATOMS:
            return "BPH", False
        return None, None

    # 3. O2' + base → SUGAR_sO
    if dn in SUGAR_O2P or an in SUGAR_O2P:
        if dn in BASE_ATOMS or an in BASE_ATOMS:
            return "SUGAR_sO", False
        return None, None

    # 4. Ribosa sin O2' → excluir
    if dn in RIBOSE_ATOMS or an in RIBOSE_ATOMS:
        return None, None

    # 5. Base + base → LW
    edge_d = get_edge(dn)
    edge_a = get_edge(an)
    if edge_d is None or edge_a is None:
        return f"OTHER_{dn}_{an}", False

    edges = f"{edge_d}{edge_a}"
    if edges not in VALID_EDGES:
        edges_inv = f"{edge_a}{edge_d}"
        edges = edges_inv if edges_inv in VALID_EDGES else edges

    orient = get_orientation(chain_d, num_d, chain_a, num_a, model)
    lw_key = f"{orient}{edges}"

    is_can = (edges == "WW" and orient == "c" and
              frozenset([base_d, base_a]) in CANONICAL_PAIRS)
    if is_can:
        return "cWW_CAN", True

    return LW_MAP.get(lw_key, f"OTHER_{lw_key}"), False

def get_region(chain, resnum):
    for region, residues in REGION_MAP.get(chain, {}).items():
        if residues and resnum in residues:
            return region
    return "UNASSIGNED"

# ─────────────────────────────────────────────
# 6. CARGAR MODELO Y EJECUTAR HBONDS
# ─────────────────────────────────────────────

from chimerax.core.commands import run

os.makedirs(OUTPUT_DIR, exist_ok=True)

target_model = None
for m in session.models.list():
    if hasattr(m, 'residues') and 'WT_miR7' in m.name:
        target_model = m
        break

if target_model is None:
    print(f"[INFO] Abriendo: {PDB_PATH}")
    run(session, f'open "{PDB_PATH}"')
    target_model = session.models.list()[-1]

print(f"[OK] Modelo  : {target_model.name}")
print(f"[OK] Residuos: {len(target_model.residues)}")

# Ejecutar hbonds y guardar log
print("\n[INFO] Ejecutando hbonds relax true...")
run(session, f'hbonds reveal true relax true log true saveFile "{LOG_PATH}"')
print(f"[OK] Log guardado: {LOG_PATH}")

# ─────────────────────────────────────────────
# 7. PARSEAR LOG
# ─────────────────────────────────────────────

hb_records = []
with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        rec = parse_hbond_line(line)
        if rec:
            hb_records.append(rec)

print(f"[OK] H-bonds parseados: {len(hb_records)}")

# Verificar parsing — mostrar primeros 5 y los NC base-base conocidos
print("\n[DEBUG] Muestra de pares parseados:")
for r in hb_records[:5]:
    print(f"  {r['chain_d']}{r['num_d']}({r['resname_d']}).{r['atom_d']}"
          f" → {r['chain_a']}{r['num_a']}({r['resname_a']}).{r['atom_a']}"
          f"  D..A={r['dist_da']}")

print("\n[DEBUG] Pares NC base-base esperados (G19-C8, U-A, C-G):")
for r in hb_records:
    if r['chain_d'] != r['chain_a']:  # inter-hebra
        if r['atom_d'] not in NON_BASE_ATOMS and r['atom_a'] not in NON_BASE_ATOMS:
            if not (frozenset([r['resname_d'],r['resname_a']]) in CANONICAL_PAIRS
                    and r['atom_d'] in WC_ATOMS and r['atom_a'] in WC_ATOMS):
                print(f"  NC: {r['chain_d']}{r['num_d']}({r['resname_d']}).{r['atom_d']}"
                      f" → {r['chain_a']}{r['num_a']}({r['resname_a']}).{r['atom_a']}"
                      f"  D..A={r['dist_da']}")

# ─────────────────────────────────────────────
# 8. CLASIFICACION
# ─────────────────────────────────────────────

raw_pairs  = []
seen_pairs = set()
n_excluded = 0

for rec in hb_records:
    cd, nd, ad = rec["chain_d"], rec["num_d"], rec["atom_d"]
    ca, na, aa = rec["chain_a"], rec["num_a"], rec["atom_a"]
    bd, ba     = rec["resname_d"], rec["resname_a"]

    if bd not in ("A","U","G","C") or ba not in ("A","U","G","C"):
        continue

    lw_class, is_can = classify_lw(
        ad, aa, cd, nd, ca, na, bd, ba, target_model)

    if lw_class is None:
        n_excluded += 1
        continue

    origin = f"intra_{cd}" if cd == ca else "inter_AB"

    # Dedup por par único
    key_lo = tuple(sorted([(cd, nd), (ca, na)]))
    dedup  = (key_lo[0], key_lo[1], lw_class)
    if dedup in seen_pairs:
        continue
    seen_pairs.add(dedup)

    # Registro bilateral
    for (sc, sn, sb), (dc, dn_p) in [
        ((cd, nd, bd), (ca, na)),
        ((ca, na, ba), (cd, nd)),
    ]:
        raw_pairs.append({
            "chain"      : sc,
            "resnum"     : sn,
            "base"       : sb,
            "p_chain"    : dc,
            "p_resnum"   : dn_p,
            "p_base"     : SEQ.get(dc, {}).get(dn_p, "?"),
            "lw_class"   : lw_class,
            "is_can"     : is_can,
            "origin"     : origin,
            "dist_da"    : rec["dist_da"],
            "dist_dha"   : rec["dist_dha"],
            "donor_atom" : ad,
            "acc_atom"   : aa,
        })

print(f"\n[OK] Pares clasificados (bilateral): {len(raw_pairs)}")
print(f"[OK] Pares excluidos (backbone/ribosa): {n_excluded}")

lw_counts = Counter(r["lw_class"] for r in raw_pairs)
print(f"\n[LEONTIS_CLASS encontradas]:")
for lw, cnt in sorted(lw_counts.items()):
    print(f"  {lw}: {cnt} (bilateral)")

# ─────────────────────────────────────────────
# 9. CONTEOS AGREGADOS POR NUCLEOTIDO
# ─────────────────────────────────────────────
# Estructura de nt_agg:
#   key  = (chain, resnum)
#   "interactions": lista de dicts con todos los datos de cada interacción detectada
#   Para conteos rápidos: listas de partners por tipo/origen

nt_agg = defaultdict(lambda: {
    "interactions": [],          # lista completa de interacciones (dicts)
    "bp"        : [],            # todos los partners (chain, resnum)
    "bp_intra"  : [],
    "bp_inter"  : [],
    "nc"        : [],
    "nc_intra"  : [],
    "nc_inter"  : [],
    "can"       : [],
    "can_intra" : [],
    "can_inter" : [],
    "lw_intra"  : [],            # clases LW intrahebra
    "lw_inter"  : [],            # clases LW interhebra
})

for rec in raw_pairs:
    k  = (rec["chain"], rec["resnum"])
    pk = (rec["p_chain"], rec["p_resnum"])
    og = rec["origin"]
    ic = rec["is_can"]

    nt_agg[k]["interactions"].append(rec)
    nt_agg[k]["bp"].append(pk)

    if "intra" in og:
        nt_agg[k]["bp_intra"].append(pk)
        nt_agg[k]["lw_intra"].append(rec["lw_class"])
        if ic:
            nt_agg[k]["can_intra"].append(pk)
        else:
            nt_agg[k]["nc_intra"].append(pk)
    else:
        nt_agg[k]["bp_inter"].append(pk)
        nt_agg[k]["lw_inter"].append(rec["lw_class"])
        if ic:
            nt_agg[k]["can_inter"].append(pk)
        else:
            nt_agg[k]["nc_inter"].append(pk)

    if ic:
        nt_agg[k]["can"].append(pk)
    else:
        nt_agg[k]["nc"].append(pk)

def fmt_partners(lst):
    """Lista de partners como 'A16;B9' ordenada y deduplicada."""
    return ";".join(f"{c}{n}" for c,n in sorted(set(lst))) if lst else ""

def fmt_lw_classes(lst):
    """Conteo de clases LW como 'cWW_CAN:2;SUGAR_sO:1'."""
    if not lst:
        return ""
    c = Counter(lst)
    return ";".join(f"{k}:{v}" for k,v in sorted(c.items()))

LW_COLS = ["cWW_CAN","cWW_NC","tWW","cWS","cSW","tWS","tSW",
           "cHW","tHW","cHS","tHS","cHH","tHH","cSS","tSS","SUGAR_sO","BPH"]

# ─────────────────────────────────────────────
# 10. TABLA DETALLE — una fila por interacción (bilateral)
# Incluye: qué nucleótido, con quién interactúa, qué clase LW, origen
# ─────────────────────────────────────────────

detail_rows = []
for rec in raw_pairs:
    k  = (rec["chain"], rec["resnum"])
    nd = nt_agg[k]
    detail_rows.append({
        "CHAIN"         : rec["chain"],
        "NT_INDEX"      : rec["resnum"],
        "NT_BASE"       : rec["base"],
        "REGION"        : get_region(rec["chain"], rec["resnum"]),
        "LEONTIS_CLASS" : rec["lw_class"],
        "IS_CANONICAL"  : rec["is_can"],
        "PARTNER_CHAIN" : rec["p_chain"],
        "PARTNER_NT"    : rec["p_resnum"],
        "PARTNER_BASE"  : rec["p_base"],
        "PARTNER_REGION": get_region(rec["p_chain"], rec["p_resnum"]),
        "ORIGIN"        : rec["origin"],
        "DONOR_ATOM"    : rec["donor_atom"],
        "ACCEPTOR_ATOM" : rec["acc_atom"],
        "DIST_DA"       : rec["dist_da"],
        "DIST_DHA"      : rec["dist_dha"],
    })

detail_rows.sort(key=lambda r: (r["CHAIN"], r["NT_INDEX"], r["ORIGIN"], r["LEONTIS_CLASS"]))
print(f"\n[DETAIL TABLE] {len(detail_rows)} filas")

# ─────────────────────────────────────────────
# 11. TABLA RESUMEN POR NUCLEOTIDO
# Una fila por cada nucleótido de ambas hebras (incluye nt sin interacciones)
# Columnas: conteos CAN/NC/TOTAL por intra, inter y total + clases LW detectadas
# ─────────────────────────────────────────────

def build_nt_summary():
    summary = []
    for chain_id, seq_dict in SEQ.items():
        for nt_idx in sorted(seq_dict.keys()):
            k  = (chain_id, nt_idx)
            nd = nt_agg[k]   # defaultdict → vacío si sin interacciones

            # Conteos únicos por partner (deduplicados)
            tot_total  = len(set(nd["bp"]))
            tot_intra  = len(set(nd["bp_intra"]))
            tot_inter  = len(set(nd["bp_inter"]))
            can_total  = len(set(nd["can"]))
            can_intra  = len(set(nd["can_intra"]))
            can_inter  = len(set(nd["can_inter"]))
            nc_total   = len(set(nd["nc"]))
            nc_intra   = len(set(nd["nc_intra"]))
            nc_inter   = len(set(nd["nc_inter"]))

            # Conteos por clase LW (intra / inter)
            lw_intra_cnt = Counter(nd["lw_intra"])
            lw_inter_cnt = Counter(nd["lw_inter"])

            row = {
                "CHAIN"           : chain_id,
                "NT_INDEX"        : nt_idx,
                "NT_BASE"         : seq_dict[nt_idx],
                "REGION"          : get_region(chain_id, nt_idx),
                # Totales generales
                "PAIRS_TOTAL"     : tot_total,
                "CAN_TOTAL"       : can_total,
                "NC_TOTAL"        : nc_total,
                # Intrahebra
                "PAIRS_INTRA"     : tot_intra,
                "CAN_INTRA"       : can_intra,
                "NC_INTRA"        : nc_intra,
                # Interhebra
                "PAIRS_INTER"     : tot_inter,
                "CAN_INTER"       : can_inter,
                "NC_INTER"        : nc_inter,
                # Partners como string
                "ALL_PARTNERS"    : fmt_partners(nd["bp"]),
                "CAN_PARTNERS"    : fmt_partners(nd["can"]),
                "NC_PARTNERS"     : fmt_partners(nd["nc"]),
                # Clases LW detectadas por origen
                "LW_CLASSES_INTRA": fmt_lw_classes(nd["lw_intra"]),
                "LW_CLASSES_INTER": fmt_lw_classes(nd["lw_inter"]),
            }
            # Columnas por clase LW — intra e inter separadas
            for lw in LW_COLS:
                row[f"INTRA_{lw}"] = lw_intra_cnt.get(lw, 0)
                row[f"INTER_{lw}"] = lw_inter_cnt.get(lw, 0)

            summary.append(row)
    return summary

nt_summary = build_nt_summary()
print(f"[NT SUMMARY] {len(nt_summary)} nucleótidos registrados")

# ─────────────────────────────────────────────
# 12. TABLA POR REGION
# Agregado por región × hebra × origen (intra/inter/total)
# ─────────────────────────────────────────────

def build_region_table():
    region_rows = []
    for chain_id, region_dict in REGION_MAP.items():
        for region_name, residues in region_dict.items():
            if residues is None:
                region_rows.append({
                    "CHAIN": chain_id, "REGION": region_name,
                    "NT_RANGE": "NA", "N_NT": 0,
                    **{c: "NA" for c in [
                        "PAIRS_TOTAL","CAN_TOTAL","NC_TOTAL",
                        "PAIRS_INTRA","CAN_INTRA","NC_INTRA",
                        "PAIRS_INTER","CAN_INTER","NC_INTER",
                    ]},
                    **{f"INTRA_{lw}": "NA" for lw in LW_COLS},
                    **{f"INTER_{lw}": "NA" for lw in LW_COLS},
                })
                continue

            # Interacciones únicas para esta región
            # (dedup por NT_INDEX + PARTNER_CHAIN + PARTNER_NT + LW_CLASS)
            seen = set()
            region_interactions = []
            for rec in raw_pairs:
                if rec["chain"] == chain_id and rec["resnum"] in residues:
                    dk = (rec["resnum"], rec["p_chain"], rec["p_resnum"], rec["lw_class"])
                    if dk not in seen:
                        seen.add(dk)
                        region_interactions.append(rec)

            def rc(origin_filter=None, canonical=None):
                sub = region_interactions
                if origin_filter:
                    sub = [r for r in sub if r["origin"] == origin_filter]
                if canonical is True:
                    sub = [r for r in sub if r["is_can"]]
                elif canonical is False:
                    sub = [r for r in sub if not r["is_can"]]
                return len(sub)

            def lw_cnt(lw, origin_filter=None):
                sub = region_interactions
                if origin_filter:
                    sub = [r for r in sub if r["origin"] == origin_filter]
                return sum(1 for r in sub if r["lw_class"] == lw)

            intra_key = f"intra_{chain_id}"
            row = {
                "CHAIN"       : chain_id,
                "REGION"      : region_name,
                "NT_RANGE"    : f"nt{residues[0]}-nt{residues[-1]}",
                "N_NT"        : len(residues),
                # Total (intra + inter)
                "PAIRS_TOTAL" : rc(),
                "CAN_TOTAL"   : rc(canonical=True),
                "NC_TOTAL"    : rc(canonical=False),
                # Intrahebra
                "PAIRS_INTRA" : rc(origin_filter=intra_key),
                "CAN_INTRA"   : rc(origin_filter=intra_key, canonical=True),
                "NC_INTRA"    : rc(origin_filter=intra_key, canonical=False),
                # Interhebra
                "PAIRS_INTER" : rc(origin_filter="inter_AB"),
                "CAN_INTER"   : rc(origin_filter="inter_AB", canonical=True),
                "NC_INTER"    : rc(origin_filter="inter_AB", canonical=False),
            }
            for lw in LW_COLS:
                row[f"INTRA_{lw}"] = lw_cnt(lw, origin_filter=intra_key)
                row[f"INTER_{lw}"] = lw_cnt(lw, origin_filter="inter_AB")

            region_rows.append(row)
    return region_rows

region_table = build_region_table()
print(f"[REGION TABLE] {len(region_table)} regiones registradas")

# ─────────────────────────────────────────────
# 13. VARIABLES GLOBALES DEL DUPLEX
# ─────────────────────────────────────────────

def global_counts_from_raw(origin_f=None, chain_f=None):
    """Conteos únicos de pares (dedup por par + clase)."""
    sub = raw_pairs
    if chain_f:   sub = [r for r in sub if r["chain"]  == chain_f]
    if origin_f:  sub = [r for r in sub if r["origin"] == origin_f]
    seen = set()
    uniq = []
    for r in sub:
        k = tuple(sorted([(r["chain"],r["resnum"]),
                          (r["p_chain"],r["p_resnum"])]) + [r["lw_class"]])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    can = sum(1 for r in uniq if r["is_can"])
    lw_c = Counter(r["lw_class"] for r in uniq)
    base = {"PAIRS_TOTAL": len(uniq), "CAN_PAIRS": can, "NC_PAIRS": len(uniq)-can}
    for lw in LW_COLS:
        base[lw] = lw_c.get(lw, 0)
    return base

globals_data = [
    {"SCOPE": "GLOBAL_DUPLEX",   **global_counts_from_raw()},
    {"SCOPE": "INTRA_CYR",       **global_counts_from_raw(chain_f="A", origin_f="intra_A")},
    {"SCOPE": "INTRA_MIR",       **global_counts_from_raw(chain_f="B", origin_f="intra_B")},
    {"SCOPE": "INTERMOLECULAR",  **global_counts_from_raw(origin_f="inter_AB")},
]

print("\n[GLOBAL]")
for g in globals_data:
    print(f"  {g['SCOPE']:<20} TOTAL={g['PAIRS_TOTAL']}  "
          f"CAN={g['CAN_PAIRS']}  NC={g['NC_PAIRS']}")

# ─────────────────────────────────────────────
# 14. EXPORTACION CSV
# ─────────────────────────────────────────────

def write_csv(fname, fieldnames, data):
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)
    print(f"[OUTPUT] {fname}")

# Columnas para cada tabla
DETAIL_COLS = [
    "CHAIN","NT_INDEX","NT_BASE","REGION",
    "LEONTIS_CLASS","IS_CANONICAL",
    "PARTNER_CHAIN","PARTNER_NT","PARTNER_BASE","PARTNER_REGION",
    "ORIGIN","DONOR_ATOM","ACCEPTOR_ATOM","DIST_DA","DIST_DHA",
]

NT_SUMMARY_COLS = (
    ["CHAIN","NT_INDEX","NT_BASE","REGION",
     "PAIRS_TOTAL","CAN_TOTAL","NC_TOTAL",
     "PAIRS_INTRA","CAN_INTRA","NC_INTRA",
     "PAIRS_INTER","CAN_INTER","NC_INTER",
     "ALL_PARTNERS","CAN_PARTNERS","NC_PARTNERS",
     "LW_CLASSES_INTRA","LW_CLASSES_INTER"]
    + [f"INTRA_{lw}" for lw in LW_COLS]
    + [f"INTER_{lw}" for lw in LW_COLS]
)

REGION_COLS = (
    ["CHAIN","REGION","NT_RANGE","N_NT",
     "PAIRS_TOTAL","CAN_TOTAL","NC_TOTAL",
     "PAIRS_INTRA","CAN_INTRA","NC_INTRA",
     "PAIRS_INTER","CAN_INTER","NC_INTER"]
    + [f"INTRA_{lw}" for lw in LW_COLS]
    + [f"INTER_{lw}" for lw in LW_COLS]
)

GLOBAL_COLS = ["SCOPE","PAIRS_TOTAL","CAN_PAIRS","NC_PAIRS"] + LW_COLS

write_csv(f"{PREFIX}_per_nt_detail.csv",  DETAIL_COLS,     detail_rows)
write_csv(f"{PREFIX}_per_nt_summary.csv", NT_SUMMARY_COLS, nt_summary)
write_csv(f"{PREFIX}_per_region.csv",     REGION_COLS,     region_table)
write_csv(f"{PREFIX}_global.csv",         GLOBAL_COLS,     globals_data)

# ─────────────────────────────────────────────
# 15. REPORTE .TXT
# ─────────────────────────────────────────────

today = datetime.now().strftime("%d/%m/%Y %H:%M")
rlines = [
    "="*70,
    "REPORTE BASE_PAIRS — DUPLEX WT_miR7  [v6]",
    "="*70, "",
    f"HERRAMIENTA : UCSF ChimeraX 1.9",
    f"FECHA       : {today}",
    f"INPUT       : {PDB_PATH}",
    f"OUTPUT DIR  : {OUTPUT_DIR}",
    f"METODO      : Log 'hbonds relax true' ({len(hb_records)} H-bonds parseados)",
    "              Filtro: excluye ribosa-ribosa (O2'→O4', etc.)",
    "              LW: edges por atomos + orientacion vectores C1'→N1/N9",
    "", "ESTRUCTURA:",
    "  Hebra A — CYR_WT  (26 nt): ANCHOR(26) | P1(19-25) | B(15-18) | P2(1-14)",
    "  Hebra B — miR-7-5p(24 nt): ANCHOR(1)  | P1(2-8)   | B(9-10)  | P2(11-24)",
    "", "-"*70, "CLASES LW DETECTADAS (pares únicos)", "-"*70,
]
all_lw = Counter(r["lw_class"] for r in raw_pairs)
for lw, c in sorted(all_lw.items()):
    rlines.append(f"  {lw:<15}: {c//2} pares únicos ({c} bilateral)")

rlines += ["", "-"*70, "VARIABLES GLOBALES", "-"*70]
for g in globals_data:
    rlines.append(f"  {g['SCOPE']:<20} TOTAL={g['PAIRS_TOTAL']}  "
                  f"CAN={g['CAN_PAIRS']}  NC={g['NC_PAIRS']}")
rlines.append("")

rlines += ["-"*70, "POR REGION", "-"*70]
for r in region_table:
    if r["PAIRS_TOTAL"] == "NA":
        rlines.append(f"  {r['CHAIN']}.{r['REGION']:<15}: NA")
        continue
    rlines.append(
        f"  {r['CHAIN']}.{r['REGION']:<15} ({r['NT_RANGE']}, n={r['N_NT']})  "
        f"TOTAL={r['PAIRS_TOTAL']}  CAN={r['CAN_TOTAL']}  NC={r['NC_TOTAL']}  "
        f"[INTRA: CAN={r['CAN_INTRA']} NC={r['NC_INTRA']}]  "
        f"[INTER: CAN={r['CAN_INTER']} NC={r['NC_INTER']}]"
    )

rlines += [
    "", "-"*70, "NOTAS", "-"*70,
    "  - cWW_CAN : A-U / G-C / G-U en orientacion cis WW.",
    "  - SUGAR_sO: O2' ribosa contactando atomo de BASE.",
    "  - BPH     : fosfato contactando base.",
    "  - Excluidos: O2'→O4', ribosa-ribosa (backbone no-LW).",
    "  - _per_nt_detail.csv  : una fila por interacción (bilateral).",
    "  - _per_nt_summary.csv : una fila por nucleótido (todos los nt).",
    "  - _per_region.csv     : agregado por región × hebra.",
    "  - _global.csv         : conteos globales del dúplex.",
    "="*70,
]

rpath = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte.txt")
with open(rpath, "w", encoding="utf-8") as f:
    f.write("\n".join(rlines))
print(f"[OUTPUT] {PREFIX}_reporte.txt")
print(f"\n[DONE] {len(detail_rows)} interacciones | {len(nt_summary)} nucleótidos | {len(region_table)} regiones")
