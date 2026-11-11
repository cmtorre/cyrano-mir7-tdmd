# =============================================================================
# MRE_CYR_M1_ChimeraX_chi_sugar.py
# HERRAMIENTA : ChimeraX (UCSF ChimeraX) — Python API interna
# VARIABLES   : CHI_ANGLE, SUGAR_PUCKER (pseudorotación P, clasificación)
# NIVELES     : Global | Por elemento (ANCHOR/P1/BULGE/P2) | Por nucleótido
# SECUENCIA   : AACAACAAAAUCACCAAUCUUCCA (24 nt, 3'→5')
# DIRECCIÓN   : 3'→5' — numeración PDB desde extremo 3' del MRE
# EJECUCIÓN   : En consola ChimeraX → open "ruta\MRE_CYR_M1_ChimeraX_chi_sugar.py"
# AUTOR       : Pipeline OE2 — MRE_CYR_M1
# =============================================================================

import csv
import math
import os
from datetime import datetime
import numpy as np

MODEL_INDEX = 0
OUTPUT_DIR = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-2\Var_E-Ter-CYR\2. ChimeraX\2. MRE_CYR_M1\CHI_SUGAR_M1"
PDB_INPUT  = "MRE_CYR_M1.pdb"

# REGION_MAP — numeración 3'→5' (nt 1 = extremo 3', nt 24 = extremo 5'/ANCHOR)
REGION_MAP = {}
REGION_MAP[24] = "ANCHOR"
for i in range(18, 24): REGION_MAP[i] = "P1"
for i in range(15, 18):  REGION_MAP[i] = "BULGE"
for i in range(1, 15):  REGION_MAP[i] = "P2"

REGION_RANGES = {
    "ANCHOR": (24, 24),
    "P1":     (18,  23),
    "BULGE":  (15,  17),
    "P2":     (1,  14),
}

# ---------------------------------------------------------------------------
def get_atom(res, name):
    for atom in res.atoms:
        if atom.name == name:
            return atom
    return None

def calc_dihedral(a1, a2, a3, a4):
    p = [np.array(a.coord) for a in [a1, a2, a3, a4]]
    b1 = p[1] - p[0]; b2 = p[2] - p[1]; b3 = p[3] - p[2]
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return math.degrees(math.atan2(np.dot(m1, n2), np.dot(n1, n2)))

def get_chi(res):
    o4p = get_atom(res, "O4'"); c1p = get_atom(res, "C1'")
    rn  = res.name.strip().upper()
    if rn in ("A", "G", "DA", "DG"):
        n_at = get_atom(res, "N9"); c_at = get_atom(res, "C4")
    elif rn in ("C", "U", "DC", "DT"):
        n_at = get_atom(res, "N1"); c_at = get_atom(res, "C2")
    else:
        return None
    if None in (o4p, c1p, n_at, c_at):
        return None
    return calc_dihedral(o4p, c1p, n_at, c_at)

def classify_chi(chi):
    if chi is None: return "N/A"
    return "anti" if (chi >= 90 or chi <= -90) else "syn"

def get_pseudorotation(res):
    names = ["C4'", "O4'", "C1'", "C2'", "C3'"]
    coords = {}
    for nm in names:
        a = get_atom(res, nm)
        if a is None: return None
        coords[nm] = np.array(a.coord)
    def dih(p0, p1, p2, p3):
        b1=p1-p0; b2=p2-p1; b3=p3-p2
        n1=np.cross(b1,b2); n2=np.cross(b2,b3)
        m1=np.cross(n1,b2/np.linalg.norm(b2))
        return math.degrees(math.atan2(np.dot(m1,n2),np.dot(n1,n2)))
    v0=dih(coords["C4'"],coords["O4'"],coords["C1'"],coords["C2'"])
    v1=dih(coords["O4'"],coords["C1'"],coords["C2'"],coords["C3'"])
    v2=dih(coords["C1'"],coords["C2'"],coords["C3'"],coords["C4'"])
    v3=dih(coords["C2'"],coords["C3'"],coords["C4'"],coords["O4'"])
    v4=dih(coords["C3'"],coords["C4'"],coords["O4'"],coords["C1'"])
    sin36=math.sin(math.radians(36)); sin72=math.sin(math.radians(72))
    denom=2.0*v2*(sin36+sin72)
    if denom==0: return None
    P=math.degrees(math.atan2((v4+v1)-(v3+v0),denom))
    return P+360.0 if P<0 else P

def classify_pucker(P):
    if P is None: return "N/A"
    P=P%360
    if P<=36 or P>=324: return "C3'-endo"
    elif 144<=P<=180:   return "C2'-endo"
    elif 36<P<90:       return "C4'-exo"
    elif 90<=P<126:     return "O4'-endo"
    elif 126<=P<144:    return "C1'-exo"
    elif 180<P<216:     return "C3'-exo"
    elif 216<=P<252:    return "C4'-endo"
    elif 252<=P<288:    return "O4'-exo"
    elif 288<=P<324:    return "C1'-endo"
    else:               return "other"

def compute_stats(rows_subset, label):
    chi_vals=[r["CHI_ANGLE"] for r in rows_subset if isinstance(r["CHI_ANGLE"],float)]
    p_vals  =[r["PSEUDO_P"]  for r in rows_subset if isinstance(r["PSEUDO_P"], float)]
    n=len(rows_subset)
    n_anti=sum(1 for r in rows_subset if r["CHI_CLASS"]=="anti")
    n_c3  =sum(1 for r in rows_subset if r["SUGAR_PUCKER"]=="C3'-endo")
    return {
        "REGION": label, "N_NT": n,
        "CHI_ANGLE_PCT_ANTI":       round(n_anti/n*100,2)                          if n        else "N/A",
        "CHI_ANGLE_MEAN":           round(float(np.mean(chi_vals)),3)               if chi_vals else "N/A",
        "CHI_ANGLE_SD":             round(float(np.std(chi_vals,ddof=1)),3)         if len(chi_vals)>1 else "N/A",
        "SUGAR_PUCKER_PCT_C3_ENDO": round(n_c3/n*100,2)                            if n        else "N/A",
        "SUGAR_PUCKER_MEAN_P":      round(float(np.mean(p_vals)),3)                 if p_vals   else "N/A",
        "SUGAR_PUCKER_SD_P":        round(float(np.std(p_vals,ddof=1)),3)           if len(p_vals)>1  else "N/A",
    }

# ---------------------------------------------------------------------------
def run(session):
    out_path = OUTPUT_DIR
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        print("DEBUG carpeta creada OK")
    else:
        print("DEBUG carpeta encontrada OK")

    model=session.models[MODEL_INDEX]; residues=model.residues
    today=datetime.now().strftime("%d/%m/%Y")

    rows=[]
    for res in residues:
        seqnum=res.number; chi=get_chi(res); P_ang=get_pseudorotation(res)
        rows.append({
            "RESNUM":seqnum, "RESNAME":res.name.strip().upper(),
            "REGION":REGION_MAP.get(seqnum,"UNK"),
            "CHI_ANGLE":round(chi,3)   if chi   is not None else "N/A",
            "CHI_CLASS":classify_chi(chi),
            "PSEUDO_P": round(P_ang,3) if P_ang is not None else "N/A",
            "SUGAR_PUCKER":classify_pucker(P_ang),
        })

    csv_nt=os.path.join(out_path,"MRE_CYR_M1_ChimeraX_chi_sugar_per_nt.csv")
    with open(csv_nt,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    stats=[]
    for region in ["ANCHOR", "P1", "BULGE", "P2"]:
        subset=[r for r in rows if r["REGION"]==region]
        stats.append(compute_stats(subset,region))
    stats.append(compute_stats(rows,"GLOBAL"))

    csv_stats=os.path.join(out_path,"MRE_CYR_M1_ChimeraX_chi_sugar_stats.csv")
    with open(csv_stats,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=stats[0].keys()); w.writeheader(); w.writerows(stats)

    global_stats=[s for s in stats if s["REGION"]=="GLOBAL"][0]
    region_stats=[s for s in stats if s["REGION"]!="GLOBAL"]

    report_path=os.path.join(out_path,"MRE_CYR_M1_ChimeraX_chi_sugar_reporte.txt")
    with open(report_path,"w",encoding="utf-8") as f:
        f.write("="*70+"\n")
        f.write("REPORTE CHI ANGLE + SUGAR PUCKER - MRE_CYR_M1\n")
        f.write("="*70+"\n\n")
        f.write(f"HERRAMIENTA      : UCSF ChimeraX — Python API interna\n")
        f.write(f"VERSION          : verificar en ChimeraX → Help > About\n")
        f.write(f"FECHA            : {today}\n")
        f.write(f"INPUT            : MRE_CYR_M1.pdb\n")
        f.write(f"PARAMETROS       : default\n")
        f.write(f"                   CHI: O4'-C1'-N9-C4 (A/G) | O4'-C1'-N1-C2 (C/U)\n")
        f.write(f"                   anti = chi ≥ 90° o ≤ -90° | syn = chi en [-90°, 90°]\n")
        f.write(f"                   Pseudorotación P: Altona & Sundaralingam 1972\n")
        f.write(f"                   SD calculada con ddof=1 (muestral)\n\n")
        f.write(f"SECUENCIA        : 3'-AACAACAAAAUCACCAAUCUUCCA-5' (24 nt)\n")
        f.write(f"DIRECCIÓN        : 3'→5' — numeración PDB desde extremo 3' del MRE\n\n")
        f.write(f"ELEMENTOS ESTRUCTURALES (MRE_CYR_M1 — 24 nt)\n")
        f.write(f"                   ANCHOR : nt 24\n")
        f.write(f"                   P1     : nt 18–23\n")
        f.write(f"                   BULGE  : nt 15–17\n")
        f.write(f"                   P2     : nt 1–14\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("ARCHIVOS DE SALIDA\n")
        f.write("-"*70+"\n")
        f.write(f"  CSV por nt      : MRE_CYR_M1_ChimeraX_chi_sugar_per_nt.csv\n")
        f.write(f"  CSV estadísticas: MRE_CYR_M1_ChimeraX_chi_sugar_stats.csv\n\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES EXTRAIDAS - GLOBAL\n")
        f.write("-"*70+"\n")
        for k,v2 in global_stats.items():
            if k in ("REGION","N_NT"): continue
            f.write(f"  {k:<30}: {v2}\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES EXTRAIDAS - POR ELEMENTO ESTRUCTURAL\n")
        f.write("-"*70+"\n")
        rng_lbl={
            "ANCHOR": "nt 24",
            "P1":     "nt 18-23",
            "BULGE":  "nt 15-17",
            "P2":     "nt 1-14",
        }
        for s in region_stats:
            f.write(f"\n  ELEMENTO : {s['REGION']}  ({rng_lbl[s['REGION']]}, n={s['N_NT']})\n")
            for k,v2 in s.items():
                if k in ("REGION","N_NT"): continue
                f.write(f"    {k:<30}: {v2}\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES EXTRAIDAS - POR NUCLEOTIDO\n")
        f.write("-"*70+"\n")
        f.write(f"  {'NT':<5} {'RES':<5} {'REGION':<8} {'CHI_ANGLE':>10} {'CHI_CLASS':<10} {'PSEUDO_P':>10} SUGAR_PUCKER\n")
        for r in rows:
            f.write(f"  {r['RESNUM']:<5} {r['RESNAME']:<5} {r['REGION']:<8} {str(r['CHI_ANGLE']):>10} {r['CHI_CLASS']:<10} {str(r['PSEUDO_P']):>10}  {r['SUGAR_PUCKER']}\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("NOTAS GENERALES\n")
        f.write("-"*70+"\n")
        f.write("  - CHI: Saenger 1984, Principles of Nucleic Acid Structure.\n")
        f.write("  - Pseudorotacion P: Altona & Sundaralingam 1972.\n")
        f.write("  - C3'-endo: pucker canonico RNA A-form.\n")
        f.write("  - Numeracion 3'→5': nt 1 = extremo 3', nt 24 = ANCHOR (extremo 5').\n")
        f.write("  - Verificar VERSION en ChimeraX → Help > About.\n")
        f.write("="*70+"\n")

    print("\n"+"="*70)
    print("CHI ANGLE + SUGAR PUCKER — MRE_CYR_M1")
    print("="*70)
    for r in rows:
        print(f"{r['RESNUM']:<5} {r['RESNAME']:<5} {r['REGION']:<8} {str(r['CHI_ANGLE']):>10} {r['CHI_CLASS']:<10} {str(r['PSEUDO_P']):>10}  {r['SUGAR_PUCKER']}")
    print(f"\n✅ CSV per_nt     → {csv_nt}")
    print(f"✅ CSV stats      → {csv_stats}")
    print(f"✅ Reporte .txt   → {report_path}")
    print("="*70+"\n")

print("VERSION_TEST_V2")
run(session)
