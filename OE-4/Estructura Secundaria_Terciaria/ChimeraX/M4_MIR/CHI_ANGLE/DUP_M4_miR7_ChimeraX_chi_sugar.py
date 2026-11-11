# =============================================================================
# DUP_M4_miR7_ChimeraX_chi_sugar.py
# HERRAMIENTA : UCSF ChimeraX 1.9 — Python API interna
# VARIABLES   : CHI_ANGLE, SUGAR_PUCKER
# DUPLEX      : M4 (Chain A, 20 nt) : miR-7-5p (Chain B, 24 nt)
# SECUENCIA A : 5-AACAACAAAAUCAUCUUCCA-3 (20 nt)
# SECUENCIA B : 5-UGGAAGACUAGUGAUUUUGUUGUU-3 (24 nt)
# PIPELINE    : OE-4 — Tesis Maestria Bioquimica UNAM
# =============================================================================

import csv, math, os
from datetime import datetime
import numpy as np

MODEL_INDEX    = 0
OUTPUT_DIR_CHI = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-4\6. ChimeraX\5.M4_MIR\CHI_SUGAR"
PDB_INPUT      = "M4_miR7.pdb"

REGION_MAP_A = {}
REGION_MAP_A[20] = 'ANCHOR_CYR'
for i in range(14, 20): REGION_MAP_A[i] = 'P1_CYR'
for i in range(1, 14): REGION_MAP_A[i] = 'P2_CYR'

REGION_MAP_B = {}
REGION_MAP_B[1] = 'ANCHOR_MIR'
for i in range(2, 8): REGION_MAP_B[i] = 'P1_MIR'
for i in range(8, 12): REGION_MAP_B[i] = 'B_MIR'
for i in range(12, 25): REGION_MAP_B[i] = 'P2_MIR'

REGION_ORDER_A = ['ANCHOR_CYR', 'P1_CYR', 'B_CYR', 'P2_CYR', 'OVERHANG_CYR']
REGION_ORDER_B = ['ANCHOR_MIR', 'P1_MIR', 'B_MIR', 'P2_MIR', 'OVERHANG_MIR']

REGION_RANGES_A = {
    'ANCHOR_CYR': (20, 20),
    'P1_CYR': (14, 19),
    'B_CYR': None,
    'P2_CYR': (1, 13),
    'OVERHANG_CYR': None,
}
REGION_RANGES_B = {
    'ANCHOR_MIR': (1, 1),
    'P1_MIR': (2, 7),
    'B_MIR': (8, 11),
    'P2_MIR': (12, 24),
    'OVERHANG_MIR': None,
}

REGION_LABEL_A = {
    'ANCHOR_CYR': 'A:20  (n=1) (extremo 3p)',
    'P1_CYR': 'A:14-19  (n=6)',
    'B_CYR': 'NA',
    'P2_CYR': 'A:1-13  (n=13) (extremo 5p)',
    'OVERHANG_CYR': 'NA',
}
REGION_LABEL_B = {
    'ANCHOR_MIR': 'B:1  (n=1) (extremo 5p)',
    'P1_MIR': 'B:2-7  (n=6)',
    'B_MIR': 'B:8-11  (n=4)',
    'P2_MIR': 'B:12-24  (n=13) (extremo 3p)',
    'OVERHANG_MIR': 'NA',
}

COLORS = {
    "ANCHOR_CYR":"#E63946","P1_CYR":"#2196F3","B_CYR":"#FF9800",
    "P2_CYR":"#4CAF50","OVERHANG_CYR":"#9C27B0",
    "ANCHOR_MIR":"#C62828","P1_MIR":"#1565C0","B_MIR":"#E65100",
    "P2_MIR":"#2E7D32","OVERHANG_MIR":"#6A1B9A",
}

def get_atom(res, name):
    for atom in res.atoms:
        if atom.name == name: return atom
    return None

def calc_dihedral(a1, a2, a3, a4):
    p = [np.array(a.coord) for a in [a1, a2, a3, a4]]
    b1=p[1]-p[0]; b2=p[2]-p[1]; b3=p[3]-p[2]
    n1=np.cross(b1,b2); n2=np.cross(b2,b3)
    m1=np.cross(n1,b2/np.linalg.norm(b2))
    return math.degrees(math.atan2(np.dot(m1,n2),np.dot(n1,n2)))

def get_chi(res):
    o4p=get_atom(res,"O4'"); c1p=get_atom(res,"C1'")
    rn=res.name.strip().upper()
    if rn in ("A","G","DA","DG"):
        n_at=get_atom(res,"N9"); c_at=get_atom(res,"C4")
    elif rn in ("C","U","DC","DT"):
        n_at=get_atom(res,"N1"); c_at=get_atom(res,"C2")
    else: return None
    if None in (o4p,c1p,n_at,c_at): return None
    return calc_dihedral(o4p,c1p,n_at,c_at)

def classify_chi(chi):
    if chi is None: return "NA"
    return "anti" if (chi>=90 or chi<=-90) else "syn"

def get_pseudorotation(res):
    names=["C4'","O4'","C1'","C2'","C3'"]
    coords={}
    for nm in names:
        a=get_atom(res,nm)
        if a is None: return None
        coords[nm]=np.array(a.coord)
    def dih(p0,p1,p2,p3):
        b1=p1-p0;b2=p2-p1;b3=p3-p2
        n1=np.cross(b1,b2);n2=np.cross(b2,b3)
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
    if P is None: return "NA"
    P=P%360
    if P<=36 or P>=324:   return "C3-endo"
    elif 144<=P<=180:     return "C2-endo"
    elif 36<P<90:         return "C4-exo"
    elif 90<=P<126:       return "O4-endo"
    elif 126<=P<144:      return "C1-exo"
    elif 180<P<216:       return "C3-exo"
    elif 216<=P<252:      return "C4-endo"
    elif 252<=P<288:      return "O4-exo"
    elif 288<=P<324:      return "C1-endo"
    else:                 return "other"

def compute_stats(rows_subset, label):
    chi_vals=[r["CHI_ANGLE"] for r in rows_subset if isinstance(r["CHI_ANGLE"],float)]
    p_vals  =[r["PSEUDO_P"]  for r in rows_subset if isinstance(r["PSEUDO_P"],float)]
    n=len(rows_subset)
    n_anti=sum(1 for r in rows_subset if r["CHI_CLASS"]=="anti")
    n_c3  =sum(1 for r in rows_subset if r["SUGAR_PUCKER"]=="C3-endo")
    return {
        "REGION":label,"N_NT":n,
        "CHI_ANGLE_PCT_ANTI":       round(n_anti/n*100,2)                     if n           else "NA",
        "CHI_ANGLE_MEAN":           round(float(np.mean(chi_vals)),3)          if chi_vals    else "NA",
        "CHI_ANGLE_SD":             round(float(np.std(chi_vals,ddof=1)),3)    if len(chi_vals)>1 else "NA",
        "SUGAR_PUCKER_PCT_C3_ENDO": round(n_c3/n*100,2)                       if n           else "NA",
        "SUGAR_PUCKER_MEAN_P":      round(float(np.mean(p_vals)),3)            if p_vals      else "NA",
        "SUGAR_PUCKER_SD_P":        round(float(np.std(p_vals,ddof=1)),3)      if len(p_vals)>1 else "NA",
    }

def na_stats(label):
    return {"REGION":label,"N_NT":"NA",
            "CHI_ANGLE_PCT_ANTI":"NA","CHI_ANGLE_MEAN":"NA","CHI_ANGLE_SD":"NA",
            "SUGAR_PUCKER_PCT_C3_ENDO":"NA","SUGAR_PUCKER_MEAN_P":"NA","SUGAR_PUCKER_SD_P":"NA"}

def cx(session,cmd):
    from chimerax.core.commands import run as cx_run
    cx_run(session,cmd)


def apply_colors(session):
    cx(session, "color #1/A:20 " + COLORS["ANCHOR_CYR"])
    cx(session, "color #1/A:14-19 " + COLORS["P1_CYR"])
    cx(session, "color #1/A:1-13 " + COLORS["P2_CYR"])
    cx(session, "color #1/B:1 " + COLORS["ANCHOR_MIR"])
    cx(session, "color #1/B:2-7 " + COLORS["P1_MIR"])
    cx(session, "color #1/B:8-11 " + COLORS["B_MIR"])
    cx(session, "color #1/B:12-24 " + COLORS["P2_MIR"])

def run(session):
    os.makedirs(OUTPUT_DIR_CHI, exist_ok=True)
    model  = session.models[MODEL_INDEX]
    today  = datetime.now().strftime("%d/%m/%Y")
    res_A  = [r for r in model.residues if r.chain_id=="A"]
    res_B  = [r for r in model.residues if r.chain_id=="B"]

    def build_rows(residues, region_map, chain_id):
        rows=[]
        for res in residues:
            chi=get_chi(res); P=get_pseudorotation(res)
            rows.append({"CHAIN":chain_id,"RESNUM":res.number,
                "RESNAME":res.name.strip().upper(),
                "REGION":region_map.get(res.number,"UNK"),
                "CHI_ANGLE":round(chi,3) if chi is not None else "NA",
                "CHI_CLASS":classify_chi(chi),
                "PSEUDO_P":round(P,3) if P is not None else "NA",
                "SUGAR_PUCKER":classify_pucker(P)})
        return rows

    rows_A   = build_rows(res_A, REGION_MAP_A, "A")
    rows_B   = build_rows(res_B, REGION_MAP_B, "B")
    rows_all = rows_A + rows_B

    csv_nt = os.path.join(OUTPUT_DIR_CHI, "DUP_M4_miR7_chi_sugar_per_nt.csv")
    with open(csv_nt,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows_all[0].keys()); w.writeheader(); w.writerows(rows_all)

    stats=[]
    for region in REGION_ORDER_A:
        if REGION_RANGES_A[region] is None: stats.append(na_stats(region))
        else: stats.append(compute_stats([r for r in rows_A if r["REGION"]==region],region))
    for region in REGION_ORDER_B:
        if REGION_RANGES_B[region] is None: stats.append(na_stats(region))
        else: stats.append(compute_stats([r for r in rows_B if r["REGION"]==region],region))
    stats.append(compute_stats(rows_A,"GLOBAL_CYR"))
    stats.append(compute_stats(rows_B,"GLOBAL_MIR"))
    stats.append(compute_stats(rows_all,"GLOBAL_DUPLEX"))

    csv_stats = os.path.join(OUTPUT_DIR_CHI, "DUP_M4_miR7_chi_sugar_stats.csv")
    with open(csv_stats,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=stats[0].keys()); w.writeheader(); w.writerows(stats)

    g_dup=next(s for s in stats if s["REGION"]=="GLOBAL_DUPLEX")
    g_cyr=next(s for s in stats if s["REGION"]=="GLOBAL_CYR")
    g_mir=next(s for s in stats if s["REGION"]=="GLOBAL_MIR")
    reg_A=[s for s in stats if s["REGION"] in REGION_ORDER_A]
    reg_B=[s for s in stats if s["REGION"] in REGION_ORDER_B]
    all_lbl={**REGION_LABEL_A,**REGION_LABEL_B}

    report_path = os.path.join(OUTPUT_DIR_CHI, "DUP_M4_miR7_chi_sugar_reporte.txt")
    with open(report_path,"w",encoding="utf-8") as f:
        f.write("="*70+"\n")
        f.write("REPORTE CHI ANGLE + SUGAR PUCKER - DUPLEX M4_miR7\n")
        f.write("="*70+"\n\n")
        f.write("HERRAMIENTA      : UCSF ChimeraX - Python API interna\n")
        f.write("VERSION          : verificar en ChimeraX -> Help > About\n")
        f.write("FECHA            : " + today + "\n")
        f.write("INPUT            : M4_miR7.pdb\n")
        f.write("PARAMETROS       : CHI: O4-C1-N9-C4 (A/G) | O4-C1-N1-C2 (C/U)\n")
        f.write("                   anti = chi >= 90 o <= -90 | syn = [-90, 90]\n")
        f.write("                   Pseudorotacion P: Altona & Sundaralingam 1972\n")
        f.write("                   SD ddof=1 (muestral)\n\n")
        f.write("CADENAS\n")
        f.write("  Chain A - M4  : 5-AACAACAAAAUCAUCUUCCA-3 (20 nt)\n")
        f.write("                       nt 1=extremo 5p, nt 20=extremo 3p\n")
        f.write("  Chain B - miR-7-5p: 5-UGGAAGACUAGUGAUUUUGUUGUU-3 (24 nt)\n")
        f.write("                       nt 1=extremo 5p, nt 24=extremo 3p\n\n")
        f.write("ELEMENTOS ESTRUCTURALES\n")
        f.write("  Chain A (M4)\n")
        f.write("    ANCHOR_CYR     : A:20\\n")
        f.write("    P1_CYR         : A:14-19\\n")
        f.write("    B_CYR          : NA\\n")
        f.write("    P2_CYR         : A:1-13\\n")
        f.write("    OVERHANG_CYR   : NA\\n")
        f.write("  Chain B (miR-7-5p)\n")
        f.write("    ANCHOR_MIR     : B:1\\n")
        f.write("    P1_MIR         : B:2-7\\n")
        f.write("    B_MIR          : B:8-11\\n")
        f.write("    P2_MIR         : B:12-24\\n")
        f.write("    OVERHANG_MIR   : NA\\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("ARCHIVOS DE SALIDA\n")
        f.write("-"*70+"\n")
        f.write("  CSV per nt  : DUP_M4_miR7_chi_sugar_per_nt.csv\n")
        f.write("  CSV stats   : DUP_M4_miR7_chi_sugar_stats.csv\n\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES - GLOBAL DUPLEX\n")
        f.write("-"*70+"\n")
        for k,v in g_dup.items():
            if k in ("REGION","N_NT"): continue
            f.write("  " + k.ljust(30) + ": " + str(v) + "\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES - GLOBAL CHAIN A (M4)\n")
        f.write("-"*70+"\n")
        for k,v in g_cyr.items():
            if k in ("REGION","N_NT"): continue
            f.write("  " + k.ljust(30) + ": " + str(v) + "\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES - GLOBAL CHAIN B (miR-7-5p)\n")
        f.write("-"*70+"\n")
        for k,v in g_mir.items():
            if k in ("REGION","N_NT"): continue
            f.write("  " + k.ljust(30) + ": " + str(v) + "\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES - POR ELEMENTO ESTRUCTURAL\n")
        f.write("-"*70+"\n")
        f.write("\n  -- Chain A (M4) --\n")
        for s in reg_A:
            lbl=all_lbl.get(s["REGION"],"")
            f.write("\n  ELEMENTO : " + s["REGION"] + "  (" + lbl + ")\n")
            if s["N_NT"]=="NA":
                f.write("    [Region ausente - NA]\n")
            else:
                for k,v in s.items():
                    if k in ("REGION","N_NT"): continue
                    f.write("    " + k.ljust(30) + ": " + str(v) + "\n")
        f.write("\n  -- Chain B (miR-7-5p) --\n")
        for s in reg_B:
            lbl=all_lbl.get(s["REGION"],"")
            f.write("\n  ELEMENTO : " + s["REGION"] + "  (" + lbl + ")\n")
            if s["N_NT"]=="NA":
                f.write("    [Region ausente - NA]\n")
            else:
                for k,v in s.items():
                    if k in ("REGION","N_NT"): continue
                    f.write("    " + k.ljust(30) + ": " + str(v) + "\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES - POR NUCLEOTIDO\n")
        f.write("-"*70+"\n")
        f.write("  CH  NT    RES   REGION         CHI_ANGLE CHI_CLASS    PSEUDO_P SUGAR_PUCKER\n")
        for r in rows_all:
            f.write("  " + r["CHAIN"].ljust(3) + " " + str(r["RESNUM"]).ljust(5) + " " +
                    r["RESNAME"].ljust(5) + " " + r["REGION"].ljust(14) + " " +
                    str(r["CHI_ANGLE"]).rjust(10) + " " + r["CHI_CLASS"].ljust(10) + " " +
                    str(r["PSEUDO_P"]).rjust(10) + "  " + r["SUGAR_PUCKER"] + "\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("NOTAS\n")
        f.write("-"*70+"\n")
        f.write("  - CHI: Saenger 1984.\n")
        f.write("  - Pseudorotacion P: Altona & Sundaralingam 1972.\n")
        f.write("  - C3-endo: pucker canonico RNA A-form.\n")
        f.write("  - Ambas cadenas numeradas 5p->3p: nt 1=extremo 5p.\n")
        f.write("  - Region ausente=NA.\n")
        f.write("="*70+"\n")

    cx(session,"set bgcolor white")
    cx(session,"lighting simple")
    cx(session,"nucleotides #1 tube/slab")
    cx(session,"show #1 cartoons")
    apply_colors(session)
    cx(session,"view #1")

    print("="*70)
    print("CHI ANGLE + SUGAR PUCKER - DUPLEX M4_miR7")
    print("="*70)
    print("  CH  NT    RES   REGION         CHI_ANGLE CHI_CLASS    PSEUDO_P  SUGAR_PUCKER")
    for r in rows_all:
        print("  " + r["CHAIN"].ljust(3) + " " + str(r["RESNUM"]).ljust(5) + " " +
              r["RESNAME"].ljust(5) + " " + r["REGION"].ljust(14) + " " +
              str(r["CHI_ANGLE"]).rjust(10) + " " + r["CHI_CLASS"].ljust(10) + " " +
              str(r["PSEUDO_P"]).rjust(10) + "  " + r["SUGAR_PUCKER"])
    print("  CSV per_nt -> " + csv_nt)
    print("  CSV stats  -> " + csv_stats)
    print("  Reporte    -> " + report_path)
    print("="*70)

run(session)
