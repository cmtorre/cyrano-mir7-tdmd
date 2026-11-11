# =============================================================================
# DUP_M4_miR7_ChX_rg_enddist.py
# HERRAMIENTA : UCSF ChimeraX 1.9 — Python API interna
# VARIABLES   : RG, END_DIST
# DUPLEX      : M4 (Chain A, 20 nt) : miR-7-5p (Chain B, 24 nt)
# SECUENCIA A : 5-AACAACAAAAUCAUCUUCCA-3 (20 nt)
# SECUENCIA B : 5-UGGAAGACUAGUGAUUUUGUUGUU-3 (24 nt)
# PIPELINE    : OE-4 — Tesis Maestria Bioquimica UNAM
# =============================================================================

import csv, math, os
from datetime import datetime
import numpy as np

MODEL_INDEX   = 0
PDB_INPUT     = "M4_miR7.pdb"
OUTPUT_DIR_RG = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-4\6. ChimeraX\5.M4_MIR\RG_END_DIST"

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

COLORS = {
    "ANCHOR_CYR":"#E63946","P1_CYR":"#2196F3","B_CYR":"#FF9800",
    "P2_CYR":"#4CAF50","OVERHANG_CYR":"#9C27B0",
    "ANCHOR_MIR":"#C62828","P1_MIR":"#1565C0","B_MIR":"#E65100",
    "P2_MIR":"#2E7D32","OVERHANG_MIR":"#6A1B9A",
}

ATOMIC_MASS={"H":1.008,"C":12.011,"N":14.007,"O":15.999,"P":30.974,"S":32.06,"MG":24.305,"ZN":65.38,"FE":55.845}

def get_mass(atom):
    elem=atom.element.name.upper() if hasattr(atom,"element") else atom.name[0].upper()
    return ATOMIC_MASS.get(elem,12.0)

def calc_rg(atoms_list):
    if not atoms_list: return None
    coords=np.array([a.coord for a in atoms_list])
    masses=np.array([get_mass(a) for a in atoms_list])
    total_mass=masses.sum()
    if total_mass==0: return None
    cm=np.average(coords,weights=masses,axis=0)
    dist_sq=np.sum((coords-cm)**2,axis=1)
    return round(math.sqrt(np.dot(masses,dist_sq)/total_mass),3)

def get_c4p(res):
    for atom in res.atoms:
        if atom.name=="C4'": return atom
    return None

def calc_end_dist(res_first,res_last):
    a1=get_c4p(res_first); a2=get_c4p(res_last)
    if a1 is None or a2 is None: return None
    return round(float(np.linalg.norm(np.array(a1.coord)-np.array(a2.coord))),3)

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
    os.makedirs(OUTPUT_DIR_RG, exist_ok=True)
    model  = session.models[MODEL_INDEX]
    today  = datetime.now().strftime("%d/%m/%Y")
    res_A  = [r for r in model.residues if r.chain_id=="A"]
    res_B  = [r for r in model.residues if r.chain_id=="B"]
    rnA    = {r.number:r for r in res_A}
    rnB    = {r.number:r for r in res_B}
    atA    = [a for r in res_A for a in r.atoms]
    atB    = [a for r in res_B for a in r.atoms]
    at_all = atA+atB

    rg_dup = calc_rg(at_all)
    ed_dup = calc_end_dist(rnA.get(1), rnB.get(24))
    rg_A   = calc_rg(atA); rg_B=calc_rg(atB)
    ed_A   = calc_end_dist(rnA.get(1), rnA.get(20))
    ed_B   = calc_end_dist(rnB.get(1), rnB.get(24))

    def region_rows(region_order, region_ranges, res_by_num, chain_id):
        rows=[]
        for region in region_order:
            rng=region_ranges[region]
            if rng is None:
                rows.append({"CHAIN":chain_id,"REGION":region,"NT_START":"NA","NT_END":"NA",
                              "N_NT":"NA","N_ATOMS":"NA","RG":"NA","END_DIST":"NA"})
            else:
                s,e=rng
                res_reg=[res_by_num[i] for i in range(s,e+1) if i in res_by_num]
                at_reg=[a for r in res_reg for a in r.atoms]
                rg_r=calc_rg(at_reg)
                ed_r=calc_end_dist(res_by_num[s],res_by_num[e]) if s!=e and s in res_by_num and e in res_by_num else "NA (n=1)"
                rows.append({"CHAIN":chain_id,"REGION":region,"NT_START":s,"NT_END":e,
                              "N_NT":e-s+1,"N_ATOMS":len(at_reg),
                              "RG":rg_r if rg_r is not None else "NA","END_DIST":ed_r})
        return rows

    rows_reg = region_rows(REGION_ORDER_A,REGION_RANGES_A,rnA,"A") +                region_rows(REGION_ORDER_B,REGION_RANGES_B,rnB,"B")

    def nt_rows(residues,region_map,chain_id):
        rows=[]
        for res in residues:
            at=list(res.atoms)
            rows.append({"CHAIN":chain_id,"RESNUM":res.number,"RESNAME":res.name.strip().upper(),
                          "REGION":region_map.get(res.number,"UNK"),
                          "N_ATOMS":len(at),"RG_NT":calc_rg(at) if at else "NA"})
        return rows

    rows_nt = nt_rows(res_A,REGION_MAP_A,"A")+nt_rows(res_B,REGION_MAP_B,"B")

    csv_global=os.path.join(OUTPUT_DIR_RG,"DUP_M4_miR7_rg_enddist_global.csv")
    with open(csv_global,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["VARIABLE","CHAIN","VALOR","UNIDAD","REFERENCIA"])
        w.writerow(["RG_GLOBAL_DUPLEX","A+B",rg_dup,"A",str(len(at_all))+" atomos"])
        w.writerow(["END_DIST_DUPLEX","A+B",ed_dup,"A","C4(A:nt1)->C4(B:nt24)"])
        w.writerow(["RG_GLOBAL_CYR","A",rg_A,"A",str(len(atA))+" atomos"])
        w.writerow(["END_DIST_CYR","A",ed_A,"A","C4(A:nt1)->C4(A:nt20)"])
        w.writerow(["RG_GLOBAL_MIR","B",rg_B,"A",str(len(atB))+" atomos"])
        w.writerow(["END_DIST_MIR","B",ed_B,"A","C4(B:nt1)->C4(B:nt24)"])

    csv_reg=os.path.join(OUTPUT_DIR_RG,"DUP_M4_miR7_rg_enddist_region.csv")
    with open(csv_reg,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows_reg[0].keys()); w.writeheader(); w.writerows(rows_reg)

    csv_nt=os.path.join(OUTPUT_DIR_RG,"DUP_M4_miR7_rg_enddist_per_nt.csv")
    with open(csv_nt,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows_nt[0].keys()); w.writeheader(); w.writerows(rows_nt)

    report_path=os.path.join(OUTPUT_DIR_RG,"DUP_M4_miR7_rg_enddist_reporte.txt")
    with open(report_path,"w",encoding="utf-8") as f:
        f.write("="*70+"\n")
        f.write("REPORTE RG + END_DIST - DUPLEX M4_miR7\n")
        f.write("="*70+"\n\n")
        f.write("HERRAMIENTA : UCSF ChimeraX - Python API interna\n")
        f.write("VERSION     : verificar en ChimeraX -> Help > About\n")
        f.write("FECHA       : " + today + "\n")
        f.write("INPUT       : M4_miR7.pdb\n")
        f.write("RG          : radio de giro masa-pesado\n")
        f.write("END_DIST    : C4(nt_inicio)->C4(nt_fin) por hebra\n\n")
        f.write("CADENAS\n")
        f.write("  Chain A - M4  : 5-AACAACAAAAUCAUCUUCCA-3 (20 nt)\n")
        f.write("  Chain B - miR-7-5p: 5-UGGAAGACUAGUGAUUUUGUUGUU-3 (24 nt)\n\n")
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
        f.write("  CSV global : DUP_M4_miR7_rg_enddist_global.csv\n")
        f.write("  CSV region : DUP_M4_miR7_rg_enddist_region.csv\n")
        f.write("  CSV per nt : DUP_M4_miR7_rg_enddist_per_nt.csv\n\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES - GLOBAL\n")
        f.write("-"*70+"\n")
        f.write("  VARIABLE                  CHAIN       VALOR UNIDAD  REFERENCIA\n")
        f.write("  RG_GLOBAL_DUPLEX          A+B   " + str(rg_dup).rjust(10) + " A\n")
        f.write("  END_DIST_DUPLEX           A+B   " + str(ed_dup).rjust(10) + " A      C4(A:nt1)->C4(B:nt24)\n")
        f.write("  RG_GLOBAL_CYR             A     " + str(rg_A).rjust(10) + " A\n")
        f.write("  END_DIST_CYR              A     " + str(ed_A).rjust(10) + " A      C4(A:nt1)->C4(A:nt20)\n")
        f.write("  RG_GLOBAL_MIR             B     " + str(rg_B).rjust(10) + " A\n")
        f.write("  END_DIST_MIR              B     " + str(ed_B).rjust(10) + " A      C4(B:nt1)->C4(B:nt24)\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES - POR REGION\n")
        f.write("-"*70+"\n")
        f.write("  CH  REGION           NT       N_NT   N_ATOMS     RG(A)   END_DIST(A)\n")
        for r in rows_reg:
            nt_r=str(r["NT_START"])+"-"+str(r["NT_END"]) if r["NT_START"]!="NA" else "NA"
            f.write("  " + r["CHAIN"].ljust(3) + " " + r["REGION"].ljust(14) + " " +
                    nt_r.rjust(8) + " " + str(r["N_NT"]).rjust(5) + " " +
                    str(r["N_ATOMS"]).rjust(8) + " " + str(r["RG"]).rjust(10) + " " +
                    str(r["END_DIST"]).rjust(13) + "\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("VARIABLES - POR NUCLEOTIDO\n")
        f.write("-"*70+"\n")
        f.write("  CH  NT    RES   REGION         N_ATOMS   RG_NT(A)\n")
        for r in rows_nt:
            f.write("  " + r["CHAIN"].ljust(3) + " " + str(r["RESNUM"]).ljust(5) + " " +
                    r["RESNAME"].ljust(5) + " " + r["REGION"].ljust(14) + " " +
                    str(r["N_ATOMS"]).rjust(8) + " " + str(r["RG_NT"]).rjust(10) + "\n")
        f.write("\n")
        f.write("-"*70+"\n")
        f.write("NOTAS\n")
        f.write("-"*70+"\n")
        f.write("  - RG masa-pesado.\n")
        f.write("  - END_DIST C4->C4 extremos de cada hebra.\n")
        f.write("  - Region ausente=NA.\n")
        f.write("  - Para video: movie record / turn y 3 120 / movie encode output ruta.mp4\n")
        f.write("="*70+"\n")

    cx(session,"set bgcolor white")
    cx(session,"lighting simple")
    cx(session,"nucleotides #1 tube/slab")
    cx(session,"show #1 cartoons")
    apply_colors(session)
    cx(session,"view #1")

    cx(session, "distance #1/A:1@C4' #1/B:24@C4'")
    cx(session, "distance #1/A:1@C4' #1/A:20@C4'")
    cx(session, "distance #1/B:1@C4' #1/B:24@C4'")
    cx(session, "distance #1/A:14@C4' #1/A:19@C4'")
    cx(session, "distance #1/A:1@C4' #1/A:13@C4'")
    cx(session, "distance #1/B:2@C4' #1/B:7@C4'")
    cx(session, "distance #1/B:8@C4' #1/B:11@C4'")
    cx(session, "distance #1/B:12@C4' #1/B:24@C4'")

    cx(session, "measure inertia #1/A:14-19 showEllipsoid true")
    cx(session, "measure inertia #1/A:1-13 showEllipsoid true")
    cx(session, "measure inertia #1/B:2-7 showEllipsoid true")
    cx(session, "measure inertia #1/B:8-11 showEllipsoid true")
    cx(session, "measure inertia #1/B:12-24 showEllipsoid true")

    cx(session,"view #1")

    print("="*70)
    print("RG + END_DIST - DUPLEX M4_miR7")
    print("="*70)
    print("  GLOBAL DUPLEX: RG=" + str(rg_dup) + " A  END_DIST=" + str(ed_dup) + " A  [C4(A:1)->C4(B:24)]")
    print("  GLOBAL CYR   : RG=" + str(rg_A) + " A  END_DIST=" + str(ed_A) + " A  [C4(A:1)->C4(A:20)]")
    print("  GLOBAL MIR   : RG=" + str(rg_B) + " A  END_DIST=" + str(ed_B) + " A  [C4(B:1)->C4(B:24)]")
    print("  CH  REGION           NT       N_NT   N_ATOMS     RG(A)   END_DIST(A)")
    for r in rows_reg:
        nt_r=str(r["NT_START"])+"-"+str(r["NT_END"]) if r["NT_START"]!="NA" else "NA"
        print("  " + r["CHAIN"].ljust(3) + " " + r["REGION"].ljust(14) + " " +
              nt_r.rjust(8) + " " + str(r["N_NT"]).rjust(5) + " " +
              str(r["N_ATOMS"]).rjust(8) + " " + str(r["RG"]).rjust(10) + " " +
              str(r["END_DIST"]).rjust(13))
    print("  CSV global -> " + csv_global)
    print("  CSV region -> " + csv_reg)
    print("  CSV per nt -> " + csv_nt)
    print("  Reporte    -> " + report_path)
    print("  CAPAS ACTIVAS: colores + distancias END_DIST + elipsoides RG")
    print("  VIDEO: movie record / turn y 3 120 / movie encode output ruta.mp4 format h264 framerate 15")
    print("="*70)

run(session)
