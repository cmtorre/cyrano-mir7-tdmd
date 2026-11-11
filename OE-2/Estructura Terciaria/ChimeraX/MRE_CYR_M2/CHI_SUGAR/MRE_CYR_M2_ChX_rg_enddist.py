# =============================================================================
# MRE_CYR_M2_ChX_rg_enddist.py
# HERRAMIENTA : UCSF ChimeraX — Python API interna
# VARIABLES   : RG, END_DIST
# SECUENCIA   : AACAACAAAAUCACCAAUGCUCCA (24 nt, 3'→5')
# DIRECCIÓN   : 3'→5' — numeración PDB desde extremo 3' del MRE
# EJECUCIÓN   : open "ruta\MRE_CYR_M2_ChX_rg_enddist.py"
# AUTOR       : Pipeline OE2 — MRE_CYR_M2
# =============================================================================

import csv, math, os
from datetime import datetime
import numpy as np

MODEL_INDEX = 0
PDB_INPUT   = "MRE_CYR_M2.pdb"

DIR_DATA = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-2\Var_E-Ter-CYR\2. ChimeraX\3.MRE_CYR_M2\RG_END_DIST"
DIR_VIZ  = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-2\Var_E-Ter-CYR\2. ChimeraX\3.MRE_CYR_M2\VISUALIZACION"

IMG_W=1800; IMG_H=1800; IMG_SUPERSAMPLE=3
VID_FRAMERATE=15; VID_FRAMES=120; VID_TURN_STEP=3

# Regiones — 24 nt, dirección 3'→5'
REGIONS = {
    "ANCHOR": (24, 24),
    "P1":     (19,  23),
    "BULGE":  (15,  18),
    "P2":     (1,  14),
}

REGION_MAP = {}
REGION_MAP[24] = "ANCHOR"
for i in range(19, 24): REGION_MAP[i] = "P1"
for i in range(15, 19):  REGION_MAP[i] = "BULGE"
for i in range(1, 15):  REGION_MAP[i] = "P2"

COLORS = {
    "ANCHOR": "#E63946",
    "P1":     "#2196F3",
    "BULGE":  "#FF9800",
    "P2":     "#4CAF50",
    "GRAY":   "#C0C0C0",
}

ATOMIC_MASS = {"H":1.008,"C":12.011,"N":14.007,"O":15.999,"P":30.974,"S":32.06,"MG":24.305,"ZN":65.38,"FE":55.845}

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

def close_model(session,mid):
    try: cx(session,f"close #{mid}")
    except: pass

def close_distances(session):
    try:
        from chimerax.core.commands import run as cx_run
        cx_run(session,"distance delete all")
    except: pass
    for mid in [8,7,6,5,4,3]: close_model(session,mid)

def close_ellipsoids(session):
    for mid in [7,6,5,4,3]: close_model(session,mid)

def save_img(session,filepath):
    cx(session,f'save "{filepath}" width {IMG_W} height {IMG_H} supersample {IMG_SUPERSAMPLE}')

def setup_base_view(session):
    cx(session,"set bgcolor white")
    cx(session,"lighting simple")
    cx(session,"graphics silhouettes false")

def apply_style(session):
    cx(session,"nucleotides #1 tube/slab")
    cx(session,"show #1 cartoons")

def color_all_regions(session):
    cx(session, f"color #1:24       {COLORS['ANCHOR']}")
    cx(session, f"color #1:19-23    {COLORS['P1']}")
    cx(session, f"color #1:15-18    {COLORS['BULGE']}")
    cx(session, f"color #1:1-14    {COLORS['P2']}")

def highlight_region(session,start,end,color):
    cx(session,f"color #1 {COLORS['GRAY']}")
    cx(session,f"color #1:{start}-{end} {color}")

def generate_enddist_images(session,res_by_num,dir_viz):
    imgs={}
    setup_base_view(session); apply_style(session)

    # Global (nt 1 → nt 24)
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "distance #1:1@C4' #1:24@C4'")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_END_DIST_GRL.png")
    save_img(session, p)
    close_distances(session)
    imgs["END_DIST_GRL"] = p
    print(f"✅ {os.path.basename(p)}")

    # ANCHOR (n=1, sin línea)
    highlight_region(session, 24, 24, COLORS["ANCHOR"])
    cx(session, "view #1:24 pad 0.8")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_END_DIST_ANCHOR.png")
    save_img(session, p)
    imgs["END_DIST_ANCHOR"] = p
    print(f"✅ {os.path.basename(p)}  [n=1 — sin línea END_DIST]")

    # P1
    highlight_region(session, 19, 23, COLORS["P1"])
    cx(session, "view #1:19-23 pad 0.4")
    cx(session, "distance #1:19@C4' #1:23@C4'")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_END_DIST_P1.png")
    save_img(session, p)
    close_distances(session)
    imgs["END_DIST_P1"] = p
    print(f"✅ {os.path.basename(p)}")

    # BULGE
    highlight_region(session, 15, 18, COLORS["BULGE"])
    cx(session, "view #1:15-18 pad 0.4")
    cx(session, "distance #1:15@C4' #1:18@C4'")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_END_DIST_B.png")
    save_img(session, p)
    close_distances(session)
    imgs["END_DIST_B"] = p
    print(f"✅ {os.path.basename(p)}")

    # P2
    highlight_region(session, 1, 14, COLORS["P2"])
    cx(session, "view #1:1-14 pad 0.4")
    cx(session, "distance #1:1@C4' #1:14@C4'")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_END_DIST_P2.png")
    save_img(session, p)
    close_distances(session)
    imgs["END_DIST_P2"] = p
    print(f"✅ {os.path.basename(p)}")

    # Todas simultáneas
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "distance #1:1@C4' #1:24@C4'")
    cx(session, "distance #1:19@C4' #1:23@C4'")
    cx(session, "distance #1:15@C4' #1:18@C4'")
    cx(session, "distance #1:1@C4' #1:14@C4'")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_END_DIST_TODAS.png")
    save_img(session, p)
    imgs["END_DIST_TODAS"] = p
    print(f"✅ {os.path.basename(p)}")

    return imgs

def generate_rg_images(session,dir_viz):
    imgs={}
    setup_base_view(session); apply_style(session); close_ellipsoids(session)

    close_ellipsoids(session)
    highlight_region(session, 19, 23, COLORS["P1"])
    cx(session, "view #1:19-23 pad 0.4")
    cx(session, "measure inertia #1:19-23 showEllipsoid true")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_RG_P1.png")
    save_img(session, p)
    imgs["RG_P1"] = p
    print(f"✅ {os.path.basename(p)}")

    close_ellipsoids(session)
    highlight_region(session, 15, 18, COLORS["BULGE"])
    cx(session, "view #1:15-18 pad 0.4")
    cx(session, "measure inertia #1:15-18 showEllipsoid true")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_RG_B.png")
    save_img(session, p)
    imgs["RG_B"] = p
    print(f"✅ {os.path.basename(p)}")

    close_ellipsoids(session)
    highlight_region(session, 1, 14, COLORS["P2"])
    cx(session, "view #1:1-14 pad 0.4")
    cx(session, "measure inertia #1:1-14 showEllipsoid true")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_RG_P2.png")
    save_img(session, p)
    imgs["RG_P2"] = p
    print(f"✅ {os.path.basename(p)}")

    close_ellipsoids(session)
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "measure inertia #1:19-23 showEllipsoid true")
    cx(session, "measure inertia #1:15-18 showEllipsoid true")
    cx(session, "measure inertia #1:1-14 showEllipsoid true")
    p = os.path.join(dir_viz, "MRE_CYR_M2_ChX_RG_TODAS.png")
    save_img(session, p)
    imgs["RG_TODAS"] = p
    print(f"✅ {os.path.basename(p)}")

    return imgs

def setup_video_scenes(session,dir_viz):
    mp4_paths={}

    setup_base_view(session)
    apply_style(session)
    close_distances(session)
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "distance #1:1@C4' #1:24@C4'")
    cx(session, "distance #1:19@C4' #1:23@C4'")
    cx(session, "distance #1:15@C4' #1:18@C4'")
    cx(session, "distance #1:1@C4' #1:14@C4'")
    mp4_ed = os.path.join(dir_viz, "MRE_CYR_M2_ChX_END_DIST_TODAS.mp4")
    print("\nVIDEO END_DIST_TODAS — ejecutar manualmente en consola ChimeraX:")
    print("   movie record")
    print(f"   turn y {VID_TURN_STEP} {VID_FRAMES}")
    print(f'   movie encode output "{mp4_ed}" format h264 framerate {VID_FRAMERATE}')
    mp4_paths["END_DIST_TODAS"] = mp4_ed

    setup_base_view(session)
    apply_style(session)
    close_ellipsoids(session)
    color_all_regions(session)
    cx(session, "view #1")
    cx(session, "measure inertia #1:19-23 showEllipsoid true")
    cx(session, "measure inertia #1:15-18 showEllipsoid true")
    cx(session, "measure inertia #1:1-14 showEllipsoid true")
    mp4_rg = os.path.join(dir_viz, "MRE_CYR_M2_ChX_RG_TODAS.mp4")
    print("\nVIDEO RG_TODAS — ejecutar manualmente en consola ChimeraX:")
    print("   movie record")
    print(f"   turn y {VID_TURN_STEP} {VID_FRAMES}")
    print(f'   movie encode output "{mp4_rg}" format h264 framerate {VID_FRAMERATE}')
    mp4_paths["RG_TODAS"] = mp4_rg

    return mp4_paths

def export_glb(session,dir_viz):
    glb_path=os.path.join(dir_viz,"MRE_CYR_M2_ChX.glb")
    try:
        cx(session,f'save "{glb_path}"')
        print(f"✅ GLB: {os.path.basename(glb_path)}")
        return glb_path
    except Exception as e:
        print(f"⚠️  GLB no generado: {e}")
        return None

def run(session):
    today=datetime.now().strftime("%d/%m/%Y")
    for d in [DIR_DATA,DIR_VIZ]:
        if not os.path.exists(d):
            os.makedirs(d,exist_ok=True); print(f"DEBUG creada: {d}")
        else:
            print(f"DEBUG encontrada: {d}")

    model=session.models[MODEL_INDEX]
    residues=list(model.residues)
    res_by_num={res.number:res for res in residues}

    all_atoms=[a for res in residues for a in res.atoms]
    rg_global=calc_rg(all_atoms)
    ed_global=calc_end_dist(res_by_num[1],res_by_num[24])

    rows_region=[]
    for region,(start,end) in REGIONS.items():
        res_reg=[res_by_num[i] for i in range(start,end+1) if i in res_by_num]
        atoms_reg=[a for res in res_reg for a in res.atoms]
        rg_reg=calc_rg(atoms_reg)
        ed_reg=calc_end_dist(res_by_num[start],res_by_num[end]) if start!=end else "N/A (n=1)"
        rows_region.append({
            "REGION":region,"NT_START":start,"NT_END":end,
            "N_NT":end-start+1,"N_ATOMS":len(atoms_reg),
            "RG":rg_reg if rg_reg is not None else "N/A",
            "END_DIST":ed_reg,
        })


    rows_nt=[]
    for res in residues:
        atoms=list(res.atoms)
        rows_nt.append({
            "RESNUM":res.number,"RESNAME":res.name.strip().upper(),
            "REGION":REGION_MAP.get(res.number,"UNK"),
            "N_ATOMS":len(atoms),"RG_NT":calc_rg(atoms) if atoms else "N/A",
        })

    csv_global=os.path.join(DIR_DATA,"MRE_CYR_M2_ChX_rg_enddist_global.csv")
    with open(csv_global,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["VARIABLE","VALOR","UNIDAD","REFERENCIA"])
        w.writerow(["RG_GLOBAL",rg_global,"Å",f"{len(all_atoms)} átomos"])
        w.writerow(["END_DIST_GLOBAL",ed_global,"Å","C4'(nt1)→C4'(nt24) 3'→5' MRE"])

    csv_reg=os.path.join(DIR_DATA,"MRE_CYR_M2_ChX_rg_enddist_region.csv")
    with open(csv_reg,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows_region[0].keys()); w.writeheader(); w.writerows(rows_region)

    csv_nt=os.path.join(DIR_DATA,"MRE_CYR_M2_ChX_rg_enddist_nt.csv")
    with open(csv_nt,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows_nt[0].keys()); w.writeheader(); w.writerows(rows_nt)

    print("\n--- Generando imágenes END_DIST ---")
    imgs_ed=generate_enddist_images(session,res_by_num,DIR_VIZ)
    print("\n--- Generando imágenes RG ---")
    imgs_rg=generate_rg_images(session,DIR_VIZ)
    print("\n--- Escenas de video (ejecución manual) ---")
    mp4s=setup_video_scenes(session,DIR_VIZ)
    print("\n--- Exportando GLB ---")
    setup_base_view(session); apply_style(session); color_all_regions(session)
    cx(session,"view #1")
    glb_path=export_glb(session,DIR_VIZ)

    print("\n"+"="*70)
    print("RG + END_DIST — MRE_CYR_M2")
    print("="*70)
    print(f"\nGLOBAL: RG={rg_global} Å  |  END_DIST={ed_global} Å")
    print(f"\n{'REGION':<8} {'NT':>8} {'N_NT':>5} {'N_ATOMS':>8} {'RG(Å)':>10} {'END_DIST(Å)':>13}")
    for r in rows_region:
        nt_range=f"{r['NT_START']}-{r['NT_END']}" if r["NT_START"]!="N/A" else "N/A"
        print(f"{r['REGION']:<8} {nt_range:>8} {str(r['N_NT']):>5} {str(r['N_ATOMS']):>8} {str(r['RG']):>10} {str(r['END_DIST']):>13}")
    print(f"\n✅ CSV global     → {csv_global}")
    print(f"✅ CSV región     → {csv_reg}")
    print(f"✅ CSV por nt     → {csv_nt}")
    print("="*70+"\n")

run(session)
