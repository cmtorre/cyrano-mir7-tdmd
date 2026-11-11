# -*- coding: utf-8 -*-
"""
F2_I1_EDA_integridad_v01.py
Fase II (Embudo de Tukey), Iteración 1: EDA macro e integridad mecanística.
Poda determinista (constantes, identidades, duplicados de linaje), auditoría de
límites biofísicos, perfilado del mecanismo de nulos, diagnóstico NZV anticipado
y análisis de sensibilidad de tolerancia de nulos en tres niveles.

Ejecución (Windows, Conda tesis_rnas):
    conda activate tesis_rnas
    python F2_I1_EDA_integridad_v01.py                       # modo exploratorio (3 ramas)
    python F2_I1_EDA_integridad_v01.py --nivel INTERMEDIO    # consolidación tras la decisión
"""

#==============================================================================
# MÓDULO 0. IMPORTACIONES, RUTAS Y PARÁMETROS
# Objetivo: centralizar rutas, versión y todos los parámetros editables.
# Inputs: argumentos opcionales de terminal.
# Proceso: define rutas con pathlib.Path, niveles de umbral y reglas V2, V6, V7.
# Outputs: constantes globales.
#==============================================================================
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

VERSION = "v01"
DATE_TAG = datetime.now().strftime("%Y%m%d")

BASE = Path(r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-6\Nuevo_Flujo\4.FASE 2_EDA\Ite_1")
DEFAULT_INPUT_CSV = BASE / "00_DATASET" / "F1_RAW_DATASET_v01_20260925.csv"
DEFAULT_DICT_CSV = BASE / "00_DATASET" / "F1_DATA_DICTIONARY_v01.csv"
DEFAULT_NAMAP_CSV = BASE / "00_DATASET" / "F1_NA_MAP_v01.csv"   # opcional
DEFAULT_OUTPUT_DIR = BASE / "02_TXT"
DEFAULT_FIGURES_DIR = BASE / "01_FIGURAS"

VAR_ORDER = ["WT", "M1", "M2", "M3", "M4", "M5", "M6"]
Y_COLS = ["Y_MIR7_REL", "Y_LOG2_MIR7REL"]

# Tres niveles de tolerancia de nulos: máximo de NaN por columna (N = 7)
LEVELS = {"CONSERVADOR": 0, "INTERMEDIO": 2, "PERMISIVO": 3}

# V2: base termodinámica aprobada {NET_DG, UNF_A, UNF_B}
V2_KEEP = "DPX_TD_NET_DG_GLB"
V2_DROP = ["DPX_TD_HYBRIDIZATION_DG_GLB", "DPX_TD_CYR_MIR_ACCES_DG_COST_GLB"]
V2_COMPONENTS = ["DPX_TD_UNFOLDING_A_DG_GLB", "DPX_TD_UNFOLDING_B_DG_GLB"]

# V6: pares intercatenarios en el bulge asimétrico (M4, M5) = 0
V6_COLS = ["DPX_3D_BP_CAN_B", "DPX_3D_BP_NC_B"]
V6_PRESENCE_FLAG = "FLAG_DPX_HAS_B"

# V7: definición de END_DIST para hebras (editable si el script de extracción difiere)
END_DIST_DEFINITION = ("Distancia euclidiana entre C5' del nucleótido terminal 5' y C3' "
                       "(o fosfato terminal) del nucleótido terminal 3' de la hebra (V7). "
                       "Confirmar frente al script de extracción de ChimeraX.")

# V5: nota de convención para la pseudorrotación
P_CONVENTION_NOTE = ("P con desfase de convención cercano a 180° respecto a Altona y "
                     "Sundaralingam (1972); un desfase uniforme preserva el orden entre "
                     "variantes, pero no corrige casos con cruce de 0/360° (ver CIRC_RISK).")

# Parámetros de auditoría biofísica (editables)
TOL = 1e-6
CONTOUR_A_PER_NT = 7.0          # límite superior aproximado de distancia por nt (Å)
SASA_PER_NT_RANGE = (100, 400)  # intervalo de plausibilidad (Å²/nt), heurístico
CIRC_SD_LIMIT = 90.0            # SD lineal de un ángulo por encima de este valor -> CIRC_RISK
CIRC_SECTOR_JUMP = 90.0         # diferencia media regional vs global -> CIRC_RISK


def parse_args():
    p = argparse.ArgumentParser(description="F2 Iteración 1: integridad y poda")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    p.add_argument("--dict-csv", type=Path, default=DEFAULT_DICT_CSV)
    p.add_argument("--namap-csv", type=Path, default=DEFAULT_NAMAP_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--nivel", default="EXPLORATORIO",
                   choices=["EXPLORATORIO"] + list(LEVELS))
    return p.parse_args()


#==============================================================================
# MÓDULO 1. REGISTRO (LOG)
# Objetivo: trazabilidad de cada decisión en consola y archivo.
# Inputs: OUTPUT_DIR.
# Proceso: logger dual (consola + F2_I1_log_v01.txt).
# Outputs: F2_I1_log_v01.txt.
#==============================================================================
def setup_logger(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("F2_I1")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(out_dir / f"F2_I1_log_{VERSION}.txt", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


#==============================================================================
# MÓDULO 2. CARGA Y VALIDACIÓN DE ENTRADAS
# Objetivo: leer RAW_DATASET y diccionario de Fase I y verificar su coherencia.
# Inputs: F1_RAW_DATASET (csv), F1_DATA_DICTIONARY (csv), F1_NA_MAP (opcional).
# Proceso: valida 7 variantes, correspondencia 1:1 columnas-diccionario y
#          deriva el nombre base (VARIABLE) de cada columna.
# Outputs: RAW (DataFrame), DIC (DataFrame indexado por VAR_NAME), NA_MAP o None.
#==============================================================================
def load_inputs(args, log):
    for p in (args.input_csv, args.dict_csv):
        if not p.exists():
            log.error(f"No se encontró: {p}")
            log.error("Copie F1_RAW_DATASET y F1_DATA_DICTIONARY de la Fase I a la carpeta 00_DATASET.")
            sys.exit(1)
    raw = pd.read_csv(args.input_csv, index_col=0, encoding="utf-8-sig")
    raw.index = raw.index.astype(str)
    raw = raw.reindex(VAR_ORDER)
    if raw.index.isna().any() or raw.shape[0] != 7:
        raise ValueError("El dataset no contiene exactamente WT y M1 a M6")
    dic = pd.read_csv(args.dict_csv, encoding="utf-8-sig", dtype=str).fillna("")
    dic = dic.set_index("VAR_NAME")
    miss_d, miss_r = set(raw.columns) - set(dic.index), set(dic.index) - set(raw.columns)
    if miss_d or miss_r:
        raise ValueError(f"Columnas sin diccionario: {sorted(miss_d)[:5]} | "
                         f"entradas sin columna: {sorted(miss_r)[:5]}")
    dic = dic.loc[raw.columns]
    dic["BASE"] = [base_of(c, dic.loc[c]) for c in dic.index]
    na_map = None
    if args.namap_csv.exists():
        na_map = pd.read_csv(args.namap_csv, index_col=0, encoding="utf-8-sig").reindex(VAR_ORDER)
        log.info("  NA_MAP de Fase I cargado (se usa para confirmar códigos de ausencia)")
    else:
        log.info("  NA_MAP de Fase I no encontrado (opcional); el mecanismo se infiere de las banderas")
    log.info(f"  RAW_DATASET: N={raw.shape[0]} x P={raw.shape[1]}")
    return raw, dic, na_map


def base_of(name, row):
    pre, reg = f"{row['MOL_CTX']}_{row['NIV']}_", row["REG"]
    if name.startswith(pre) and name.endswith("_" + reg):
        return name[len(pre):-(len(reg) + 1)]
    return name


def strand_of(row):
    return row["MOL_CTX"].split("_")[0]


def num(raw, c):
    return pd.to_numeric(raw[c], errors="coerce").astype(float)


#==============================================================================
# MÓDULO 3. SEPARACIÓN DE ROLES E IMPUTACIÓN V6
# Objetivo: aislar respuestas y metadatos; corregir pares del bulge asimétrico.
# Inputs: RAW, DIC.
# Proceso: ROLE PREDICTOR/FLAG -> espacio analítico; META_* -> archivo propio;
#          V6: BP_CAN_B y BP_NC_B = 0 donde el bulge existe en el dúplex y es NaN.
# Outputs: listas de columnas por rol, RAW corregido, F2_I1_META_v01.csv.
#==============================================================================
def split_roles(raw, dic, out_dir, log):
    analytic = [c for c in raw.columns if dic.at[c, "ROLE"] in ("PREDICTOR", "FLAG")]
    meta = [c for c in raw.columns if c.startswith("META_")]
    ys = [c for c in Y_COLS if c in raw.columns]
    other = set(raw.columns) - set(analytic) - set(meta) - set(ys)
    if other:
        raise ValueError(f"Columnas sin rol asignable: {sorted(other)}")
    raw[meta].to_csv(out_dir / f"F2_I1_META_{VERSION}.csv", encoding="utf-8-sig")
    log.info(f"  Espacio analítico: {len(analytic)} | metadatos: {len(meta)} | respuestas: {len(ys)}")
    return analytic, meta, ys


def apply_v6(raw, dic, log):
    n = 0
    for c in V6_COLS:
        mask = raw[c].isna() & (num(raw, V6_PRESENCE_FLAG) == 1)
        raw.loc[mask, c] = 0
        n += int(mask.sum())
        dic.at[c, "NOTES"] = (dic.at[c, "NOTES"] + " | " if dic.at[c, "NOTES"] else "") + \
            "V6: 0 en M4 y M5 (bulge asimétrico sin cadena complementaria)"
        log.info(f"  [V6] {c}: {int(mask.sum())} celdas imputadas con 0 en {list(raw.index[mask])}")
    return raw, n


#==============================================================================
# MÓDULO 4. PODA DETERMINISTA S1-S3
# Objetivo: eliminar constantes, identidades exactas y duplicados de linaje.
# Inputs: RAW, DIC, espacio analítico.
# Proceso: S1 recalcula constantes (<= 1 valor único); S2 elimina dependientes
#          del diccionario con la excepción V2; S3 clasifica igualdades exactas en
#          linaje (misma magnitud, misma hebra, misma región) o alias.
# Outputs: columnas retenidas S3, registro de poda y grupos de alias.
#==============================================================================
def prune(raw, dic, analytic, log):
    plog = []

    # --- S1 constantes
    s1 = []
    for c in analytic:
        if raw[c].nunique(dropna=True) <= 1:
            plog.append(dict(VAR_NAME=c, STEP="S1", REASON="Constante absoluta", KEEPER=""))
        else:
            s1.append(c)
    f1_const = {c for c in analytic if dic.at[c, "FLAG_CONST"] == "1"}
    now_const = set(analytic) - set(s1)
    if f1_const != now_const:
        log.info(f"  [S1] diferencias frente a FLAG_CONST de Fase I: "
                 f"nuevas={sorted(now_const - f1_const)} | ya no constantes={sorted(f1_const - now_const)}")
    log.info(f"  S1 constantes: -{len(analytic) - len(s1)} -> {len(s1)}")

    # --- S2 identidades con excepción V2
    dep = {c for c in s1 if dic.at[c, "FLAG_IDENTITY"] != ""}
    dep.discard(V2_KEEP)
    dep |= {c for c in V2_DROP if c in s1}
    resid = (num(raw, V2_DROP[0]) - (num(raw, V2_KEEP) - num(raw, V2_COMPONENTS[0]) - num(raw, V2_COMPONENTS[1]))).abs().max()
    log.info(f"  [V2] HYB_DG = NET_DG - UNF_A - UNF_B verificado (máx |residuo| = {resid:.2e})")
    s2 = []
    for c in s1:
        if c in dep:
            fam = dic.at[c, "FLAG_IDENTITY"] or "NET_DG = HYB_DG + UNF_A + UNF_B (V2)"
            src = dic.at[c, "DERIVED_FROM"] if dic.at[c, "FLAG_IDENTITY"] else f"{V2_KEEP} ; {' ; '.join(V2_COMPONENTS)}"
            plog.append(dict(VAR_NAME=c, STEP="S2", REASON=f"Identidad exacta: {fam}", KEEPER=src))
        else:
            s2.append(c)
    log.info(f"  S2 identidades: -{len(s1) - len(s2)} -> {len(s2)}")

    # --- S3 duplicados de linaje y alias
    seen, lineage, alias = {}, {}, {}
    for c in s2:
        sig = tuple("NA" if pd.isna(v) else round(float(v), 6) for v in num(raw, c))
        if sig not in seen:
            seen[sig] = c
            continue
        k = seen[sig]
        rc, rk = dic.loc[c], dic.loc[k]
        is_lineage = (not c.startswith("FLAG") and rc["BASE"] == rk["BASE"] and rc["REG"] == rk["REG"]
                      and (strand_of(rc) == strand_of(rk) or rc["BASE"] == "N_MOD"))
        if is_lineage:
            lineage[c] = k
            plog.append(dict(VAR_NAME=c, STEP="S3", REASON="Duplicado de linaje", KEEPER=k))
        else:
            alias.setdefault(k, []).append(c)
    s3 = [c for c in s2 if c not in lineage]
    log.info(f"  S3 duplicados de linaje: -{len(lineage)} -> {len(s3)}")
    alias_rows = []
    for i, (k, members) in enumerate(alias.items(), 1):
        gid = f"ALIAS_{i:02d}"
        for m in [k] + members:
            rel = "complementariedad WC" if (strand_of(dic.loc[m]) != strand_of(dic.loc[k])
                                              and dic.at[m, "REG"] == dic.at[k, "REG"]
                                              and not m.startswith("FLAG")) else "coincidencia numérica"
            alias_rows.append(dict(ALIAS_GROUP=gid, VAR_NAME=m, ROLE_IN_GROUP="representante" if m == k else "alias",
                                   RELATION="" if m == k else rel))
    alias_df = pd.DataFrame(alias_rows)
    log.info(f"  Alias conservados: {sum(len(v) for v in alias.values())} columnas en {len(alias)} grupos")
    return s1, s2, s3, pd.DataFrame(plog), alias_df


#==============================================================================
# MÓDULO 5. AUDITORÍA DE LÍMITES BIOFÍSICOS
# Objetivo: verificar coherencia física de las variables remanentes (S3).
# Inputs: RAW, DIC, columnas S3.
# Proceso: reglas univariadas (signo de ΔG, rangos, no negatividad, enteros) y
#          cruzadas (ΔG ensamble <= MFE, SASA/nt, END_DIST/contorno, pares/nt);
#          bandera CIRC_RISK para ángulos con SD > 90° o saltos de sector.
# Outputs: tabla BOUNDS_AUDIT (una fila por variable y regla), estado por variable.
#==============================================================================
def bounds_audit(raw, dic, s3, log):
    rows = []

    def add(c, rule, ok, detail="", level="FAIL", bad=None):
        rows.append(dict(VAR_NAME=c, RULE=rule, STATUS="PASS" if ok else level,
                         DETAIL=detail, VIOLATING=",".join(bad) if bad is not None and len(bad) else ""))

    for c in s3:
        b, ctx, reg = dic.at[c, "BASE"], dic.at[c, "MOL_CTX"], dic.at[c, "REG"]
        if c.startswith("FLAG") or c.startswith("DLT"):
            continue
        v = num(raw, c)
        vv = v.dropna()
        bad = lambda m: list(v.index[m.fillna(False)])
        if b in ("HYBRIDIZATION_DG", "MFE_DG", "ENSEMBLE_DG", "FINAL_DG", "NET_DG", "DG_ENSEMBLE", "DG_FOLD"):
            add(c, "DG_NO_POSITIVO", (vv <= TOL).all(), f"máx={vv.max():.3g}", bad=bad(v > TOL))
        if b in ("UNFOLDING_A_DG", "UNFOLDING_B_DG", "CYR_MIR_ACCES_DG_COST"):
            add(c, "DG_NO_NEGATIVO", (vv >= -TOL).all(), f"mín={vv.min():.3g}", bad=bad(v < -TOL))
        if b in ("MFE_FREQ", "FREQ_MFE"):
            add(c, "FRECUENCIA_0_1", vv.between(0, 1).all(), f"[{vv.min():.3g}, {vv.max():.3g}]",
                bad=bad(~v.between(0, 1) & v.notna()))
        if b.startswith("PCT_") or b.endswith("_PCT") or b in ("PROB_PAIR", "ACCES_POS"):
            add(c, "PORCENTAJE_0_100", vv.between(-TOL, 100 + TOL).all(), f"[{vv.min():.3g}, {vv.max():.3g}]",
                bad=bad(~v.between(-TOL, 100 + TOL) & v.notna()))
        if b.startswith(("RG", "END_DIST", "SASA", "ENT_")) or b == "ENS_DIV" or b.endswith("_SD"):
            add(c, "NO_NEGATIVO", (vv >= -TOL).all(), f"mín={vv.min():.3g}", bad=bad(v < -TOL))
        if b.startswith("LEN") or b in ("BP_CAN", "BP_NC", "BP_TOTAL", "STACK", "N_MOD"):
            ok = (vv >= 0).all() and np.allclose(vv, vv.round())
            add(c, "CONTEO_ENTERO_NO_NEG", ok, f"[{vv.min():.3g}, {vv.max():.3g}]")
        if b.startswith("PSEUDO_P_SUGAR") and not b.endswith("_SD"):
            add(c, "P_RANGO_0_360", vv.between(0, 360).all(), f"[{vv.min():.3g}, {vv.max():.3g}]",
                bad=bad(~v.between(0, 360) & v.notna()))
            south = vv.between(144, 216).mean()
            rows.append(dict(VAR_NAME=c, RULE="P_SECTOR_INFO", STATUS="INFO",
                             DETAIL=f"fracción en sector Sur 144-216°: {south:.2f} ({P_CONVENTION_NOTE[:40]}...)",
                             VIOLATING=""))
        if b.startswith("CHI_ANGLE") and not b.endswith("_SD"):
            add(c, "CHI_RANGO_-180_360", vv.between(-180, 360).all(), f"[{vv.min():.3g}, {vv.max():.3g}]")

    # --- reglas cruzadas (se evalúan en RAW y se asignan a la columna retenida)
    def cross(target, rule, ok_series, detail, level="FAIL"):
        if target in s3:
            add(target, rule, bool(ok_series.fillna(True).all()), detail, level,
                bad=list(ok_series.index[~ok_series.fillna(True)]))

    ens, mfe = num(raw, "DPX_TD_ENSEMBLE_DG_GLB"), num(raw, "DPX_TD_MFE_DG_GLB")
    for t in ("DPX_TD_ENSEMBLE_DG_GLB", "DPX_TD_MFE_DG_GLB"):
        cross(t, "DG_ENSAMBLE_LE_MFE", ens <= mfe + TOL, f"máx(ens-mfe)={(ens - mfe).max():.3g}")
    if "CYR_ISO_2D_DG_ENSEMBLE_GLB" in raw:
        e, f = num(raw, "CYR_ISO_2D_DG_ENSEMBLE_GLB"), num(raw, "CYR_ISO_2D_DG_FOLD_GLB")
        cross("CYR_ISO_2D_DG_ENSEMBLE_GLB", "DG_ENSAMBLE_LE_MFE", e <= f + TOL, f"máx(ens-mfe)={(e - f).max():.3g}")

    len_cyr = num(raw, "CYR_ISO_1D_LEN_REAL_GLB")
    len_mir = num(raw, "MIR_ISO_1D_LEN_REAL_GLB")
    lens = {"CYR": len_cyr, "MIR": len_mir}
    for ctx in ("CYR_ISO", "MIR_ISO", "CYR_DPX", "MIR_DPX"):
        L = lens[ctx.split("_")[0]]
        sasa = f"{ctx}_3D_SASA_TOTAL_GLB"
        if sasa in raw:
            r = num(raw, sasa) / L
            cross(sasa, "SASA_POR_NT_PLAUSIBLE", r.between(*SASA_PER_NT_RANGE) | r.isna(),
                  f"[{r.min():.1f}, {r.max():.1f}] Å²/nt", level="WARN")
        for reg in ("GLB", "P1", "B", "P2", "O"):
            ed = f"{ctx}_3D_END_DIST_{reg}"
            Lr = L if reg == "GLB" else (num(raw, f"{ctx}_1D_LEN_{reg}") if f"{ctx}_1D_LEN_{reg}" in raw else None)
            if ed in raw and Lr is not None:
                cont = CONTOUR_A_PER_NT * (Lr - 1)
                r = num(raw, ed) / cont.where(cont > 0)
                # celdas sin contorno definido (región ausente o de 1 nt) no se evalúan
                cross(ed, "END_DIST_LE_CONTORNO", (r <= 1 + TOL) | r.isna(), f"máx razón={r.max():.2f}")
    bp = num(raw, "DPX_3D_BP_CAN_GLB") + num(raw, "DPX_3D_BP_NC_GLB")
    r = bp / (len_cyr + len_mir)
    for t in ("DPX_3D_BP_CAN_GLB", "DPX_3D_BP_NC_GLB"):
        cross(t, "PARES_POR_RESIDUO_LE_1", (r <= 1 + TOL) | r.isna(), f"máx={r.max():.2f}")

    # --- CIRC_RISK
    circ = {}
    ang = [c for c in s3 if dic.at[c, "BASE"].startswith(("CHI_ANGLE", "PSEUDO_P_SUGAR"))
           and not c.startswith("DLT")]
    for c in ang:
        b, ctx, reg = dic.at[c, "BASE"], dic.at[c, "MOL_CTX"], dic.at[c, "REG"]
        v = num(raw, c)
        reasons = []
        if b.endswith("_SD") and (v > CIRC_SD_LIMIT).any():
            reasons.append(f"SD > {CIRC_SD_LIMIT:.0f}° en {list(v.index[v > CIRC_SD_LIMIT])}")
        if b.startswith("CHI") and not b.endswith("_SD") and (v < 0).any() and (v > 0).any():
            reasons.append("signos mixtos en χ")
        if not b.endswith("_SD") and reg != "GLB":
            g = f"{ctx}_3D_{b.replace('_T1', '')}_GLB" if b in ("CHI_ANGLE", "PSEUDO_P_SUGAR") else f"{ctx}_3D_{b}_GLB"
            if b == "CHI_ANGLE":
                g = f"{ctx}_3D_CHI_ANGLE_MEAN_GLB"
            if b == "PSEUDO_P_SUGAR":
                g = f"{ctx}_3D_PSEUDO_P_SUGAR_MEAN_GLB"
            if g in raw:
                jump = (v - num(raw, g)).abs()
                if (jump > CIRC_SECTOR_JUMP).any():
                    reasons.append(f"salto > {CIRC_SECTOR_JUMP:.0f}° frente a la media global en "
                                   f"{list(v.index[jump > CIRC_SECTOR_JUMP])}")
        if reasons:
            circ[c] = "; ".join(reasons)
    # propagación a la media/SD pareada de la misma región
    for c in list(circ):
        b = dic.at[c, "BASE"]
        pair = c.replace("_SD_", "_MEAN_") if "_SD_" in c else c.replace("_MEAN_", "_SD_")
        if pair in s3 and pair not in circ:
            circ[pair] = f"pareada con {c}"
    # propagación a deltas
    for c in s3:
        if c.startswith("DLT"):
            src = dic.at[c, "DERIVED_FROM"].replace(" ", "").split("-")
            if any(s in circ for s in src):
                circ[c] = "delta derivado de una variable con CIRC_RISK"
    for c, why in circ.items():
        rows.append(dict(VAR_NAME=c, RULE="CIRC_RISK", STATUS="WARN", DETAIL=why, VIOLATING=""))
    aud = pd.DataFrame(rows)
    order = {"PASS": 0, "INFO": 0, "WARN": 1, "FAIL": 2}
    status = aud.assign(o=aud.STATUS.map(order)).groupby("VAR_NAME").o.max().map({0: "PASS", 1: "WARN", 2: "FAIL"})
    n_fail = int((status == "FAIL").sum())
    log.info(f"  Reglas evaluadas: {len(aud)} | variables auditadas: {status.size} | "
             f"PASS={int((status == 'PASS').sum())} WARN={int((status == 'WARN').sum())} FAIL={n_fail}")
    for _, r in aud[aud.STATUS == "FAIL"].iterrows():
        log.warning(f"  [FAIL] {r.VAR_NAME} :: {r.RULE} :: {r.DETAIL} :: {r.VIOLATING}")
    for c, why in circ.items():
        log.info(f"  [CIRC_RISK] {c}: {why}")
    return aud, status, circ


#==============================================================================
# MÓDULO 6. MECANISMO DE NULOS
# Objetivo: confirmar empíricamente que cada patrón de nulos es estructural.
# Inputs: RAW, columnas S3, banderas de región, NA_MAP (opcional).
# Proceso: patrón binario por variante; se compara con el complemento de cada
#          bandera FLAG_*; si coincide -> MNAR_ESTRUCTURAL; si no -> MNAR_UNDEF
#          (o TECH si NA_MAP reporta TECH_NA).
# Outputs: tabla NA_MECHANISM y resumen de patrones.
#==============================================================================
def na_mechanism(raw, s3, na_map, log):
    flags = {c: "".join("1" if v == 0 else "0" for v in num(raw, c)) for c in raw.columns if c.startswith("FLAG")}
    rows = []
    for c in s3:
        m = raw[c].isna()
        if not m.any():
            continue
        pat = "".join("1" if x else "0" for x in m)
        match = [f for f, p in flags.items() if p == pat]
        codes = ""
        if na_map is not None and c in na_map.columns:
            codes = ";".join(sorted(set(na_map.loc[m, c].astype(str))))
        mech = "MNAR_ESTRUCTURAL" if match else ("TECH_NA" if "TECH_NA" in codes else "MNAR_UNDEF")
        rows.append(dict(VAR_NAME=c, N_NA=int(m.sum()), PATTERN=pat, MECHANISM=mech,
                         MATCHED_FLAG=";".join(match), F1_NA_CODES=codes))
    df = pd.DataFrame(rows)
    summ = df.groupby(["PATTERN", "MECHANISM", "MATCHED_FLAG"]).size().reset_index(name="N_COLS") \
        .sort_values("N_COLS", ascending=False)
    log.info(f"  Columnas con nulos: {len(df)} | patrones únicos: {df.PATTERN.nunique()}")
    for _, r in summ.iterrows():
        log.info(f"  [patrón {r.PATTERN}] {r.N_COLS} cols -> {r.MECHANISM} {r.MATCHED_FLAG.split(';')[0] if r.MATCHED_FLAG else ''}")
    if (df.MECHANISM == "TECH_NA").any():
        log.warning("  Se detectaron faltantes técnicos: revisar antes de continuar")
    return df, summ


#==============================================================================
# MÓDULO 7. DIAGNÓSTICO NZV ANTICIPADO (NO SE APLICA)
# Objetivo: documentar variables con muy pocos valores distintos del modal.
# Inputs: RAW, columnas S3.
# Proceso: n_minoría = observaciones no nulas - frecuencia del valor modal;
#          marca si la única observación discrepante es WT.
# Outputs: DataFrame con NZV_N_MINORITY y NZV_WT_ONLY.
#==============================================================================
def nzv_diagnostic(raw, s3, log):
    rows = []
    for c in s3:
        v = num(raw, c).dropna().round(6)
        vc = v.value_counts()
        minority = int(len(v) - vc.iloc[0])
        wt_only = bool(minority == 1 and "WT" in v.index and v["WT"] != vc.index[0])
        rows.append(dict(VAR_NAME=c, NZV_N_MINORITY=minority, NZV_WT_ONLY=wt_only))
    df = pd.DataFrame(rows)
    log.info(f"  n_minoría=1: {int((df.NZV_N_MINORITY == 1).sum())} (WT como único discrepante: "
             f"{int(df.NZV_WT_ONLY.sum())}) | n_minoría=2: {int((df.NZV_N_MINORITY == 2).sum())}")
    return df


#==============================================================================
# MÓDULO 8. RAMAS DE SENSIBILIDAD (TRES NIVELES DE TOLERANCIA DE NULOS)
# Objetivo: construir y comparar los datasets Conservador, Intermedio y Permisivo.
# Inputs: RAW, DIC, columnas S3.
# Proceso: retiene columnas con NaN <= umbral; resume por región y bloque.
#          La decisión no usa Y (evita data snooping).
# Outputs: diccionario nivel -> columnas, tabla comparativa.
#==============================================================================
def build_branches(raw, dic, s3, log):
    n_na = raw[s3].isna().sum()
    branches, rows = {}, []
    for lvl, mx in LEVELS.items():
        keep = [c for c in s3 if n_na[c] <= mx]
        branches[lvl] = keep
        regs = pd.Series([dic.at[c, "REG"] for c in keep]).value_counts()
        rows.append(dict(NIVEL=lvl, MAX_NA=mx, N_EFECTIVO_MIN=7 - mx,
                         P_MIN_EXACTO_SPEARMAN=round(2 / float(np.prod(range(1, 8 - mx))), 5),
                         P_RETENIDO=len(keep), **{f"REG_{r}": int(regs.get(r, 0)) for r in
                                                  ["GLB", "T1", "P1", "B", "P2", "O"]}))
        log.info(f"  [{lvl}] NaN <= {mx}: P={len(keep)} | " +
                 ", ".join(f"{r}={int(regs.get(r, 0))}" for r in ["GLB", "T1", "P1", "B", "P2", "O"]))
    excl = [c for c in s3 if n_na[c] > max(LEVELS.values())]
    log.info(f"  Excluidas en los tres niveles (NaN > {max(LEVELS.values())}): {len(excl)}")
    return branches, pd.DataFrame(rows)


#==============================================================================
# MÓDULO 9. DICCIONARIO ACTUALIZADO
# Objetivo: extender el linaje de Fase I con el estado de la Iteración 1.
# Inputs: DIC, registro de poda, alias, auditoría, nulos, NZV, ramas.
# Proceso: añade STATUS_I1, PRUNE_STEP, PRUNE_REASON, KEEPER, ALIAS_GROUP,
#          BOUNDS_STATUS, BOUNDS_RULES, CIRC_RISK, NA_MECHANISM, NA_PATTERN,
#          NZV_*, KEEP_<NIVEL> y notas V5 y V7.
# Outputs: DataFrame F2_I1_DATA_DICTIONARY.
#==============================================================================
def update_dictionary(dic, plog, alias_df, aud, status, circ, nam, nzv, branches, s3):
    d = dic.copy()
    d["STATUS_I1"] = ""
    d.loc[[c for c in d.index if d.at[c, "ROLE"] == "RESPONSE"], "STATUS_I1"] = "RESPONSE"
    d.loc[[c for c in d.index if c.startswith("META_")], "STATUS_I1"] = "META"
    d.loc[s3, "STATUS_I1"] = "RETAINED_S3"
    for col in ("PRUNE_STEP", "PRUNE_REASON", "KEEPER", "ALIAS_GROUP", "BOUNDS_RULES", "NA_MECHANISM", "NA_PATTERN"):
        d[col] = ""
    names = {"S1": "PRUNED_CONST", "S2": "PRUNED_IDENTITY", "S3": "PRUNED_LINEAGE"}
    for _, r in plog.iterrows():
        d.at[r.VAR_NAME, "STATUS_I1"] = names[r.STEP]
        d.at[r.VAR_NAME, "PRUNE_STEP"] = r.STEP
        d.at[r.VAR_NAME, "PRUNE_REASON"] = r.REASON
        d.at[r.VAR_NAME, "KEEPER"] = r.KEEPER
    for _, r in alias_df.iterrows():
        d.at[r.VAR_NAME, "ALIAS_GROUP"] = r.ALIAS_GROUP
    d["BOUNDS_STATUS"] = d.index.map(status).fillna("")
    for c, g in aud[aud.STATUS != "PASS"].groupby("VAR_NAME"):
        d.at[c, "BOUNDS_RULES"] = " | ".join(f"{r.RULE}:{r.STATUS}" for _, r in g.iterrows())
    d["CIRC_RISK"] = [int(c in circ) for c in d.index]
    for _, r in nam.iterrows():
        d.at[r.VAR_NAME, "NA_MECHANISM"] = r.MECHANISM
        d.at[r.VAR_NAME, "NA_PATTERN"] = r.PATTERN
    nz = nzv.set_index("VAR_NAME")
    d["NZV_N_MINORITY"] = d.index.map(nz["NZV_N_MINORITY"]).astype("Int64")
    d["NZV_WT_ONLY"] = d.index.map(nz["NZV_WT_ONLY"])
    for lvl, keep in branches.items():
        d[f"KEEP_{lvl}"] = [int(c in keep) for c in d.index]
    for c in d.index:
        b = d.at[c, "BASE"]
        add = []
        if b.startswith(("PSEUDO_P_SUGAR", "SUGAR_PUCKER")):
            add.append("V5: " + P_CONVENTION_NOTE)
        if b.startswith("END_DIST") and not b.startswith("END_DIST_INTER") and d.at[c, "MOL_CTX"] != "DPX":
            add.append("V7: " + END_DIST_DEFINITION)
        if c in circ:
            add.append("CIRC_RISK: " + circ[c])
        if add:
            d.at[c, "NOTES"] = " | ".join(x for x in [d.at[c, "NOTES"]] + add if x)
    return d.reset_index().rename(columns={"index": "VAR_NAME"})


#==============================================================================
# MÓDULO 10. FIGURAS (UN GRÁFICO POR ARCHIVO, 300 DPI)
# Objetivo: visual tracking de poda, límites, ángulos, nulos, umbrales y NZV.
# Inputs: resultados de los módulos 4 a 8.
# Proceso: ocho rutinas independientes.
# Outputs: F2_I1_FIG01 a F2_I1_FIG08 en FIGURES_DIR.
#==============================================================================
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10})
CTX_COLORS = {"CYR_ISO": "#55A868", "CYR_DPX": "#4C72B0", "MIR_DPX": "#C44E52", "MIR_ISO": "#8172B3", "DPX": "#CCB974"}


def save(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=300)
    plt.close(fig)


def fig01_funnel(counts, fig_dir):
    labels = list(counts)
    vals = list(counts.values())
    colors = ["#8C8C8C"] + ["#4C72B0"] * 3 + ["#C44E52", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(labels, vals, color=colors[:len(vals)])
    for i, v in enumerate(vals):
        ax.text(i, v + 6, str(v), ha="center", fontsize=9)
        if 0 < i <= 3:
            ax.text(i, v / 2, f"-{vals[i - 1] - v}", ha="center", color="white", fontsize=9, fontweight="bold")
    ax.axvline(3.5, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Número de predictores (P)")
    ax.set_title("Embudo de poda determinista y niveles de tolerancia de nulos")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylim(0, max(vals) * 1.12)
    save(fig, fig_dir, "F2_I1_FIG01_embudo_poda_deterministica.png")


def fig02_blocks(dic_up, fig_dir):
    d = dic_up[dic_up.STATUS_I1.isin(["PRUNED_CONST", "PRUNED_IDENTITY", "PRUNED_LINEAGE", "RETAINED_S3"])].copy()
    d["BLOQUE"] = d.MOL_CTX + "_" + d.NIV
    t = d.groupby(["BLOQUE", "STATUS_I1"]).size().unstack(fill_value=0)
    order = ["RETAINED_S3", "PRUNED_LINEAGE", "PRUNED_IDENTITY", "PRUNED_CONST"]
    lab = {"RETAINED_S3": "Retenidas (S3)", "PRUNED_LINEAGE": "Duplicado de linaje",
           "PRUNED_IDENTITY": "Identidad exacta", "PRUNED_CONST": "Constante"}
    col = {"RETAINED_S3": "#4C72B0", "PRUNED_LINEAGE": "#8172B3", "PRUNED_IDENTITY": "#DD8452", "PRUNED_CONST": "#C44E52"}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(len(t))
    for s in order:
        if s in t:
            ax.bar(t.index, t[s], bottom=bottom, color=col[s], label=lab[s])
            bottom += t[s].values
    ax.set_ylabel("Número de variables analíticas")
    ax.set_xlabel("Bloque (molécula en contexto y nivel)")
    ax.set_title("Destino de las variables por bloque de origen tras la poda determinista")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I1_FIG02_poda_por_bloque.png")


def fig03_bounds(aud, fig_dir):
    a = aud[aud.STATUS != "INFO"]
    t = a.groupby(["RULE", "STATUS"]).size().unstack(fill_value=0)
    for s in ("PASS", "WARN", "FAIL"):
        if s not in t:
            t[s] = 0
    t = t[["PASS", "WARN", "FAIL"]].sort_values("PASS")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    left = np.zeros(len(t))
    for s, c in (("PASS", "#55A868"), ("WARN", "#DD8452"), ("FAIL", "#C44E52")):
        ax.barh(t.index, t[s], left=left, color=c, label=s)
        left += t[s].values
    ax.set_xlabel("Número de variables evaluadas")
    ax.set_title("Auditoría de límites biofísicos por regla")
    ax.legend(frameon=False, loc="lower right")
    save(fig, fig_dir, "F2_I1_FIG03_auditoria_limites_biofisicos.png")


def _polar(raw, dic, cols, circ, fig_dir, fname, title, sectors, to360):
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for (a0, a1, lab, colr) in sectors:
        ax.bar(np.deg2rad((a0 + a1) / 2), 5.2, width=np.deg2rad(a1 - a0), bottom=0,
               color=colr, alpha=0.18, edgecolor="none")
        ax.text(np.deg2rad((a0 + a1) / 2), 5.45, lab, ha="center", va="center", fontsize=8)
    ctxs = [c for c in CTX_COLORS if any(dic.at[x, "MOL_CTX"] == c for x in cols)]
    rpos = {c: i + 1 for i, c in enumerate(ctxs)}
    regoff = {"GLB": 0.0, "T1": 0.15, "P1": 0.3, "B": 0.45, "P2": 0.6, "O": 0.75}
    for c in cols:
        ctx, reg = dic.at[c, "MOL_CTX"], dic.at[c, "REG"]
        v = num(raw, c).dropna()
        th = np.deg2rad(v % 360 if to360 else v)
        r = np.full(len(v), rpos[ctx] + regoff.get(reg, 0))
        ax.scatter(th, r, s=26, color=CTX_COLORS[ctx], marker="x" if c in circ else "o",
                   alpha=0.85, linewidths=1.2)
    ax.set_rticks([rpos[c] for c in ctxs])
    ax.set_yticklabels(ctxs, fontsize=7)
    ax.set_rlim(0, 5.7)
    handles = [Patch(color=CTX_COLORS[c], label=c) for c in ctxs] + \
              [plt.Line2D([], [], marker="x", ls="", color="k", label="CIRC_RISK")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    ax.set_title(title, pad=24)
    save(fig, fig_dir, fname)


def fig04_pseudo(raw, dic, s3, circ, fig_dir):
    cols = [c for c in s3 if dic.at[c, "BASE"] in ("PSEUDO_P_SUGAR_MEAN", "PSEUDO_P_SUGAR")
            and dic.at[c, "MOL_CTX"] in CTX_COLORS]
    sectors = [(0, 36, "Norte\nC3'-endo", "#4C72B0"), (144, 180, "C2'-endo", "#C44E52"),
               (180, 216, "C3'-exo", "#DD8452")]
    _polar(raw, dic, cols, circ, fig_dir, "F2_I1_FIG04_pseudorrotacion_polar.png",
           "Fase de pseudorrotación P por variable y variante", sectors, to360=False)


def fig05_chi(raw, dic, s3, circ, fig_dir):
    cols = [c for c in s3 if dic.at[c, "BASE"] in ("CHI_ANGLE_MEAN", "CHI_ANGLE")
            and dic.at[c, "MOL_CTX"] in CTX_COLORS]
    sectors = [(90, 270, "anti", "#4C72B0"), (270, 360, "syn", "#C44E52"), (0, 90, "syn", "#C44E52")]
    _polar(raw, dic, cols, circ, fig_dir, "F2_I1_FIG05_chi_polar.png",
           "Ángulo glicosídico χ por variable y variante (escala 0-360°)", sectors, to360=True)


def fig06_patterns(summ, fig_dir):
    s = summ.copy()
    mat = np.array([[int(ch) for ch in p] for p in s.PATTERN])
    fig, ax = plt.subplots(figsize=(8.5, 0.7 * len(s) + 1.8))
    ax.imshow(mat, cmap=ListedColormap(["#E8E8E8", "#4C72B0"]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(7))
    ax.set_xticklabels(VAR_ORDER)
    labels = [f"{r.N_COLS} cols · {r.MATCHED_FLAG.split(';')[0] if r.MATCHED_FLAG else 'sin bandera (UNDEF)'}"
              for _, r in s.iterrows()]
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Variante del MRE")
    ax.set_title("Patrones de nulos y su correspondencia con las banderas de región")
    ax.legend(handles=[Patch(color="#4C72B0", label="Nulo"), Patch(color="#E8E8E8", label="Observado")],
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I1_FIG06_patrones_nulos_vs_banderas.png")


def fig07_thresholds(dic, s3, branches, fig_dir):
    regs = ["GLB", "T1", "P1", "B", "P2", "O"]
    series = {"S3 (antes del umbral)": s3, **branches}
    cols = {"S3 (antes del umbral)": "#B0B0B0", "CONSERVADOR": "#C44E52", "INTERMEDIO": "#DD8452", "PERMISIVO": "#55A868"}
    x = np.arange(len(regs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (k, keep) in enumerate(series.items()):
        cnt = pd.Series([dic.at[c, "REG"] for c in keep]).value_counts()
        vals = [int(cnt.get(r, 0)) for r in regs]
        ax.bar(x + (i - 1.5) * w, vals, w, color=cols[k], label=f"{k} (P={len(keep)})")
    ax.set_xticks(x)
    ax.set_xticklabels(regs)
    ax.set_xlabel("Región estructural")
    ax.set_ylabel("Variables retenidas")
    ax.set_title("Sensibilidad de la retención por región a la tolerancia de nulos")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I1_FIG07_sensibilidad_umbral_nulos.png")


def fig08_nzv(nzv, fig_dir):
    t = nzv.groupby("NZV_N_MINORITY").agg(total=("VAR_NAME", "size"), wt=("NZV_WT_ONLY", "sum"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(t.index, t.total - t.wt, color="#4C72B0", label="Otras variantes discrepantes")
    ax.bar(t.index, t.wt, bottom=t.total - t.wt, color="#C44E52", label="WT como único discrepante")
    for k, v in t.total.items():
        ax.text(k, v + 1, str(v), ha="center", fontsize=9)
    ax.set_xlabel("Observaciones distintas del valor modal (n_minoría)")
    ax.set_ylabel("Número de variables")
    ax.set_title("Diagnóstico anticipado de varianza cercana a cero (no aplicado)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I1_FIG08_diagnostico_nzv.png")


#==============================================================================
# MÓDULO 11. ORQUESTACIÓN Y EXPORTACIÓN
# Objetivo: ejecutar el flujo completo y escribir los artefactos.
# Inputs: rutas y nivel (--nivel).
# Proceso: carga -> roles/V6 -> poda S1-S3 -> límites -> nulos -> NZV -> ramas ->
#          diccionario -> figuras -> exportación (3 ramas o dataset final).
# Outputs: datasets, registros, diccionario, log y ocho figuras.
#==============================================================================
def main():
    args = parse_args()
    out, figd = args.output_dir, args.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out)
    log.info(f"F2 Iteración 1 {VERSION} | modo {args.nivel} | fecha {DATE_TAG} | pandas {pd.__version__}")

    log.info("MÓDULO 2 | Carga y validación")
    raw, dic, na_map = load_inputs(args, log)

    log.info("MÓDULO 3 | Roles e imputación V6")
    analytic, meta, ys = split_roles(raw, dic, out, log)
    raw, _ = apply_v6(raw, dic, log)

    log.info("MÓDULO 4 | Poda determinista")
    s1, s2, s3, plog, alias_df = prune(raw, dic, analytic, log)

    log.info("MÓDULO 5 | Auditoría de límites biofísicos")
    aud, status, circ = bounds_audit(raw, dic, s3, log)

    log.info("MÓDULO 6 | Mecanismo de nulos")
    nam, summ = na_mechanism(raw, s3, na_map, log)

    log.info("MÓDULO 7 | Diagnóstico NZV (no aplicado)")
    nzv = nzv_diagnostic(raw, s3, log)

    log.info("MÓDULO 8 | Ramas de sensibilidad")
    branches, comp = build_branches(raw, dic, s3, log)

    log.info("MÓDULO 9 | Diccionario")
    dic_up = update_dictionary(dic, plog, alias_df, aud, status, circ, nam, nzv, branches, s3)

    log.info("MÓDULO 10 | Figuras")
    counts = {"S0 analítico": len(analytic), "S1 sin constantes": len(s1), "S2 sin identidades": len(s2),
              "S3 sin linaje": len(s3), **{k.capitalize(): len(v) for k, v in branches.items()}}
    fig01_funnel(counts, figd)
    fig02_blocks(dic_up, figd)
    fig03_bounds(aud, figd)
    fig04_pseudo(raw, dic, s3, circ, figd)
    fig05_chi(raw, dic, s3, circ, figd)
    fig06_patterns(summ, figd)
    fig07_thresholds(dic, s3, branches, figd)
    fig08_nzv(nzv, figd)

    log.info("MÓDULO 11 | Exportación")
    plog.to_csv(out / f"F2_I1_PRUNING_LOG_{VERSION}.csv", index=False, encoding="utf-8-sig")
    aud.to_csv(out / f"F2_I1_BOUNDS_AUDIT_{VERSION}.csv", index=False, encoding="utf-8-sig")
    nam.to_csv(out / f"F2_I1_NA_MECHANISM_{VERSION}.csv", index=False, encoding="utf-8-sig")
    alias_df.to_csv(out / f"F2_I1_ALIAS_GROUPS_{VERSION}.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out / f"F2_I1_THRESHOLD_COMPARISON_{VERSION}.csv", index=False, encoding="utf-8-sig")
    dic_up.to_csv(out / f"F2_I1_DATA_DICTIONARY_{VERSION}.csv", index=False, encoding="utf-8-sig")
    if args.nivel == "EXPLORATORIO":
        for lvl, keep in branches.items():
            raw[ys + keep].to_csv(out / f"F2_I1_DATASET_{lvl}_{VERSION}.csv", encoding="utf-8-sig")
        log.info("  Modo exploratorio: 3 ramas exportadas. Tras la decisión, ejecutar con --nivel <NIVEL>.")
    else:
        keep = branches[args.nivel]
        raw[ys + keep].to_csv(out / f"F2_I1_DATASET_FINAL_{VERSION}_{DATE_TAG}.csv", encoding="utf-8-sig")
        log.info(f"  Dataset final ({args.nivel}): N=7 x P={len(ys) + len(keep)} (2 respuestas + {len(keep)} predictores)")
    log.info(f"RESUMEN | S0={len(analytic)} S1={len(s1)} S2={len(s2)} S3={len(s3)} | " +
             " ".join(f"{k}={len(v)}" for k, v in branches.items()) +
             f" | FAIL={int((status == 'FAIL').sum())} CIRC_RISK={len(circ)} alias={int((alias_df.ROLE_IN_GROUP == 'alias').sum())} en {alias_df.ALIAS_GROUP.nunique()} grupos")


if __name__ == "__main__":
    main()
