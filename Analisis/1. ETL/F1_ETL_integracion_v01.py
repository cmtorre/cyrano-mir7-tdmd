# -*- coding: utf-8 -*-
"""
F1_ETL_integracion_v01.py
Fase I (ETL) del flujo OE6: integración de las matrices OE1 a OE5 del sistema
Cyrano (variantes del MRE) / miR-7-5p en una Matriz Maestra (RAW_DATASET),
con diccionario de datos, mapa de nulos tipificados y figuras de control.

Ejecución (Windows, Conda tesis_rnas):
    conda activate tesis_rnas
    python F1_ETL_integracion_v01.py

Opcional, para probar con otras rutas:
    python F1_ETL_integracion_v01.py --input-dir "..." --output-dir "..." --figures-dir "..."
"""

#==============================================================================
# MÓDULO 0. IMPORTACIONES, RUTAS Y PARÁMETROS GLOBALES
# Objetivo: centralizar rutas, versión, fecha y registro de archivos de entrada.
# Inputs: ninguno (constantes del script y argumentos opcionales de terminal).
# Proceso: define INPUT_DIR, OUTPUT_DIR y FIGURES_DIR con pathlib.Path y el
#          registro FILE_REGISTRY, que asigna a cada hoja su OE, su molécula en
#          contexto (MOL_CTX) y su nivel estructural (NIV).
# Outputs: constantes globales usadas por el resto de módulos.
#==============================================================================
import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: solo guarda archivos
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

VERSION = "v01"
DATE_TAG = datetime.now().strftime("%Y%m%d")

DEFAULT_INPUT_DIR = Path(r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-6\Nuevo_Flujo\3.FASE 1_ETL\00_matrices_crudas")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-6\Nuevo_Flujo\3.FASE 1_ETL\02_output_txt")
DEFAULT_FIGURES_DIR = Path(r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-6\Nuevo_Flujo\3.FASE 1_ETL\01_outputs_figuras")

VAR_ORDER = ["WT", "M1", "M2", "M3", "M4", "M5", "M6"]
REGIONS = ["GLB", "T1", "P1", "B", "P2", "O"]

# Cada entrada: archivo, hoja, OE, MOL_CTX, NIV por defecto
FILE_REGISTRY = [
    ("1.OE1_Resultados_Cyranos y miR7-5-P.xlsx",                     "A_FINAL_MREs",   "OE1", "EXP",     "EXP"),
    ("2.1.OE2_Result_est_prim_MATRIZ_MREs_CYR.xlsx",                 "A_FINAL_MREs",   "OE2", "CYR_ISO", "1D"),
    ("2.2.OE2_Result_est-sec_MATRIZ_MREs.xlsx",                      "A_FINAL_MREs",   "OE2", "CYR_ISO", "2D"),
    ("2.3.OE2_Result-est-terc__Matriz_MREs.xlsx",                    "A_FINAL_MREs",   "OE2", "CYR_ISO", "3D"),
    ("3.1.OE3_Result_est_prim_MIR.xlsx",                             "A_FINAL_MIR",    "OE3", "MIR_ISO", "1D"),
    ("3.2.OE3_Result_est_sec_MIR.xlsx",                              "A_FINAL_MIR",    "OE3", "MIR_ISO", "2D"),
    ("3.3.OE3_Result_est_terc_MIR.xlsx",                             "A_FINAL_MIR",    "OE3", "MIR_ISO", "3D"),
    ("4.1.OE4_Result_est_prim_MATRIZ_MREs_MIR.xlsx",                 "A_FINAL_MREs",   "OE4", "CYR_DPX", "1D"),
    ("4.1.OE4_Result_est_prim_MATRIZ_MREs_MIR.xlsx",                 "A_FINAL_MIRs",   "OE4", "MIR_DPX", "1D"),
    ("4.2.OE4_Result_est_sec_terc_MATRIZ_MREs_MIR.xlsx",             "A_FINAL_MREs",   "OE4", "CYR_DPX", "3D"),
    ("4.2.OE4_Result_est_sec_terc_MATRIZ_MREs_MIR.xlsx",             "A_FINAL_MIRs",   "OE4", "MIR_DPX", "3D"),
    ("5.1.OE5_Result_termodinamica_est_sec_terc_Matriz_Duplex.xlsx", "A_FINAL_DUPLEX", "OE5", "DPX",     "3D"),
]
# Los nombres corresponden a los archivos locales. Si un nombre difiere solo en puntos,
# espacios o guiones (por ejemplo, 2.1.OE2 frente a 2_1_OE2), resolve_input_files()
# lo encuentra igualmente mediante un nombre normalizado.
# Hoja A_FINAL_MIR de OE1 no se ingiere: no tiene ID_VAR y su secuencia ya está en OE3.

# Variables de OE5 que pertenecen al nivel termodinámico (TD)
TD_BASES = {"NET_DG", "HYBRIDIZATION_DG", "UNFOLDING_A_DG", "UNFOLDING_B_DG",
            "CYR_MIR_ACCES_DG_COST", "MFE_DG", "ENSEMBLE_DG", "MFE_FREQ", "FINAL_DG"}

# Unificación de sinónimos (se aplica al nombre base ya limpio)
SYNONYMS = {
    "SUGAR_PUCKER_C3_EXO_PCT": "SUGAR_PUCKER_C3EXO_PCT",
    "BASE_PAIRS_TOTAL": "BP_TOTAL", "TOTAL_BASE_PAIRS": "BP_TOTAL", "BASE_PAIR_TOTAL": "BP_TOTAL",
    "CAN_BASE_PAIRS": "BP_CAN", "CAN_BASE_PAIR_TOTAL": "BP_CAN",
    "NC_BASE_PAIRS": "BP_NC", "NC_BASE_PAIR_TOTAL": "BP_NC", "NC_PAIRS_TOTAL": "BP_NC", "NC_PAIR": "BP_NC",
    "STACKING": "STACK", "STACKING_TOTAL": "STACK",
    "RG_GLOBAL_DUPLEX": "RG", "END_DIST_GLOBAL_DUPLEX": "END_DIST",
    "RG_GLOBAL_INTER": "RG_INTER",
}
SYNONYMS_GLB_ONLY = {"SEQ": "SEQ_REAL", "LEN": "LEN_REAL"}

# Variables de conteo o tamaño: se imputan con 0 cuando la región no existe
COUNT_BASES = {"LEN", "BP_TOTAL", "BP_CAN", "BP_NC", "STACK", "N_MOD"}
# Variables que no aplican a la secuencia silvestre (sin modificación)
NOT_APPLIC_BASES = {"POS_MOD", "TYPE_MOD", "REG_MIR_MOD"}
# Variables globales 3D para los deltas de acoplamiento (decisión 4 aprobada)
DELTA_BASES = ["RG", "END_DIST", "CHI_ANGLE_MEAN", "CHI_ANGLE_SD",
               "PSEUDO_P_SUGAR_MEAN", "PSEUDO_P_SUGAR_SD", "SASA_TOTAL", "STACK"]

NA_CODES = ["OBS", "STRUCT_ABS", "UNDEF", "NOT_APPLIC", "TECH_NA"]


def parse_args():
    p = argparse.ArgumentParser(description="Fase I ETL OE1-OE5")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return p.parse_args()


def normalize_filename(name: str) -> str:
    """Minúsculas y cualquier secuencia no alfanumérica convertida en un solo '_'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def resolve_input_files(in_dir: Path):
    """Relaciona cada archivo del registro con un archivo real de INPUT_DIR.
    Primero busca el nombre exacto; si no existe, compara nombres normalizados.
    Ignora archivos temporales de Excel (~$...)."""
    available = [f for f in in_dir.glob("*.xlsx") if not f.name.startswith("~$")]
    by_norm = {normalize_filename(f.name): f for f in available}
    resolved, missing = {}, []
    for fname in sorted({f for f, *_ in FILE_REGISTRY}):
        exact = in_dir / fname
        if exact.exists():
            resolved[fname] = exact
        elif normalize_filename(fname) in by_norm:
            resolved[fname] = by_norm[normalize_filename(fname)]
        else:
            missing.append(fname)
    return resolved, missing, [f.name for f in available]


#==============================================================================
# MÓDULO 1. CONFIGURACIÓN DEL REGISTRO (LOG)
# Objetivo: dejar trazabilidad escrita de cada decisión y advertencia del ETL.
# Inputs: OUTPUT_DIR.
# Proceso: crea un logger que escribe a consola y a F1_ETL_log_v01.txt.
# Outputs: F1_ETL_log_v01.txt en OUTPUT_DIR.
#==============================================================================
def setup_logger(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("F1_ETL")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(output_dir / f"F1_ETL_log_{VERSION}.txt", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


#==============================================================================
# MÓDULO 2. EXTRACCIÓN (E): LECTURA ROBUSTA DE HOJAS CON ENCABEZADOS MULTINIVEL
# Objetivo: leer cada hoja sin depender de la fila o columna donde inicia la tabla.
# Inputs: archivo .xlsx y nombre de hoja.
# Proceso: 1) lee la hoja sin encabezado; 2) localiza la celda 'ID_VAR';
#          3) toma las filas superiores como etiquetas de bloque y las propaga
#          hacia la derecha (celdas combinadas); 4) conserva la última etiqueta
#          de bloque de cada columna, que indica su región.
# Outputs: DataFrame de datos y lista paralela de etiquetas de bloque.
#==============================================================================
def read_sheet(path: Path, sheet: str):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    pos = [(i, j) for i in range(raw.shape[0]) for j in range(raw.shape[1])
           if isinstance(raw.iat[i, j], str) and raw.iat[i, j].strip() == "ID_VAR"]
    if not pos:
        raise ValueError(f"No se encontró 'ID_VAR' en {path.name}/{sheet}")
    hr, _ = pos[0]
    header = raw.iloc[hr]
    cols = [j for j in range(raw.shape[1]) if pd.notna(header.iat[j])]
    blocks = raw.iloc[:hr].ffill(axis=1) if hr > 0 else None
    block_labels = []
    for j in cols:
        lab = ""
        if blocks is not None:
            for r in range(hr - 1, -1, -1):  # etiqueta más cercana al encabezado
                v = blocks.iat[r, j]
                if pd.notna(v):
                    lab = str(v)
                    break
        block_labels.append(lab)
    body = raw.iloc[hr + 1:, cols].dropna(how="all").reset_index(drop=True)
    body.columns = [str(header.iat[j]).strip() for j in cols]
    return body, block_labels


#==============================================================================
# MÓDULO 3. TRANSFORMACIÓN (T1): LLAVE COMÚN, NOMBRES Y REGIONES
# Objetivo: resolver los cinco esquemas de ID y las colisiones de nombre.
# Inputs: DataFrame de una hoja, etiquetas de bloque y datos del registro.
# Proceso: extrae VAR_KEY (WT, M1..M6); limpia caracteres no válidos; asigna
#          región desde la etiqueta de bloque y la contrasta con el sufijo del
#          nombre; unifica sinónimos; construye {MOL_CTX}_{NIV}_{BASE}_{REG}.
# Outputs: DataFrame renombrado indexado por VAR_KEY y registros de linaje.
#==============================================================================
KEY_RE = re.compile(r"(?<![A-Z0-9])(WT|M[1-6])(?![0-9])")
SUFFIX_RE = re.compile(r"_(T1|P1|P2|P3|B|O)$")


def extract_key(id_value) -> str:
    m = KEY_RE.search(str(id_value).upper())
    return m.group(1) if m else None


def clean_name(name: str) -> str:
    s = str(name).strip().upper().replace("'", "")
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def region_from_block(label: str) -> str:
    s = re.sub(r"\s+", "", str(label).upper())
    if "OVERHANG" in s:
        return "O"
    if "BULGE" in s:
        return "B"
    if "T1" in s or "ANCHOR" in s:
        return "T1"
    if "P1" in s or "SEED" in s:
        return "P1"
    if "P2" in s or "3'" in s:
        return "P2"
    return "GLB"


def is_text_column(s: pd.Series) -> bool:
    v = s.dropna()
    if v.empty:
        return False
    return bool(pd.to_numeric(v, errors="coerce").isna().any())


def transform_sheet(df, block_labels, reg_entry, log):
    fname, sheet, oe, mol_ctx, niv_default = reg_entry
    df = df.copy()
    records = []
    label_of = dict(zip(df.columns, block_labels))  # nombre original -> etiqueta de bloque

    # --- llave común
    if mol_ctx == "MIR_ISO":
        if len(df) != 1:
            raise ValueError(f"OE3 debe tener N=1 y tiene N={len(df)}")
        df = pd.concat([df] * len(VAR_ORDER), ignore_index=True)  # replicado de referencia
        df["VAR_KEY"] = VAR_ORDER
        log.info(f"  {fname}/{sheet}: N=1 replicado a 7 filas como estado de referencia")
    else:
        df["VAR_KEY"] = df["ID_VAR"].map(extract_key)
        if df["VAR_KEY"].isna().any():
            raise ValueError(f"IDs sin llave en {fname}/{sheet}: {df.loc[df.VAR_KEY.isna(),'ID_VAR'].tolist()}")
        if df["VAR_KEY"].duplicated().any():
            raise ValueError(f"Llaves duplicadas en {fname}/{sheet}")
    df = df.set_index("VAR_KEY").drop(columns=["ID_VAR"])

    new_cols = {}
    for orig in df.columns:
        lab = label_of[orig]
        cname = clean_name(orig)
        reg_block = region_from_block(lab)
        m = SUFFIX_RE.search(cname)
        suf = m.group(1) if m else None
        base = SUFFIX_RE.sub("", cname) if suf else cname
        note = ""
        if reg_block == "GLB" and suf:
            reg = "P2" if suf == "P3" else suf
            note = f"región tomada del sufijo {suf} (bloque GLB)"
        elif reg_block != "GLB":
            reg = reg_block
            if suf is None:
                note = f"sufijo de región ausente; asignado por bloque ({reg})"
            elif suf != reg_block:
                note = f"sufijo {suf} corregido a {reg_block} según bloque"
        else:
            reg = "GLB"
        if note:
            log.info(f"  [región] {fname}/{sheet} :: {orig} -> {note}")

        if reg == "GLB" and base in SYNONYMS_GLB_ONLY:
            base = SYNONYMS_GLB_ONLY[base]
        base = SYNONYMS.get(base, base)

        text = is_text_column(df[orig])
        if mol_ctx == "EXP":
            niv = "EXP"
            if base == "MIR7_REL":
                name, role = "Y_MIR7_REL", "RESPONSE"
            elif base == "TDMD_SCORE":
                name, role = "META_EXP_TDMD_SCORE", "META_REDUNDANT"
            else:
                name, role = f"META_EXP_{base}", "META"
        else:
            niv = ("TD" if base in TD_BASES else "3D") if mol_ctx == "DPX" else niv_default
            hier = f"{mol_ctx}_{niv}_{base}_{reg}"
            name, role = (f"META_{hier}", "META") if text else (hier, "PREDICTOR")
        if name in new_cols.values():
            raise ValueError(f"Colisión de nombre tras renombrar: {name} ({fname}/{sheet})")
        new_cols[orig] = name
        records.append(dict(VAR_NAME=name, ORIG_NAME=orig, OE=oe, SOURCE_FILE=fname,
                            SOURCE_SHEET=sheet, MOL_CTX=mol_ctx, NIV=niv, REG=reg,
                            BASE=base, ROLE=role, NOTES=note))
    df = df.rename(columns=new_cols)
    return df, records


#==============================================================================
# MÓDULO 4. TRANSFORMACIÓN (T2): CASTEO ESTRICTO DE TIPOS
# Objetivo: homogeneizar tipos (float64, Int64 para conteos, category para texto).
# Inputs: DataFrame integrado y registros de linaje.
# Proceso: convierte N_BIO ('n=4') a entero; castea columnas numéricas a float64
#          y las de conteo con valores enteros a Int64 (entero con nulos).
# Outputs: DataFrame con tipos definitivos.
#==============================================================================
def cast_types(df, meta):
    for c in df.columns:
        base = meta[c]["BASE"]
        if c == "META_EXP_N_BIO":
            df[c] = df[c].astype(str).str.extract(r"(\d+)")[0].astype("Int64")
            continue
        if is_text_column(df[c]):
            df[c] = df[c].astype("category")
            continue
        num = pd.to_numeric(df[c], errors="coerce").astype("float64")
        v = num.dropna()
        if (base in COUNT_BASES or base.startswith("LEN")) and len(v) and np.allclose(v, v.round()):
            df[c] = num.round().astype("Int64")
        else:
            df[c] = num
    return df


#==============================================================================
# MÓDULO 5. TIPIFICACIÓN DE NULOS, BANDERAS DE REGIÓN E IMPUTACIÓN DUAL
# Objetivo: distinguir ausencia biológica, métrica indefinida, no aplicable y
#           faltante técnico; imputar 0 solo en conteos de regiones ausentes.
# Inputs: DataFrame tipado y registros de linaje.
# Proceso: 1) presencia de región = LEN_{REG} > 0 en el mismo MOL_CTX (para DPX,
#          presencia en al menos una hebra); 2) cada NaN recibe un código;
#          3) conteos/tamaños con STRUCT_ABS pasan a 0; geométricas quedan NaN.
# Outputs: DataFrame imputado, NA_MAP (códigos por celda) y banderas FLAG_*_HAS_*.
#==============================================================================
def build_presence(df):
    pres = {}
    for ctx in ["CYR_ISO", "MIR_ISO", "CYR_DPX", "MIR_DPX"]:
        for reg in ["T1", "P1", "B", "P2", "O"]:
            c = f"{ctx}_1D_LEN_{reg}"
            if c in df.columns:
                pres[(ctx, reg)] = df[c].fillna(0).astype(float) > 0
    for reg in ["T1", "P1", "B", "P2", "O"]:
        a, b = pres.get(("CYR_DPX", reg)), pres.get(("MIR_DPX", reg))
        if a is not None and b is not None:
            pres[("DPX", reg)] = a | b
    return pres


def classify_na(df, meta, pres, log):
    na_map = pd.DataFrame("OBS", index=df.index, columns=df.columns, dtype=object)
    n_mod = df.get("CYR_ISO_1D_N_MOD_GLB")
    for c in df.columns:
        m = meta[c]
        miss = df[c].isna()
        if not miss.any():
            continue
        all_na = miss.all()
        p = pres.get((m["MOL_CTX"], m["REG"]))
        for k in df.index[miss]:
            if m["BASE"] in NOT_APPLIC_BASES and n_mod is not None and n_mod.loc[k] == 0:
                code = "NOT_APPLIC"
            elif m["REG"] != "GLB" and p is not None and not bool(p.loc[k]):
                code = "STRUCT_ABS"
            elif m["REG"] != "GLB" or all_na:
                code = "UNDEF"
            else:
                code = "TECH_NA"
            na_map.at[k, c] = code
    n_tech = int((na_map == "TECH_NA").sum().sum())
    log.info(f"  Celdas por código: " + ", ".join(f"{x}={int((na_map == x).sum().sum())}" for x in NA_CODES))
    if n_tech:
        log.warning(f"  Se detectaron {n_tech} faltantes técnicos (TECH_NA): revisar")
    und = [(c, k) for c in df.columns for k in df.index if na_map.at[k, c] == "UNDEF"]
    for c in sorted(set(c for c, _ in und)):
        ks = [k for cc, k in und if cc == c]
        log.info(f"  [UNDEF] {c} en {ks}")
    return na_map


def impute_counts(df, meta, na_map, log):
    n_imp = {}
    for c in df.columns:
        m = meta[c]
        if m["ROLE"].startswith("META"):
            continue
        if m["BASE"] in COUNT_BASES or m["BASE"].startswith("LEN"):
            mask = na_map[c] == "STRUCT_ABS"
            if mask.any():
                df.loc[mask, c] = 0
                n_imp[c] = int(mask.sum())
    log.info(f"  Imputación con 0 (conteos en regiones ausentes): {len(n_imp)} columnas, "
             f"{sum(n_imp.values())} celdas")
    return df, n_imp


def region_flags(df, pres):
    flags = {}
    for (ctx, reg), s in pres.items():
        if reg in ("P1", "B", "O") and ctx != "MIR_ISO":  # MIR_ISO: N=1, siempre presente
            flags[f"FLAG_{ctx}_HAS_{reg}"] = s.astype("Int64")
    return pd.DataFrame(flags, index=df.index)


#==============================================================================
# MÓDULO 6. VARIABLES DERIVADAS: RESPUESTA TRANSFORMADA Y DELTAS DE ACOPLAMIENTO
# Objetivo: crear Y_LOG2_MIR7REL y DLT_{CYR|MIR}_3D_{VAR}_GLB.
# Inputs: DataFrame imputado.
# Proceso: log2 de miR7_rel (cociente relativo a WT); delta = valor en dúplex
#          menos valor aislado, solo en variables globales 3D (decisión 4).
# Outputs: DataFrame de variables derivadas y sus registros de linaje.
#==============================================================================
def derived_variables(df, log):
    out, recs = {}, []
    y = df["Y_MIR7_REL"].astype(float)
    if (y <= 0).any():
        raise ValueError("miR7_rel contiene valores <= 0; log2 no definido")
    out["Y_LOG2_MIR7REL"] = np.log2(y)
    recs.append(dict(VAR_NAME="Y_LOG2_MIR7REL", OE="OE1", MOL_CTX="EXP", NIV="EXP", REG="GLB",
                     BASE="LOG2_MIR7_REL", ROLE="RESPONSE", DERIVED_FROM="log2(Y_MIR7_REL)"))
    for strand in ["CYR", "MIR"]:
        for b in DELTA_BASES:
            a, i = f"{strand}_DPX_3D_{b}_GLB", f"{strand}_ISO_3D_{b}_GLB"
            if a in df.columns and i in df.columns:
                name = f"DLT_{strand}_3D_{b}_GLB"
                out[name] = df[a].astype(float) - df[i].astype(float)
                recs.append(dict(VAR_NAME=name, OE="OE4-" + ("OE2" if strand == "CYR" else "OE3"),
                                 MOL_CTX=f"DLT_{strand}", NIV="3D", REG="GLB", BASE=b,
                                 ROLE="PREDICTOR", DERIVED_FROM=f"{a} - {i}"))
            else:
                log.warning(f"  Delta omitido por columna ausente: {a} o {i}")
    log.info(f"  Variables derivadas creadas: {len(out)}")
    return pd.DataFrame(out, index=df.index), recs


#==============================================================================
# MÓDULO 7. AUDITORÍA DE IDENTIDADES LINEALES, CONSTANTES Y DUPLICADOS
# Objetivo: marcar (sin eliminar) columnas redundantes para la Iteración 1.
# Inputs: RAW_DATASET completo.
# Proceso: verifica numéricamente cada identidad (tolerancia 1e-3); marca
#          constantes (≤1 valor único) y duplicados exactos entre fuentes.
# Outputs: diccionarios FLAG_IDENTITY/DERIVED_FROM, FLAG_CONST, DUPLICATE_OF y
#          tabla resumen de identidades.
#==============================================================================
def check_identities(raw, log, tol=1e-3):
    ident, rows = {}, []
    cols = set(raw.columns)

    def test(target, expr_cols, fn, family):
        if target not in cols or not all(c in cols for c in expr_cols):
            return
        sub = raw[[target] + expr_cols].astype(float).dropna()
        if sub.empty:
            return
        resid = (sub[target] - fn(sub)).abs()
        ok = bool(resid.max() <= tol)
        rows.append(dict(FAMILY=family, TARGET=target, N=len(sub), MAX_ABS_RESID=float(resid.max()), HOLDS=ok))
        if ok:
            ident[target] = (family, " ; ".join(expr_cols))

    for c in raw.columns:
        if c.startswith("META") or c.startswith("FLAG"):
            continue
        if "_ACCES_POS_" in c:
            p = c.replace("_ACCES_POS_", "_PROB_PAIR_")
            test(c, [p], lambda s, p=p: 100 - s[p], "ACCES_POS = 100 - PROB_PAIR")
        if "_SASA_BASES_PCT_" in c:
            p = c.replace("_SASA_BASES_PCT_", "_SASA_BB_PCT_")
            test(c, [p], lambda s, p=p: 100 - s[p], "SASA_BASES_PCT = 100 - SASA_BB_PCT")
        if "_PCT_PIR_" in c:
            p = c.replace("_PCT_PIR_", "_PCT_PUR_")
            test(c, [p], lambda s, p=p: 100 - s[p], "PCT_PIR = 100 - PCT_PUR")
        if "_PCT_C_" in c:
            ps = [c.replace("_PCT_C_", f"_PCT_{n}_") for n in "AUG"]
            test(c, ps, lambda s, ps=ps: 100 - s[ps].sum(axis=1), "PCT_C = 100 - (A+U+G)")
        if "_BP_TOTAL_" in c:
            ps = [c.replace("_BP_TOTAL_", "_BP_CAN_"), c.replace("_BP_TOTAL_", "_BP_NC_")]
            test(c, ps, lambda s, ps=ps: s[ps].sum(axis=1), "BP_TOTAL = BP_CAN + BP_NC")
    test("DPX_TD_NET_DG_GLB", ["DPX_TD_HYBRIDIZATION_DG_GLB", "DPX_TD_CYR_MIR_ACCES_DG_COST_GLB"],
         lambda s: s.iloc[:, 1] + s.iloc[:, 2], "NET_DG = HYB_DG + ACCES_COST")
    test("DPX_TD_CYR_MIR_ACCES_DG_COST_GLB", ["DPX_TD_UNFOLDING_A_DG_GLB", "DPX_TD_UNFOLDING_B_DG_GLB"],
         lambda s: s.iloc[:, 1] + s.iloc[:, 2], "ACCES_COST = UNF_A + UNF_B")
    test("META_EXP_TDMD_SCORE", ["Y_MIR7_REL"], lambda s: 1 - s["Y_MIR7_REL"], "TDMD_SCORE = 1 - MIR7_REL")

    rep = pd.DataFrame(rows)
    for fam, g in rep.groupby("FAMILY"):
        log.info(f"  [identidad] {fam}: {int(g.HOLDS.sum())}/{len(g)} se cumplen "
                 f"(máx |residuo| = {g.MAX_ABS_RESID.max():.2e})")
    fails = rep[~rep.HOLDS]
    for _, r in fails.iterrows():
        log.warning(f"  Identidad NO cumplida: {r.TARGET} ({r.FAMILY}), residuo {r.MAX_ABS_RESID:.3g}")
    return ident, rep


def flag_constants_duplicates(raw):
    const = {c: raw[c].nunique(dropna=True) <= 1 for c in raw.columns}
    seen, dup = {}, {}
    for c in raw.columns:
        if const[c]:
            continue
        sig = tuple("NA" if pd.isna(v) else (round(float(v), 6) if isinstance(v, (int, float, np.number)) else str(v))
                    for v in raw[c].tolist())
        if sig in seen:
            dup[c] = seen[sig]
        else:
            seen[sig] = c
    return const, dup


#==============================================================================
# MÓDULO 8. DICCIONARIO DE DATOS (LINAJE COMPLETO)
# Objetivo: permitir rastrear cada columna hasta su OE, archivo y hoja de origen.
# Inputs: registros de linaje, RAW_DATASET, NA_MAP y banderas de auditoría.
# Proceso: añade tipo, unidades, conteo de nulos, códigos de ausencia,
#          imputaciones, constantes, identidades y duplicados.
# Outputs: DataFrame DATA_DICTIONARY.
#==============================================================================
def units_for(base: str, ctx: str) -> str:
    b = base.upper()
    if b.endswith("_DG") or b.startswith("DG_") or "DG_COST" in b:
        return "kcal/mol"
    if b in ("MFE_FREQ", "FREQ_MFE"):
        return "fracción [0-1]"
    if b.startswith("RG") or b.startswith("END_DIST"):
        return "Å"
    if b.startswith("CHI_ANGLE") or b.startswith("PSEUDO_P_SUGAR"):
        return "grados"
    if b.endswith("_PCT") or b.startswith("PCT_") or b in ("ACCES_POS", "PROB_PAIR"):
        return "%"
    if b in ("SASA_TOTAL", "SASA_BASE"):
        return "Å²"
    if b.startswith("LEN"):
        return "nt"
    if b in ("BP_TOTAL", "BP_CAN", "BP_NC", "STACK", "N_MOD"):
        return "conteo"
    if b.startswith("ENT_") or b == "ENS_DIV":
        return "adimensional"
    if b in ("MIR7_REL", "LOG2_MIR7_REL", "TDMD_SCORE"):
        return "relativo a WT" if b != "LOG2_MIR7_REL" else "log2 relativo a WT"
    return ""


def notes_for(name, base, ctx):
    n = []
    if base == "SASA_BASE":
        n.append("Tratada como Å² por no llevar sufijo _PCT (decisión 7)")
    if ctx == "DPX" and base in ("BP_TOTAL", "BP_CAN", "BP_NC"):
        n.append("Suma de interacciones por residuo de ambas hebras (decisión 5); no equivale a pares únicos")
    if ctx == "MIR_ISO":
        n.append("OE3 N=1 replicado: estado de referencia para deltas; constante por diseño")
    if name == "META_EXP_TDMD_SCORE":
        n.append("Redundante: 1 - Y_MIR7_REL; excluida del espacio analítico")
    if ctx == "DPX" and name.endswith("_B"):
        n.append("Bulge asimétrico en M4 y M5 (solo hebra miR, decisión 6)")
    return " | ".join(n)


def build_dictionary(raw, meta, na_map, n_imp, ident, const, dup):
    rows = []
    for c in raw.columns:
        m = meta.get(c, {})
        codes = na_map[c].value_counts()
        na_txt = ";".join(f"{k}:{int(v)}" for k, v in codes.items())
        fam = ident.get(c)
        notes = " | ".join(x for x in [m.get("NOTES", ""), notes_for(c, m.get("BASE", ""), m.get("MOL_CTX", ""))] if x)
        rows.append(dict(
            VAR_NAME=c, ORIG_NAME=m.get("ORIG_NAME", ""), OE=m.get("OE", ""),
            SOURCE_FILE=m.get("SOURCE_FILE", "derivada"), SOURCE_SHEET=m.get("SOURCE_SHEET", ""),
            MOL_CTX=m.get("MOL_CTX", ""), NIV=m.get("NIV", ""), REG=m.get("REG", ""),
            DTYPE=str(raw[c].dtype), UNITS=units_for(m.get("BASE", ""), m.get("MOL_CTX", "")),
            ROLE=m.get("ROLE", ""), NA_TYPE=na_txt, N_NA=int(raw[c].isna().sum()),
            N_IMPUTED_0=n_imp.get(c, 0), N_UNIQUE=int(raw[c].nunique(dropna=True)),
            FLAG_CONST=int(const.get(c, False)), FLAG_IDENTITY=fam[0] if fam else "",
            DERIVED_FROM=(fam[1] if fam else m.get("DERIVED_FROM", "")),
            DUPLICATE_OF=dup.get(c, ""), NOTES=notes))
    return pd.DataFrame(rows)


#==============================================================================
# MÓDULO 9. FIGURAS DE CONTROL (UN GRÁFICO POR ARCHIVO, 300 DPI)
# Objetivo: evidencia visual del estado antes y después del ETL.
# Inputs: resumen de hojas, NA_MAP, diccionario, reporte de identidades y RAW.
# Proceso: seis rutinas independientes, cada una genera y guarda una figura.
# Outputs: F1_FIG01 a F1_FIG06 en FIGURES_DIR.
#==============================================================================
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10})


def fig01_dimensions(summary, fig_dir):
    s = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"OE1": "#4C72B0", "OE2": "#55A868", "OE3": "#C44E52", "OE4": "#8172B3", "OE5": "#CCB974"}
    ax.barh(s["LABEL"], s["P"], color=[colors[o] for o in s["OE"]])
    for i, (p, n) in enumerate(zip(s["P"], s["N"])):
        ax.text(p + 1, i, f"P={p}  N={n}", va="center", fontsize=8)
    ax.set_xlabel("Número de descriptores (P)")
    ax.set_title("Dimensionalidad de las matrices de entrada por OE")
    ax.legend(handles=[Patch(color=v, label=k) for k, v in colors.items()], loc="lower right", frameon=False)
    ax.set_xlim(0, s["P"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_FIG01_dimensiones_por_OE.png", dpi=300)
    plt.close(fig)


def fig02_na_map(na_map, fig_dir):
    cols = [c for c in na_map.columns if (na_map[c] != "OBS").any()]
    sub = na_map[cols]
    code_num = sub.apply(lambda s: s.map({k: i for i, k in enumerate(NA_CODES)})).astype(int)
    cmap = ListedColormap(["#E8E8E8", "#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 0.13), 4.8))
    ax.imshow(code_num.values, aspect="auto", cmap=cmap, vmin=0, vmax=len(NA_CODES) - 1, interpolation="nearest")
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels(sub.index)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=4.5)
    ax.set_xlabel("Variables con al menos una celda no observada")
    ax.set_ylabel("Variante del MRE")
    ax.set_title("Mapa de ausencias tipificadas antes de la imputación")
    ax.legend(handles=[Patch(color=cmap(i), label=k) for i, k in enumerate(NA_CODES)],
              loc="upper left", bbox_to_anchor=(1.001, 1), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_FIG02_mapa_nulos_tipificados.png", dpi=300)
    plt.close(fig)


def fig03_constants(dic, fig_dir):
    d = dic[~dic.ROLE.str.startswith("META") & (dic.MOL_CTX != "")].copy()
    d["BLOQUE"] = d["MOL_CTX"] + "_" + d["NIV"]
    g = d.groupby("BLOQUE")["FLAG_CONST"].agg(["sum", "count"])
    g["var"] = g["count"] - g["sum"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(g.index, g["var"], color="#4C72B0", label="Con variación")
    ax.bar(g.index, g["sum"], bottom=g["var"], color="#C44E52", label="Constantes")
    ax.set_ylabel("Número de variables analíticas")
    ax.set_xlabel("Bloque (molécula en contexto y nivel)")
    ax.set_title("Variables constantes por bloque de origen")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_FIG03_constantes_por_bloque.png", dpi=300)
    plt.close(fig)


def fig04_identities(rep, fig_dir):
    g = rep.groupby("FAMILY").agg(n=("HOLDS", "sum"), resid=("MAX_ABS_RESID", "max")).sort_values("n")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(g.index, g["n"], color="#8172B3")
    for i, (n, r) in enumerate(zip(g["n"], g["resid"])):
        ax.text(n + 0.2, i, f"máx |residuo| = {r:.1e}", va="center", fontsize=8)
    ax.set_xlabel("Columnas que cumplen la identidad (marcadas como redundantes)")
    ax.set_title("Identidades lineales exactas verificadas en RAW_DATASET")
    ax.set_xlim(0, g["n"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_FIG04_identidades_lineales.png", dpi=300)
    plt.close(fig)


def fig05_response(raw, fig_dir, log):
    """Respuesta lineal (barras, eje izquierdo) y log2 (línea, eje derecho) en un solo gráfico."""
    y = raw["Y_MIR7_REL"].astype(float)
    ly = raw["Y_LOG2_MIR7REL"].astype(float)
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x, y, color="#4C72B0", alpha=0.85, label="Y_MIR7_REL (escala lineal)")
    for xi, v in zip(x, y):  # valor lineal dentro de la barra, lejos de la línea log2
        ax.text(xi, max(v * 0.5, 2.5), f"{v:g}", ha="center", va="center", fontsize=9,
                color="white" if v > 5 else "#4C72B0", fontweight="bold")
    ax.set_ylabel("miR-7-5p relativo a WT (lineal)", color="#4C72B0")
    ax.set_ylim(0, y.max() * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(y.index)
    ax.set_xlabel("Variante del MRE de Cyrano")
    ax2 = ax.twinx()
    ax2.plot(x, ly, color="#C44E52", marker="o", lw=1.8, clip_on=False, zorder=5,
             label="Y_LOG2_MIR7REL (escala log2)")
    for xi, v in zip(x, ly):
        ax2.text(xi + 0.12, v + 0.12, f"{v:.2f}", fontsize=7.5, color="#C44E52")
    ax2.set_ylabel("log2(miR-7-5p relativo a WT)", color="#C44E52")
    ax2.set_ylim(0, ly.max() * 1.25)  # el cero de ambos ejes coincide (WT = referencia)
    # empates en la respuesta
    for val, grp in y.groupby(y):
        if len(grp) > 1:
            idx = [list(y.index).index(k) for k in grp.index]
            ax.axvspan(min(idx) - 0.45, max(idx) + 0.45, color="#DD8452", alpha=0.15, lw=0)
            ax.text(np.mean(idx), y.max() * 1.17, f"Empate: {' = '.join(grp.index)} ({val:g})",
                    ha="center", fontsize=8, color="#8C4A1F")
            log.info(f"  [respuesta] empate en {list(grp.index)} con valor {val:g}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=8)
    ax.set_title("Respuesta TDMD en escala lineal y logarítmica por variante")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_FIG05_distribucion_respuesta_Y.png", dpi=300)
    plt.close(fig)


def fig06_deltas(raw, fig_dir, log):
    """Mapa de calor de los 16 deltas. Color = delta escalado por fila (Δ / máx|Δ| de esa
    variable), que conserva signo y cero y hace comparables variables con unidades distintas.
    Anotación = delta en unidades nativas, para auditar magnitudes."""
    units = {"RG": "Å", "END_DIST": "Å", "CHI_ANGLE_MEAN": "°", "CHI_ANGLE_SD": "°",
             "PSEUDO_P_SUGAR_MEAN": "°", "PSEUDO_P_SUGAR_SD": "°", "SASA_TOTAL": "Å²", "STACK": "n"}
    rows, labels, rel = [], [], []
    for strand in ["CYR", "MIR"]:
        for b in DELTA_BASES:
            d, iso = f"DLT_{strand}_3D_{b}_GLB", f"{strand}_ISO_3D_{b}_GLB"
            if d not in raw.columns:
                continue
            dv = raw[d].astype(float)
            mx = dv.abs().max()
            rows.append(dv.values)
            rel.append((dv / mx).values if mx > 0 else np.zeros(len(dv)))
            labels.append(f"Δ{strand} {b} ({units[b]})")
            log.info(f"  [delta] {d}: min={dv.min():.3g} max={dv.max():.3g} "
                     f"signo(+/-/0)={int((dv > 0).sum())}/{int((dv < 0).sum())}/{int((dv == 0).sum())}")
    vals, relm = np.array(rows), np.array(rel)
    lim = 1.0
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(relm, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if abs(relm[i, j]) > 0.6 * lim else "black")
    ax.axhline(len(DELTA_BASES) - 0.5, color="black", lw=1.2)
    ax.set_xticks(range(len(raw.index)))
    ax.set_xticklabels(raw.index)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Variante del MRE de Cyrano")
    ax.set_title("Deltas de acoplamiento 3D globales (dúplex menos hebra aislada)")

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Δ escalado por variable (Δ / máx|Δ|)")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_FIG06_auditoria_deltas_acoplamiento.png", dpi=300)
    plt.close(fig)


#==============================================================================
# MÓDULO 10. CARGA (L): ENSAMBLE, ORDEN Y EXPORTACIÓN
# Objetivo: orquestar el flujo completo y escribir los artefactos a disco.
# Inputs: rutas de trabajo.
# Proceso: E -> T1 -> unión por VAR_KEY (validación 1:1) -> T2 -> nulos ->
#          derivadas -> auditoría -> diccionario -> figuras -> exportación.
# Outputs: F1_RAW_DATASET_v01_YYYYMMDD.csv/.xlsx, F1_DATA_DICTIONARY_v01.csv,
#          F1_NA_MAP_v01.csv, F1_ETL_log_v01.txt y seis figuras PNG.
#==============================================================================
def main():
    args = parse_args()
    in_dir, out_dir, fig_dir = args.input_dir, args.output_dir, args.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out_dir)
    log.info(f"F1 ETL {VERSION} | fecha {DATE_TAG} | pandas {pd.__version__}")
    log.info(f"INPUT_DIR  = {in_dir}")

    if not in_dir.is_dir():
        log.error(f"INPUT_DIR no existe o no es una carpeta: {in_dir}")
        sys.exit(1)
    resolved, missing, available = resolve_input_files(in_dir)
    if missing:
        log.error(f"Archivos no encontrados: {missing}")
        log.error(f"Archivos .xlsx presentes en INPUT_DIR: {available}")
        sys.exit(1)
    for k, v in resolved.items():
        log.info(f"  [archivo] {k} -> {v.name}")

    # --- E + T1
    log.info("MÓDULO 2-3 | Extracción y renombrado")
    frames, meta, summary = [], {}, []
    for entry in FILE_REGISTRY:
        f, sh, oe, ctx, niv = entry
        body, labels = read_sheet(resolved[f], sh)
        summary.append(dict(LABEL=f"{f.split('_Result')[0]} | {sh}", OE=oe, N=len(body), P=body.shape[1] - 1))
        log.info(f"  {f}/{sh}: N={len(body)} P={body.shape[1] - 1}")
        df, recs = transform_sheet(body, labels, entry, log)
        frames.append(df)
        for r in recs:
            meta[r["VAR_NAME"]] = r
    summary = pd.DataFrame(summary)
    fig01_dimensions(summary, fig_dir)

    raw = frames[0]
    for df in frames[1:]:
        clash = set(raw.columns) & set(df.columns)
        if clash:
            raise ValueError(f"Colisión entre bloques: {sorted(clash)[:5]}")
        raw = raw.join(df, how="outer", validate="one_to_one")
    raw = raw.reindex(VAR_ORDER)
    if raw.index.isna().any() or len(raw) != 7:
        raise ValueError("La matriz integrada no contiene exactamente las 7 variantes")
    log.info(f"  Matriz unida: {raw.shape[0]} x {raw.shape[1]}")

    # --- T2
    log.info("MÓDULO 4 | Casteo de tipos")
    raw = cast_types(raw, meta)

    # --- nulos
    log.info("MÓDULO 5 | Tipificación de nulos e imputación dual")
    pres = build_presence(raw)
    na_map = classify_na(raw, meta, pres, log)
    fig02_na_map(na_map, fig_dir)
    raw, n_imp = impute_counts(raw, meta, na_map, log)
    flags = region_flags(raw, pres)
    for c in flags.columns:
        ctx, reg = c.replace("FLAG_", "").rsplit("_HAS_", 1)
        meta[c] = dict(VAR_NAME=c, OE="derivada", MOL_CTX=ctx, NIV="FLAG", REG=reg, BASE="HAS_REGION",
                       ROLE="FLAG", DERIVED_FROM=f"{ctx}_1D_LEN_{reg} > 0" if ctx != "DPX"
                       else f"CYR_DPX o MIR_DPX LEN_{reg} > 0")

    # --- derivadas
    log.info("MÓDULO 6 | Variables derivadas")
    der, drecs = derived_variables(raw, log)
    for r in drecs:
        meta[r["VAR_NAME"]] = r

    # --- orden final de columnas
    y_cols = ["Y_MIR7_REL"]
    pred = [c for c in raw.columns if not c.startswith("META") and c not in y_cols]
    meta_cols = [c for c in raw.columns if c.startswith("META")]
    RAW = pd.concat([raw[y_cols], der[["Y_LOG2_MIR7REL"]], flags, raw[pred],
                     der.drop(columns="Y_LOG2_MIR7REL"), raw[meta_cols]], axis=1)
    RAW.index.name = "VAR_KEY"
    na_full = na_map.reindex(columns=RAW.columns, fill_value="OBS")
    na_full.index.name = "VAR_KEY"

    # --- auditoría
    log.info("MÓDULO 7 | Identidades, constantes y duplicados")
    ident, rep = check_identities(RAW, log)
    const, dup = flag_constants_duplicates(RAW)
    log.info(f"  Constantes: {sum(const.values())} | duplicados exactos: {len(dup)}")
    fig04_identities(rep, fig_dir)
    log.info("MÓDULO 9 | Figuras de respuesta y deltas")
    fig05_response(RAW, fig_dir, log)
    fig06_deltas(RAW, fig_dir, log)

    # --- diccionario
    log.info("MÓDULO 8 | Diccionario de datos")
    dic = build_dictionary(RAW, meta, na_full, n_imp, ident, const, dup)
    fig03_constants(dic, fig_dir)

    # --- exportación
    log.info("MÓDULO 10 | Exportación")
    base = f"F1_RAW_DATASET_{VERSION}_{DATE_TAG}"
    RAW.to_csv(out_dir / f"{base}.csv", encoding="utf-8-sig")
    RAW.to_excel(out_dir / f"{base}.xlsx", sheet_name="RAW_DATASET")
    dic.to_csv(out_dir / f"F1_DATA_DICTIONARY_{VERSION}.csv", index=False, encoding="utf-8-sig")
    na_full.to_csv(out_dir / f"F1_NA_MAP_{VERSION}.csv", encoding="utf-8-sig")

    roles = dic.ROLE.value_counts().to_dict()
    log.info(f"RAW_DATASET final: N={RAW.shape[0]} x P={RAW.shape[1]} | roles: {roles}")
    log.info(f"Predictores no constantes y no redundantes (candidatos a Iteración 1): "
             f"{int(((dic.ROLE == 'PREDICTOR') & (dic.FLAG_CONST == 0) & (dic.FLAG_IDENTITY == '') & (dic.DUPLICATE_OF == '')).sum())}")
    log.info(f"Artefactos escritos en {out_dir} y {fig_dir}")


if __name__ == "__main__":
    main()
