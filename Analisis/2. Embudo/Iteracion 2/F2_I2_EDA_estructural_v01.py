# -*- coding: utf-8 -*-
"""
F2_I2_EDA_estructural_v01.py
Fase II (Embudo de Tukey), Iteración 2: EDA estructural y no supervisado.
Tamizado por varianza con criterio estructural (NZV), auditoría del espacio
latente (PCA), diagnóstico de artefactos técnicos por bloque de extracción y
mapeo de módulos de colinealidad con agrupamiento jerárquico en tres cortes.

Ejecución (Windows, Conda tesis_rnas):
    conda activate tesis_rnas
    python F2_I2_EDA_estructural_v01.py                    # modo exploratorio
    python F2_I2_EDA_estructural_v01.py --corte 0.90       # consolidación
"""

#==============================================================================
# MÓDULO 0. IMPORTACIONES, RUTAS Y PARÁMETROS
# Objetivo: centralizar rutas, semilla, cortes de colinealidad y reglas V1, V2, V6.
# Inputs: argumentos opcionales de terminal.
# Proceso: define rutas con pathlib.Path, los tres cortes de |rho| y el mapa de
#          complementariedad Watson-Crick usado para detectar alias deterministas.
# Outputs: constantes globales.
#==============================================================================
import argparse
import itertools
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from scipy.spatial.distance import squareform

VERSION = "v01"
DATE_TAG = datetime.now().strftime("%Y%m%d")
SEED = 20260925

BASE = Path(r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-6\Nuevo_Flujo\4.FASE 2_EDA\Ite_2")
DEFAULT_INPUT_CSV = BASE / "00_DATASET" / "F2_I1_DATASET_FINAL_v01_20260925.csv"
DEFAULT_DICT_CSV = BASE / "00_DATASET" / "F2_I1_DATA_DICTIONARY_v01.csv"
DEFAULT_ALIAS_CSV = BASE / "00_DATASET" / "F2_I1_ALIAS_GROUPS_v01.csv"
DEFAULT_OUTPUT_DIR = BASE / "02_TXT"
DEFAULT_FIGURES_DIR = BASE / "01_FIGURAS"

VAR_ORDER = ["WT", "M1", "M2", "M3", "M4", "M5", "M6"]
Y_COLS = ["Y_MIR7_REL", "Y_LOG2_MIR7REL"]
N_OBS = 7

# Cortes de colinealidad que definen el grano de los módulos (no podan variables)
CUTS = {"CONSERVADOR": 0.95, "INTERMEDIO": 0.90, "PERMISIVO": 0.85}

# V1 y V2: ninguna variable se elimina por varianza; se etiqueta su tipo
WT_KEY = "WT"
STRUCT_PAIRS = {("M3", "M6"): "OVERHANG", ("M4", "M5"): "BULGE_ASIMETRICO"}

# V6: complementariedad Watson-Crick como única equivalencia determinista
WC_COMPLEMENT = {"PCT_A": "PCT_U", "PCT_U": "PCT_A", "PCT_G": "PCT_C", "PCT_C": "PCT_G",
                 "PCT_PUR": "PCT_PIR", "PCT_PIR": "PCT_PUR", "LEN": "LEN"}

N_PERM = 5000          # permutaciones para la banda nula del scree y el diagnóstico de bloques
TOP_LOADINGS = 20      # vectores dibujados en el biplot
MIN_MODULE_SIZE = 5    # tamaño mínimo para considerar un módulo en las figuras de composición


def parse_args():
    p = argparse.ArgumentParser(description="F2 Iteración 2: EDA estructural")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    p.add_argument("--dict-csv", type=Path, default=DEFAULT_DICT_CSV)
    p.add_argument("--alias-csv", type=Path, default=DEFAULT_ALIAS_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--corte", default="EXPLORATORIO",
                   help="EXPLORATORIO (default) o el corte elegido: 0.95, 0.90 o 0.85")
    return p.parse_args()


#==============================================================================
# MÓDULO 1. REGISTRO (LOG)
# Objetivo: trazabilidad de cada decisión en consola y archivo.
# Inputs: OUTPUT_DIR y modo de ejecución.
# Proceso: logger dual; el nombre del archivo incluye el modo para no sobrescribir
#          el log exploratorio al consolidar (hallazgo de la Iteración 1).
# Outputs: F2_I2_log_<MODO>_v01.txt.
#==============================================================================
def setup_logger(out_dir: Path, modo: str) -> logging.Logger:
    log = logging.getLogger("F2_I2")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(out_dir / f"F2_I2_log_{modo}_{VERSION}.txt", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


#==============================================================================
# MÓDULO 2. CARGA Y VALIDACIÓN
# Objetivo: leer el dataset consolidado de la Iteración 1 y su diccionario.
# Inputs: F2_I1_DATASET_FINAL (csv), F2_I1_DATA_DICTIONARY (csv),
#         F2_I1_ALIAS_GROUPS (csv, opcional pero recomendado).
# Proceso: valida las 7 variantes, separa respuestas de predictores, comprueba
#          que cada predictor tenga entrada en el diccionario y separa el
#          subespacio completo (sin nulos) del subespacio con nulos estructurales.
# Outputs: Y, P (predictores), DIC, ALIAS, listas de columnas completas e incompletas.
#==============================================================================
def load_inputs(args, log):
    for p in (args.input_csv, args.dict_csv):
        if not p.exists():
            log.error(f"No se encontró: {p}")
            log.error("Copie el dataset y el diccionario de la Iteración 1 a la carpeta 00_DATASET.")
            sys.exit(1)
    X = pd.read_csv(args.input_csv, index_col=0, encoding="utf-8-sig")
    X.index = X.index.astype(str)
    X = X.reindex(VAR_ORDER)
    if X.index.isna().any() or len(X) != N_OBS:
        raise ValueError("El dataset no contiene exactamente WT y M1 a M6")
    ys = [c for c in Y_COLS if c in X.columns]
    if len(ys) != 2:
        raise ValueError(f"Faltan variables respuesta: {set(Y_COLS) - set(X.columns)}")
    Y = X[ys].astype(float)
    P = X.drop(columns=ys).apply(pd.to_numeric, errors="coerce")

    dic = pd.read_csv(args.dict_csv, encoding="utf-8-sig", dtype=str).fillna("").set_index("VAR_NAME")
    falta = set(P.columns) - set(dic.index)
    if falta:
        raise ValueError(f"Predictores sin entrada en el diccionario: {sorted(falta)[:5]}")
    dic = dic.loc[list(P.columns)]
    dic["BLOQUE"] = dic["MOL_CTX"] + "_" + dic["NIV"]

    alias = pd.DataFrame(columns=["ALIAS_GROUP", "VAR_NAME", "ROLE_IN_GROUP", "RELATION"])
    if args.alias_csv.exists():
        alias = pd.read_csv(args.alias_csv, encoding="utf-8-sig").fillna("")
        alias = alias[alias.VAR_NAME.isin(P.columns)]
        log.info(f"  Grupos de alias heredados: {alias.ALIAS_GROUP.nunique()} "
                 f"({len(alias)} columnas)")
    else:
        log.info("  F2_I1_ALIAS_GROUPS no encontrado: el colapso de alias deterministas se omite")

    completas = [c for c in P.columns if P[c].notna().all()]
    incompletas = [c for c in P.columns if c not in completas]
    log.info(f"  Predictores: {P.shape[1]} | completos: {len(completas)} | "
             f"con nulos estructurales: {len(incompletas)}")
    log.info(f"  Respuestas blindadas: {ys}")
    return Y, P, dic, alias, completas, incompletas


#==============================================================================
# MÓDULO 3. TAMIZADO POR VARIANZA CON CRITERIO ESTRUCTURAL (NZV)
# Objetivo: caracterizar las variables cuasi-constantes sin eliminarlas (V1, V2).
# Inputs: P, DIC.
# Proceso: n_minoría = observaciones no nulas menos la frecuencia del valor modal;
#          identifica qué variantes se apartan del modo; etiqueta WT_SIGNATURE
#          (solo WT discrepa), STRUCT_M3M6 y STRUCT_M4M5 (pares arquitectónicos)
#          y MUTANTE_UNICO. El criterio clásico de razón de frecuencias no
#          discrimina con N = 7, porque toda variable con n_minoría = 1 tiene
#          razón 6 sea cual sea la variante discrepante.
# Outputs: tabla NZV con n_minoría, variantes discrepantes y etiqueta.
#==============================================================================
def nzv_structural(P, dic, log):
    rows = []
    for c in P.columns:
        v = P[c].dropna().round(6)
        vc = v.value_counts()
        modo = vc.index[0]
        dev = [k for k in v.index if v[k] != modo]
        mn = len(dev)
        etiqueta, motivo = "VARIABLE", ""
        if mn == 1:
            if dev == [WT_KEY]:
                etiqueta = "WT_SIGNATURE"
                motivo = "Solo WT difiere del valor modal (V1): conservada con advertencia"
            else:
                etiqueta = "MUTANTE_UNICO"
                motivo = f"Solo {dev[0]} difiere del valor modal (V2): conservada"
        elif mn == 2:
            clave = tuple(sorted(dev, key=VAR_ORDER.index))
            if clave in STRUCT_PAIRS:
                etiqueta = f"STRUCT_{STRUCT_PAIRS[clave]}"
                motivo = f"Par arquitectónico {clave[0]} y {clave[1]} (V2): codifica una diferencia estructural"
            else:
                etiqueta = "PAR_NO_ESTRUCTURAL"
                motivo = f"Discrepan {dev[0]} y {dev[1]} sin correspondencia arquitectónica"
        rows.append(dict(VAR_NAME=c, BLOQUE=dic.at[c, "BLOQUE"], REG=dic.at[c, "REG"],
                         N_OBS=int(len(v)), N_MINORIA=int(mn),
                         VARIANTES_DISCREPANTES=",".join(dev), N_UNICOS=int(vc.size),
                         RAZON_FRECUENCIA=round(float(vc.iloc[0] / vc.iloc[1]), 3) if vc.size > 1 else np.inf,
                         NZV_ETIQUETA=etiqueta, NZV_DECISION="RETENIDA", MOTIVO=motivo))
    nzv = pd.DataFrame(rows)
    log.info("  n_minoría: " + ", ".join(f"{k}={v}" for k, v in
                                          sorted(nzv.N_MINORIA.value_counts().items())))
    for et, g in nzv[nzv.NZV_ETIQUETA != "VARIABLE"].groupby("NZV_ETIQUETA"):
        log.info(f"  [{et}] {len(g)} variables")
    for c in nzv[nzv.NZV_ETIQUETA == "WT_SIGNATURE"].VAR_NAME:
        log.info(f"  [WT_SIGNATURE] {c}")
    log.info(f"  Variables eliminadas por varianza: 0 (V1 y V2: se conservan todas)")
    return nzv


#==============================================================================
# MÓDULO 4. ALIAS DETERMINISTAS (V6)
# Objetivo: distinguir equivalencias impuestas por definición de coincidencias
#           numéricas debidas al tamaño muestral.
# Inputs: P, DIC, tabla de alias de la Iteración 1.
# Proceso: un alias es determinista si comparte región, pertenece a la hebra
#          complementaria y su variable base es el complemento Watson-Crick de la
#          del representante (A/U, G/C, PUR/PIR, LEN/LEN). Cualquier otra igualdad
#          se trata como coincidencia numérica y conserva su identidad propia.
#          El colapso afecta solo a la matriz de agrupamiento; el dataset exportado
#          conserva todas las columnas.
# Outputs: tabla de alias clasificada y lista de columnas colapsadas.
#==============================================================================
def classify_alias(P, dic, alias, log):
    if alias.empty:
        return pd.DataFrame(columns=["ALIAS_GROUP", "REPRESENTANTE", "VAR_NAME", "TIPO", "COLAPSADA"]), []
    rows = []
    for g, sub in alias.groupby("ALIAS_GROUP"):
        rep_row = sub[sub.ROLE_IN_GROUP == "representante"]
        if rep_row.empty:
            continue
        rep = rep_row.VAR_NAME.iloc[0]
        for m in sub[sub.ROLE_IN_GROUP == "alias"].VAR_NAME:
            b_rep, b_ali = dic.at[rep, "BASE"], dic.at[m, "BASE"]
            hebra_rep = dic.at[rep, "MOL_CTX"].split("_")[0]
            hebra_ali = dic.at[m, "MOL_CTX"].split("_")[0]
            det = (dic.at[m, "REG"] == dic.at[rep, "REG"] and hebra_ali != hebra_rep
                   and WC_COMPLEMENT.get(b_rep) == b_ali)
            rows.append(dict(ALIAS_GROUP=g, REPRESENTANTE=rep, VAR_NAME=m,
                             TIPO="DETERMINISTA_WC" if det else "COINCIDENCIA_NUMERICA",
                             COLAPSADA=int(det)))
    tab = pd.DataFrame(rows)
    coll = tab[tab.COLAPSADA == 1].VAR_NAME.tolist() if not tab.empty else []
    log.info(f"  Alias deterministas (complementariedad WC) colapsados para el agrupamiento: {len(coll)}")
    for c in coll:
        log.info(f"    {c} == {tab.loc[tab.VAR_NAME == c, 'REPRESENTANTE'].iloc[0]}")
    n_coinc = int((tab.TIPO == "COINCIDENCIA_NUMERICA").sum()) if not tab.empty else 0
    log.info(f"  Coincidencias numéricas conservadas como variables distintas: {n_coinc}")
    return tab, coll


#==============================================================================
# MÓDULO 5. CORRELACIONES Y DISTRIBUCIÓN NULA EXACTA
# Objetivo: cuantificar la redundancia y separarla del ruido esperado con N = 7.
# Inputs: P, columnas colapsadas.
# Proceso: Spearman (referencia) y Pearson (diagnóstico) por pares, con mínimo de
#          5 pares completos. La distribución nula exacta de Spearman se obtiene
#          enumerando las 5040 permutaciones de rangos, lo que da P(|rho| >= t)
#          sin aproximaciones asintóticas.
# Outputs: matrices de correlación, tabla de la distribución nula y comparación
#          observado vs. esperado por corte.
#==============================================================================
def exact_null_spearman(n=N_OBS):
    base = np.arange(n)
    perms = np.array(list(itertools.permutations(base)))
    d2 = ((perms - base) ** 2).sum(axis=1)
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def correlations(P, coll, log):
    cols = [c for c in P.columns if c not in coll]
    R_s = P[cols].rank().corr(method="pearson", min_periods=5)   # Spearman sobre pares completos
    R_p = P[cols].corr(method="pearson", min_periods=5)
    iu = np.triu_indices(len(cols), 1)
    s, p = np.abs(R_s.values[iu]), np.abs(R_p.values[iu])
    n_pares = int(np.sum(~np.isnan(s)))
    nulo = np.abs(exact_null_spearman())
    rows = []
    for lvl, t in CUTS.items():
        pr = float(np.mean(nulo >= t - 1e-9))
        obs = int(np.nansum(s >= t - 1e-9))
        rows.append(dict(NIVEL=lvl, CORTE=t, P_NULO_EXACTO=round(pr, 5),
                         PARES_ESPERADOS=round(pr * n_pares, 1), PARES_OBS_SPEARMAN=obs,
                         PARES_OBS_PEARSON=int(np.nansum(p >= t - 1e-9)),
                         RAZON_SENAL_RUIDO=round(obs / (pr * n_pares), 2) if pr > 0 else np.inf))
    comp = pd.DataFrame(rows)
    log.info(f"  Pares evaluados: {n_pares} (de {len(iu[0])}; {len(iu[0]) - n_pares} sin soporte suficiente)")
    for _, r in comp.iterrows():
        log.info(f"  [|rho| >= {r.CORTE}] observados={r.PARES_OBS_SPEARMAN} "
                 f"esperados por azar={r.PARES_ESPERADOS} razón={r.RAZON_SENAL_RUIDO} "
                 f"| Pearson={r.PARES_OBS_PEARSON}")
    return R_s, R_p, comp, s, p, nulo


#==============================================================================
# MÓDULO 6. MÓDULOS DE COLINEALIDAD (AGRUPAMIENTO JERÁRQUICO)
# Objetivo: hacer emerger los bloques densos de redundancia en tres granos.
# Inputs: matriz de Spearman.
# Proceso: distancia 1 - |rho| con enlace promedio (no impone forma a los módulos)
#          y corte a 1 - t para cada nivel. Esta iteración no elimina variables por
#          colinealidad (V3): los módulos se exportan como estructura para la
#          selección de representantes de la Iteración 4.
# Outputs: enlace, asignación de módulos por corte y resumen por módulo.
#==============================================================================
def build_modules(R_s, dic, log):
    cols = list(R_s.columns)
    D = 1 - np.abs(R_s.fillna(0).values)
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), method="average")
    asign = pd.DataFrame(index=cols)
    resumen = []
    for lvl, t in CUTS.items():
        cl = fcluster(Z, 1 - t, criterion="distance")
        col = f"MODULO_{int(t * 100)}"
        asign[col] = [f"M{int(t * 100)}_{k:03d}" for k in cl]
        vc = pd.Series(cl).value_counts()
        log.info(f"  [{lvl} | corte {t}] módulos={cl.max()} singletons={int((vc == 1).sum())} "
                 f"mayor={int(vc.max())} con>={MIN_MODULE_SIZE}: {int((vc >= MIN_MODULE_SIZE).sum())}")
        for m, idx in asign.groupby(col).groups.items():
            miembros = list(idx)
            bloques = sorted({dic.at[c, "BLOQUE"] for c in miembros})
            regs = sorted({dic.at[c, "REG"] for c in miembros})
            sub = np.abs(R_s.loc[miembros, miembros].values)
            intra = sub[np.triu_indices(len(miembros), 1)]
            resumen.append(dict(NIVEL=lvl, CORTE=t, MODULO=m, TAMANO=len(miembros),
                                N_BLOQUES=len(bloques), BLOQUES=";".join(bloques),
                                REGIONES=";".join(regs),
                                RHO_INTRA_MIN=round(float(np.nanmin(intra)), 3) if len(intra) else np.nan,
                                MIEMBROS=";".join(miembros)))
    return Z, asign, pd.DataFrame(resumen)


#==============================================================================
# MÓDULO 7. ESPACIO LATENTE (PCA) Y SIGNIFICANCIA POR PERMUTACIÓN
# Objetivo: describir la geometría global de las 7 variantes sin usar la respuesta.
# Inputs: P, columnas completas.
# Proceso: PCA por descomposición en valores singulares sobre Z-scores de las
#          variables completas (V5: sin imputación). Con N = 7 solo existen 6
#          componentes no triviales, por lo que la retención no se decide con el
#          criterio de Kaiser sino contra una banda nula obtenida permutando de
#          forma independiente cada columna (Horn, 1965; Peres-Neto et al., 2005).
#          Se repite el PCA sobre rangos estandarizados como control de robustez
#          frente a valores extremos.
# Outputs: puntuaciones, cargas, varianza explicada, banda nula y control por rangos.
#==============================================================================
def run_pca(P, completas, log, seed=SEED):
    C = P[completas]
    sd = C.std(ddof=1)
    if (sd == 0).any():
        cero = list(sd.index[sd == 0])
        log.warning(f"  Variables completas con varianza nula excluidas del PCA: {cero}")
        C = C.drop(columns=cero)
    def pca(M):
        Zs = (M - M.mean()) / M.std(ddof=1)
        u, s, vt = np.linalg.svd(Zs.values, full_matrices=False)
        k = min(N_OBS - 1, len(s))
        return u[:, :k] * s[:k], vt[:k].T, (s ** 2 / np.sum(s ** 2))[:k]
    sc, ld, ev = pca(C)
    sc_r, _, ev_r = pca(C.rank())
    rng = np.random.default_rng(seed)
    nulo = np.empty((N_PERM, len(ev)))
    V = C.values
    for i in range(N_PERM):
        Pm = np.column_stack([rng.permutation(V[:, j]) for j in range(V.shape[1])])
        Pm = (Pm - Pm.mean(0)) / Pm.std(0, ddof=1)
        s = np.linalg.svd(Pm, compute_uv=False)
        nulo[i] = (s ** 2 / np.sum(s ** 2))[:len(ev)]
    pvals = [(np.sum(nulo[:, k] >= ev[k]) + 1) / (N_PERM + 1) for k in range(len(ev))]
    idx = [f"PC{k + 1}" for k in range(len(ev))]
    var = pd.DataFrame(dict(COMPONENTE=idx, VAR_EXPLICADA=np.round(ev, 4),
                            ACUMULADA=np.round(np.cumsum(ev), 4),
                            NULO_P50=np.round(np.percentile(nulo, 50, axis=0), 4),
                            NULO_P95=np.round(np.percentile(nulo, 95, axis=0), 4),
                            P_PERMUTACION=np.round(pvals, 4),
                            VAR_EXPLICADA_RANGOS=np.round(ev_r, 4)))
    scores = pd.DataFrame(sc, index=C.index, columns=idx)
    scores_r = pd.DataFrame(sc_r, index=C.index, columns=idx)
    loadings = pd.DataFrame(ld, index=C.columns, columns=idx)
    for _, r in var.iterrows():
        marca = "significativo" if r.P_PERMUTACION < 0.05 else "no distinguible del nulo"
        log.info(f"  {r.COMPONENTE}: {r.VAR_EXPLICADA:.1%} (acum {r.ACUMULADA:.1%}) "
                 f"| nulo p95={r.NULO_P95:.1%} | p={r.P_PERMUTACION:.4f} -> {marca}")
    for k in ("PC1", "PC2"):
        rho = scores[k].rank().corr(scores_r[k].rank())
        log.info(f"  Control por rangos {k}: correlación de puntuaciones = {rho:.3f}")
        log.info(f"  Puntuaciones {k}: " + ", ".join(f"{i}={v:.2f}" for i, v in scores[k].items()))
    return scores, scores_r, loadings, var, C.columns.tolist()


#==============================================================================
# MÓDULO 8. DIAGNÓSTICO DE ARTEFACTOS TÉCNICOS POR BLOQUE
# Objetivo: verificar que ningún componente esté dominado por un bloque de extracción.
# Inputs: cargas del PCA y diccionario.
# Proceso: los grupos técnicos agrupan columnas, no filas (las 7 variantes pasaron
#          por el mismo procedimiento), de modo que un efecto de lote se
#          manifestaría como un componente monopolizado por un bloque. Se compara
#          la suma de cargas al cuadrado de cada bloque con su proporción esperada
#          por tamaño, y se obtiene un valor p permutando las etiquetas de bloque.
#          Como se evalúan decenas de combinaciones de bloque y componente, los
#          valores p se corrigen por tasa de falso descubrimiento (Benjamini &
#          Hochberg, 1995); solo se advierte sobre los que superan esa corrección.
# Outputs: tabla de contribución observada, esperada y p por bloque y componente.
#==============================================================================
def benjamini_hochberg(p):
    """Valores p corregidos por tasa de falso descubrimiento (Benjamini & Hochberg, 1995)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    orden = np.argsort(p)
    ajust = np.minimum.accumulate((p[orden] * m / np.arange(m, 0, -1))[::-1])[::-1]
    out = np.empty(m)
    out[orden] = np.clip(ajust, 0, 1)
    return out


def batch_diagnostic(loadings, dic, log, seed=SEED):
    rng = np.random.default_rng(seed)
    bl = np.array([dic.at[c, "BLOQUE"] for c in loadings.index])
    niv = np.array([dic.at[c, "NIV"] for c in loadings.index])
    rows = []
    for etiqueta, grupos in (("BLOQUE", bl), ("NIVEL", niv)):
        for pc in ["PC1", "PC2", "PC3"]:
            if pc not in loadings:
                continue
            w = loadings[pc].values ** 2
            for g in np.unique(grupos):
                m = grupos == g
                obs = w[m].sum()
                esp = m.mean()
                nulo = np.array([w[rng.permutation(len(w))][:m.sum()].sum() for _ in range(N_PERM)])
                pv = (np.sum(nulo >= obs) + 1) / (N_PERM + 1)
                rows.append(dict(AGRUPACION=etiqueta, GRUPO=g, COMPONENTE=pc, N_VARS=int(m.sum()),
                                 CONTRIB_OBS=round(float(obs), 4), CONTRIB_ESPERADA=round(float(esp), 4),
                                 RAZON=round(float(obs / esp), 2), P_PERMUTACION=round(float(pv), 4)))
    tab = pd.DataFrame(rows)
    tab["P_FDR"] = np.nan
    for etiqueta, g in tab.groupby("AGRUPACION"):
        tab.loc[g.index, "P_FDR"] = benjamini_hochberg(g.P_PERMUTACION.values).round(4)
    sig = tab[(tab.P_FDR < 0.05) & (tab.RAZON > 1)]
    if sig.empty:
        log.info("  Ningún bloque ni nivel contribuye a PC1 a PC3 por encima de su tamaño")
    else:
        log.info("  Concentraciones por encima del tamaño del grupo (FDR < 0.05). Una concentración "
                 "indica un artefacto solo si el grupo corresponde a una herramienta de extracción "
                 "distinta; si agrupa una hebra o un nivel con biología propia, es señal, no sesgo.")
        for _, r in sig.iterrows():
            log.info(f"  [concentración] {r.AGRUPACION}={r.GRUPO} en {r.COMPONENTE}: "
                        f"contribución {r.CONTRIB_OBS:.1%} frente a {r.CONTRIB_ESPERADA:.1%} "
                        f"esperado (razón {r.RAZON}, p={r.P_PERMUTACION}, FDR={r.P_FDR})")
    return tab


#==============================================================================
# MÓDULO 9. FIGURAS (UN GRÁFICO POR ARCHIVO, 300 DPI)
# Objetivo: visual tracking de varianza, espacio latente, artefactos y módulos.
# Inputs: resultados de los módulos 3 a 8.
# Proceso: catorce rutinas independientes.
# Outputs: F2_I2_FIG01 a F2_I2_FIG14 en FIGURES_DIR.
#==============================================================================
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10})
COL = {"WT_SIGNATURE": "#C44E52", "MUTANTE_UNICO": "#DD8452", "STRUCT_OVERHANG": "#55A868",
       "STRUCT_BULGE_ASIMETRICO": "#4C72B0", "PAR_NO_ESTRUCTURAL": "#8172B3", "VARIABLE": "#B0B0B0"}
CUT_COL = {"CONSERVADOR": "#C44E52", "INTERMEDIO": "#DD8452", "PERMISIVO": "#55A868"}


def save(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=300)
    plt.close(fig)


def fig01_nzv(nzv, fig_dir):
    sub = nzv[nzv.N_MINORIA <= 2].copy()
    sub["DEV"] = sub.VARIANTES_DISCREPANTES
    t = sub.groupby(["DEV", "NZV_ETIQUETA"]).size().unstack(fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bottom = np.zeros(len(t))
    for et in t.columns:
        ax.bar(t.index, t[et], bottom=bottom, color=COL.get(et, "#B0B0B0"), label=et)
        bottom += t[et].values
    for i, v in enumerate(t.sum(axis=1)):
        ax.text(i, v + 0.4, str(int(v)), ha="center", fontsize=9)
    ax.set_xlabel("Variantes que se apartan del valor modal")
    ax.set_ylabel("Número de variables")
    ax.set_title("Variables cuasi-constantes según la variante discrepante")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I2_FIG01_nzv_variante_discrepante.png")


def fig02_nzv_bloque(nzv, fig_dir):
    sub = nzv[nzv.N_MINORIA <= 2]
    t = sub.groupby(["BLOQUE", "NZV_ETIQUETA"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(len(t))
    for et in t.columns:
        ax.bar(t.index, t[et], bottom=bottom, color=COL.get(et, "#B0B0B0"), label=et)
        bottom += t[et].values
    ax.set_ylabel("Número de variables")
    ax.set_xlabel("Bloque (molécula en contexto y nivel)")
    ax.set_title("Origen de las variables cuasi-constantes por bloque")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I2_FIG02_nzv_por_bloque.png")


def fig03_scree(var, fig_dir):
    x = np.arange(1, len(var) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x, var.VAR_EXPLICADA * 100, color="#4C72B0", label="Varianza explicada")
    ax.plot(x, var.NULO_P95 * 100, color="#C44E52", marker="o", lw=1.6,
            label="Percentil 95 de la distribución nula")
    ax.plot(x, var.NULO_P50 * 100, color="#C44E52", ls=":", lw=1.2, label="Mediana nula")
    for i, r in var.iterrows():
        ax.text(i + 1, r.VAR_EXPLICADA * 100 + 1, f"p={r.P_PERMUTACION:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(var.COMPONENTE)
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_xlabel("Componente principal")
    ax.set_title("Varianza explicada frente a la distribución nula por permutación")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I2_FIG03_scree_varianza_explicada.png")


def _scores_plot(scores, Y, var, a, b, fig_dir, fname, titulo):
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    y = Y["Y_LOG2_MIR7REL"]
    sc = ax.scatter(scores[a], scores[b], c=y, cmap="viridis", s=140, edgecolor="k", zorder=3)
    for k in scores.index:
        ax.annotate(k, (scores.at[k, a], scores.at[k, b]), xytext=(7, 6),
                    textcoords="offset points", fontsize=9)
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    va = var.set_index("COMPONENTE").VAR_EXPLICADA
    ax.set_xlabel(f"{a} ({va[a]:.1%} de la varianza)")
    ax.set_ylabel(f"{b} ({va[b]:.1%} de la varianza)")
    ax.set_title(titulo)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label("log2(miR-7-5p relativo a WT)")
    save(fig, fig_dir, fname)


def fig04_scores12(scores, Y, var, fig_dir):
    _scores_plot(scores, Y, var, "PC1", "PC2", fig_dir, "F2_I2_FIG04_pca_scores_pc1_pc2.png",
                 "Variantes en el plano PC1-PC2 del espacio estructural")


def fig05_scores13(scores, Y, var, fig_dir):
    _scores_plot(scores, Y, var, "PC1", "PC3", fig_dir, "F2_I2_FIG05_pca_scores_pc1_pc3.png",
                 "Variantes en el plano PC1-PC3 del espacio estructural")


def fig06_contrib(batch, fig_dir):
    b = batch[(batch.AGRUPACION == "BLOQUE") & (batch.COMPONENTE.isin(["PC1", "PC2", "PC3"]))]
    piv = b.pivot(index="GRUPO", columns="COMPONENTE", values="CONTRIB_OBS").fillna(0) * 100
    esp = b.drop_duplicates("GRUPO").set_index("GRUPO").CONTRIB_ESPERADA * 100
    piv = piv.loc[esp.sort_values(ascending=False).index]
    x = np.arange(len(piv))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for i, pc in enumerate(["PC1", "PC2", "PC3"]):
        if pc in piv:
            ax.bar(x + (i - 1) * w, piv[pc], w, label=pc)
    ax.plot(x, esp.loc[piv.index], "k_", ms=22, mew=2, label="Esperado por tamaño del bloque")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=45, ha="right")
    ax.set_ylabel("Contribución al componente (% de la suma de cargas al cuadrado)")
    ax.set_xlabel("Bloque de extracción")
    ax.set_title("Contribución de cada bloque a los componentes principales")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I2_FIG06_pca_cargas_por_bloque.png")


def fig07_biplot(scores, loadings, dic, var, fig_dir):
    """Biplot con doble juego de ejes: las variantes en la escala de puntuaciones
    (ejes inferior e izquierdo) y las cargas en su propia escala (ejes superior y
    derecho), de modo que ambas magnitudes se leen sin reescalados arbitrarios."""
    L = loadings[["PC1", "PC2"]]
    norma = np.sqrt((L ** 2).sum(axis=1)).sort_values(ascending=False)
    top = norma.index[:TOP_LOADINGS]
    fig, ax = plt.subplots(figsize=(10.5, 8))
    lim_s = np.abs(scores[["PC1", "PC2"]].values).max() * 1.25
    ax.set_xlim(-lim_s, lim_s)
    ax.set_ylim(-lim_s, lim_s)
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.scatter(scores.PC1, scores.PC2, s=170, color="black", zorder=5)
    for k in scores.index:
        ax.annotate(k, (scores.at[k, "PC1"], scores.at[k, "PC2"]), color="white",
                    ha="center", va="center", fontsize=8.5, zorder=6)
    va = var.set_index("COMPONENTE").VAR_EXPLICADA
    ax.set_xlabel(f"PC1 ({va['PC1']:.1%}): puntuaciones de las variantes")
    ax.set_ylabel(f"PC2 ({va['PC2']:.1%}): puntuaciones de las variantes")

    ax2 = ax.twinx().twiny()
    lim_l = float(np.abs(L.loc[top].values).max()) * 1.25
    ax2.set_xlim(-lim_l, lim_l)
    ax2.set_ylim(-lim_l, lim_l)
    bloques = sorted({dic.at[c, "BLOQUE"] for c in top})
    cmap = plt.get_cmap("tab10")
    colb = {b: cmap(i % 10) for i, b in enumerate(bloques)}
    for c in top:
        x, y = L.at[c, "PC1"], L.at[c, "PC2"]
        col = colb[dic.at[c, "BLOQUE"]]
        ax2.annotate("", xy=(x, y), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=0.9))
        ha = "left" if x >= 0 else "right"
        ax2.text(x * 1.03, y * 1.03, c, fontsize=6, color=col, ha=ha,
                 va="bottom" if y >= 0 else "top", clip_on=True)
    ax2.set_xlabel("Carga en PC1", fontsize=9)
    ax2.set_ylabel("Carga en PC2", fontsize=9)
    ax.set_title(f"Biplot PC1-PC2 con las {TOP_LOADINGS} cargas de mayor magnitud", pad=32)
    ax.legend(handles=[Patch(color=colb[b], label=b) for b in bloques],
              loc="upper left", bbox_to_anchor=(1.09, 1), frameon=False, fontsize=7.5)
    save(fig, fig_dir, "F2_I2_FIG07_biplot_pc1_pc2.png")


def fig08_robustez(scores, scores_r, fig_dir):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for pc, c in (("PC1", "#4C72B0"), ("PC2", "#C44E52")):
        ax.scatter(scores[pc], scores_r[pc], color=c, s=90, label=pc)
        for k in scores.index:
            ax.annotate(k, (scores.at[k, pc], scores_r.at[k, pc]), xytext=(5, 4),
                        textcoords="offset points", fontsize=7.5, color=c)
    lim = max(np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max())
    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Puntuación con variables estandarizadas (Z-score)")
    ax.set_ylabel("Puntuación con rangos estandarizados")
    ax.set_title("Control de robustez del PCA frente a valores extremos")
    ax.legend(frameon=False)
    save(fig, fig_dir, "F2_I2_FIG08_pca_sensibilidad_zscore_rango.png")


def fig09_heatmap(R_s, Z, dic, fig_dir):
    orden = leaves_list(Z)
    cols = [R_s.columns[i] for i in orden]
    M = R_s.loc[cols, cols].values
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{len(cols)} predictores ordenados por agrupamiento jerárquico")
    ax.set_ylabel("Predictores (mismo orden)")
    ax.set_title("Matriz de correlación de Spearman reordenada por módulos")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Coeficiente de Spearman")
    save(fig, fig_dir, "F2_I2_FIG09_heatmap_spearman_ordenado.png")


def fig10_dendro(Z, fig_dir):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    dendrogram(Z, no_labels=True, color_threshold=1 - CUTS["INTERMEDIO"], ax=ax,
               above_threshold_color="#B0B0B0")
    for lvl, t in CUTS.items():
        ax.axhline(1 - t, color=CUT_COL[lvl], ls="--", lw=1.2, label=f"{lvl} (|rho| >= {t})")
    ax.set_ylabel("Distancia (1 - |rho| de Spearman)")
    ax.set_xlabel("Predictores")
    ax.set_title("Dendrograma de redundancia con los tres cortes evaluados")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I2_FIG10_dendrograma_modulos.png")


def fig11_null(s, nulo, comp, fig_dir):
    obs = s[~np.isnan(s)]
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(obs, bins=bins, density=True, color="#4C72B0", alpha=0.85, label="Pares observados")
    ax.hist(np.abs(nulo), bins=bins, density=True, histtype="step", color="#C44E52", lw=1.8,
            label="Distribución nula exacta (n = 7)")
    for _, r in comp.iterrows():
        ax.axvline(r.CORTE, color=CUT_COL[r.NIVEL], ls="--", lw=1.1)
        ax.text(r.CORTE, ax.get_ylim()[1] * 0.92, f"{r.CORTE}\nrazón {r.RAZON_SENAL_RUIDO}",
                ha="center", fontsize=7.5, color=CUT_COL[r.NIVEL])
    ax.set_xlabel("|rho| de Spearman")
    ax.set_ylabel("Densidad")
    ax.set_title("Correlaciones observadas frente a la distribución nula exacta")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I2_FIG11_distribucion_correlaciones_vs_nulo.png")


def fig12_composicion(resumen, asign, dic, fig_dir):
    nivel = "INTERMEDIO"
    t = CUTS[nivel]
    r = resumen[(resumen.NIVEL == nivel) & (resumen.TAMANO >= MIN_MODULE_SIZE)] \
        .sort_values("TAMANO", ascending=False).head(12)
    col = f"MODULO_{int(t * 100)}"
    bloques = sorted({dic.at[c, "BLOQUE"] for c in asign.index})
    cmap = plt.get_cmap("tab20")
    colb = {b: cmap(i % 20) for i, b in enumerate(bloques)}
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for i, (_, row) in enumerate(r.iterrows()):
        miembros = row.MIEMBROS.split(";")
        cnt = pd.Series([dic.at[c, "BLOQUE"] for c in miembros]).value_counts()
        left = 0
        for b, n in cnt.items():
            ax.barh(i, n, left=left, color=colb[b])
            left += n
        ax.text(left + 0.2, i, f"{row.MODULO} (n={row.TAMANO}, {row.N_BLOQUES} bloques)",
                va="center", fontsize=7.5)
    ax.set_yticks([])
    ax.set_xlabel("Número de variables en el módulo")
    ax.set_title(f"Composición por bloque de los módulos mayores (corte |rho| >= {t})")
    ax.set_xlim(0, r.TAMANO.max() * 1.9)
    ax.legend(handles=[Patch(color=colb[b], label=b) for b in bloques],
              loc="lower right", frameon=False, fontsize=7)
    save(fig, fig_dir, "F2_I2_FIG12_composicion_modulos_por_bloque.png")


def fig13_sensibilidad(resumen, fig_dir):
    r = resumen.groupby(["NIVEL", "CORTE"]).agg(
        modulos=("MODULO", "size"), singletons=("TAMANO", lambda s: int((s == 1).sum())),
        mayor=("TAMANO", "max"), grandes=("TAMANO", lambda s: int((s >= MIN_MODULE_SIZE).sum()))
    ).reset_index().sort_values("CORTE", ascending=False)
    x = np.arange(len(r))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, (k, lab) in enumerate([("modulos", "Módulos"), ("singletons", "Singletons"),
                                  ("grandes", f"Módulos con >= {MIN_MODULE_SIZE}"), ("mayor", "Tamaño del mayor")]):
        ax.bar(x + (i - 1.5) * w, r[k], w, label=lab)
        for xi, v in zip(x + (i - 1.5) * w, r[k]):
            ax.text(xi, v + 1, str(int(v)), ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n|rho| >= {c}" for n, c in zip(r.NIVEL, r.CORTE)])
    ax.set_ylabel("Número de módulos o variables")
    ax.set_title("Sensibilidad de la modularidad al corte de colinealidad")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I2_FIG13_sensibilidad_modularidad.png")


def fig14_pearson(s, p, fig_dir):
    m = ~np.isnan(s) & ~np.isnan(p)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(s[m], p[m], s=3, alpha=0.18, color="#4C72B0", edgecolors="none")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.9, ls=":")
    for lvl, t in CUTS.items():
        ax.axvline(t, color=CUT_COL[lvl], ls="--", lw=0.9)
        ax.axhline(t, color=CUT_COL[lvl], ls="--", lw=0.9, alpha=0.5)
    disc = int(np.sum((p[m] >= 0.95) & (s[m] < 0.95)))
    ax.set_xlabel("|rho| de Spearman (referencia)")
    ax.set_ylabel("|r| de Pearson (diagnóstico)")
    ax.set_title(f"Pearson frente a Spearman por par de variables\n"
                 f"{disc} pares con |r| >= 0.95 y |rho| < 0.95 (apalancamiento por valores extremos)")
    save(fig, fig_dir, "F2_I2_FIG14_pearson_vs_spearman.png")


#==============================================================================
# MÓDULO 10. DICCIONARIO ACTUALIZADO
# Objetivo: extender el linaje con el resultado estructural de la Iteración 2.
# Inputs: diccionario de la Iteración 1, NZV, alias, módulos y cargas.
# Proceso: añade NZV_N_MINORIA_I2, NZV_ETIQUETA_I2, NZV_DECISION_I2, WT_SIGNATURE,
#          ALIAS_TIPO_I2, ALIAS_COLAPSADA_I2, MODULO_95/90/85 y PC1 a PC3.
# Outputs: DataFrame F2_I2_DATA_DICTIONARY.
#==============================================================================
def update_dictionary(dic, nzv, alias_tab, asign, loadings, coll):
    d = dic.copy()
    n = nzv.set_index("VAR_NAME")
    d["NZV_N_MINORIA_I2"] = d.index.map(n.N_MINORIA).astype("Int64")
    d["NZV_VARIANTES_DISC_I2"] = d.index.map(n.VARIANTES_DISCREPANTES).fillna("")
    d["NZV_ETIQUETA_I2"] = d.index.map(n.NZV_ETIQUETA).fillna("")
    d["NZV_DECISION_I2"] = d.index.map(n.NZV_DECISION).fillna("")
    d["WT_SIGNATURE"] = (d["NZV_ETIQUETA_I2"] == "WT_SIGNATURE").astype(int)
    d["ALIAS_TIPO_I2"] = ""
    d["ALIAS_COLAPSADA_I2"] = 0
    if not alias_tab.empty:
        a = alias_tab.set_index("VAR_NAME")
        d.loc[a.index, "ALIAS_TIPO_I2"] = a.TIPO
        d.loc[a.index, "ALIAS_COLAPSADA_I2"] = a.COLAPSADA
    for col in asign.columns:
        d[col] = d.index.map(asign[col]).fillna("")
    for c in coll:
        if c in d.index:
            d.at[c, "NOTES"] = (d.at[c, "NOTES"] + " | " if d.at[c, "NOTES"] else "") + \
                "V6: alias determinista por complementariedad WC; colapsada solo para el agrupamiento"
    for pc in ["PC1", "PC2", "PC3"]:
        if pc in loadings:
            d[f"CARGA_{pc}"] = d.index.map(loadings[pc]).astype(float).round(4)
    for c in d.index[d["WT_SIGNATURE"] == 1]:
        d.at[c, "NOTES"] = (d.at[c, "NOTES"] + " | " if d.at[c, "NOTES"] else "") + \
            "WT_SIGNATURE (V1): WT es la única variante discrepante; toda asociación con Y " \
            "puede reflejar la separación WT/mutantes y no un mecanismo transferible"
    return d.reset_index().rename(columns={"index": "VAR_NAME"})


#==============================================================================
# MÓDULO 11. ORQUESTACIÓN Y EXPORTACIÓN
# Objetivo: ejecutar el flujo y escribir los artefactos.
# Inputs: rutas y corte (--corte).
# Proceso: carga -> NZV -> alias -> correlaciones -> módulos -> PCA -> artefactos
#          técnicos -> diccionario -> figuras -> exportación.
# Outputs: dataset, matrices, módulos, PCA, diccionario, log y catorce figuras.
#          El dataset no cambia de dimensión: V1 y V2 conservan las 232 variables
#          y V3 aplaza la reducción por módulos a la Iteración 4. Lo que el corte
#          selecciona es la asignación de módulos que hereda la Iteración 3.
#==============================================================================
def main():
    args = parse_args()
    modo = "EXPLORATORIO" if args.corte == "EXPLORATORIO" else f"CORTE{args.corte}"
    out, figd = args.output_dir, args.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out, modo)
    log.info(f"F2 Iteración 2 {VERSION} | modo {modo} | fecha {DATE_TAG} | "
             f"pandas {pd.__version__} | semilla {SEED}")

    log.info("MÓDULO 2 | Carga y validación")
    Y, P, dic, alias, completas, incompletas = load_inputs(args, log)

    log.info("MÓDULO 3 | Tamizado por varianza con criterio estructural")
    nzv = nzv_structural(P, dic, log)

    log.info("MÓDULO 4 | Clasificación de alias (V6)")
    alias_tab, coll = classify_alias(P, dic, alias, log)

    log.info("MÓDULO 5 | Correlaciones y distribución nula exacta")
    R_s, R_p, comp, s_abs, p_abs, nulo = correlations(P, coll, log)

    log.info("MÓDULO 6 | Módulos de colinealidad")
    Z, asign, resumen = build_modules(R_s, dic, log)

    log.info("MÓDULO 7 | Espacio latente (PCA)")
    scores, scores_r, loadings, var, usadas = run_pca(P, completas, log)

    log.info("MÓDULO 8 | Diagnóstico de artefactos técnicos")
    batch = batch_diagnostic(loadings, dic, log)

    log.info("MÓDULO 10 | Diccionario")
    dic_up = update_dictionary(dic, nzv, alias_tab, asign, loadings, coll)

    log.info("MÓDULO 9 | Figuras")
    fig01_nzv(nzv, figd)
    fig02_nzv_bloque(nzv, figd)
    fig03_scree(var, figd)
    fig04_scores12(scores, Y, var, figd)
    fig05_scores13(scores, Y, var, figd)
    fig06_contrib(batch, figd)
    fig07_biplot(scores, loadings, dic, var, figd)
    fig08_robustez(scores, scores_r, figd)
    fig09_heatmap(R_s, Z, dic, figd)
    fig10_dendro(Z, figd)
    fig11_null(s_abs, nulo, comp, figd)
    fig12_composicion(resumen, asign, dic, figd)
    fig13_sensibilidad(resumen, figd)
    fig14_pearson(s_abs, p_abs, figd)

    log.info("MÓDULO 11 | Exportación")
    nzv.to_csv(out / f"F2_I2_NZV_LOG_{VERSION}.csv", index=False, encoding="utf-8-sig")
    alias_tab.to_csv(out / f"F2_I2_ALIAS_CLASIFICADOS_{VERSION}.csv", index=False, encoding="utf-8-sig")
    R_s.to_csv(out / f"F2_I2_CORR_SPEARMAN_{VERSION}.csv", encoding="utf-8-sig")
    R_p.to_csv(out / f"F2_I2_CORR_PEARSON_{VERSION}.csv", encoding="utf-8-sig")
    comp.to_csv(out / f"F2_I2_CORR_VS_NULO_{VERSION}.csv", index=False, encoding="utf-8-sig")
    resumen.to_csv(out / f"F2_I2_MODULES_{VERSION}.csv", index=False, encoding="utf-8-sig")
    asign.rename_axis("VAR_NAME").to_csv(out / f"F2_I2_MODULE_ASSIGNMENT_{VERSION}.csv", encoding="utf-8-sig")
    scores.rename_axis("VAR_KEY").to_csv(out / f"F2_I2_PCA_SCORES_{VERSION}.csv", encoding="utf-8-sig")
    scores_r.rename_axis("VAR_KEY").to_csv(out / f"F2_I2_PCA_SCORES_RANGOS_{VERSION}.csv", encoding="utf-8-sig")
    loadings.rename_axis("VAR_NAME").to_csv(out / f"F2_I2_PCA_LOADINGS_{VERSION}.csv", encoding="utf-8-sig")
    var.to_csv(out / f"F2_I2_PCA_VARIANCE_{VERSION}.csv", index=False, encoding="utf-8-sig")
    batch.to_csv(out / f"F2_I2_BATCH_DIAGNOSTIC_{VERSION}.csv", index=False, encoding="utf-8-sig")
    dic_up.to_csv(out / f"F2_I2_DATA_DICTIONARY_{VERSION}.csv", index=False, encoding="utf-8-sig")

    full = pd.concat([Y, P], axis=1)
    if args.corte == "EXPLORATORIO":
        full.rename_axis("VAR_KEY").to_csv(out / f"F2_I2_DATASET_{VERSION}.csv", encoding="utf-8-sig")
        log.info("  Modo exploratorio: dataset sin cambios de dimensión (V1, V2 y V3). "
                 "Tras elegir el corte, ejecutar con --corte 0.95, 0.90 o 0.85.")
    else:
        t = float(args.corte)
        if not any(abs(t - c) < 1e-9 for c in CUTS.values()):
            log.error(f"Corte no reconocido: {args.corte}. Use 0.95, 0.90 o 0.85.")
            sys.exit(1)
        col = f"MODULO_{int(round(t * 100))}"
        full.rename_axis("VAR_KEY").to_csv(
            out / f"F2_I2_DATASET_FINAL_{VERSION}_{DATE_TAG}.csv", encoding="utf-8-sig")
        asign[[col]].rename_axis("VAR_NAME").to_csv(
            out / f"F2_I2_MODULE_ASSIGNMENT_FINAL_{VERSION}_{DATE_TAG}.csv", encoding="utf-8-sig")
        n_mod = asign[col].nunique()
        log.info(f"  Corte consolidado |rho| >= {t}: {n_mod} módulos sobre {len(asign)} variables agrupadas")
    log.info(f"RESUMEN | P={P.shape[1]} (completas={len(completas)}, con nulos={len(incompletas)}) "
             f"| eliminadas por NZV=0 | alias colapsados={len(coll)} "
             f"| PC1+PC2={var.ACUMULADA.iloc[1]:.1%} "
             f"| módulos " + " ".join(f"{k}={asign[f'MODULO_{int(v * 100)}'].nunique()}" for k, v in CUTS.items()))


if __name__ == "__main__":
    main()
