# -*- coding: utf-8 -*-
"""
F2_I3_EDA_tipologico_v01.py
Fase II (Embudo de Tukey), Iteración 3: EDA tipológico y subespacios.
Organiza el espacio de 232 predictores en subespacios estructurales con cobertura
homogénea, perfila los 70 módulos de covariación del corte 0.90 y explora
tipologías entre las 7 variantes con un protocolo de sensibilidad explícito.

No selecciona representantes ni construye CYRMIR_DATASET o MEC_DATASET: ambas
tareas corresponden a la Iteración 4.

Ejecución (Windows, Conda tesis_rnas):
    conda activate tesis_rnas
    python F2_I3_EDA_tipologico_v01.py
"""

#==============================================================================
# MÓDULO 0. IMPORTACIONES, RUTAS Y PARÁMETROS
# Objetivo: centralizar rutas, semilla, rejilla de configuraciones y umbrales.
# Inputs: argumentos opcionales de terminal.
# Proceso: define rutas con pathlib.Path, la configuración de referencia (V1),
#          la rejilla de sensibilidad y el número de réplicas de remuestreo (V6).
# Outputs: constantes globales.
#==============================================================================
import argparse
import logging
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform

VERSION = "v01"
DATE_TAG = datetime.now().strftime("%Y%m%d")
SEED = 20260926

BASE = Path(r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-6\Nuevo_Flujo\4.FASE 2_EDA\Ite_3")
DEFAULT_INPUT_DIR = BASE / "00_DATASET"
DEFAULT_OUTPUT_DIR = BASE / "02_TXT"
DEFAULT_FIGURES_DIR = BASE / "01_FIGURAS"

# Prefijos de los archivos de entrada: el sufijo de fecha se resuelve solo
PREFIX_DATASET = "F2_I2_DATASET_FINAL"
PREFIX_DICT = "F2_I2_DATA_DICTIONARY"
PREFIX_MODULES = "F2_I2_MODULE_ASSIGNMENT_FINAL"
PREFIX_MODULES_ALT = "F2_I2_MODULE_ASSIGNMENT"

VAR_ORDER = ["WT", "M1", "M2", "M3", "M4", "M5", "M6"]
Y_COLS = ["Y_MIR7_REL", "Y_LOG2_MIR7REL"]
MODULE_COL = "MODULO_90"

# Tipología arquitectónica documentada (V7: único contraste permitido en esta iteración)
ARQUITECTURA = {"WT": "SIN_SEGMENTO_EXTRA", "M1": "SIN_SEGMENTO_EXTRA", "M2": "SIN_SEGMENTO_EXTRA",
                "M3": "OVERHANG", "M6": "OVERHANG",
                "M4": "BULGE_ASIMETRICO", "M5": "BULGE_ASIMETRICO"}

# Rejilla de configuraciones: (escalado, métrica, enlace). Ward solo con euclídea.
ESCALADOS = ["rangos", "zscore"]
METRICAS = ["correlation", "euclidean"]
ENLACES = ["average", "complete", "ward"]
CONFIG_REFERENCIA = ("rangos", "correlation", "average")   # V1
K_VALUES = [2, 3]
N_BOOT = 1000            # V6
MIN_MODULE_SIZE = 5      # tamaño mínimo para las figuras de composición
MIN_VARS_CLUSTER = 10    # un subespacio necesita al menos esto para agruparse


def parse_args():
    p = argparse.ArgumentParser(description="F2 Iteración 3: EDA tipológico y subespacios")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    return p.parse_args()


#==============================================================================
# MÓDULO 1. REGISTRO (LOG)
# Objetivo: trazabilidad de cada decisión en consola y archivo.
# Inputs: OUTPUT_DIR.
# Proceso: logger dual; el nombre del archivo lleva el modo, como en la Iteración 2.
# Outputs: F2_I3_log_EXPLORATORIO_v01.txt.
#==============================================================================
def setup_logger(out_dir: Path, modo: str) -> logging.Logger:
    log = logging.getLogger("F2_I3")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(out_dir / f"F2_I3_log_{modo}_{VERSION}.txt", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


#==============================================================================
# MÓDULO 2. CARGA Y VALIDACIÓN
# Objetivo: leer los artefactos de la Iteración 2 sin depender del sufijo de fecha.
# Inputs: carpeta 00_DATASET con el dataset, el diccionario y la asignación de módulos.
# Proceso: resuelve cada archivo por prefijo (el más reciente si hay varios), valida
#          las 7 variantes y las 2 respuestas, comprueba que cada predictor tenga
#          entrada en el diccionario y recupera la columna de módulo del corte 0.90.
# Outputs: Y, P (predictores), DIC con módulo asignado.
#==============================================================================
def resolve(in_dir: Path, prefix: str, log, obligatorio=True, alt=None):
    cand = sorted(in_dir.glob(f"{prefix}*.csv"))
    if not cand and alt:
        cand = sorted(in_dir.glob(f"{alt}*.csv"))
    if not cand:
        if obligatorio:
            log.error(f"No se encontró ningún archivo con el prefijo '{prefix}' en {in_dir}")
            log.error(f"Archivos .csv presentes: {[f.name for f in in_dir.glob('*.csv')]}")
            sys.exit(1)
        return None
    elegido = cand[-1]
    if len(cand) > 1:
        log.info(f"  [archivo] {prefix}: {len(cand)} candidatos, se usa {elegido.name}")
    else:
        log.info(f"  [archivo] {prefix} -> {elegido.name}")
    return elegido


def load_inputs(args, log):
    in_dir = args.input_dir
    if not in_dir.is_dir():
        log.error(f"INPUT_DIR no existe: {in_dir}")
        sys.exit(1)
    f_data = resolve(in_dir, PREFIX_DATASET, log)
    f_dic = resolve(in_dir, PREFIX_DICT, log)
    f_mod = resolve(in_dir, PREFIX_MODULES, log, obligatorio=False, alt=PREFIX_MODULES_ALT)

    X = pd.read_csv(f_data, index_col=0, encoding="utf-8-sig")
    X.index = X.index.astype(str)
    X = X.reindex(VAR_ORDER)
    if X.index.isna().any() or len(X) != len(VAR_ORDER):
        raise ValueError("El dataset no contiene exactamente WT y M1 a M6")
    ys = [c for c in Y_COLS if c in X.columns]
    if len(ys) != 2:
        raise ValueError(f"Faltan variables respuesta: {set(Y_COLS) - set(X.columns)}")
    Y = X[ys].astype(float)
    P = X.drop(columns=ys).apply(pd.to_numeric, errors="coerce")

    dic = pd.read_csv(f_dic, encoding="utf-8-sig", dtype=str).fillna("").set_index("VAR_NAME")
    falta = set(P.columns) - set(dic.index)
    if falta:
        raise ValueError(f"Predictores sin entrada en el diccionario: {sorted(falta)[:5]}")
    dic = dic.loc[list(P.columns)].copy()
    dic["BLOQUE"] = dic["MOL_CTX"] + "_" + dic["NIV"]

    if MODULE_COL not in dic.columns or (dic[MODULE_COL] == "").all():
        if f_mod is None:
            raise ValueError(f"No se encontró la asignación de módulos ni la columna {MODULE_COL}")
        m = pd.read_csv(f_mod, index_col=0, encoding="utf-8-sig")
        col = MODULE_COL if MODULE_COL in m.columns else m.columns[0]
        dic[MODULE_COL] = dic.index.map(m[col]).fillna("")
    sin_mod = int((dic[MODULE_COL] == "").sum())
    log.info(f"  Predictores: {P.shape[1]} | con módulo asignado: {P.shape[1] - sin_mod} "
             f"| sin módulo (alias colapsados): {sin_mod}")
    log.info(f"  Módulos distintos en el corte 0.90: {dic.loc[dic[MODULE_COL] != '', MODULE_COL].nunique()}")
    log.info(f"  Respuestas blindadas: {ys} (no intervienen en ningún análisis no supervisado)")
    return Y, P, dic


#==============================================================================
# MÓDULO 3. PATRONES DE COBERTURA Y SUBESPACIOS (V3, V4)
# Objetivo: delimitar los subespacios analizables y su N efectivo.
# Inputs: P, DIC.
# Proceso: agrupa las columnas por patrón de ausencia; construye dos particiones
#          ortogonales (por región estructural y por molécula en contexto) y, para
#          cada subespacio, calcula dos escenarios bajo la regla de cobertura
#          homogénea: COMPLETAS (solo variables sin ausencias, N = 7) y AMPLIADO
#          (todas las variables del subespacio restringidas a la intersección de
#          variantes disponibles). Un escenario solo es apto para agrupamiento si
#          su N efectivo es 7 o 5 y reúne al menos MIN_VARS_CLUSTER variables.
# Outputs: mapa variable-subespacio y tabla de escenarios con N efectivo.
#==============================================================================
def coverage_patterns(P, log):
    pat = P.isna().apply(lambda c: "".join("1" if v else "0" for v in c), axis=0)
    vc = pat.value_counts()
    for p, n in vc.items():
        faltan = [v for v, ch in zip(VAR_ORDER, p) if ch == "1"]
        log.info(f"  [cobertura] patrón {p}: {n} variables"
                 + (f" (ausentes en {', '.join(faltan)})" if faltan else " (completas)"))
    return pat


def build_subspaces(P, dic, pat, log):
    mapa, escenarios = [], []
    particiones = {"REGION": dic["REG"], "CONTEXTO": dic["MOL_CTX"]}
    for part, serie in particiones.items():
        for nombre, sub in serie.groupby(serie):
            cols = list(sub.index)
            completas = [c for c in cols if pat[c] == "0" * len(VAR_ORDER)]
            filas_amp = [v for i, v in enumerate(VAR_ORDER)
                         if all(pat[c][i] == "0" for c in cols)]
            for c in cols:
                mapa.append(dict(VAR_NAME=c, PARTICION=part, SUBESPACIO=f"S_{nombre}",
                                 COMPLETA=int(pat[c] == "0" * len(VAR_ORDER)),
                                 PATRON_NA=pat[c]))
            for esc, vs, filas in (("COMPLETAS", completas, VAR_ORDER),
                                   ("AMPLIADO", cols, filas_amp)):
                apto = (len(vs) >= MIN_VARS_CLUSTER and len(filas) >= 5)
                escenarios.append(dict(PARTICION=part, SUBESPACIO=f"S_{nombre}", ESCENARIO=esc,
                                       N_VARS=len(vs), N_EFECTIVO=len(filas),
                                       VARIANTES=",".join(filas),
                                       APTO_CLUSTER=int(apto and len(vs) > 0)))
    esc = pd.DataFrame(escenarios).drop_duplicates(subset=["PARTICION", "SUBESPACIO", "ESCENARIO"])
    for _, r in esc.iterrows():
        log.info(f"  [{r.PARTICION}] {r.SUBESPACIO} {r.ESCENARIO}: {r.N_VARS} variables, "
                 f"N efectivo={r.N_EFECTIVO} -> {'apto' if r.APTO_CLUSTER else 'solo descripción'}")
    return pd.DataFrame(mapa), esc


#==============================================================================
# MÓDULO 4. PERFIL DE LOS MÓDULOS DE COVARIACIÓN
# Objetivo: caracterizar cada módulo del corte 0.90 sin seleccionar representantes.
# Inputs: P, DIC con MODULO_90.
# Proceso: por módulo calcula cohesión estadística (rho mínimo, mediano y medio
#          entre sus miembros, y diámetro 1 - |rho|), cobertura (variables completas,
#          soporte mínimo de las aristas internas), integración (bloques, niveles,
#          regiones y contextos representados), alineamiento latente (suma de cargas
#          al cuadrado sobre PC1 y PC2) y diagnóstico de medoide (variable que
#          minimiza la distancia media interna, con el margen frente al segundo
#          candidato). Estas cuatro dimensiones se mantienen separadas: un módulo
#          puede ser cohesionado y tener soporte débil, o integrador y poco cohesivo.
# Outputs: tabla MODULE_PROFILE.
#==============================================================================
def module_profile(P, dic, log):
    rows = []
    mods = dic.loc[dic[MODULE_COL] != "", MODULE_COL]
    R = P.rank()
    for m, idx in mods.groupby(mods).groups.items():
        miembros = list(idx)
        sub = P[miembros]
        n = len(miembros)
        if n > 1:
            rho = sub.rank().corr(method="pearson", min_periods=5)
            vals = np.abs(rho.values[np.triu_indices(n, 1)])
            soporte = [int(sub[[a, b]].dropna().shape[0]) for a, b in combinations(miembros, 2)]
            D = 1 - np.abs(rho.fillna(0).values)
            np.fill_diagonal(D, 0)
            medias = D.sum(axis=1) / (n - 1)
            orden = np.argsort(medias)
            medoide = miembros[orden[0]]
            margen = float(medias[orden[1]] - medias[orden[0]])
            rho_min, rho_med, rho_mean = np.nanmin(vals), np.nanmedian(vals), np.nanmean(vals)
            diam = float(np.nanmax(1 - vals))
        else:
            medoide, margen = miembros[0], np.nan
            rho_min = rho_med = rho_mean = np.nan
            diam = 0.0
            soporte = [int(sub[miembros[0]].notna().sum())]
        info = dic.loc[miembros]
        cargas = {pc: float((pd.to_numeric(info.get(f"CARGA_{pc}"), errors="coerce") ** 2).sum())
                  for pc in ("PC1", "PC2") if f"CARGA_{pc}" in info}
        etiquetas = sorted({e for e in info.get("NZV_ETIQUETA_I2", pd.Series(dtype=str))
                            if e and e != "VARIABLE"})
        rows.append(dict(
            MODULO=m, TAMANO=n,
            RHO_MIN=round(float(rho_min), 3) if n > 1 else np.nan,
            RHO_MEDIANO=round(float(rho_med), 3) if n > 1 else np.nan,
            RHO_MEDIO=round(float(rho_mean), 3) if n > 1 else np.nan,
            DIAMETRO=round(diam, 3),
            N_COMPLETAS=int(sub.notna().all().sum()), N_CON_NA=int((~sub.notna().all()).sum()),
            SOPORTE_MIN=int(min(soporte)), MIXTO=int(0 < int((~sub.notna().all()).sum()) < n),
            N_BLOQUES=info.BLOQUE.nunique(), N_NIVELES=info.NIV.nunique(),
            N_REGIONES=info.REG.nunique(), N_CONTEXTOS=info.MOL_CTX.nunique(),
            BLOQUES=";".join(sorted(set(info.BLOQUE))), REGIONES=";".join(sorted(set(info.REG))),
            CARGA_PC1=round(cargas.get("PC1", np.nan), 4), CARGA_PC2=round(cargas.get("PC2", np.nan), 4),
            ETIQUETAS_NZV=";".join(etiquetas),
            MEDOIDE_CANDIDATO=medoide, MEDOIDE_MARGEN=round(margen, 4) if n > 1 else np.nan,
            MIEMBROS=";".join(miembros)))
    prof = pd.DataFrame(rows).sort_values("TAMANO", ascending=False).reset_index(drop=True)
    log.info(f"  Módulos perfilados: {len(prof)} | completos: {int((prof.MIXTO == 0).sum())} "
             f"| mixtos: {int(prof.MIXTO.sum())}")
    log.info(f"  Cohesión (rho mediano interno) en módulos de 2 o más: "
             f"mín={prof.RHO_MEDIANO.min():.3f}, mediana={prof.RHO_MEDIANO.median():.3f}")
    log.info(f"  Integración: módulos de un solo bloque={int((prof.N_BLOQUES == 1).sum())}, "
             f"de 4 o más bloques={int((prof.N_BLOQUES >= 4).sum())}")
    amb = prof[(prof.TAMANO > 2) & (prof.MEDOIDE_MARGEN < 0.01)]
    log.info(f"  Medoide con margen menor a 0.01 (candidatos en competencia): {len(amb)} módulos")
    return prof


#==============================================================================
# MÓDULO 5. TIPOLOGÍAS DE VARIANTES: REJILLA DE CONFIGURACIONES
# Objetivo: explorar agrupamientos naturales entre las 7 variantes sin usar Y.
# Inputs: matriz de un subespacio y sus variantes disponibles.
# Proceso: escala las columnas (Z-score o rangos estandarizados), calcula la
#          distancia entre perfiles de variantes (1 - rho de Pearson sobre el perfil,
#          o euclídea) y agrupa con enlace promedio, completo o Ward (Ward solo con
#          euclídea, donde su criterio de varianza está definido). Cada partición se
#          evalúa en k = 2 y k = 3. Con N = 7 todo agrupamiento es exploratorio.
# Outputs: asignaciones por configuración y matrices de distancia.
#==============================================================================
def scale_matrix(M, escalado):
    if escalado == "rangos":
        M = M.rank(axis=0)
    sd = M.std(ddof=1)
    M = M.loc[:, sd > 0]
    return (M - M.mean()) / M.std(ddof=1)


def cluster_grid(M, escalados=ESCALADOS, metricas=METRICAS, enlaces=ENLACES, ks=K_VALUES):
    out, dists = [], {}
    for esc in escalados:
        Z = scale_matrix(M, esc)
        if Z.shape[1] < 2:
            continue
        for met in metricas:
            D = pdist(Z.values, metric=met)
            dists[(esc, met)] = pd.DataFrame(squareform(D), index=M.index, columns=M.index)
            for enl in enlaces:
                if enl == "ward" and met != "euclidean":
                    continue
                L = linkage(D, method=enl)
                for k in ks:
                    if k >= len(M.index):
                        continue
                    cl = fcluster(L, k, criterion="maxclust")
                    out.append(dict(ESCALADO=esc, METRICA=met, ENLACE=enl, K=k,
                                    N_VARS=Z.shape[1], N_EFECTIVO=len(M.index),
                                    PARTICION=particion_texto(M.index, cl),
                                    ASIGNACION={v: int(c) for v, c in zip(M.index, cl)},
                                    LINKAGE=L))
    return out, dists


def particion_texto(indice, etiquetas):
    grupos = {}
    for v, c in zip(indice, etiquetas):
        grupos.setdefault(int(c), []).append(v)
    ordenados = sorted(grupos.values(), key=lambda g: VAR_ORDER.index(g[0]))
    return " | ".join("{" + ",".join(sorted(g, key=VAR_ORDER.index)) + "}" for g in ordenados)


#==============================================================================
# MÓDULO 6. CONCORDANCIA ENTRE CONFIGURACIONES Y ESTABILIDAD (V6)
# Objetivo: medir si la tipología depende de la configuración o de unas pocas variables.
# Inputs: asignaciones de la rejilla y la matriz del subespacio.
# Proceso: índice de Rand ajustado entre cada par de configuraciones (Hubert &
#          Arabie, 1985), que corrige la coincidencia esperada por azar. La
#          estabilidad frente a las variables se estima remuestreando columnas con
#          reemplazo (1000 réplicas) y contando con qué frecuencia cada par de
#          variantes queda en el mismo grupo. Con N = 7 no es viable remuestrear
#          filas, de modo que el remuestreo evalúa la dependencia del resultado
#          respecto del conjunto de descriptores, no respecto de la muestra.
# Outputs: matriz de concordancia y tabla de estabilidad por par de variantes.
#==============================================================================
def adjusted_rand(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cont = pd.crosstab(a, b).values
    comb = lambda x: x * (x - 1) / 2
    idx = comb(cont).sum()
    sa, sb = comb(cont.sum(axis=1)).sum(), comb(cont.sum(axis=0)).sum()
    total = comb(len(a))
    esperado = sa * sb / total
    maximo = (sa + sb) / 2
    return float((idx - esperado) / (maximo - esperado)) if maximo != esperado else 1.0


def config_agreement(asignaciones, log):
    rows = []
    for i, a in enumerate(asignaciones):
        for j, b in enumerate(asignaciones):
            if a["K"] != b["K"]:
                continue
            va = [a["ASIGNACION"][v] for v in a["ASIGNACION"]]
            vb = [b["ASIGNACION"][v] for v in a["ASIGNACION"]]
            rows.append(dict(K=a["K"], CONFIG_A=nombre_config(a), CONFIG_B=nombre_config(b),
                             ARI=round(adjusted_rand(va, vb), 3)))
    tab = pd.DataFrame(rows)
    for k in sorted(tab.K.unique()):
        sub = tab[(tab.K == k) & (tab.CONFIG_A != tab.CONFIG_B)]
        log.info(f"  [k={k}] ARI entre configuraciones: mediana={sub.ARI.median():.3f}, "
                 f"pares idénticos={int((sub.ARI == 1).sum() // 2)} de {len(sub) // 2}")
    return tab


def nombre_config(a):
    return f"{a['ESCALADO']}+{a['METRICA']}+{a['ENLACE']}"


def particion_modal(asignaciones, k, log):
    sub = [a for a in asignaciones if a["K"] == k]
    cuenta = pd.Series([a["PARTICION"] for a in sub]).value_counts()
    log.info(f"  [k={k}] particiones distintas: {len(cuenta)} en {len(sub)} configuraciones")
    for p, n in cuenta.items():
        log.info(f"    {n}/{len(sub)} configuraciones -> {p}")
    return cuenta


def pair_stability(M, escalado, metrica, enlace, k, n_boot, log, seed=SEED):
    rng = np.random.default_rng(seed)
    variantes = list(M.index)
    pares = list(combinations(variantes, 2))
    juntos = {p: 0 for p in pares}
    validas = 0
    cols = np.arange(M.shape[1])
    for _ in range(n_boot):
        muestra = rng.choice(cols, size=len(cols), replace=True)
        B = M.iloc[:, muestra]
        Z = scale_matrix(B, escalado)
        if Z.shape[1] < 2:
            continue
        D = pdist(Z.values, metric=metrica)
        if not np.all(np.isfinite(D)):
            continue
        cl = fcluster(linkage(D, method=enlace), k, criterion="maxclust")
        asign = dict(zip(variantes, cl))
        for a, b in pares:
            if asign[a] == asign[b]:
                juntos[(a, b)] += 1
        validas += 1
    rows = [dict(VAR_A=a, VAR_B=b, FRECUENCIA=round(juntos[(a, b)] / validas, 3),
                 MISMA_ARQUITECTURA=int(ARQUITECTURA[a] == ARQUITECTURA[b]))
            for a, b in pares]
    tab = pd.DataFrame(rows).sort_values("FRECUENCIA", ascending=False)
    log.info(f"  Remuestreo de columnas: {validas}/{n_boot} réplicas válidas (k={k})")
    for _, r in tab.head(4).iterrows():
        log.info(f"    {r.VAR_A}-{r.VAR_B}: coagrupan en {r.FRECUENCIA:.1%} de las réplicas")
    return tab


#==============================================================================
# MÓDULO 7. CONTRASTE DESCRIPTIVO CON LA ARQUITECTURA (V7)
# Objetivo: comparar las tipologías obtenidas con la arquitectura documentada.
# Inputs: asignaciones de la rejilla.
# Proceso: índice de Rand ajustado entre cada partición y la tipología
#          arquitectónica (sin segmento adicional, overhang, bulge asimétrico).
#          El contraste es posterior a la construcción de los agrupamientos y
#          descriptivo: con 7 observaciones y 3 grupos no sustenta inferencia.
#          El contraste con la respuesta TDMD queda reservado para la Fase III.
# Outputs: tabla con el ARI de cada configuración frente a la arquitectura.
#==============================================================================
def architecture_contrast(asignaciones, log):
    rows = []
    for a in asignaciones:
        vs = list(a["ASIGNACION"].keys())
        arq = [ARQUITECTURA[v] for v in vs]
        rows.append(dict(CONFIG=nombre_config(a), K=a["K"], PARTICION=a["PARTICION"],
                         ARI_ARQUITECTURA=round(adjusted_rand([a["ASIGNACION"][v] for v in vs], arq), 3)))
    tab = pd.DataFrame(rows).sort_values(["K", "ARI_ARQUITECTURA"], ascending=[True, False])
    for k in sorted(tab.K.unique()):
        sub = tab[tab.K == k]
        log.info(f"  [k={k}] ARI frente a la arquitectura: máx={sub.ARI_ARQUITECTURA.max():.3f}, "
                 f"mediana={sub.ARI_ARQUITECTURA.median():.3f}, "
                 f"configuraciones con ARI=1: {int((sub.ARI_ARQUITECTURA == 1).sum())}")
    return tab


#==============================================================================
# MÓDULO 8. TIPOLOGÍAS POR SUBESPACIO
# Objetivo: determinar si la tipología es global o emerge de una región concreta.
# Inputs: P, mapa de subespacios, escenarios aptos.
# Proceso: repite la configuración de referencia en cada subespacio apto y compara
#          su partición con la obtenida en el espacio global y con la arquitectura.
# Outputs: tabla de particiones por subespacio.
#==============================================================================
def subspace_typologies(P, mapa, escenarios, log):
    rows = []
    for _, r in escenarios[escenarios.APTO_CLUSTER == 1].iterrows():
        cols = mapa[(mapa.PARTICION == r.PARTICION) & (mapa.SUBESPACIO == r.SUBESPACIO)].VAR_NAME
        if r.ESCENARIO == "COMPLETAS":
            cols = [c for c in cols if P[c].notna().all()]
        filas = r.VARIANTES.split(",")
        M = P.loc[filas, list(cols)].dropna(axis=1, how="any")
        if M.shape[1] < MIN_VARS_CLUSTER:
            continue
        esc, met, enl = CONFIG_REFERENCIA
        res, _ = cluster_grid(M, [esc], [met], [enl], K_VALUES)
        for a in res:
            arq = [ARQUITECTURA[v] for v in a["ASIGNACION"]]
            rows.append(dict(PARTICION_TIPO=r.PARTICION, SUBESPACIO=r.SUBESPACIO,
                             ESCENARIO=r.ESCENARIO, N_VARS=M.shape[1], N_EFECTIVO=len(filas),
                             K=a["K"], PARTICION=a["PARTICION"],
                             ARI_ARQUITECTURA=round(adjusted_rand(
                                 [a["ASIGNACION"][v] for v in a["ASIGNACION"]], arq), 3)))
    tab = pd.DataFrame(rows)
    for _, r in tab[tab.K == 3].iterrows():
        log.info(f"  [{r.SUBESPACIO} {r.ESCENARIO}] N={r.N_EFECTIVO}, {r.N_VARS} vars, k=3: "
                 f"{r.PARTICION} (ARI arquitectura={r.ARI_ARQUITECTURA})")
    return tab


#==============================================================================
# MÓDULO 9. FIGURAS (UN GRÁFICO POR ARCHIVO, 300 DPI)
# Objetivo: visual tracking de cobertura, módulos y tipologías.
# Inputs: resultados de los módulos 3 a 8.
# Proceso: once rutinas independientes.
# Outputs: F2_I3_FIG01 a F2_I3_FIG11 en FIGURES_DIR.
#==============================================================================
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10})
ARQ_COLOR = {"SIN_SEGMENTO_EXTRA": "#4C72B0", "OVERHANG": "#55A868", "BULGE_ASIMETRICO": "#C44E52"}


def save(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=300)
    plt.close(fig)


def fig01_cobertura(esc, fig_dir):
    e = esc.copy()
    e["ETQ"] = e.SUBESPACIO + " (" + e.ESCENARIO.str[:4] + ")"
    e = e[e.N_VARS > 0].sort_values(["PARTICION", "N_VARS"], ascending=[True, False])
    colores = {7: "#4C72B0", 5: "#DD8452", 3: "#C44E52"}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(e))
    ax.barh(y, e.N_VARS, color=[colores.get(n, "#B0B0B0") for n in e.N_EFECTIVO])
    for i, (v, n) in enumerate(zip(e.N_VARS, e.N_EFECTIVO)):
        ax.text(v + 1, i, f"N efectivo = {n}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(e.ETQ, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Número de variables del subespacio")
    ax.set_title("Cobertura y tamaño de los subespacios estructurales")
    ax.set_xlim(0, e.N_VARS.max() * 1.35)
    ax.legend(handles=[Patch(color=c, label=f"N efectivo = {k}") for k, c in colores.items()],
              loc="lower right", frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I3_FIG01_cobertura_por_subespacio.png")


def fig02_cohesion(prof, fig_dir):
    p = prof[prof.TAMANO > 1]
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(p.N_BLOQUES, p.RHO_MEDIANO, s=p.TAMANO * 22, c=p.RHO_MIN,
                    cmap="viridis", alpha=0.85, edgecolor="k", linewidth=0.5)
    for _, r in p[p.TAMANO >= 7].iterrows():
        ax.annotate(f"{r.MODULO} (n={r.TAMANO})", (r.N_BLOQUES, r.RHO_MEDIANO),
                    xytext=(7, 5), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Bloques de origen representados en el módulo (integración)")
    ax.set_ylabel("Coeficiente de Spearman mediano interno (cohesión)")
    ax.set_title("Cohesión estadística frente a integración entre bloques por módulo")
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label("Coeficiente mínimo interno")
    save(fig, fig_dir, "F2_I3_FIG02_modulos_cohesion_vs_integracion.png")


def fig03_cobertura_modulos(prof, fig_dir):
    p = prof[prof.TAMANO >= 3].sort_values("TAMANO")
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.28 * len(p))))
    ax.barh(y, p.N_COMPLETAS, color="#4C72B0", label="Variables completas (N = 7)")
    ax.barh(y, p.N_CON_NA, left=p.N_COMPLETAS, color="#DD8452",
            label="Variables con ausencia estructural (N = 5)")
    ax.set_yticks(y)
    ax.set_yticklabels(p.MODULO, fontsize=6.5)
    ax.set_xlabel("Número de variables")
    ax.set_title("Cobertura de los módulos con tres o más miembros")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, fig_dir, "F2_I3_FIG03_perfil_modulos_cobertura.png")


def fig04_alineamiento(prof, fig_dir):
    p = prof.dropna(subset=["CARGA_PC1", "CARGA_PC2"]).copy()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(p.CARGA_PC1 * 100, p.CARGA_PC2 * 100, s=p.TAMANO * 22,
               c=p.MIXTO.map({0: "#4C72B0", 1: "#DD8452"}), alpha=0.85, edgecolor="k", linewidth=0.5)
    lim = max((p.CARGA_PC1.max(), p.CARGA_PC2.max())) * 100
    ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls=":")
    for _, r in p[p.TAMANO >= 7].iterrows():
        ax.annotate(r.MODULO, (r.CARGA_PC1 * 100, r.CARGA_PC2 * 100), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Contribución del módulo a PC1 (% de cargas al cuadrado)")
    ax.set_ylabel("Contribución del módulo a PC2 (%)")
    ax.set_title("Alineamiento de los módulos con los componentes retenidos")
    ax.legend(handles=[Patch(color="#4C72B0", label="Módulo completo"),
                       Patch(color="#DD8452", label="Módulo mixto")], frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I3_FIG04_modulos_alineamiento_pc.png")


def fig05_composicion(prof, dic, fig_dir):
    p = prof[prof.TAMANO >= MIN_MODULE_SIZE].sort_values("TAMANO")
    regs = ["GLB", "T1", "P1", "B", "P2", "O"]
    cmap = plt.get_cmap("tab10")
    colr = {r: cmap(i) for i, r in enumerate(regs)}
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, r) in enumerate(p.iterrows()):
        cnt = pd.Series([dic.at[c, "REG"] for c in r.MIEMBROS.split(";")]).value_counts()
        left = 0
        for reg in regs:
            if reg in cnt:
                ax.barh(i, cnt[reg], left=left, color=colr[reg])
                left += cnt[reg]
        ax.text(left + 0.25, i, f"{r.MODULO} (n={r.TAMANO}, rho med={r.RHO_MEDIANO})",
                va="center", fontsize=7.5)
    ax.set_yticks([])
    ax.set_xlim(0, p.TAMANO.max() * 2.1)
    ax.set_xlabel("Número de variables en el módulo")
    ax.set_title(f"Composición regional de los módulos con al menos {MIN_MODULE_SIZE} miembros")
    ax.legend(handles=[Patch(color=colr[r], label=r) for r in regs],
              loc="lower right", frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I3_FIG05_composicion_modulos_por_region.png")


def fig06_dendro_variantes(L, indice, fig_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    dendrogram(L, labels=list(indice), ax=ax, color_threshold=0, above_threshold_color="#4C72B0")
    for t in ax.get_xticklabels():
        t.set_color(ARQ_COLOR[ARQUITECTURA[t.get_text()]])
        t.set_fontsize(11)
    ax.set_ylabel("Distancia entre perfiles (1 - rho de Spearman)")
    ax.set_xlabel("Variante del MRE (color según su arquitectura documentada)")
    ax.set_title("Tipología de variantes en la configuración de referencia")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in ARQ_COLOR.items()],
              frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I3_FIG06_dendrograma_variantes_referencia.png")


def fig07_concordancia(agree, fig_dir, k=3):
    a = agree[agree.K == k]
    piv = a.pivot(index="CONFIG_A", columns="CONFIG_B", values="ARI")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_title(f"Concordancia entre configuraciones de agrupamiento (k = {k})")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Índice de Rand ajustado")
    save(fig, fig_dir, "F2_I3_FIG07_matriz_concordancia_configuraciones.png")


def fig08_estabilidad(stab, fig_dir, k):
    M = pd.DataFrame(0.0, index=VAR_ORDER, columns=VAR_ORDER)
    for _, r in stab.iterrows():
        M.at[r.VAR_A, r.VAR_B] = M.at[r.VAR_B, r.VAR_A] = r.FRECUENCIA
    for v in VAR_ORDER:
        M.at[v, v] = 1.0
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M.values, cmap="viridis", vmin=0, vmax=1)
    for i in range(len(VAR_ORDER)):
        for j in range(len(VAR_ORDER)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M.values[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(VAR_ORDER)))
    ax.set_xticklabels(VAR_ORDER)
    ax.set_yticks(range(len(VAR_ORDER)))
    ax.set_yticklabels(VAR_ORDER)
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_color(ARQ_COLOR[ARQUITECTURA[t.get_text()]])
    ax.set_title(f"Frecuencia de coagrupamiento por par de variantes (k = {k})")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(
        "Proporción de réplicas de remuestreo de columnas")
    save(fig, fig_dir, "F2_I3_FIG08_estabilidad_pares_variantes.png")


def fig09_subespacios(sub, fig_dir):
    s = sub[sub.K == 3].copy()
    if s.empty:
        return
    s["ETQ"] = s.SUBESPACIO + " (" + s.ESCENARIO.str[:4] + ", N=" + s.N_EFECTIVO.astype(str) + ")"
    s = s.sort_values("ARI_ARQUITECTURA")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.45 * len(s))))
    ax.barh(y, s.ARI_ARQUITECTURA, color="#4C72B0")
    for i, (p, v) in enumerate(zip(s.PARTICION, s.ARI_ARQUITECTURA)):
        ax.text(max(v, 0) + 0.02, i, p, va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(s.ETQ, fontsize=7.5)
    ax.set_xlabel("Índice de Rand ajustado frente a la arquitectura documentada")
    ax.set_xlim(min(0, s.ARI_ARQUITECTURA.min() * 1.2), 1.75)
    ax.set_title("Tipología obtenida en cada subespacio (configuración de referencia, k = 3)")
    save(fig, fig_dir, "F2_I3_FIG09_particiones_por_subespacio.png")


def fig10_distancias(D, fig_dir):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(D.values, cmap="magma_r")
    for i in range(len(D)):
        for j in range(len(D)):
            ax.text(j, i, f"{D.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if D.values[i, j] > D.values.max() * 0.6 else "black")
    ax.set_xticks(range(len(D)))
    ax.set_xticklabels(D.columns)
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels(D.index)
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_color(ARQ_COLOR[ARQUITECTURA[t.get_text()]])
    ax.set_title("Distancia entre perfiles estructurales de las variantes")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("1 - rho de Spearman entre perfiles")
    save(fig, fig_dir, "F2_I3_FIG10_distancias_entre_variantes.png")


def fig11_arquitectura(contr, fig_dir):
    c = contr[contr.K == 3].sort_values("ARI_ARQUITECTURA")
    y = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.45 * len(c))))
    ax.barh(y, c.ARI_ARQUITECTURA, color=["#55A868" if v == 1 else "#4C72B0"
                                          for v in c.ARI_ARQUITECTURA])
    for i, (p, v) in enumerate(zip(c.PARTICION, c.ARI_ARQUITECTURA)):
        ax.text(max(v, 0) + 0.02, i, p, va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(c.CONFIG, fontsize=7.5)
    ax.set_xlabel("Índice de Rand ajustado frente a la arquitectura documentada")
    ax.set_xlim(min(0, c.ARI_ARQUITECTURA.min() * 1.2), 1.9)
    ax.set_title("Correspondencia de cada configuración con la arquitectura (k = 3)")
    save(fig, fig_dir, "F2_I3_FIG11_tipologia_vs_arquitectura.png")


#==============================================================================
# MÓDULO 10. DICCIONARIO ACTUALIZADO
# Objetivo: extender el linaje con la organización tipológica.
# Inputs: diccionario de la Iteración 2, mapa de subespacios y perfil de módulos.
# Proceso: añade SUBESPACIO_REGION, SUBESPACIO_CONTEXTO, PATRON_NA_I3,
#          N_EFECTIVO_VARIABLE, MODULO_TAMANO, MODULO_RHO_MEDIANO, MODULO_N_BLOQUES,
#          MODULO_MIXTO y MODULO_MEDOIDE_CANDIDATO.
# Outputs: DataFrame F2_I3_DATA_DICTIONARY.
#==============================================================================
def update_dictionary(dic, mapa, prof, P):
    d = dic.copy()
    reg = mapa[mapa.PARTICION == "REGION"].set_index("VAR_NAME")
    ctx = mapa[mapa.PARTICION == "CONTEXTO"].set_index("VAR_NAME")
    d["SUBESPACIO_REGION"] = d.index.map(reg.SUBESPACIO).fillna("")
    d["SUBESPACIO_CONTEXTO"] = d.index.map(ctx.SUBESPACIO).fillna("")
    d["PATRON_NA_I3"] = d.index.map(reg.PATRON_NA).fillna("")
    d["N_EFECTIVO_VARIABLE"] = [int(P[c].notna().sum()) for c in d.index]
    info = {}
    for _, r in prof.iterrows():
        for m in r.MIEMBROS.split(";"):
            info[m] = r
    d["MODULO_TAMANO"] = [info[c].TAMANO if c in info else pd.NA for c in d.index]
    d["MODULO_RHO_MEDIANO"] = [info[c].RHO_MEDIANO if c in info else pd.NA for c in d.index]
    d["MODULO_N_BLOQUES"] = [info[c].N_BLOQUES if c in info else pd.NA for c in d.index]
    d["MODULO_MIXTO"] = [info[c].MIXTO if c in info else pd.NA for c in d.index]
    d["MODULO_MEDOIDE_CANDIDATO"] = [info[c].MEDOIDE_CANDIDATO if c in info else "" for c in d.index]
    d["ES_MEDOIDE_CANDIDATO"] = [int(c in info and info[c].MEDOIDE_CANDIDATO == c) for c in d.index]
    return d.reset_index().rename(columns={"index": "VAR_NAME"})


#==============================================================================
# MÓDULO 11. ORQUESTACIÓN Y EXPORTACIÓN
# Objetivo: ejecutar el flujo y escribir los artefactos.
# Inputs: rutas y número de réplicas.
# Proceso: carga -> cobertura y subespacios -> perfil de módulos -> rejilla de
#          agrupamiento -> concordancia y estabilidad -> tipologías por subespacio
#          -> contraste con la arquitectura -> diccionario -> figuras -> exportación.
# Outputs: tablas, diccionario, log y once figuras. El dataset no cambia: la
#          Iteración 3 organiza y describe, no selecciona ni reduce.
#==============================================================================
def main():
    args = parse_args()
    out, figd = args.output_dir, args.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out, "EXPLORATORIO")
    log.info(f"F2 Iteración 3 {VERSION} | fecha {DATE_TAG} | pandas {pd.__version__} | semilla {SEED}")
    log.info(f"  Configuración de referencia (V1): {'+'.join(CONFIG_REFERENCIA)}")

    log.info("MÓDULO 2 | Carga y validación")
    Y, P, dic = load_inputs(args, log)

    log.info("MÓDULO 3 | Cobertura y subespacios")
    pat = coverage_patterns(P, log)
    mapa, esc = build_subspaces(P, dic, pat, log)

    log.info("MÓDULO 4 | Perfil de los módulos de covariación")
    prof = module_profile(P, dic, log)

    log.info("MÓDULO 5 | Tipologías de variantes (espacio global, variables completas)")
    completas = [c for c in P.columns if P[c].notna().all()]
    M_glob = P[completas]
    log.info(f"  Espacio de agrupamiento global: N={len(M_glob)} x {M_glob.shape[1]} variables completas")
    grid, dists = cluster_grid(M_glob)
    asign = pd.DataFrame([dict(CONFIG=nombre_config(a), ESCALADO=a["ESCALADO"], METRICA=a["METRICA"],
                               ENLACE=a["ENLACE"], K=a["K"], N_VARS=a["N_VARS"],
                               N_EFECTIVO=a["N_EFECTIVO"], PARTICION=a["PARTICION"],
                               **{f"GRUPO_{v}": a["ASIGNACION"][v] for v in VAR_ORDER})
                          for a in grid])
    modal = {k: particion_modal(grid, k, log) for k in K_VALUES}

    log.info("MÓDULO 6 | Concordancia y estabilidad")
    agree = config_agreement(grid, log)
    esc_r, met_r, enl_r = CONFIG_REFERENCIA
    stab = pair_stability(M_glob, esc_r, met_r, enl_r, 3, args.n_boot, log)

    log.info("MÓDULO 7 | Contraste descriptivo con la arquitectura")
    contr = architecture_contrast(grid, log)

    log.info("MÓDULO 8 | Tipologías por subespacio")
    subt = subspace_typologies(P, mapa, esc, log)

    log.info("MÓDULO 10 | Diccionario")
    dic_up = update_dictionary(dic, mapa, prof, P)

    log.info("MÓDULO 9 | Figuras")
    ref = [a for a in grid if nombre_config(a) == "+".join(CONFIG_REFERENCIA) and a["K"] == 3][0]
    D_ref = dists[(esc_r, met_r)]
    fig01_cobertura(esc, figd)
    fig02_cohesion(prof, figd)
    fig03_cobertura_modulos(prof, figd)
    fig04_alineamiento(prof, figd)
    fig05_composicion(prof, dic, figd)
    fig06_dendro_variantes(ref["LINKAGE"], M_glob.index, figd)
    fig07_concordancia(agree, figd, k=3)
    fig08_estabilidad(stab, figd, k=3)
    fig09_subespacios(subt, figd)
    fig10_distancias(D_ref, figd)
    fig11_arquitectura(contr, figd)

    log.info("MÓDULO 11 | Exportación")
    prof.to_csv(out / f"F2_I3_MODULE_PROFILE_{VERSION}.csv", index=False, encoding="utf-8-sig")
    mapa.to_csv(out / f"F2_I3_SUBSPACES_{VERSION}.csv", index=False, encoding="utf-8-sig")
    esc.to_csv(out / f"F2_I3_SUBSPACE_SCENARIOS_{VERSION}.csv", index=False, encoding="utf-8-sig")
    asign.to_csv(out / f"F2_I3_VARIANT_CLUSTERS_{VERSION}.csv", index=False, encoding="utf-8-sig")
    agree.to_csv(out / f"F2_I3_CONFIG_AGREEMENT_{VERSION}.csv", index=False, encoding="utf-8-sig")
    stab.to_csv(out / f"F2_I3_PAIR_STABILITY_{VERSION}.csv", index=False, encoding="utf-8-sig")
    contr.to_csv(out / f"F2_I3_ARCHITECTURE_CONTRAST_{VERSION}.csv", index=False, encoding="utf-8-sig")
    subt.to_csv(out / f"F2_I3_SUBSPACE_TYPOLOGIES_{VERSION}.csv", index=False, encoding="utf-8-sig")
    for (e, m), D in dists.items():
        D.rename_axis("VAR_KEY").to_csv(out / f"F2_I3_VARIANT_DISTANCES_{e}_{m}_{VERSION}.csv",
                                        encoding="utf-8-sig")
    dic_up.to_csv(out / f"F2_I3_DATA_DICTIONARY_{VERSION}.csv", index=False, encoding="utf-8-sig")

    top3, n3, tot3 = modal[3].index[0], int(modal[3].iloc[0]), int(modal[3].sum())
    log.info(f"RESUMEN | P={P.shape[1]} sin cambios (la Iteración 3 no selecciona ni reduce) "
             f"| subespacios aptos={int(esc.APTO_CLUSTER.sum())} | módulos perfilados={len(prof)} "
             f"(mixtos={int(prof.MIXTO.sum())}) | partición modal k=3 en {n3}/{tot3} configuraciones: {top3} "
             f"| ARI máx frente a arquitectura={contr[contr.K == 3].ARI_ARQUITECTURA.max():.3f}")


if __name__ == "__main__":
    main()
