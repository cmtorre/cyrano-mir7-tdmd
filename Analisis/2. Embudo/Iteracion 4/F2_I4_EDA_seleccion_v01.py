# -*- coding: utf-8 -*-
"""
F2_I4_EDA_seleccion_v01.py
Fase II (Embudo de Tukey), Iteración 4: EDA micro y quirúrgico.
Selecciona un descriptor representativo por módulo mediante dos estrategias
independientes y construye las dos matrices analíticas finales:

  CYRMIR_DATASET  selección supervisada por la respuesta TDMD (A1 primario,
                  A2 y A3 de sensibilidad).
  MEC_DATASET     selección no supervisada restringida a descriptores cuya
                  definición no depende de la identidad de secuencia de Cyrano
                  (B1 primario, B2 y B3 de sensibilidad).

Con N = 7, toda selección supervisada es exploratoria y está condicionada por
selección: la asociación del representante sobrestima la de su módulo.

Ejecución (Windows, Conda tesis_rnas):
    conda activate tesis_rnas
    python F2_I4_EDA_seleccion_v01.py
"""

#==============================================================================
# MÓDULO 0. IMPORTACIONES, RUTAS Y PARÁMETROS
# Objetivo: centralizar rutas, semilla, reglas de elegibilidad y desempate.
# Inputs: argumentos opcionales de terminal.
# Proceso: define rutas con pathlib.Path, la respuesta supervisada, las familias
#          excluidas del MEC_DATASET (V2) y el orden de los criterios de desempate.
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

VERSION = "v01"
DATE_TAG = datetime.now().strftime("%Y%m%d")

BASE = Path(r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-6\Nuevo_Flujo\4.FASE 2_EDA\Ite_4")
DEFAULT_INPUT_DIR = BASE / "00_DATASET"
DEFAULT_OUTPUT_DIR = BASE / "02_TXT"
DEFAULT_FIGURES_DIR = BASE / "01_FIGURAS"

PREFIX_DATASET = "F2_I2_DATASET_FINAL"
PREFIX_DICT = "F2_I3_DATA_DICTIONARY"
PREFIX_MODULES = "F2_I2_MODULE_ASSIGNMENT_FINAL"
PREFIX_CORR = "F2_I2_CORR_SPEARMAN"
PREFIX_PROFILE = "F2_I3_MODULE_PROFILE"
PREFIX_SUBSPACES = "F2_I3_SUBSPACES"

VAR_ORDER = ["WT", "M1", "M2", "M3", "M4", "M5", "M6"]
Y_COLS = ["Y_MIR7_REL", "Y_LOG2_MIR7REL"]
Y_SUPERVISADA = "Y_LOG2_MIR7REL"     # invariante por rangos; coherente con la Fase I
MODULE_COL = "MODULO_90"

# V2: familias excluidas del MEC_DATASET por depender de la identidad de secuencia
MEC_EXCLUIR_BASES = {"SEQ_REAL", "DOTBR", "LEN_REAL"}          # secuencia literal y longitud total
MEC_EXCLUIR_PREFIJO_PCT = ("PCT_A", "PCT_U", "PCT_G", "PCT_C")  # composición nucleotídica individual
MEC_MANTENER = {"PCT_PUR", "PCT_PIR"}                           # purinas y pirimidinas: geometría del apareamiento

# Orden de preferencia por nivel estructural en el desempate (más directo primero)
ORDEN_NIVEL = {"1D": 0, "2D": 1, "3D": 2, "TD": 3, "FLAG": 4, "EXP": 5}

# V7: penalizaciones en el desempate
PENALIZA_WT_SIGNATURE = True
PENALIZA_CIRC_RISK = True

MIN_MIEMBROS_PC1 = 3     # B3 requiere al menos tres miembros
N_OBS = 7


def parse_args():
    p = argparse.ArgumentParser(description="F2 Iteración 4: selección de representantes")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return p.parse_args()


#==============================================================================
# MÓDULO 1. REGISTRO (LOG)
# Objetivo: trazabilidad de cada decisión en consola y archivo.
# Inputs: OUTPUT_DIR.
# Proceso: logger dual.
# Outputs: F2_I4_log_v01.txt.
#==============================================================================
def setup_logger(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("F2_I4")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(out_dir / f"F2_I4_log_{VERSION}.txt", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


#==============================================================================
# MÓDULO 2. CARGA Y AUDITORÍA DE COMPATIBILIDAD
# Objetivo: leer los seis artefactos y verificar que sus identificadores casan.
# Inputs: carpeta 00_DATASET.
# Proceso: resuelve cada archivo por prefijo; filtra el diccionario (superconjunto
#          de 597 entradas) a las 232 columnas del dataset; recupera el módulo del
#          corte 0.90; comprueba que la matriz de Spearman cubre las 227 variables
#          con módulo y que el perfil de módulos coincide con la asignación.
# Outputs: Y, P, DIC, matriz de correlación, perfil de módulos, mapa de subespacios.
#==============================================================================
def resolve(in_dir: Path, prefix: str, log, obligatorio=True):
    cand = sorted(in_dir.glob(f"{prefix}*.csv"))
    if not cand:
        if obligatorio:
            log.error(f"No se encontró ningún archivo con el prefijo '{prefix}' en {in_dir}")
            log.error(f"Archivos .csv presentes: {[f.name for f in in_dir.glob('*.csv')]}")
            sys.exit(1)
        return None
    log.info(f"  [archivo] {prefix} -> {cand[-1].name}")
    return cand[-1]


def load_inputs(args, log):
    in_dir = args.input_dir
    if not in_dir.is_dir():
        log.error(f"INPUT_DIR no existe: {in_dir}")
        sys.exit(1)
    f_data = resolve(in_dir, PREFIX_DATASET, log)
    f_dic = resolve(in_dir, PREFIX_DICT, log)
    f_mod = resolve(in_dir, PREFIX_MODULES, log)
    f_corr = resolve(in_dir, PREFIX_CORR, log)
    f_prof = resolve(in_dir, PREFIX_PROFILE, log, obligatorio=False)
    f_sub = resolve(in_dir, PREFIX_SUBSPACES, log, obligatorio=False)

    X = pd.read_csv(f_data, index_col=0, encoding="utf-8-sig")
    X.index = X.index.astype(str)
    X = X.reindex(VAR_ORDER)
    if X.index.isna().any() or len(X) != N_OBS:
        raise ValueError("El dataset no contiene exactamente WT y M1 a M6")
    if Y_SUPERVISADA not in X.columns:
        raise ValueError(f"Falta la respuesta {Y_SUPERVISADA}")
    Y = X[[c for c in Y_COLS if c in X.columns]].astype(float)
    P = X.drop(columns=Y.columns).apply(pd.to_numeric, errors="coerce")

    dic = pd.read_csv(f_dic, encoding="utf-8-sig", dtype=str).fillna("").set_index("VAR_NAME")
    log.info(f"  Diccionario: {len(dic)} entradas (superconjunto); se filtra a las {P.shape[1]} del dataset")
    falta = set(P.columns) - set(dic.index)
    if falta:
        raise ValueError(f"Predictores sin entrada en el diccionario: {sorted(falta)[:5]}")
    dic = dic.loc[list(P.columns)].copy()
    dic["BLOQUE"] = dic["MOL_CTX"] + "_" + dic["NIV"]

    mod = pd.read_csv(f_mod, index_col=0, encoding="utf-8-sig")
    col = MODULE_COL if MODULE_COL in mod.columns else mod.columns[0]
    dic["MODULO"] = dic.index.map(mod[col]).fillna("")
    if (dic["MODULO"] == "").sum() != 5:
        log.warning(f"  Variables sin módulo: {int((dic['MODULO'] == '').sum())} (se esperaban 5 alias colapsados)")

    R = pd.read_csv(f_corr, index_col=0, encoding="utf-8-sig")
    R.index = R.index.astype(str)
    falta_r = set(dic.index[dic.MODULO != ""]) - set(R.columns)
    if falta_r:
        raise ValueError(f"Variables con módulo ausentes de la matriz de Spearman: {sorted(falta_r)[:5]}")

    prof = pd.read_csv(f_prof, encoding="utf-8-sig") if f_prof else None
    if prof is not None:
        d_mods = set(dic.loc[dic.MODULO != "", "MODULO"])
        p_mods = set(prof.MODULO)
        if d_mods != p_mods:
            log.warning(f"  Módulos en el perfil pero no en la asignación: {sorted(p_mods - d_mods)[:5]}")
    sub = pd.read_csv(f_sub, encoding="utf-8-sig") if f_sub else None

    n_mod = dic.loc[dic.MODULO != "", "MODULO"].nunique()
    log.info(f"  Predictores: {P.shape[1]} | con módulo: {int((dic.MODULO != '').sum())} | módulos: {n_mod}")
    log.info(f"  Matriz de Spearman: {R.shape[0]} x {R.shape[1]}")
    log.info(f"  Respuesta supervisada (solo para CYRMIR): {Y_SUPERVISADA}")
    return Y, P, dic, R, prof, sub


#==============================================================================
# MÓDULO 3. ALIAS DETERMINISTAS Y ELEGIBILIDAD MECANÍSTICA (V6, V2)
# Objetivo: asignar los alias por trazabilidad y marcar qué variables pueden
#           representar a un módulo en el MEC_DATASET.
# Inputs: DIC.
# Proceso: V6 asigna cada alias determinista al módulo de su representante y lo
#          excluye de la competencia, sin duplicarlo en los datasets finales.
#          V2 declara no elegible la composición nucleotídica individual, la
#          secuencia literal y la longitud total específica de Cyrano. Permanecen
#          elegibles el porcentaje de purinas y pirimidinas, las longitudes de
#          regiones funcionales y los descriptores geométricos, termodinámicos,
#          estructurales y conformacionales. Esto no implica que su
#          transferibilidad esté demostrada.
# Outputs: DIC con MODULO_TRAZA, COMPITE y MEC_ELEGIBLE; tabla de elegibilidad.
#==============================================================================
def assign_alias(dic, log):
    dic["MODULO_TRAZA"] = dic["MODULO"]
    dic["COMPITE"] = 1
    dic["MOTIVO_NO_COMPITE"] = ""
    sin_mod = list(dic.index[dic.MODULO == ""])
    for c in sin_mod:
        grupo = dic.at[c, "ALIAS_GROUP"]
        rep = ""
        if grupo:
            hermanos = dic.index[(dic.ALIAS_GROUP == grupo) & (dic.MODULO != "")]
            if len(hermanos):
                rep = hermanos[0]
        dic.at[c, "MODULO_TRAZA"] = dic.at[rep, "MODULO"] if rep else ""
        dic.at[c, "COMPITE"] = 0
        dic.at[c, "MOTIVO_NO_COMPITE"] = (
            f"Alias determinista de {rep} (V6): trazado a su módulo, sin competir ni duplicarse"
            if rep else "Alias determinista sin representante identificable")
        log.info(f"  [alias] {c} -> módulo {dic.at[c, 'MODULO_TRAZA'] or 'no resuelto'} "
                 f"(representante {rep or 'no identificado'})")
    return dic


def mec_eligibility(dic, log):
    filas = []
    for c in dic.index:
        b = dic.at[c, "BASE"]
        if b in MEC_EXCLUIR_BASES:
            eleg, cat, motivo = 0, "SECUENCIA_O_LONGITUD_TOTAL", "Propiedad de la molécula concreta (V2)"
        elif b.startswith(MEC_EXCLUIR_PREFIJO_PCT) and b not in MEC_MANTENER:
            eleg, cat, motivo = 0, "COMPOSICION_NT_INDIVIDUAL", "Depende de la identidad de secuencia (V2)"
        elif b in MEC_MANTENER:
            eleg, cat, motivo = 1, "COMPOSICION_PURINA_PIRIMIDINA", "Determina la geometría del apareamiento"
        elif b.startswith("LEN"):
            eleg, cat, motivo = 1, "LONGITUD_DE_REGION", "Parámetro arquitectónico del sitio"
        elif b == "HAS_REGION":
            eleg, cat, motivo = 1, "ARQUITECTURA_REGION", "Presencia de una región funcional"
        else:
            eleg, cat, motivo = 1, "BIOFISICO_GENERAL", "Definición aplicable a otros sistemas"
        filas.append(dict(VAR_NAME=c, BASE=b, BLOQUE=dic.at[c, "BLOQUE"], REG=dic.at[c, "REG"],
                          CATEGORIA=cat, MEC_ELEGIBLE=eleg, MOTIVO=motivo))
    tab = pd.DataFrame(filas)
    dic["MEC_ELEGIBLE"] = dic.index.map(tab.set_index("VAR_NAME").MEC_ELEGIBLE)
    dic["MEC_CATEGORIA"] = dic.index.map(tab.set_index("VAR_NAME").CATEGORIA)
    log.info("  Elegibilidad MEC: " + ", ".join(f"{k}={v}" for k, v in tab.CATEGORIA.value_counts().items()))
    log.info(f"  Elegibles: {int(tab.MEC_ELEGIBLE.sum())} | no elegibles: {int((tab.MEC_ELEGIBLE == 0).sum())}")
    return dic, tab


#==============================================================================
# MÓDULO 4. ASOCIACIÓN CON LA RESPUESTA (A1, A2) Y NULO EXACTO POR PERMUTACIÓN
# Objetivo: cuantificar la asociación de cada variable con TDMD y situarla frente
#           a la distribución nula correspondiente a su cobertura.
# Inputs: P, Y.
# Proceso: A1 calcula |rho| de Spearman y A2 |tau_b| de Kendall sobre las
#          observaciones disponibles de cada variable. El valor p de cada variable
#          se obtiene enumerando las 5040 permutaciones de la respuesta, lo que da
#          un nulo exacto y distinto para N = 7 y para N = 5 (V1, FIG01).
#          El nulo del máximo intramódulo (V5) se construye con las mismas
#          permutaciones: en cada una se toma el máximo |rho| del módulo, lo que
#          conserva la dependencia real entre sus miembros y evita la corrección
#          de Šidák, inválida bajo correlación.
# Outputs: tabla de asociación por variable y nulo del máximo por módulo.
#==============================================================================
def kendall_tau_b(x, y):
    n = len(x)
    conc = dis = 0
    tx = ty = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                dis += 1
    n0 = n * (n - 1) / 2
    den = np.sqrt((n0 - tx) * (n0 - ty))
    return (conc - dis) / den if den > 0 else np.nan


def ranks_std(v):
    r = pd.Series(v).rank().values
    s = r.std(ddof=1)
    return (r - r.mean()) / s if s > 0 else np.zeros_like(r)


def strata_from_patterns(patrones):
    """Estratos de cobertura: partición más gruesa de las 7 variantes tal que todo
    patrón de ausencia presente en el módulo es unión de estratos. Se construye
    automáticamente dando a cada variante la firma de su disponibilidad en cada
    patrón y agrupando las variantes con la misma firma."""
    firmas = {}
    for v in VAR_ORDER:
        firma = tuple(int(v in pat) for pat in patrones)
        firmas.setdefault(firma, []).append(v)
    return [sorted(g, key=VAR_ORDER.index) for g in firmas.values()]


def stratified_perms(estratos):
    """Todas las permutaciones de la respuesta que permutan dentro de cada estrato
    y no entre estratos. Devuelve una lista de reordenamientos del vector de 7."""
    pos = [[VAR_ORDER.index(v) for v in g] for g in estratos]
    bloques = [list(itertools.permutations(g)) for g in pos]
    salida = []
    for combo in itertools.product(*bloques):
        orden = list(range(N_OBS))
        for g, perm in zip(pos, combo):
            for destino, origen in zip(g, perm):
                orden[destino] = origen
        salida.append(orden)
    return salida


def spearman(zx, zy, n):
    """Correlación de rangos a partir de rangos estandarizados con ddof = 1.
    La normalización es n - 1, no n: con ddof = 1 dividir entre n comprime el
    coeficiente por un factor (n-1)/n y deja el nulo en una escala distinta de la
    del valor observado."""
    return float(zy @ zx) / (n - 1)


def association(P, Y, dic, log):
    """Asociación marginal y nulo exacto de cada variable sobre sus propias
    respuestas disponibles: 5040 permutaciones para N efectivo 7 y 120 para N
    efectivo 5. El valor p es la proporción de permutaciones con |rho| mayor o
    igual al observado; como la permutación identidad pertenece a la enumeración,
    ningún p exacto puede ser cero."""
    y = Y[Y_SUPERVISADA].astype(float)
    patrones = {}
    for c in P.columns:
        patrones.setdefault(tuple(P[c].dropna().index), []).append(c)
    filas, control = [], []
    for idx, cols in patrones.items():
        filas_idx = list(idx)
        n = len(filas_idx)
        yv = y.loc[filas_idx].values
        perms = list(itertools.permutations(range(n)))
        Zy = np.array([ranks_std(yv[list(pp)]) for pp in perms])
        Zx = {c: ranks_std(P.loc[filas_idx, c].values) for c in cols}
        log.info(f"  [nulo marginal] N efectivo {n}: {len(cols)} variables, "
                 f"{len(perms)} permutaciones de sus respuestas disponibles")
        # --- control interno: la identidad debe reproducir el rho observado
        id_pos = perms.index(tuple(range(n)))
        for c in cols:
            v = P.loc[filas_idx, c]
            if v.nunique() < 2:
                filas.append(dict(VAR_NAME=c, N_EFECTIVO=n, RHO_SPEARMAN=np.nan, ABS_RHO=np.nan,
                                  TAU_KENDALL=np.nan, ABS_TAU=np.nan, P_EXACTO=np.nan,
                                  N_PERM=len(perms)))
                continue
            rho = float(np.corrcoef(pd.Series(v).rank(), pd.Series(y.loc[filas_idx]).rank())[0, 1])
            rho_id = spearman(Zx[c], Zy[id_pos], n)
            control.append(dict(VAR_NAME=c, N_EFECTIVO=n, RHO_OBSERVADO=round(rho, 6),
                                RHO_IDENTIDAD=round(rho_id, 6),
                                COINCIDE=int(abs(rho - rho_id) < 1e-9)))
            nulo = np.abs(Zy @ Zx[c]) / (n - 1)
            p = float(np.mean(nulo >= abs(rho) - 1e-9))
            tau = float(kendall_tau_b(v.values, y.loc[filas_idx].values))
            filas.append(dict(VAR_NAME=c, N_EFECTIVO=n, RHO_SPEARMAN=round(rho, 4),
                              ABS_RHO=round(abs(rho), 4), TAU_KENDALL=round(tau, 4),
                              ABS_TAU=round(abs(tau), 4), P_EXACTO=round(p, 6),
                              N_PERM=len(perms)))
    asoc = pd.DataFrame(filas).set_index("VAR_NAME")
    ctrl = pd.DataFrame(control)
    malos = ctrl[ctrl.COINCIDE == 0]
    if len(malos):
        log.error(f"  CONTROL FALLIDO: la identidad no reproduce el rho observado en "
                  f"{len(malos)} variables: {list(malos.VAR_NAME[:5])}")
        sys.exit(1)
    log.info(f"  [control] la permutación identidad reproduce el rho observado en las "
             f"{len(ctrl)} variables evaluadas, en todos los patrones de cobertura")
    if (asoc.P_EXACTO.dropna() <= 0).any():
        log.error("  CONTROL FALLIDO: hay valores p exactos iguales a cero")
        sys.exit(1)
    log.info(f"  Asociación: {int((asoc.P_EXACTO <= 0.05).sum())} variables con p exacto <= 0.05 "
             f"de {int(asoc.P_EXACTO.notna().sum())} | p mínimo = {asoc.P_EXACTO.min():.6f}")
    log.info(f"  |rho| máximo del corpus: {asoc.ABS_RHO.max():.3f}")
    return asoc, ctrl


def module_null(P, Y, dic, asoc, log):
    """Nulo del máximo intramódulo con permutación estratificada por cobertura.
    Los estratos se derivan de los patrones de ausencia presentes en el módulo, de
    modo que una misma permutación se aplica a todas sus variables a la vez y cada
    variable ve exactamente las permutaciones de sus propias respuestas. En un
    módulo homogéneo de N efectivo 7 el único estrato son las siete variantes y el
    nulo recupera las 5040 permutaciones completas."""
    y = Y[Y_SUPERVISADA].astype(float)
    filas = []
    for m, idx in dic.loc[dic.COMPITE == 1].groupby("MODULO_TRAZA").groups.items():
        if not m:
            continue
        miembros = [c for c in idx]
        pats = sorted({tuple(P[c].dropna().index) for c in miembros})
        estratos = strata_from_patterns(pats)
        perms = stratified_perms(estratos)
        Zy = {}
        for pat in pats:
            pos = [VAR_ORDER.index(v) for v in pat]
            Zy[pat] = np.array([ranks_std(y.values[[o[k] for k in pos]]) for o in perms])
        col = []
        for c in miembros:
            pat = tuple(P[c].dropna().index)
            n = len(pat)
            zx = ranks_std(P.loc[list(pat), c].values)
            col.append(np.abs(Zy[pat] @ zx) / (n - 1))
        maxima = np.max(np.column_stack(col), axis=1)
        # El observado se toma de la permutación identidad, que pertenece al nulo:
        # así el estadístico observado y el nulo comparten exactamente la misma
        # aritmética y se evita comparar contra un valor redondeado.
        id_pos = perms.index(list(range(N_OBS)))
        obs = float(maxima[id_pos])
        obs_tabla = float(asoc.loc[miembros, "ABS_RHO"].max())
        if abs(obs - obs_tabla) > 5e-4:
            log.warning(f"  [{m}] máximo por identidad {obs:.6f} frente a tabla {obs_tabla:.4f}")
        esquema = ("COMPLETA" if len(estratos) == 1 else
                   "ESTRATIFICADA:" + "|".join("+".join(g) for g in estratos))
        filas.append(dict(MODULO=m, TAMANO=len(miembros), N_PATRONES=len(pats),
                          ESQUEMA=esquema, N_ESTRATOS=len(estratos), N_PERM=len(perms),
                          MAX_ABS_RHO=round(obs, 4),
                          P_PERM_MAX=round(float(np.mean(maxima >= obs - 1e-9)), 6),
                          NULO_P95=round(float(np.percentile(maxima, 95)), 4)))
    nul = pd.DataFrame(filas)
    for esq, g in nul.groupby(nul.ESQUEMA.str.split(":").str[0]):
        log.info(f"  [nulo intramódulo] esquema {esq}: {len(g)} módulos, "
                 f"N_PERM en {sorted(set(g.N_PERM))}")
    if (nul.P_PERM_MAX <= 0).any():
        log.error("  CONTROL FALLIDO: hay p permutacionales iguales a cero")
        sys.exit(1)
    log.info(f"  Módulos con p permutacional <= 0.05: {int((nul.P_PERM_MAX <= 0.05).sum())} "
             f"de {len(nul)} | p mínimo = {nul.P_PERM_MAX.min():.6f}")
    return nul


#==============================================================================
# MÓDULO 5. TOPOLOGÍA INTRAMÓDULO (B1, B2, B3)
# Objetivo: calcular centralidad, silueta y carga sobre el primer componente.
# Inputs: matriz de Spearman, P, DIC.
# Proceso: la distancia es 1 - |rho|. B1 toma la distancia media de cada variable
#          al resto de su módulo (medoide). B2 calcula el ancho de silueta, que
#          añade la distancia media al módulo vecino más próximo y discrimina donde
#          la centralidad empata (Rousseeuw, 1987). B3 calcula un PCA sobre las
#          variables estandarizadas del módulo y toma la carga absoluta sobre su
#          primer componente; requiere al menos tres miembros y cobertura común.
# Outputs: tabla con DIST_MEDIA_INTRA, SILUETA y CARGA_PC1_MODULO por variable.
#==============================================================================
def topology(P, R, dic, log):
    compiten = dic.index[(dic.COMPITE == 1) & (dic.MODULO_TRAZA != "")]
    D = 1 - R.loc[compiten, compiten].abs()
    np.fill_diagonal(D.values, 0.0)
    mods = dic.loc[compiten, "MODULO_TRAZA"]
    filas = []
    for m, idx in mods.groupby(mods).groups.items():
        miembros = list(idx)
        otros = {mm: list(i) for mm, i in mods.groupby(mods).groups.items() if mm != m}
        # --- B3: primer componente del módulo
        cargas = {}
        if len(miembros) >= MIN_MIEMBROS_PC1:
            M = P[miembros].dropna(axis=0, how="any")
            if len(M) >= 3:
                Z = (M - M.mean()) / M.std(ddof=1)
                Z = Z.loc[:, Z.notna().all()]
                if Z.shape[1] >= 2:
                    _, _, vt = np.linalg.svd(Z.values - Z.values.mean(axis=0), full_matrices=False)
                    cargas = dict(zip(Z.columns, np.abs(vt[0])))
        for c in miembros:
            if len(miembros) > 1:
                a = float(D.loc[c, [x for x in miembros if x != c]].mean())
                b = min(float(D.loc[c, v].mean()) for v in otros.values()) if otros else np.nan
                sil = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
            else:
                a, b, sil = 0.0, np.nan, np.nan
            filas.append(dict(VAR_NAME=c, MODULO=m, TAMANO=len(miembros),
                              DIST_MEDIA_INTRA=round(a, 5),
                              DIST_VECINO=round(b, 5) if b == b else np.nan,
                              SILUETA=round(sil, 5) if sil == sil else np.nan,
                              CARGA_PC1_MODULO=round(float(cargas.get(c, np.nan)), 5)
                              if c in cargas else np.nan))
    top = pd.DataFrame(filas).set_index("VAR_NAME")
    n_pc1 = int(top.CARGA_PC1_MODULO.notna().any() and top.groupby("MODULO").CARGA_PC1_MODULO
                .apply(lambda s: s.notna().any()).sum())
    log.info(f"  Topología calculada para {len(top)} variables en {top.MODULO.nunique()} módulos")
    log.info(f"  B3 aplicable en {n_pc1} módulos (>= {MIN_MIEMBROS_PC1} miembros con cobertura común)")
    return top


#==============================================================================
# MÓDULO 6. SELECCIÓN CON REGLAS DE DESEMPATE
# Objetivo: elegir un representante por módulo bajo un criterio dado.
# Inputs: candidatos del módulo, métrica a maximizar y tabla de atributos.
# Proceso: ordena por la métrica; ante empate aplica, en orden: cobertura (mayor
#          N efectivo), centralidad (menor distancia media intramódulo), ausencia
#          de marcas heredadas (V7: CIRC_RISK se evita siempre que exista una
#          alternativa válida; WT_SIGNATURE pierde el desempate frente a una
#          alternativa equivalente), nivel estructural más directo y orden
#          alfabético como criterio determinista final.
# Outputs: variable elegida, valor de la métrica y nivel de desempate aplicado.
#==============================================================================
def elegir(cands, metrica, attr, log=None):
    if not len(cands):
        return None, np.nan, "SIN_CANDIDATOS"
    vals = attr.loc[cands, metrica].astype(float)
    if vals.notna().sum() == 0:
        top = list(cands)
        nivel = "METRICA_NO_DISPONIBLE"
    else:
        mx = vals.max()
        top = list(vals.index[(vals - mx).abs() < 1e-9])
        nivel = "METRICA" if len(top) == 1 else ""
    if len(top) == 1:
        return top[0], float(vals.get(top[0], np.nan)), nivel or "METRICA"
    criterios = [
        ("COBERTURA", lambda c: -int(attr.at[c, "N_EFECTIVO"])),
        ("CENTRALIDAD", lambda c: float(attr.at[c, "DIST_MEDIA_INTRA"])
         if attr.at[c, "DIST_MEDIA_INTRA"] == attr.at[c, "DIST_MEDIA_INTRA"] else 9.0),
        ("SIN_CIRC_RISK", lambda c: int(attr.at[c, "CIRC_RISK"]) if PENALIZA_CIRC_RISK else 0),
        ("SIN_WT_SIGNATURE", lambda c: int(attr.at[c, "WT_SIGNATURE"]) if PENALIZA_WT_SIGNATURE else 0),
        ("NIVEL", lambda c: ORDEN_NIVEL.get(attr.at[c, "NIV"], 9)),
        ("ALFABETICO", lambda c: c),
    ]
    aplicado = []
    for nombre, clave in criterios:
        valores = {c: clave(c) for c in top}
        mejor = min(valores.values())
        nuevos = [c for c in top if valores[c] == mejor]
        if len(nuevos) < len(top):
            aplicado.append(nombre)
        top = nuevos
        if len(top) == 1:
            break
    elegido = sorted(top)[0]
    return elegido, float(vals.get(elegido, np.nan)), "+".join(["METRICA"] + aplicado) if aplicado else "ALFABETICO"


def build_attr(dic, asoc, top):
    attr = dic.copy()
    for col in ("N_EFECTIVO", "ABS_RHO", "ABS_TAU", "P_EXACTO"):
        attr[col] = attr.index.map(asoc[col]) if col in asoc.columns else np.nan
    for col in ("DIST_MEDIA_INTRA", "SILUETA", "CARGA_PC1_MODULO", "TAMANO"):
        attr[col] = attr.index.map(top[col]) if col in top.columns else np.nan
    attr["CIRC_RISK"] = pd.to_numeric(attr["CIRC_RISK"], errors="coerce").fillna(0).astype(int)
    attr["WT_SIGNATURE"] = pd.to_numeric(attr["WT_SIGNATURE"], errors="coerce").fillna(0).astype(int)
    return attr


#==============================================================================
# MÓDULO 7. CYRMIR_DATASET: A1 PRIMARIO, A2 Y A3 DE SENSIBILIDAD
# Objetivo: seleccionar un representante supervisado por módulo y medir su
#           dependencia del algoritmo y de las observaciones.
# Inputs: attr, P, Y.
# Proceso: A1 maximiza |rho| de Spearman; A2 maximiza |tau_b| de Kendall; A3
#          repite A1 en las siete réplicas de dejar una variante fuera y registra
#          con qué frecuencia cada variable resulta elegida. Los singletons entran
#          como SINGLETON_FORCED (V4), sin competencia y sin exigir asociación.
# Outputs: selección por módulo, tabla de estabilidad y registro de decisiones.
#==============================================================================
def select_cyrmir(attr, P, Y, log):
    y = Y[Y_SUPERVISADA].astype(float)
    sel, stab = [], []
    mods = attr.loc[attr.COMPITE == 1].groupby("MODULO_TRAZA")
    for m, sub in mods:
        if not m:
            continue
        cands = list(sub.index)
        singleton = len(cands) == 1
        a1, v1, why1 = elegir(cands, "ABS_RHO", attr)
        a2, v2, _ = elegir(cands, "ABS_TAU", attr)
        # --- A3: dejar una variante fuera
        cuenta = {c: 0 for c in cands}
        for fuera in VAR_ORDER:
            filas = [v for v in VAR_ORDER if v != fuera]
            rhos = {}
            for c in cands:
                v = P.loc[filas, c].dropna()
                if v.nunique() < 2:
                    continue
                rhos[c] = abs(float(np.corrcoef(pd.Series(v).rank(),
                                                pd.Series(y.loc[v.index]).rank())[0, 1]))
            if not rhos:
                continue
            mx = max(rhos.values())
            ganadores = [c for c, r in rhos.items() if abs(r - mx) < 1e-9]
            tmp = attr.copy()
            tmp["TMP"] = pd.Series(rhos)
            g, _, _ = elegir(ganadores, "TMP", tmp)
            cuenta[g] += 1
        a3 = max(cuenta, key=lambda c: (cuenta[c], -ORDEN_NIVEL.get(attr.at[c, "NIV"], 9)))
        frec = cuenta[a1] / sum(cuenta.values()) if sum(cuenta.values()) else np.nan
        sel.append(dict(MODULO=m, TAMANO=len(cands), REPRESENTANTE=a1,
                        ALGORITMO="A1_SPEARMAN", METRICA_ABS_RHO=round(v1, 4) if v1 == v1 else np.nan,
                        ABS_TAU=attr.at[a1, "ABS_TAU"], P_EXACTO=attr.at[a1, "P_EXACTO"],
                        N_EFECTIVO=int(attr.at[a1, "N_EFECTIVO"]),
                        DESEMPATE=("SINGLETON_FORCED" if singleton else why1),
                        A2_KENDALL=a2, A3_LOO=a3,
                        COINCIDE_A2=int(a1 == a2), COINCIDE_A3=int(a1 == a3),
                        ESTABILIDAD_LOO=round(frec, 3) if frec == frec else np.nan,
                        WT_SIGNATURE=int(attr.at[a1, "WT_SIGNATURE"]),
                        CIRC_RISK=int(attr.at[a1, "CIRC_RISK"])))
        for c, k in cuenta.items():
            stab.append(dict(MODULO=m, VAR_NAME=c, VECES_ELEGIDA=k,
                             FRECUENCIA=round(k / len(VAR_ORDER), 3),
                             ES_REPRESENTANTE=int(c == a1)))
    s = pd.DataFrame(sel)
    ns = s[s.TAMANO > 1]
    log.info(f"  CYRMIR: {len(s)} representantes ({int((s.TAMANO == 1).sum())} SINGLETON_FORCED)")
    log.info(f"  Coincidencia A1-A2 (sin singletons): {int(ns.COINCIDE_A2.sum())}/{len(ns)} "
             f"| A1-A3: {int(ns.COINCIDE_A3.sum())}/{len(ns)}")
    log.info(f"  Estabilidad LOO del representante: mediana={ns.ESTABILIDAD_LOO.median():.3f}, "
             f"módulos con estabilidad 1.0: {int((ns.ESTABILIDAD_LOO == 1).sum())}")
    log.info(f"  Representantes con p exacto <= 0.05: {int((s.P_EXACTO <= 0.05).sum())} de {len(s)}")
    return s, pd.DataFrame(stab)


#==============================================================================
# MÓDULO 8. MEC_DATASET: B1 PRIMARIO, B2 Y B3 DE SENSIBILIDAD
# Objetivo: seleccionar un representante no supervisado por módulo, restringido a
#           variables elegibles, sin que la respuesta intervenga en ningún paso.
# Inputs: attr.
# Proceso: B1 elige el medoide (mínima distancia media intramódulo); B2 la máxima
#          silueta; B3 la mayor carga absoluta sobre el primer componente del
#          módulo. Los desempates añaden, tras la cobertura, la preferencia por
#          descriptores del dúplex o de la hebra miR sobre los exclusivos de
#          Cyrano. Un módulo sin ninguna variable elegible queda ausente del
#          dataset, con el motivo registrado.
# Outputs: selección por módulo y registro de módulos ausentes.
#==============================================================================
def select_mec(attr, log):
    sel, ausentes = [], []
    attr = attr.copy()
    attr["NEG_DIST"] = -attr["DIST_MEDIA_INTRA"].astype(float)   # elegir() maximiza
    mods = attr.loc[attr.COMPITE == 1].groupby("MODULO_TRAZA")
    for m, sub in mods:
        if not m:
            continue
        cands = [c for c in sub.index if int(attr.at[c, "MEC_ELEGIBLE"]) == 1]
        if not cands:
            ausentes.append(dict(MODULO=m, TAMANO=len(sub),
                                 MOTIVO="Ningún descriptor elegible (V2)",
                                 MIEMBROS=";".join(sub.index),
                                 CATEGORIAS=";".join(sorted(set(sub.MEC_CATEGORIA)))))
            continue
        singleton = len(cands) == 1
        b1, v1, why1 = elegir(cands, "NEG_DIST", attr)
        b2, _, _ = elegir(cands, "SILUETA", attr)
        b3, _, _ = elegir(cands, "CARGA_PC1_MODULO", attr)
        sel.append(dict(MODULO=m, TAMANO=len(sub), N_ELEGIBLES=len(cands), REPRESENTANTE=b1,
                        ALGORITMO="B1_MEDOIDE", DIST_MEDIA_INTRA=attr.at[b1, "DIST_MEDIA_INTRA"],
                        SILUETA=attr.at[b1, "SILUETA"], CARGA_PC1=attr.at[b1, "CARGA_PC1_MODULO"],
                        N_EFECTIVO=int(attr.at[b1, "N_EFECTIVO"]),
                        DESEMPATE=("SINGLETON_FORCED" if singleton else why1),
                        B2_SILUETA=b2, B3_PC1=b3,
                        COINCIDE_B2=int(b1 == b2), COINCIDE_B3=int(b1 == b3),
                        MEC_CATEGORIA=attr.at[b1, "MEC_CATEGORIA"],
                        WT_SIGNATURE=int(attr.at[b1, "WT_SIGNATURE"]),
                        CIRC_RISK=int(attr.at[b1, "CIRC_RISK"])))
    s = pd.DataFrame(sel)
    ns = s[s.N_ELEGIBLES > 1]
    log.info(f"  MEC: {len(s)} representantes ({int((s.N_ELEGIBLES == 1).sum())} SINGLETON_FORCED) "
             f"| módulos ausentes: {len(ausentes)}")
    log.info(f"  Coincidencia B1-B2 (sin singletons): {int(ns.COINCIDE_B2.sum())}/{len(ns)} "
             f"| B1-B3: {int(ns.COINCIDE_B3.sum())}/{len(ns)}")
    for a in ausentes:
        log.info(f"  [ausente] {a['MODULO']} ({a['TAMANO']} vars): {a['MOTIVO']} -> {a['MIEMBROS']}")
    return s, pd.DataFrame(ausentes)


#==============================================================================
# MÓDULO 9. CONCORDANCIA ENTRE ALGORITMOS
# Objetivo: cuantificar cuánto depende la selección del algoritmo empleado.
# Inputs: selecciones de ambos datasets y matriz de Spearman.
# Proceso: por instrucción del usuario no se usa kappa de Cohen. Se reportan la
#          coincidencia exacta de los tres algoritmos, la coincidencia de dos de
#          tres, la concordancia par a par y, cuando dos algoritmos difieren, el
#          |rho| entre los dos representantes: si supera 0.95, la discrepancia es
#          intrascendente porque ambos portan la misma información. Los singletons
#          se excluyen, ya que su coincidencia es forzada.
# Outputs: tabla de concordancia por dataset y por módulo.
#==============================================================================
def agreement(sel, R, cols, etiqueta, log):
    filas = []
    prim, s1, s2 = cols
    for _, r in sel.iterrows():
        a, b, c = r[prim], r[s1], r[s2]
        n_dist = len({a, b, c})
        sim = {}
        for nombre, alt in ((s1, b), (s2, c)):
            if alt != a and a in R.index and alt in R.columns:
                sim[nombre] = round(float(abs(R.at[a, alt])), 4)
        filas.append(dict(DATASET=etiqueta, MODULO=r.MODULO, TAMANO=r.TAMANO,
                          SINGLETON=int(r.TAMANO == 1 or r.get("N_ELEGIBLES", 2) == 1),
                          PRIMARIO=a, ALT1=b, ALT2=c,
                          ACUERDO_3_DE_3=int(n_dist == 1), ACUERDO_2_DE_3=int(n_dist == 2),
                          RHO_VS_ALT1=sim.get(s1, np.nan), RHO_VS_ALT2=sim.get(s2, np.nan)))
    t = pd.DataFrame(filas)
    ns = t[t.SINGLETON == 0]
    if len(ns):
        sims = pd.concat([ns.RHO_VS_ALT1, ns.RHO_VS_ALT2]).dropna()
        log.info(f"  [{etiqueta}] sin singletons: {len(ns)} módulos | 3/3={int(ns.ACUERDO_3_DE_3.sum())} "
                 f"({ns.ACUERDO_3_DE_3.mean():.1%}) | 2/3={int(ns.ACUERDO_2_DE_3.sum())}")
        if len(sims):
            log.info(f"  [{etiqueta}] discrepancias: {len(sims)} | |rho| entre representantes: "
                     f"mediana={sims.median():.3f}, con |rho| >= 0.95: {int((sims >= 0.95).sum())}")
    return t


#==============================================================================
# MÓDULO 10. FIGURAS (UN GRÁFICO POR ARCHIVO, 300 DPI)
# Objetivo: visual tracking de asociación, selección, elegibilidad y composición.
# Inputs: resultados de los módulos 3 a 9.
# Proceso: doce rutinas independientes, adaptadas a los algoritmos aprobados.
# Outputs: F2_I4_FIG01 a F2_I4_FIG12 en FIGURES_DIR.
#==============================================================================
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10})
C_CYR, C_MEC, C_AMB = "#4C72B0", "#55A868", "#DD8452"


def save(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=300)
    plt.close(fig)


def fig01_asociacion_nulo(asoc, P, Y, fig_dir):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bins = np.linspace(0, 1, 26)
    for n, col in ((7, "#4C72B0"), (5, "#DD8452")):
        obs = asoc.loc[asoc.N_EFECTIVO == n, "ABS_RHO"].dropna()
        if not len(obs):
            continue
        cols = list(asoc.index[asoc.N_EFECTIVO == n])
        idx = list(P[cols[0]].dropna().index)
        yv = Y[Y_SUPERVISADA].loc[idx].values
        Zy = np.array([ranks_std(yv[list(pp)]) for pp in itertools.permutations(range(n))])
        nulo = np.concatenate([np.abs(Zy @ ranks_std(P.loc[idx, c].values)) / (n - 1)
                               for c in cols])
        ax.hist(obs, bins=bins, density=True, alpha=0.55, color=col,
                label=f"Observado, N efectivo = {n} ({len(obs)} variables)")
        ax.hist(nulo, bins=bins, density=True, histtype="step", lw=1.8, color=col,
                label=f"Nulo exacto, N efectivo = {n} ({len(Zy)} permutaciones)")
    ax.set_xlabel("|rho| de Spearman con la respuesta TDMD")
    ax.set_ylabel("Densidad")
    ax.set_title("Asociación observada frente a la distribución nula exacta de cada cobertura")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I4_FIG01_distribucion_asociacion_vs_nulo.png")


def fig02_max_por_modulo(nul, fig_dir):
    n = nul.sort_values("MAX_ABS_RHO")
    y = np.arange(len(n))
    fig, ax = plt.subplots(figsize=(9.5, max(6, 0.17 * len(n))))
    col = [C_CYR if p <= 0.05 else "#B0B0B0" for p in n.P_PERM_MAX]
    ax.barh(y, n.MAX_ABS_RHO, color=col)
    ax.plot(n.NULO_P95, y, "k.", ms=4, label="Percentil 95 del nulo permutacional")
    for i, (p, t, np_) in enumerate(zip(n.P_PERM_MAX, n.TAMANO, n.N_PERM)):
        ax.text(n.MAX_ABS_RHO.iloc[i] + 0.01, i, f"p={p:.4f} (n={t}, {np_} perm.)", va="center", fontsize=5.5)
    ax.set_yticks(y)
    ax.set_yticklabels(n.MODULO, fontsize=5.5)
    ax.set_xlabel("|rho| máximo dentro del módulo")
    ax.set_xlim(0, 1.25)
    ax.set_title("Asociación máxima por módulo frente al nulo permutacional estratificado")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, fig_dir, "F2_I4_FIG02_asociacion_maxima_por_modulo.png")


def _fig_concordancia(t, etiqueta, nombres, fig_dir, fname):
    ns = t[t.SINGLETON == 0]
    if not len(ns):
        return
    cat = pd.Series(np.where(ns.ACUERDO_3_DE_3 == 1, "3 de 3",
                             np.where(ns.ACUERDO_2_DE_3 == 1, "2 de 3", "Los tres difieren")))
    cuenta = cat.value_counts().reindex(["3 de 3", "2 de 3", "Los tres difieren"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(cuenta.index, cuenta.values, color=[C_MEC, C_AMB, "#C44E52"])
    for i, v in enumerate(cuenta.values):
        ax.text(i, v + 0.3, f"{int(v)} ({v / len(ns):.0%})", ha="center", fontsize=9)
    ax.set_ylabel("Número de módulos con más de un candidato")
    ax.set_title(f"Concordancia entre algoritmos de selección, {etiqueta}\n{nombres}")
    ax.set_ylim(0, max(cuenta.values) * 1.25)
    save(fig, fig_dir, fname)


def fig03_concordancia_cyrmir(t, fig_dir):
    _fig_concordancia(t, "CYRMIR_DATASET", "A1 Spearman, A2 Kendall, A3 dejar una fuera",
                      fig_dir, "F2_I4_FIG03_concordancia_algoritmos_cyrmir.png")


def fig06_concordancia_mec(t, fig_dir):
    _fig_concordancia(t, "MEC_DATASET", "B1 medoide, B2 silueta, B3 carga sobre PC1 interno",
                      fig_dir, "F2_I4_FIG06_concordancia_algoritmos_mec.png")


def fig04_estabilidad(sel, fig_dir):
    s = sel[sel.TAMANO > 1].sort_values("ESTABILIDAD_LOO")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.28 * len(s))))
    ax.barh(y, s.ESTABILIDAD_LOO, color=[C_CYR if v == 1 else C_AMB for v in s.ESTABILIDAD_LOO])
    for i, (m, r) in enumerate(zip(s.MODULO, s.REPRESENTANTE)):
        ax.text(0.02, i, f"{m}: {r}", va="center", fontsize=6, color="white" if s.ESTABILIDAD_LOO.iloc[i] > 0.3 else "black")
    ax.set_yticks([])
    ax.set_xlabel("Proporción de réplicas de dejar una fuera en que se elige el mismo representante")
    ax.set_title("Estabilidad del representante supervisado frente a la omisión de una variante")
    ax.set_xlim(0, 1.05)
    save(fig, fig_dir, "F2_I4_FIG04_estabilidad_seleccion_cyrmir.png")


def fig05_silueta_centralidad(top, sel_mec, fig_dir):
    t = top[top.TAMANO > 1].dropna(subset=["SILUETA"])
    dif = set(sel_mec.loc[sel_mec.COINCIDE_B2 == 0, "MODULO"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for etiqueta, sub, col in (("Módulos donde B1 y B2 coinciden", t[~t.MODULO.isin(dif)], "#B0B0B0"),
                               ("Módulos donde B2 propone otra variable", t[t.MODULO.isin(dif)], C_AMB)):
        ax.scatter(sub.DIST_MEDIA_INTRA, sub.SILUETA, s=28, color=col, alpha=0.85,
                   edgecolor="k", linewidth=0.4, label=etiqueta)
    reps = sel_mec.set_index("MODULO")
    sel_idx = [c for c in t.index if c in set(reps.REPRESENTANTE)]
    ax.scatter(t.loc[sel_idx, "DIST_MEDIA_INTRA"], t.loc[sel_idx, "SILUETA"], s=90,
               facecolor="none", edgecolor=C_MEC, linewidth=1.6, label="Representante elegido por B1")
    ax.set_xlabel("Distancia media intramódulo (criterio B1, menor es mejor)")
    ax.set_ylabel("Ancho de silueta (criterio B2, mayor es mejor)")
    ax.set_title("Centralidad frente a silueta por variable")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I4_FIG05_silueta_vs_centralidad.png")


def fig07_elegibilidad(eleg, fig_dir):
    t = eleg.groupby(["CATEGORIA", "MEC_ELEGIBLE"]).size().unstack(fill_value=0)
    t = t.reindex(columns=[1, 0], fill_value=0).sort_values(by=list(t.columns)[0], ascending=True)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.barh(y, t[1], color=C_MEC, label="Elegible para MEC_DATASET")
    ax.barh(y, t[0], left=t[1], color="#C44E52", label="No elegible (V2)")
    for i, (a, b) in enumerate(zip(t[1], t[0])):
        ax.text(a + b + 1, i, f"{int(a + b)}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=8)
    ax.set_xlabel("Número de variables")
    ax.set_title("Elegibilidad mecanística por categoría de descriptor")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, fig_dir, "F2_I4_FIG07_elegibilidad_mec_por_categoria.png")


def _fig_composicion(sel_c, sel_m, dic, campo, titulo, fname, fig_dir):
    cc = pd.Series([dic.at[v, campo] for v in sel_c.REPRESENTANTE]).value_counts()
    cm = pd.Series([dic.at[v, campo] for v in sel_m.REPRESENTANTE]).value_counts()
    idx = sorted(set(cc.index) | set(cm.index))
    x = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x - 0.2, [cc.get(i, 0) for i in idx], 0.4, color=C_CYR, label=f"CYRMIR ({len(sel_c)})")
    ax.bar(x + 0.2, [cm.get(i, 0) for i in idx], 0.4, color=C_MEC, label=f"MEC ({len(sel_m)})")
    ax.set_xticks(x)
    ax.set_xticklabels(idx, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Número de representantes")
    ax.set_title(titulo)
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, fname)


def fig08_bloque(sel_c, sel_m, dic, fig_dir):
    _fig_composicion(sel_c, sel_m, dic, "BLOQUE", "Composición de los datasets finales por bloque de origen",
                     "F2_I4_FIG08_composicion_datasets_por_bloque.png", fig_dir)


def fig09_region(sel_c, sel_m, dic, fig_dir):
    _fig_composicion(sel_c, sel_m, dic, "REG", "Composición de los datasets finales por región estructural",
                     "F2_I4_FIG09_composicion_datasets_por_region.png", fig_dir)


def fig10_solapamiento(sel_c, sel_m, fig_dir):
    a, b = set(sel_c.REPRESENTANTE), set(sel_m.REPRESENTANTE)
    vals = [len(a - b), len(a & b), len(b - a)]
    etq = ["Solo CYRMIR", "Compartidas", "Solo MEC"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(etq, vals, color=[C_CYR, "#8172B3", C_MEC])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, str(v), ha="center", fontsize=10)
    ax.set_ylabel("Número de variables")
    ax.set_title(f"Solapamiento entre los datasets finales\n"
                 f"CYRMIR: {len(a)} variables · MEC: {len(b)} variables")
    ax.set_ylim(0, max(vals) * 1.2)
    save(fig, fig_dir, "F2_I4_FIG10_solapamiento_cyrmir_mec.png")


def fig11_cobertura(sel_c, sel_m, fig_dir):
    d = pd.DataFrame({"CYRMIR": sel_c.N_EFECTIVO.value_counts(),
                      "MEC": sel_m.N_EFECTIVO.value_counts()}).fillna(0)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, d.CYRMIR, 0.4, color=C_CYR, label="CYRMIR_DATASET")
    ax.bar(x + 0.2, d.MEC, 0.4, color=C_MEC, label="MEC_DATASET")
    for i, (a, b) in enumerate(zip(d.CYRMIR, d.MEC)):
        ax.text(i - 0.2, a + 0.4, str(int(a)), ha="center", fontsize=9)
        ax.text(i + 0.2, b + 0.4, str(int(b)), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N efectivo = {i}" for i in d.index])
    ax.set_ylabel("Número de representantes")
    ax.set_title("Cobertura de los representantes seleccionados")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I4_FIG11_cobertura_datasets.png")


def fig12_trazabilidad(sel_c, fig_dir):
    s = sel_c.copy()
    s["MARCA"] = np.where(s.CIRC_RISK == 1, "CIRC_RISK",
                          np.where(s.WT_SIGNATURE == 1, "WT_SIGNATURE",
                                   np.where(s.DESEMPATE == "SINGLETON_FORCED", "SINGLETON_FORCED",
                                            np.where(s.DESEMPATE == "METRICA", "Métrica sin empate",
                                                     "Resuelto por desempate"))))
    cmap = {"Métrica sin empate": C_CYR, "Resuelto por desempate": C_AMB,
            "SINGLETON_FORCED": "#B0B0B0", "WT_SIGNATURE": "#C44E52", "CIRC_RISK": "#8172B3"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, col in cmap.items():
        sub = s[s.MARCA == k]
        if len(sub):
            ax.scatter(sub.TAMANO, sub.METRICA_ABS_RHO, s=55, color=col, alpha=0.9,
                       edgecolor="k", linewidth=0.4, label=f"{k} ({len(sub)})")
    ax.set_xlabel("Número de miembros del módulo")
    ax.set_ylabel("|rho| del representante con la respuesta")
    ax.set_title("Trazabilidad del representante supervisado: tamaño del módulo y origen de la decisión")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, "F2_I4_FIG12_trazabilidad_modulo_representante.png")


#==============================================================================
# MÓDULO 11. DICCIONARIOS FINALES Y EXPORTACIÓN
# Objetivo: construir las dos matrices con la cadena de trazabilidad completa.
# Inputs: selecciones, attr, mapa de subespacios.
# Proceso: variable -> módulo -> subespacio -> bloque -> OE -> cobertura ->
#          algoritmo -> motivo de selección. Los alias quedan registrados como
#          equivalentes sin duplicarse en la matriz.
# Outputs: CYRMIR_DATASET, MEC_DATASET y sus diccionarios.
#==============================================================================
def build_dictionary(sel, attr, sub, etiqueta, algoritmo):
    reps = sel.set_index("REPRESENTANTE")
    filas = []
    for v in sel.REPRESENTANTE:
        a = attr.loc[v]
        r = reps.loc[v]
        s_reg = s_ctx = ""
        if sub is not None:
            q = sub[sub.VAR_NAME == v]
            s_reg = ";".join(q.loc[q.PARTICION == "REGION", "SUBESPACIO"].unique())
            s_ctx = ";".join(q.loc[q.PARTICION == "CONTEXTO", "SUBESPACIO"].unique())
        alias = list(attr.index[(attr.ALIAS_GROUP != "") & (attr.ALIAS_GROUP == a.ALIAS_GROUP) &
                                (attr.index != v)]) if a.ALIAS_GROUP else []
        filas.append(dict(
            DATASET=etiqueta, VAR_NAME=v, MODULO=r.MODULO, TAMANO_MODULO=r.TAMANO,
            SUBESPACIO_REGION=s_reg, SUBESPACIO_CONTEXTO=s_ctx,
            BLOQUE=a.BLOQUE, MOL_CTX=a.MOL_CTX, NIV=a.NIV, REG=a.REG, OE=a.OE,
            SOURCE_FILE=a.SOURCE_FILE, UNITS=a.UNITS, BASE=a.BASE,
            N_EFECTIVO=r.N_EFECTIVO, ALGORITMO=algoritmo, DESEMPATE=r.DESEMPATE,
            MEC_ELEGIBLE=int(a.MEC_ELEGIBLE), MEC_CATEGORIA=a.MEC_CATEGORIA,
            ABS_RHO_TDMD=a.ABS_RHO, P_EXACTO_TDMD=a.P_EXACTO,
            DIST_MEDIA_INTRA=a.DIST_MEDIA_INTRA, SILUETA=a.SILUETA,
            WT_SIGNATURE=int(a.WT_SIGNATURE), CIRC_RISK=int(a.CIRC_RISK),
            ALIAS_EQUIVALENTES=";".join(alias), NOTES=a.NOTES))
    return pd.DataFrame(filas)


#==============================================================================
# MÓDULO 12. ORQUESTACIÓN
# Objetivo: ejecutar el flujo y escribir los artefactos.
# Inputs: rutas.
# Proceso: carga y auditoría -> alias y elegibilidad -> asociación y nulo ->
#          topología -> selección CYRMIR -> selección MEC -> concordancia ->
#          diccionarios -> figuras -> exportación.
# Outputs: dos datasets, sus diccionarios, seis tablas, el log y doce figuras.
#==============================================================================
def main():
    args = parse_args()
    out, figd = args.output_dir, args.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out)
    log.info(f"F2 Iteración 4 {VERSION} | fecha {DATE_TAG} | pandas {pd.__version__}")

    log.info("MÓDULO 2 | Carga y auditoría de compatibilidad")
    Y, P, dic, R, prof, sub = load_inputs(args, log)

    log.info("MÓDULO 3 | Alias deterministas y elegibilidad mecanística")
    dic = assign_alias(dic, log)
    dic, eleg = mec_eligibility(dic, log)

    log.info("MÓDULO 4 | Asociación con la respuesta y nulo permutacional")
    asoc, ctrl = association(P, Y, dic, log)
    nul = module_null(P, Y, dic, asoc, log)

    log.info("MÓDULO 5 | Topología intramódulo")
    top = topology(P, R, dic, log)
    attr = build_attr(dic, asoc, top)

    log.info("MÓDULO 7 | Selección supervisada (CYRMIR_DATASET)")
    sel_c, stab = select_cyrmir(attr, P, Y, log)

    log.info("MÓDULO 8 | Selección no supervisada (MEC_DATASET)")
    sel_m, ausentes = select_mec(attr, log)

    log.info("MÓDULO 9 | Concordancia entre algoritmos")
    ag_c = agreement(sel_c, R, ("REPRESENTANTE", "A2_KENDALL", "A3_LOO"), "CYRMIR", log)
    ag_m = agreement(sel_m.rename(columns={"TAMANO": "TAMANO_MOD", "N_ELEGIBLES": "TAMANO"}),
                     R, ("REPRESENTANTE", "B2_SILUETA", "B3_PC1"), "MEC", log)

    log.info("MÓDULO 11 | Diccionarios finales")
    dic_c = build_dictionary(sel_c, attr, sub, "CYRMIR", "A1_SPEARMAN")
    dic_m = build_dictionary(sel_m, attr, sub, "MEC", "B1_MEDOIDE")
    CYR = pd.concat([Y, P[list(sel_c.REPRESENTANTE)]], axis=1).rename_axis("VAR_KEY")
    MEC = pd.concat([Y, P[list(sel_m.REPRESENTANTE)]], axis=1).rename_axis("VAR_KEY")

    log.info("MÓDULO 10 | Figuras")
    fig01_asociacion_nulo(asoc, P, Y, figd)
    fig02_max_por_modulo(nul, figd)
    fig03_concordancia_cyrmir(ag_c, figd)
    fig04_estabilidad(sel_c, figd)
    fig05_silueta_centralidad(top, sel_m, figd)
    fig06_concordancia_mec(ag_m, figd)
    fig07_elegibilidad(eleg, figd)
    fig08_bloque(sel_c, sel_m, attr, figd)
    fig09_region(sel_c, sel_m, attr, figd)
    fig10_solapamiento(sel_c, sel_m, figd)
    fig11_cobertura(sel_c, sel_m, figd)
    fig12_trazabilidad(sel_c, figd)

    log.info("MÓDULO 12 | Exportación")
    CYR.to_csv(out / f"F2_I4_CYRMIR_DATASET_{VERSION}_{DATE_TAG}.csv", encoding="utf-8-sig")
    MEC.to_csv(out / f"F2_I4_MEC_DATASET_{VERSION}_{DATE_TAG}.csv", encoding="utf-8-sig")
    dic_c.to_csv(out / f"F2_I4_CYRMIR_DICTIONARY_{VERSION}.csv", index=False, encoding="utf-8-sig")
    dic_m.to_csv(out / f"F2_I4_MEC_DICTIONARY_{VERSION}.csv", index=False, encoding="utf-8-sig")
    pd.concat([sel_c.assign(DATASET="CYRMIR"), sel_m.assign(DATASET="MEC")], ignore_index=True) \
        .to_csv(out / f"F2_I4_SELECTION_LOG_{VERSION}.csv", index=False, encoding="utf-8-sig")
    asoc.rename_axis("VAR_NAME").to_csv(out / f"F2_I4_ASSOCIATION_{VERSION}.csv", encoding="utf-8-sig")
    nul.to_csv(out / f"F2_I4_MODULE_NULL_{VERSION}.csv", index=False, encoding="utf-8-sig")
    ctrl.to_csv(out / f"F2_I4_CONTROL_IDENTIDAD_{VERSION}.csv", index=False, encoding="utf-8-sig")
    pd.concat([ag_c, ag_m], ignore_index=True).to_csv(
        out / f"F2_I4_ALGORITHM_AGREEMENT_{VERSION}.csv", index=False, encoding="utf-8-sig")
    stab.to_csv(out / f"F2_I4_SELECTION_STABILITY_{VERSION}.csv", index=False, encoding="utf-8-sig")
    eleg.to_csv(out / f"F2_I4_MEC_ELIGIBILITY_{VERSION}.csv", index=False, encoding="utf-8-sig")
    if len(ausentes):
        ausentes.to_csv(out / f"F2_I4_MEC_MODULOS_AUSENTES_{VERSION}.csv", index=False, encoding="utf-8-sig")
    top.rename_axis("VAR_NAME").to_csv(out / f"F2_I4_TOPOLOGY_{VERSION}.csv", encoding="utf-8-sig")

    comun = len(set(sel_c.REPRESENTANTE) & set(sel_m.REPRESENTANTE))
    log.info(f"RESUMEN | CYRMIR: {CYR.shape[0]} x {CYR.shape[1]} ({len(sel_c)} representantes + 2 respuestas) "
             f"| MEC: {MEC.shape[0]} x {MEC.shape[1]} ({len(sel_m)} representantes + 2 respuestas) "
             f"| compartidas: {comun} | módulos MEC ausentes: {len(ausentes)}")
    log.info("La selección de CYRMIR usa la respuesta: toda asociación posterior con estas variables "
             "es exploratoria y está condicionada por selección.")


if __name__ == "__main__":
    main()
