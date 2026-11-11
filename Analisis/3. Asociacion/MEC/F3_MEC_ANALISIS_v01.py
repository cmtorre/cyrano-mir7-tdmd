# -*- coding: utf-8 -*-
"""
F3_MEC_ANALISIS_v01.py
Fase III, rama MEC_DATASET: análisis descriptivo, de asociación, inferencial y de
robustez de la relación estructura-función.

Los 66 representantes de esta matriz se seleccionaron en la Fase II sin usar la
respuesta TDMD, de modo que cada descriptor permanece fijo durante la inferencia
y basta con permutar la respuesta entre las variantes disponibles.

Con N = 7, todo el análisis es exploratorio. No es un modelo predictivo, no
constituye validación externa ni demuestra causalidad.

Ejecución (Windows, Conda tesis_rnas):
    conda activate tesis_rnas
    python F3_MEC_ANALISIS_v01.py
"""

#==============================================================================
# MÓDULO 0. IMPORTACIONES, RUTAS Y PARÁMETROS
# Objetivo: centralizar rutas, respuesta primaria y reglas de tipo físico.
# Inputs: argumentos opcionales de terminal.
# Proceso: define rutas con pathlib.Path, la respuesta primaria declarada y el
#          mapa de tipo físico que decide cómo se calcula la diferencia con WT.
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
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

VERSION = "v01"
DATE_TAG = datetime.now().strftime("%Y%m%d")

VAR_ORDER = ["WT", "M1", "M2", "M3", "M4", "M5", "M6"]
MUTANTES = [v for v in VAR_ORDER if v != "WT"]
N_OBS = 7
Y_PRIMARIA = "Y_LOG2_MIR7REL"      # respuesta primaria declarada (V8)
Y_LINEAL = "Y_MIR7_REL"            # solo para el control de invariancia
TOL = 1e-9

# V4: tipo físico derivado de unidades y variable base; decide la diferencia con WT
ANGULARES = ("CHI_ANGLE", "PSEUDO_P_SUGAR")
NIVELES_REFERENCIA_FDR = [0.05, 0.10, 0.20]


def tipo_fisico(base, units, nombre=""):
    """Clasifica la variable para decidir cómo se calcula su diferencia con WT.
    Los indicadores de presencia de región se detectan por el prefijo FLAG_ o por
    la marca HAS_ de su variable base."""
    b, u, n = str(base).upper(), str(units), str(nombre).upper()
    if n.startswith("FLAG_") or "HAS_" in b:
        return "BINARIA"
    if b.startswith(ANGULARES) or "grados" in u:
        return "ANGULAR"
    if "kcal/mol" in u:
        return "ENERGIA"
    if u.strip() == "Å":
        return "DISTANCIA"
    if "Å²" in u:
        return "AREA"
    if "%" in u:
        return "PORCENTAJE"
    if "conteo" in u or "nt" == u.strip():
        return "CONTEO"
    return "ADIMENSIONAL"


def diferencia_wt(valores, wt, tipo):
    """Diferencia respecto a WT según el tipo físico (V4).
    Las variables angulares usan la desviación circular con signo en (-180, 180],
    de modo que dos ángulos vecinos separados por el corte de la escala no
    producen una diferencia artificial cercana a 360 grados."""
    if tipo == "BINARIA" or pd.isna(wt):
        return pd.Series([np.nan] * len(valores), index=valores.index)
    d = valores.astype(float) - float(wt)
    if tipo == "ANGULAR":
        d = ((d + 180.0) % 360.0) - 180.0
        d = d.where(d != -180.0, 180.0)
    return d


#==============================================================================
# MÓDULO 1. REGISTRO (LOG)
# Objetivo: trazabilidad de cada decisión en consola y archivo.
# Inputs: OUTPUT_DIR.
# Proceso: logger dual.
# Outputs: F3_<RAMA>_log_v01.txt.
#==============================================================================
def setup_logger(out_dir: Path, rama: str) -> logging.Logger:
    log = logging.getLogger(f"F3_{rama}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(out_dir / f"F3_{rama}_log_{VERSION}.txt", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


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


#==============================================================================
# MÓDULO 3. ESTADÍSTICA DESCRIPTIVA (III-1)
# Objetivo: caracterizar cada representante antes de mirar su relación con TDMD.
# Inputs: matriz de representantes y su diccionario.
# Proceso: por descriptor calcula N efectivo, patrón de ausencias, media,
#          desviación estándar, mediana, rango intercuartílico, mínimo, máximo,
#          rango, el valor de WT, los valores de M1 a M6 y la diferencia con WT
#          según el tipo físico. No se calcula ninguna medida de dispersión
#          relativa universal (V3): con medias cercanas a cero o valores
#          negativos carecería de interpretación.
# Outputs: tabla descriptiva por variable.
#==============================================================================
def descriptiva(P, dic, log):
    filas = []
    for c in P.columns:
        v = P[c]
        obs = v.dropna()
        tipo = tipo_fisico(dic.at[c, "BASE"], dic.at[c, "UNITS"], c)
        d = diferencia_wt(v, v.get("WT", np.nan), tipo)
        q1, q3 = obs.quantile(0.25), obs.quantile(0.75)
        fila = dict(VARIABLE=c, N_EFFECTIVE=int(obs.size),
                    NA_COUNT=int(v.isna().sum()),
                    NA_PATTERN="".join("1" if pd.isna(v[k]) else "0" for k in VAR_ORDER),
                    TIPO_FISICO=tipo, UNITS=dic.at[c, "UNITS"],
                    MEAN=round(float(obs.mean()), 4), SD=round(float(obs.std(ddof=1)), 4),
                    MEDIAN=round(float(obs.median()), 4), IQR=round(float(q3 - q1), 4),
                    MIN=round(float(obs.min()), 4), MAX=round(float(obs.max()), 4),
                    RANGE=round(float(obs.max() - obs.min()), 4),
                    WT_VALUE=(round(float(v["WT"]), 4) if pd.notna(v.get("WT", np.nan)) else np.nan),
                    MODULE_ID=dic.at[c, "MODULO"], MODULE_SIZE=int(dic.at[c, "TAMANO_MODULO"]),
                    REGION=dic.at[c, "REG"], BLOCK=dic.at[c, "BLOQUE"],
                    LEVEL=dic.at[c, "NIV"], CONTEXT=dic.at[c, "MOL_CTX"], OE=dic.at[c, "OE"])
        for k in VAR_ORDER:
            fila[f"VAL_{k}"] = (round(float(v[k]), 4) if pd.notna(v[k]) else np.nan)
        for k in MUTANTES:
            fila[f"DIFF_WT_{k}"] = (round(float(d[k]), 4) if pd.notna(d[k]) else np.nan)
        filas.append(fila)
    des = pd.DataFrame(filas).set_index("VARIABLE")
    log.info(f"  Descriptiva: {len(des)} variables | N efectivo 7: "
             f"{int((des.N_EFFECTIVE == 7).sum())} | N efectivo 5: {int((des.N_EFFECTIVE == 5).sum())}")
    log.info("  Tipos físicos: " + ", ".join(f"{k}={v}" for k, v in des.TIPO_FISICO.value_counts().items()))
    log.info(f"  Diferencia con WT omitida en {int(des.TIPO_FISICO.eq('BINARIA').sum())} variables binarias")
    return des


#==============================================================================
# MÓDULO 4. REDUNDANCIA RESIDUAL
# Objetivo: describir cuánta correlación queda entre los representantes.
# Inputs: matriz de representantes.
# Proceso: correlación de Spearman por pares con soporte mínimo de 5, y recuento
#          de pares por encima de umbrales de referencia. No se elimina ninguna
#          variable: la selección terminó en la Fase II.
# Outputs: matriz de correlación y resumen de redundancia.
#==============================================================================
def redundancia(P, log):
    R = P.rank().corr(method="pearson", min_periods=5)
    iu = np.triu_indices(len(R), 1)
    s = np.abs(R.values[iu])
    s = s[~np.isnan(s)]
    resumen = pd.DataFrame([dict(UMBRAL=t, PARES=int((s >= t).sum()),
                                 PROPORCION=round(float((s >= t).mean()), 4))
                            for t in (0.95, 0.90, 0.85, 0.70)])
    log.info(f"  Redundancia residual: {len(s)} pares | " +
             " | ".join(f"|rho|>={r.UMBRAL}: {r.PARES}" for _, r in resumen.iterrows()))
    return R, resumen


#==============================================================================
# MÓDULO 5-6. ASOCIACIÓN PRIMARIA Y DE SENSIBILIDAD (III-2)
# Objetivo: cuantificar dirección y magnitud de la relación con TDMD.
# Inputs: matriz de representantes y la respuesta primaria.
# Proceso: Spearman es el análisis principal; Kendall tau-b la sensibilidad
#          ordinal; Pearson un diagnóstico complementario de linealidad,
#          calculado sobre la misma respuesta primaria y nunca interpretado como
#          evidencia equivalente. Los tres se calculan sobre las observaciones
#          disponibles de cada variable.
# Outputs: tabla de asociación con signo, magnitud y N efectivo.
#==============================================================================
def kendall_tau_b(x, y):
    n = len(x)
    conc = dis = tx = ty = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                tx += 1; ty += 1
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


def spearman(x, y):
    if pd.Series(x).nunique() < 2 or pd.Series(y).nunique() < 2:
        return np.nan
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def asociacion(P, y, log):
    filas = []
    for c in P.columns:
        v = P[c].dropna()
        yy = y.loc[v.index]
        if v.nunique() < 2:
            filas.append(dict(VARIABLE=c, N_EFFECTIVE=len(v), SPEARMAN_RHO=np.nan,
                              KENDALL_TAU=np.nan, PEARSON_R=np.nan, RHO_RAW=np.nan))
            continue
        rho = spearman(v.values, yy.values)
        tau = kendall_tau_b(v.values, yy.values)
        r = float(np.corrcoef(v.values, yy.values)[0, 1]) if v.std(ddof=1) > 0 else np.nan
        filas.append(dict(VARIABLE=c, N_EFFECTIVE=int(len(v)),
                          SPEARMAN_RHO=round(rho, 4), KENDALL_TAU=round(tau, 4),
                          PEARSON_R=round(r, 4), RHO_RAW=rho))
    a = pd.DataFrame(filas).set_index("VARIABLE")
    a["ABS_RHO"] = a.SPEARMAN_RHO.abs()
    a["KENDALL_DIRECTION_MATCH"] = (np.sign(a.SPEARMAN_RHO) == np.sign(a.KENDALL_TAU)).astype(int)
    a["PEARSON_DIRECTION_MATCH"] = (np.sign(a.SPEARMAN_RHO) == np.sign(a.PEARSON_R)).astype(int)
    log.info(f"  Asociación: |rho| máximo={a.ABS_RHO.max():.3f}, mediana={a.ABS_RHO.median():.3f}")
    log.info(f"  Dirección coincidente con Kendall en {int(a.KENDALL_DIRECTION_MATCH.sum())}/{len(a)} "
             f"| con Pearson (diagnóstico) en {int(a.PEARSON_DIRECTION_MATCH.sum())}/{len(a)}")
    return a


#==============================================================================
# MÓDULO 7. MULTIPLICIDAD (BENJAMINI Y HOCHBERG)
# Objetivo: situar los valores p nominales frente al número de pruebas.
# Inputs: valores p exactos.
# Proceso: procedimiento de paso descendente de Benjamini y Hochberg (1995). El
#          valor q depende del conjunto completo de valores p y de la posición de
#          cada uno en el orden, no solo del mínimo, de modo que no se reporta
#          ningún límite teórico general.
# Outputs: vector de valores q. Se usa como diagnóstico, nunca como filtro.
#==============================================================================
def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    orden = np.argsort(pv)
    ajust = np.minimum.accumulate((pv[orden] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[orden] = np.clip(ajust, 0, 1)
    out[ok] = q
    return out


#==============================================================================
# MÓDULO 9. DEJAR UNA FUERA (III-4 A)
# Objetivo: medir la influencia de cada observación sobre la asociación.
# Inputs: matriz, respuesta.
# Proceso: se recalcula Spearman omitiendo una variante por vez. No es una
#          validación predictiva: mide la dependencia del coeficiente respecto de
#          observaciones individuales. En variables con N efectivo 5, omitir una
#          variante deja cuatro observaciones, y la fila lo declara.
# Outputs: tabla con las réplicas, su mediana, extremos, rango, estabilidad de
#          signo y cambio máximo respecto al coeficiente completo.
#==============================================================================
def leave_one_out(P, y, asoc, log):
    filas = []
    for c in P.columns:
        rho_full = asoc.at[c, "SPEARMAN_RHO"]
        reps = {}
        for fuera in VAR_ORDER:
            idx = [v for v in VAR_ORDER if v != fuera and pd.notna(P.at[v, c])]
            if len(idx) < 3 or P.loc[idx, c].nunique() < 2:
                reps[fuera] = np.nan
                continue
            reps[fuera] = spearman(P.loc[idx, c].values, y.loc[idx].values)
        vals = pd.Series(reps).dropna()
        n_min = min(int(P[c].dropna().index.isin([v for v in VAR_ORDER if v != f]).sum())
                    for f in VAR_ORDER)
        fila = dict(VARIABLE=c, N_EFFECTIVE=int(P[c].notna().sum()), N_LOO_MIN=n_min,
                    SPEARMAN_RHO=rho_full,
                    LOO_MEDIAN=round(float(vals.median()), 4) if len(vals) else np.nan,
                    LOO_MIN=round(float(vals.min()), 4) if len(vals) else np.nan,
                    LOO_MAX=round(float(vals.max()), 4) if len(vals) else np.nan,
                    LOO_RANGE=round(float(vals.max() - vals.min()), 4) if len(vals) else np.nan,
                    LOO_SIGN_EVALUABLE=int(len(vals) > 0 and pd.notna(rho_full) and rho_full != 0),
                    LOO_SIGN_STABILITY=(round(float((np.sign(vals) == np.sign(rho_full)).mean()), 3)
                                        if len(vals) and pd.notna(rho_full) and rho_full != 0
                                        else np.nan),
                    LOO_MAX_CHANGE=(round(float((vals - rho_full).abs().max()), 4)
                                    if len(vals) and pd.notna(rho_full) else np.nan))
        for k, val in reps.items():
            fila[f"LOO_SIN_{k}"] = round(val, 4) if pd.notna(val) else np.nan
        filas.append(fila)
    loo = pd.DataFrame(filas).set_index("VARIABLE")
    n_ev = int(loo.LOO_SIGN_EVALUABLE.sum())
    log.info(f"  Dejar una fuera: estabilidad de signo mediana={loo.LOO_SIGN_STABILITY.median():.3f} "
             f"| signo conservado en las 7 réplicas: "
             f"{int((loo.LOO_SIGN_STABILITY == 1).sum())}/{n_ev} evaluables "
             f"| no evaluables por coeficiente completo cero o no definido: {len(loo) - n_ev}")
    log.info(f"  Cambio máximo respecto al coeficiente completo: mediana="
             f"{loo.LOO_MAX_CHANGE.median():.3f}, máximo={loo.LOO_MAX_CHANGE.max():.3f}")
    return loo


#==============================================================================
# MÓDULO 10. SENSIBILIDAD A WT (III-4 B)
# Objetivo: detectar asociaciones dominadas por el contraste entre el tipo
#           silvestre y las variantes mutantes.
# Inputs: matriz, respuesta.
# Proceso: se recalcula Spearman sobre M1 a M6 y se compara con el coeficiente
#          completo. WT permanece en el análisis principal: este bloque no
#          justifica su exclusión, solo cuantifica su influencia.
# Outputs: coeficiente sin WT, delta y conservación de signo.
#==============================================================================
def sensibilidad_wt(P, y, asoc, log):
    """El signo solo está definido cuando ambos coeficientes existen y el
    coeficiente completo no es exactamente cero. Un coeficiente nulo no tiene
    signo, de modo que no puede conservarse ni cambiar: esas variables se marcan
    como no evaluables con su motivo, y quedan fuera del recuento en el log, en la
    tabla maestra y en la figura, que comparten esta única definición."""
    filas = []
    for c in P.columns:
        idx = [v for v in MUTANTES if pd.notna(P.at[v, c])]
        rho_full = asoc.at[c, "SPEARMAN_RHO"]
        rho_sin = (spearman(P.loc[idx, c].values, y.loc[idx].values)
                   if len(idx) >= 3 and P.loc[idx, c].nunique() >= 2 else np.nan)
        if pd.isna(rho_full):
            eval_, motivo = 0, "Coeficiente completo no definido"
        elif pd.isna(rho_sin):
            eval_, motivo = 0, "Sin variación entre M1 y M6: el coeficiente sin WT no existe"
        elif rho_full == 0:
            eval_, motivo = 0, "Coeficiente completo exactamente cero: no tiene signo"
        else:
            eval_, motivo = 1, ""
        filas.append(dict(VARIABLE=c, N_SIN_WT=len(idx), SPEARMAN_RHO=rho_full,
                          RHO_WITHOUT_WT=round(rho_sin, 4) if pd.notna(rho_sin) else np.nan,
                          DELTA_RHO_WITHOUT_WT=(round(float(rho_sin - rho_full), 4)
                                                if pd.notna(rho_sin) and pd.notna(rho_full) else np.nan),
                          WT_SENS_EVALUABLE=eval_, WT_SENS_MOTIVO=motivo,
                          SIGN_WITHOUT_WT=(int(np.sign(rho_sin) == np.sign(rho_full))
                                           if eval_ else np.nan)))
    w = pd.DataFrame(filas).set_index("VARIABLE")
    n_ev = int(w.WT_SENS_EVALUABLE.sum())
    n_cons = int(w.SIGN_WITHOUT_WT.sum())
    log.info(f"  Sensibilidad a WT: |delta| mediano={w.DELTA_RHO_WITHOUT_WT.abs().median():.3f}, "
             f"máximo={w.DELTA_RHO_WITHOUT_WT.abs().max():.3f}")
    log.info(f"  Signo conservado sin WT en {n_cons}/{n_ev} evaluables | "
             f"cambia en {n_ev - n_cons} | no evaluables: {len(w) - n_ev}")
    for motivo, g in w[w.WT_SENS_EVALUABLE == 0].groupby("WT_SENS_MOTIVO"):
        log.info(f"    no evaluables por '{motivo}': {len(g)} ({', '.join(g.index[:3])})")
    return w


#==============================================================================
# MÓDULO 12. FIGURAS (UN GRÁFICO POR ARCHIVO, 300 DPI)
# Objetivo: visual tracking equivalente entre ramas.
# Inputs: resultados de los módulos anteriores.
# Proceso: rutinas independientes con nombres comparables entre CYRMIR y MEC.
# Outputs: F3_<RAMA>_FIG01 a FIG12 (y FIG13 solo en CYRMIR).
#==============================================================================
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 10})
C7, C5 = "#4C72B0", "#DD8452"
CPOS, CNEG = "#C44E52", "#4C72B0"


def save(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=300)
    plt.close(fig)


def fig01_composicion(des, fig_dir, rama):
    campos = [("REGION", "Región estructural"), ("BLOCK", "Bloque de origen"),
              ("LEVEL", "Nivel estructural"), ("CONTEXT", "Contexto molecular")]
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    ax = axes
    etiquetas, v7, v5, sep = [], [], [], []
    for campo, _ in campos:
        vals = sorted(des[campo].unique())
        for v in vals:
            sub = des[des[campo] == v]
            etiquetas.append(f"{campo[:3]}: {v}")
            v7.append(int((sub.N_EFFECTIVE == 7).sum()))
            v5.append(int((sub.N_EFFECTIVE == 5).sum()))
        sep.append(len(etiquetas) - 0.5)
    x = np.arange(len(etiquetas))
    ax.bar(x, v7, color=C7, label="N efectivo = 7")
    ax.bar(x, v5, bottom=v7, color=C5, label="N efectivo = 5")
    for s in sep[:-1]:
        ax.axvline(s, color="gray", lw=0.8, ls=":")
    for i, (a, b) in enumerate(zip(v7, v5)):
        ax.text(i, a + b + 0.3, str(a + b), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=90, fontsize=7)
    ax.set_ylabel("Número de representantes")
    ax.set_title(f"Composición y cobertura de {rama}_DATASET")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, f"F3_{rama}_FIG01_composicion_y_cobertura.png")


def fig02_perfil(des, fig_dir, rama):
    d = des.copy()
    rng = (d.MAX - d.MIN).replace(0, np.nan)
    d["POS_WT"] = (d.WT_VALUE - d.MIN) / rng
    d["POS_MED"] = (d.MEDIAN - d.MIN) / rng
    d = d.sort_values(["REGION", "POS_WT"])
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, max(6, 0.16 * len(d))))
    ax.hlines(y, 0, 1, color="#D0D0D0", lw=2)
    ax.scatter(d.POS_MED, y, s=18, color="#808080", label="Mediana", zorder=3)
    ax.scatter(d.POS_WT, y, s=30, color=C5, marker="D", label="Valor de WT", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i} [{r.REGION}]" for i, r in d.iterrows()], fontsize=4.5)
    ax.set_xlabel("Posición relativa dentro del rango observado de cada variable (0 = mínimo, 1 = máximo)")
    ax.set_title(f"Perfil descriptivo por representante, {rama}_DATASET")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, fig_dir, f"F3_{rama}_FIG02_perfil_descriptivo.png")


def fig03_heatmap(P, des, y, fig_dir, rama):
    orden = des.sort_values(["REGION", "BLOCK"]).index
    Z = ((P[orden] - P[orden].mean()) / P[orden].std(ddof=1)).T
    fig, ax = plt.subplots(figsize=(7.5, max(7, 0.13 * len(orden))))
    im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(VAR_ORDER)))
    ax.set_xticklabels(VAR_ORDER)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([f"{c} [{des.at[c, 'REGION']}]" for c in orden], fontsize=4.5)
    ax.set_xlabel("Variante del MRE")
    ax.set_title(f"Valores estandarizados por variante, {rama}_DATASET\n"
                 f"(gris: ausencia estructural)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Desviaciones estándar")
    save(fig, fig_dir, f"F3_{rama}_FIG03_heatmap_estandarizado.png")


def fig04_correlacion(R, fig_dir, rama):
    D = 1 - R.abs().fillna(0).values
    np.fill_diagonal(D, 0)
    D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), "average")
    orden = [R.columns[i] for i in leaves_list(Z)]
    M = R.loc[orden, orden]
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{len(orden)} representantes ordenados por agrupamiento jerárquico")
    ax.set_title(f"Redundancia residual entre representantes, {rama}_DATASET")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Coeficiente de Spearman")
    save(fig, fig_dir, f"F3_{rama}_FIG04_correlacion_residual.png")


def fig05_asociacion(master, fig_dir, rama):
    d = master.dropna(subset=["SPEARMAN_RHO"]).sort_values("SPEARMAN_RHO")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.5, max(6, 0.17 * len(d))))
    ax.barh(y, d.SPEARMAN_RHO, color=[CPOS if v > 0 else CNEG for v in d.SPEARMAN_RHO])
    marca = {7: "", 5: "  (N=5)"}
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i}{marca.get(int(n), '')}" for i, n in zip(d.index, d.N_EFFECTIVE)],
                       fontsize=4.5)
    for i, n in enumerate(d.N_EFFECTIVE):
        if n == 5:
            ax.get_yticklabels()[i].set_color(C5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coeficiente de Spearman con la respuesta TDMD (con signo)")
    ax.set_title(f"Asociación ordenada de los representantes, {rama}_DATASET")
    ax.set_xlim(-1.05, 1.05)
    save(fig, fig_dir, f"F3_{rama}_FIG05_asociacion_ordenada.png")


def _fig_vs(master, col, etiqueta, fname, titulo, fig_dir):
    d = master.dropna(subset=["SPEARMAN_RHO", col])
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    for n, col_c, lab in ((7, C7, "N efectivo = 7"), (5, C5, "N efectivo = 5")):
        sub = d[d.N_EFFECTIVE == n]
        ax.scatter(sub.SPEARMAN_RHO, sub[col], s=40, color=col_c, alpha=0.85,
                   edgecolor="k", linewidth=0.4, label=lab)
    ax.plot([-1, 1], [-1, 1], color="gray", lw=0.9, ls=":")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    disc = int((np.sign(d.SPEARMAN_RHO) != np.sign(d[col])).sum())
    ax.set_xlabel("Coeficiente de Spearman (análisis principal)")
    ax.set_ylabel(etiqueta)
    ax.set_title(f"{titulo}\n{disc} variables con dirección discordante")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, fname)


def fig06_kendall(master, fig_dir, rama):
    _fig_vs(master, "KENDALL_TAU", "Coeficiente tau-b de Kendall (sensibilidad ordinal)",
            f"F3_{rama}_FIG06_spearman_vs_kendall.png",
            f"Concordancia ordinal, {rama}_DATASET", fig_dir)


def fig07_pearson(master, fig_dir, rama):
    _fig_vs(master, "PEARSON_R", "Coeficiente r de Pearson (solo diagnóstico)",
            f"F3_{rama}_FIG07_spearman_vs_pearson.png",
            f"Diagnóstico de linealidad, {rama}_DATASET", fig_dir)


def fig08_loo(master, fig_dir, rama):
    d = master.dropna(subset=["LOO_MIN", "LOO_MAX"]).sort_values("SPEARMAN_RHO")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.5, max(6, 0.17 * len(d))))
    ax.hlines(y, d.LOO_MIN, d.LOO_MAX, color="#B0B0B0", lw=2.2)
    ax.scatter(d.SPEARMAN_RHO, y, s=16,
               color=["#B0B0B0" if pd.isna(s) else (C7 if s == 1 else C5)
                      for s in d.LOO_SIGN_STABILITY], zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=4.5)
    ax.set_xlabel("Coeficiente de Spearman: intervalo de las réplicas de dejar una fuera")
    ax.set_title(f"Estabilidad frente a la omisión de una variante, {rama}_DATASET")
    ax.legend(handles=[Patch(color=C7, label="Signo conservado en las 7 réplicas"),
                       Patch(color=C5, label="Signo no conservado en alguna réplica"),
                       Patch(color="#B0B0B0", label="No evaluable: coeficiente completo cero")],
              frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(-1.05, 1.05)
    save(fig, fig_dir, f"F3_{rama}_FIG08_estabilidad_loo.png")


def fig09_wt(master, fig_dir, rama):
    """Usa la misma definición que la tabla maestra y el log: el cambio de signo
    solo se cuenta en las variables evaluables, es decir, aquellas con ambos
    coeficientes definidos y con coeficiente completo distinto de cero."""
    d = master.dropna(subset=["SPEARMAN_RHO", "RHO_WITHOUT_WT"]).copy()
    ev = d.WT_SENS_EVALUABLE == 1
    cambio = ev & (d.SIGN_WITHOUT_WT == 0)
    conserv = ev & (d.SIGN_WITHOUT_WT == 1)
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    ax.scatter(d.SPEARMAN_RHO[conserv], d.RHO_WITHOUT_WT[conserv], s=40, color=C7,
               edgecolor="k", linewidth=0.4, label=f"Signo conservado ({int(conserv.sum())})")
    ax.scatter(d.SPEARMAN_RHO[cambio], d.RHO_WITHOUT_WT[cambio], s=40, color=CPOS,
               edgecolor="k", linewidth=0.4, label=f"Cambio de signo ({int(cambio.sum())})")
    ax.scatter(d.SPEARMAN_RHO[~ev], d.RHO_WITHOUT_WT[~ev], s=40, color="#B0B0B0",
               edgecolor="k", linewidth=0.4,
               label=f"No evaluable ({int((~ev).sum())})")
    ax.plot([-1, 1], [-1, 1], color="gray", lw=0.9, ls=":")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("Coeficiente con WT y M1 a M6 (análisis principal)")
    ax.set_ylabel("Coeficiente solo con M1 a M6")
    ax.set_title(f"Influencia de WT sobre la asociación, {rama}_DATASET\n"
                 f"{int(cambio.sum())} de {int(ev.sum())} variables evaluables cambian de signo "
                 f"al retirar WT")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, f"F3_{rama}_FIG09_rho_con_y_sin_wt.png")


def fig10_fdr(master, fig_dir, rama):
    d = master.dropna(subset=["P_EXACT", "Q_BH"]).sort_values("P_EXACT")
    x = np.arange(1, len(d) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot(x, d.P_EXACT, color=C7, marker="o", ms=3, lw=1.2, label="p exacto nominal")
    ax.plot(x, d.Q_BH, color=CPOS, marker="s", ms=3, lw=1.2, label="q de Benjamini y Hochberg")
    for t in NIVELES_REFERENCIA_FDR:
        ax.axhline(t, color="gray", lw=0.7, ls=":")
        ax.text(len(d) * 0.99, t * 1.04, f"{t}", fontsize=7, color="gray", ha="right")
    ax.set_xlabel("Representantes ordenados por p exacto")
    ax.set_ylabel("Valor")
    ax.set_yscale("log")
    ax.set_title(f"Valor p nominal frente al ajuste por multiplicidad, {rama}_DATASET\n"
                 f"q mínimo observado: {d.Q_BH.min():.3f}")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, f"F3_{rama}_FIG10_p_exacto_vs_fdr.png")


def fig11_por_grupo(master, fig_dir, rama):
    fig, ax = plt.subplots(figsize=(11, 6))
    grupos, datos, colores, pos = [], [], [], []
    p = 0
    for campo, col in (("REGION", "#4C72B0"), ("BLOCK", "#55A868")):
        for v in sorted(master[campo].dropna().unique()):
            vals = master.loc[master[campo] == v, "SPEARMAN_RHO"].dropna()
            if len(vals) == 0:
                continue
            p += 1
            grupos.append(f"{v}")
            datos.append(vals.values)
            colores.append(col)
            pos.append(p)
    bp = ax.boxplot(datos, positions=pos, widths=0.6, patch_artist=True, showfliers=False)
    for caja, c in zip(bp["boxes"], colores):
        caja.set_facecolor(c)
        caja.set_alpha(0.55)
    for i, (xp, vals) in enumerate(zip(pos, datos)):
        ax.scatter(np.full(len(vals), xp) + np.random.default_rng(i).normal(0, 0.06, len(vals)),
                   vals, s=12, color="black", alpha=0.55, zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(grupos, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Coeficiente de Spearman con la respuesta")
    ax.set_title(f"Asociación por región (azul) y por bloque (verde), {rama}_DATASET")
    save(fig, fig_dir, f"F3_{rama}_FIG11_asociacion_por_region_bloque.png")


def fig12_magnitud_estabilidad(master, fig_dir, rama):
    d = master.dropna(subset=["ABS_RHO", "LOO_MAX_CHANGE"])
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for n, c, lab in ((7, C7, "N efectivo = 7"), (5, C5, "N efectivo = 5")):
        sub = d[d.N_EFFECTIVE == n]
        ax.scatter(sub.ABS_RHO, sub.LOO_MAX_CHANGE, s=45,
                   color=c, alpha=0.85, edgecolor="k", linewidth=0.4, label=lab)
    for _, r in d.nlargest(6, "ABS_RHO").iterrows():
        ax.annotate(r.name, (r.ABS_RHO, r.LOO_MAX_CHANGE), xytext=(5, 4),
                    textcoords="offset points", fontsize=6)
    ax.set_xlabel("Magnitud de la asociación, |rho| completo")
    ax.set_ylabel("Cambio máximo del coeficiente al omitir una variante")
    ax.set_title(f"Magnitud frente a estabilidad, {rama}_DATASET\n"
                 f"Sin clasificación categórica: ambas dimensiones se reportan de forma continua")
    ax.legend(frameon=False, fontsize=8)
    save(fig, fig_dir, f"F3_{rama}_FIG12_magnitud_vs_estabilidad.png")

RAMA = "MEC"
BASE = Path(r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-6\Nuevo_Flujo\5.FASE 3_ANALISIS\2.MEC")
PREFIX_DATASET = "F2_I4_MEC_DATASET"
PREFIX_DICT = "F2_I4_MEC_DICTIONARY"


def parse_args():
    p = argparse.ArgumentParser(description="Fase III, rama MEC")
    p.add_argument("--input-dir", type=Path, default=BASE / "00_DATASET")
    p.add_argument("--output-dir", type=Path, default=BASE / "02_TXT")
    p.add_argument("--figures-dir", type=Path, default=BASE / "01_FIGURAS")
    return p.parse_args()


#==============================================================================
# MÓDULO 2. CARGA Y AUDITORÍA
# Objetivo: leer la matriz y su diccionario, y verificar su coherencia.
# Inputs: carpeta 00_DATASET.
# Proceso: resuelve los archivos por prefijo, valida las 7 variantes y las dos
#          respuestas, comprueba que cada representante tenga entrada en el
#          diccionario y verifica la invariancia de Spearman frente a la
#          transformación logarítmica de la respuesta (control de V8).
# Outputs: Y, P, diccionario indexado por variable y tabla de controles.
#==============================================================================
def load_base(args, log):
    in_dir = args.input_dir
    if not in_dir.is_dir():
        log.error(f"INPUT_DIR no existe: {in_dir}")
        sys.exit(1)
    f_data = resolve(in_dir, PREFIX_DATASET, log)
    f_dic = resolve(in_dir, PREFIX_DICT, log)
    X = pd.read_csv(f_data, index_col=0, encoding="utf-8-sig")
    X.index = X.index.astype(str)
    X = X.reindex(VAR_ORDER)
    if X.index.isna().any() or len(X) != N_OBS:
        raise ValueError("La matriz no contiene exactamente WT y M1 a M6")
    if Y_PRIMARIA not in X.columns:
        raise ValueError(f"Falta la respuesta primaria {Y_PRIMARIA}")
    ycols = [c for c in (Y_LINEAL, Y_PRIMARIA) if c in X.columns]
    Y = X[ycols].astype(float)
    P = X.drop(columns=ycols).apply(pd.to_numeric, errors="coerce")
    dic = pd.read_csv(f_dic, encoding="utf-8-sig", dtype=str).fillna("").set_index("VAR_NAME")
    falta = set(P.columns) - set(dic.index)
    if falta:
        raise ValueError(f"Representantes sin entrada en el diccionario: {sorted(falta)[:5]}")
    dic = dic.loc[list(P.columns)].copy()
    log.info(f"  Representantes: {P.shape[1]} | respuestas: {ycols}")
    log.info(f"  Respuesta primaria declarada: {Y_PRIMARIA}")
    log.info(f"  Cobertura: N efectivo 7 en {int((P.notna().all()).sum())}, "
             f"N efectivo 5 en {int((~P.notna().all()).sum())}")
    for marca in ("SINGLETON_FORCED",):
        n = int((dic.DESEMPATE == marca).sum())
        log.info(f"  Marca heredada {marca}: {n} representantes")
    for marca in ("WT_SIGNATURE", "CIRC_RISK"):
        n = int(pd.to_numeric(dic[marca], errors="coerce").fillna(0).sum())
        log.info(f"  Marca heredada {marca}: {n} representantes")
    return Y, P, dic


def control_invariancia(P, Y, log):
    """Control computacional de V8: Spearman y Kendall son invariantes a la
    transformación logarítmica de la respuesta, Pearson no. Se verifica en una
    muestra determinista de variables y se registra."""
    if Y_LINEAL not in Y.columns:
        return pd.DataFrame()
    filas = []
    for c in sorted(P.columns)[:10]:
        v = P[c].dropna()
        if v.nunique() < 2:
            continue
        r1 = spearman(v.values, Y.loc[v.index, Y_PRIMARIA].values)
        r2 = spearman(v.values, Y.loc[v.index, Y_LINEAL].values)
        filas.append(dict(CONTROL="INVARIANCIA_SPEARMAN", VARIABLE=c,
                          RHO_LOG2=round(r1, 6), RHO_LINEAL=round(r2, 6),
                          COINCIDE=int(abs(r1 - r2) < TOL)))
    t = pd.DataFrame(filas)
    if len(t) and not t.COINCIDE.all():
        log.error("  CONTROL FALLIDO: Spearman difiere entre las dos respuestas")
        sys.exit(1)
    log.info(f"  [control] Spearman invariante a la transformación logarítmica en "
             f"{len(t)} variables verificadas")
    return t


#==============================================================================
# MÓDULO 8. INFERENCIA PERMUTACIONAL CON DESCRIPTOR FIJO (III-3 A)
# Objetivo: situar cada coeficiente observado frente a su distribución nula exacta.
# Inputs: matriz de representantes y respuesta primaria.
# Proceso: los representantes de esta rama se eligieron sin consultar la
#          respuesta, de modo que cada descriptor permanece fijo y solo se permuta
#          la respuesta entre las variantes realmente disponibles para él: 5040
#          permutaciones con N efectivo 7 y 120 con N efectivo 5. El valor p es la
#          proporción de permutaciones cuyo |rho| iguala o supera al observado.
#          Como la permutación identidad pertenece a la enumeración y la
#          comparación usa el operador de mayor o igual, ningún p puede ser cero.
# Outputs: tabla de inferencia y tabla de controles.
#==============================================================================
def inferencia(P, y, asoc, log):
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
        id_pos = perms.index(tuple(range(n)))
        log.info(f"  [nulo] N efectivo {n}: {len(cols)} variables, {len(perms)} permutaciones")
        for c in cols:
            v = P.loc[filas_idx, c]
            if v.nunique() < 2:
                filas.append(dict(VARIABLE=c, N_EFFECTIVE=n, N_PERM=len(perms),
                                  NULL_SCHEME="FIJO", P_EXACT=np.nan, NULL_P95=np.nan))
                continue
            zx = ranks_std(v.values)
            nulo = np.abs(Zy @ zx) / (n - 1)
            rho_obs = asoc.at[c, "RHO_RAW"]   # sin redondear: el nulo y el observado deben compartir aritmética
            control.append(dict(CONTROL="IDENTIDAD", VARIABLE=c,
                                RHO_OBSERVADO=round(float(rho_obs), 6),
                                RHO_IDENTIDAD=round(float((Zy[id_pos] @ zx) / (n - 1)), 6),
                                COINCIDE=int(abs((Zy[id_pos] @ zx) / (n - 1) - rho_obs) < 1e-8)))
            filas.append(dict(VARIABLE=c, N_EFFECTIVE=n, N_PERM=len(perms),
                              NULL_SCHEME="FIJO",
                              P_EXACT=round(float(np.mean(nulo >= abs(rho_obs) - TOL)), 6),
                              NULL_P95=round(float(np.percentile(nulo, 95)), 4)))
    inf = pd.DataFrame(filas).set_index("VARIABLE")
    ctrl = pd.DataFrame(control)
    if len(ctrl) and not ctrl.COINCIDE.all():
        log.error("  CONTROL FALLIDO: la identidad no reproduce el coeficiente observado")
        sys.exit(1)
    if (inf.P_EXACT.dropna() <= 0).any():
        log.error("  CONTROL FALLIDO: hay valores p exactos iguales a cero")
        sys.exit(1)
    log.info(f"  [control] la identidad reproduce el coeficiente observado en {len(ctrl)} variables")
    log.info(f"  Inferencia: {int((inf.P_EXACT <= 0.05).sum())} variables con p exacto <= 0.05 "
             f"de {int(inf.P_EXACT.notna().sum())} | p mínimo = {inf.P_EXACT.min():.6f}")
    return inf, ctrl


#==============================================================================
# MÓDULO 11 Y 13. INTEGRACIÓN, FIGURAS Y EXPORTACIÓN
# Objetivo: construir la tabla maestra, generar las figuras y escribir todo.
# Inputs: resultados de los módulos anteriores.
# Proceso: une descriptiva, asociación, inferencia, multiplicidad, dejar una
#          fuera y sensibilidad a WT en una sola tabla por variable, con el
#          linaje completo y las marcas heredadas. No se asigna ninguna
#          clasificación categórica de robustez: las métricas se reportan de
#          forma continua.
# Outputs: doce tablas y doce figuras.
#==============================================================================
def main():
    args = parse_args()
    out, figd = args.output_dir, args.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out, RAMA)
    log.info(f"Fase III, rama {RAMA} | {VERSION} | fecha {DATE_TAG} | pandas {pd.__version__}")
    log.info("  Con N = 7 el análisis es exploratorio: no es modelo predictivo ni prueba causal")

    log.info("MÓDULO 2 | Carga y auditoría")
    Y, P, dic = load_base(args, log)
    y = Y[Y_PRIMARIA]
    ctrl_inv = control_invariancia(P, Y, log)

    log.info("MÓDULO 3 | Estadística descriptiva (III-1)")
    des = descriptiva(P, dic, log)

    log.info("MÓDULO 4 | Redundancia residual")
    R, red = redundancia(P, log)

    log.info("MÓDULO 5-6 | Asociación primaria y de sensibilidad (III-2)")
    asoc = asociacion(P, y, log)

    log.info("MÓDULO 8 | Inferencia permutacional (III-3)")
    inf, ctrl_id = inferencia(P, y, asoc, log)

    log.info("MÓDULO 7 | Multiplicidad")
    inf["Q_BH"] = benjamini_hochberg(inf.P_EXACT.values)
    for t in NIVELES_REFERENCIA_FDR:
        log.info(f"  Nivel de referencia {t}: p <= {t} en {int((inf.P_EXACT <= t).sum())} variables, "
                 f"q <= {t} en {int((inf.Q_BH <= t).sum())}")
    log.info(f"  q mínimo observado con estos valores p: {inf.Q_BH.min():.4f}. "
             "Depende del conjunto completo de valores p y de su orden, no solo del mínimo, "
             "y se reporta como diagnóstico de multiplicidad, no como filtro")

    log.info("MÓDULO 9 | Dejar una fuera (III-4 A)")
    loo = leave_one_out(P, y, asoc, log)

    log.info("MÓDULO 10 | Sensibilidad a WT (III-4 B)")
    wt = sensibilidad_wt(P, y, asoc, log)

    log.info("MÓDULO 11 | Tabla maestra")
    master = des.join(asoc.drop(columns=["N_EFFECTIVE", "RHO_RAW"])) \
                .join(inf.drop(columns=[c for c in ("N_EFFECTIVE", "MODULE_ID", "MODULE_SIZE")
                                       if c in inf.columns])) \
                .join(loo.drop(columns=["N_EFFECTIVE", "SPEARMAN_RHO"])) \
                .join(wt.drop(columns=["SPEARMAN_RHO"]))
    for col, src in (("SELECTION_ALGORITHM", "ALGORITMO"), ("TIE_BREAK", "DESEMPATE"),
                     ("ALIAS_EQUIVALENTES", "ALIAS_EQUIVALENTES")):
        master[col] = master.index.map(dic[src])
    master["SINGLETON_FORCED"] = (master.TIE_BREAK == "SINGLETON_FORCED").astype(int)
    for marca in ("WT_SIGNATURE", "CIRC_RISK"):
        master[marca] = master.index.map(pd.to_numeric(dic[marca], errors="coerce")).fillna(0).astype(int)
    master["DATASET"] = RAMA
    log.info(f"  Tabla maestra: {master.shape[0]} filas x {master.shape[1]} columnas")

    log.info("MÓDULO 12 | Figuras")
    fig01_composicion(des, figd, RAMA)
    fig02_perfil(des, figd, RAMA)
    fig03_heatmap(P, des, y, figd, RAMA)
    fig04_correlacion(R, figd, RAMA)
    fig05_asociacion(master, figd, RAMA)
    fig06_kendall(master, figd, RAMA)
    fig07_pearson(master, figd, RAMA)
    fig08_loo(master, figd, RAMA)
    fig09_wt(master, figd, RAMA)
    fig10_fdr(master, figd, RAMA)
    fig11_por_grupo(master, figd, RAMA)
    fig12_magnitud_estabilidad(master, figd, RAMA)

    log.info("MÓDULO 13 | Exportación")
    master.rename_axis("VARIABLE").to_csv(out / f"F3_{RAMA}_MASTER_TABLE_{VERSION}.csv", encoding="utf-8-sig")
    des.rename_axis("VARIABLE").to_csv(out / f"F3_{RAMA}_DESCRIPTIVE_{VERSION}.csv", encoding="utf-8-sig")
    asoc.rename_axis("VARIABLE").to_csv(out / f"F3_{RAMA}_ASSOCIATION_{VERSION}.csv", encoding="utf-8-sig")
    inf.rename_axis("VARIABLE").to_csv(out / f"F3_{RAMA}_INFERENCE_{VERSION}.csv", encoding="utf-8-sig")
    loo.rename_axis("VARIABLE").to_csv(out / f"F3_{RAMA}_LOO_{VERSION}.csv", encoding="utf-8-sig")
    wt.rename_axis("VARIABLE").to_csv(out / f"F3_{RAMA}_WT_SENSITIVITY_{VERSION}.csv", encoding="utf-8-sig")
    R.rename_axis("VARIABLE").to_csv(out / f"F3_{RAMA}_RESIDUAL_CORR_{VERSION}.csv", encoding="utf-8-sig")
    red.to_csv(out / f"F3_{RAMA}_REDUNDANCY_SUMMARY_{VERSION}.csv", index=False, encoding="utf-8-sig")
    pd.concat([ctrl_inv, ctrl_id], ignore_index=True).to_csv(
        out / f"F3_{RAMA}_CONTROLS_{VERSION}.csv", index=False, encoding="utf-8-sig")

    log.info(f"RESUMEN | {RAMA}: {P.shape[1]} representantes | "
             f"|rho| máximo={master.ABS_RHO.max():.3f} | p <= 0.05: {int((master.P_EXACT <= 0.05).sum())} "
             f"| q mínimo={master.Q_BH.min():.3f} | signo conservado en las 7 réplicas: "
             f"{int((master.LOO_SIGN_STABILITY == 1).sum())}")
    log.info("Los representantes de esta rama se eligieron sin usar la respuesta: la inferencia "
             "con descriptor fijo es la que corresponde a su construcción.")


if __name__ == "__main__":
    main()
