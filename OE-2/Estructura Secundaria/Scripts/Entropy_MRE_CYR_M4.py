#!/usr/bin/env python
# coding: utf-8

# ============================================================
# OE2 — Extracción de entropía posicional desde RNAfold
# Proyecto: Análisis computacional Cyrano / miR-7-5p / TDMD
# Autor: Martin
# Fecha: 27/04/2026
# Herramienta fuente: RNAfold 2.6.3 (ViennaRNA)
# Referencia: Lorenz R et al. Algorithms Mol. Biol. 2011, 6:26
# ============================================================


# ============================================================
# MÓDULO 1 — IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import re
import math
import numpy as np
import os


# ============================================================
# MÓDULO 2 — PARÁMETROS DEL ANÁLISIS
# Para procesar otra variante: cambia solo este bloque.
# ============================================================

ID_VAR    = "CYR_M4"
SECUENCIA = "AACAACAAAAUCAUCUUCCA"    # 20 nt

ARCHIVO_EPS = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-2\Var_E-Sec-CYR\RNAfold\5. Rfold_MRE_CYR_M4\CYR_M4_dp.eps"

# Parámetros globales de la hebra — obtenidos del output RNAfold WebServer
# Fuente: RNAfold_web_server_MRE_CYR_M4.pdf (27/04/2026)
MFE_KCAL           = 0.00       # kcal/mol — energía libre mínima
DG_ENSEMBLE        = -0.00      # kcal/mol — energía libre del ensemble termodinámico
FREQ_MFE           = 99.44      # % — frecuencia de la estructura MFE en el ensemble
ENSEMBLE_DIVERSITY = 0.02       # diversidad del ensemble termodinámico
DOT_BRACKET        = "...................."  # estructura MFE (20 nt, todos desapareados)
CARPETA_OUTPUT     = r"C:\Users\cmart\OneDrive\Documentos OneDrive\1. circRNA_TESIS\Tesis\7.Metodo\OE-2\Var_E-Sec-CYR\RNAfold\5. Rfold_MRE_CYR_M4\Py_output"

# Regiones funcionales del dúplex bipartito (posiciones 1-based)
# Basado en Sheu-Gruttadauria et al. Mol. Cell 2019 + Kleaveland et al. 2018
# ANCHOR: posición 1 — contacto con AGO (región independiente desde M1)
# BULGE: ausente en M4 — registrado como (0,0)/NA para consistencia de matriz
REGION_ANCHOR = (1, 1)    # t1:   contacto AGO
REGION_SEED   = (2, 7)    # t2–t7: complementaria al seed de miR-7-5p → P1
REGION_BULGE  = (0, 0)    # AUSENTE — placeholder para consistencia en Excel
REGION_3P     = (8, 20)   # t8–t20: complementaria a región 3' del miRNA → P2


# ============================================================
# MÓDULO 3 — LECTURA Y PARSEO DEL ARCHIVO .eps
# Formato RNAfold dot plot:
#   i  j  sqrt(prob_ij)  ubox
#   prob_ij = sqrt(valor)^2
# ============================================================

def leer_probabilidades_pares(archivo_eps, n_seq):
    """
    Lee el archivo .eps del dot plot de RNAfold y extrae
    la matriz de probabilidades de pares de bases.

    Parámetros:
        archivo_eps (str): ruta al archivo .eps
        n_seq (int): longitud de la secuencia

    Retorna:
        prob_matrix (np.ndarray): matriz (n+1 x n+1), indexada desde 1
    """

    if not os.path.exists(archivo_eps):
        raise FileNotFoundError(
            f"No se encontró el archivo: {archivo_eps}\n"
            f"Verifica la ruta o descarga el archivo dp.eps del servidor RNAfold."
        )

    with open(archivo_eps, 'r', encoding='latin-1') as f:
        content = f.read()

    patron = r'(\d+)\s+(\d+)\s+([\d.eE+\-]+)\s+ubox'
    pares   = re.findall(patron, content)

    if not pares:
        raise ValueError(
            f"No se encontraron datos de probabilidades en {archivo_eps}.\n"
            f"Verifica que el archivo sea el dot plot (_dp.eps) de RNAfold."
        )

    prob_matrix = np.zeros((n_seq + 1, n_seq + 1))

    for i_str, j_str, sqrt_prob_str in pares:
        i    = int(i_str)
        j    = int(j_str)
        prob = float(sqrt_prob_str) ** 2
        prob_matrix[i][j] = prob
        prob_matrix[j][i] = prob

    print(f"[✓] Archivo leído: {archivo_eps}")
    print(f"[✓] Pares de bases encontrados en el ensemble: {len(pares)}")

    return prob_matrix


# ============================================================
# MÓDULO 4 — CÁLCULO DE PROB_PAIR Y ACC_POS POR POSICIÓN
# prob_pair(i) = Σⱼ P(i,j)  para j ≠ i
# ACC_POS(i)   = 1 − prob_pair(i)
# ============================================================

def calcular_prob_pair_y_acc(prob_matrix, n_seq):
    """
    Calcula probabilidad de apareamiento total y accesibilidad
    para cada nucleótido.

    Retorna:
        prob_pair (list): P(apareado) por posición (1-based)
        acc_pos   (list): P(desapareado) por posición (1-based)
    """

    prob_pair = []
    acc_pos   = []

    for i in range(1, n_seq + 1):
        p_paired   = sum(prob_matrix[i][j] for j in range(1, n_seq + 1) if j != i)
        p_paired   = min(p_paired, 1.0)
        p_unpaired = 1.0 - p_paired

        prob_pair.append(p_paired)
        acc_pos.append(p_unpaired)

    return prob_pair, acc_pos


# ============================================================
# MÓDULO 5 — CÁLCULO DE ENTROPÍA POSICIONAL (Shannon)
#
#   S(i) = − Σⱼ P(i,j)·log₂[P(i,j)] − P_unpaired(i)·log₂[P_unpaired(i)]
#
# Rango normalizado por RNAfold: [0, 1]
#   0 = estado conformacional perfectamente definido
#   1 = máxima ambigüedad — distribuido entre múltiples conformaciones
#
# Referencia: Lorenz R et al. Algorithms Mol. Biol. 2011, 6:26
# ============================================================

def calcular_entropia_posicional(prob_matrix, n_seq, secuencia):
    """
    Calcula la entropía posicional de Shannon por nucleótido.

    Retorna:
        entropia (list): S(i) por posición (1-based)
    """

    entropia = []

    for i in range(1, n_seq + 1):
        S = 0.0

        for j in range(1, n_seq + 1):
            if j == i:
                continue
            p_ij = prob_matrix[i][j]
            if p_ij > 0:
                S -= p_ij * math.log2(p_ij)

        p_paired_total = sum(prob_matrix[i][j] for j in range(1, n_seq + 1) if j != i)
        p_paired_total = min(p_paired_total, 1.0)
        p_unpaired     = 1.0 - p_paired_total

        if p_unpaired > 0:
            S -= p_unpaired * math.log2(p_unpaired)

        entropia.append(S)

    return entropia


# ============================================================
# MÓDULO 6 — INTERPRETACIÓN DE COLOR (escala RNAfold)
# Mapeo del valor de entropía al color del gráfico circular
# ============================================================

def interpretar_color_entropia(valor_entropia):
    """
    Retorna color e interpretación conformacional según ENT_POS.
    """

    if valor_entropia < 0.05:
        return "ROJO INTENSO",    "Estado perfectamente definido — siempre ss en el ensemble"
    elif valor_entropia < 0.15:
        return "ROJO-NARANJA",    "Estado bien definido — casi siempre ss"
    elif valor_entropia < 0.30:
        return "NARANJA",         "Mayormente definido — baja ambigüedad"
    elif valor_entropia < 0.50:
        return "NARANJA-AMARILLO","Moderadamente ambiguo"
    elif valor_entropia < 0.65:
        return "AMARILLO-VERDE",  "Ambiguo — participa en varias estructuras"
    elif valor_entropia < 0.80:
        return "VERDE",           "Muy ambiguo — alta variabilidad conformacional"
    elif valor_entropia < 0.90:
        return "AZUL CLARO",      "Extremadamente ambiguo"
    else:
        return "AZUL OSCURO/MORADO", "Entropía máxima — múltiples conformaciones"


# ============================================================
# MÓDULO 7 — ASIGNACIÓN DE REGIÓN POR POSICIÓN
# Incluye ANCHOR como región independiente (desde M1 en adelante)
# ============================================================

def asignar_region(i, region_anchor, region_seed, region_bulge, region_3p):
    """
    Retorna la etiqueta de región funcional para la posición i.
    Si region_bulge == (0,0), BULGE se omite de la asignación.
    """

    if region_anchor[0] <= i <= region_anchor[1]:
        return "ANCHOR"
    elif region_seed[0] <= i <= region_seed[1]:
        return "SEED"
    elif region_bulge != (0, 0) and region_bulge[0] <= i <= region_bulge[1]:
        return "BULGE"
    elif region_3p[0] <= i <= region_3p[1]:
        return "3P"
    else:
        return "—"


# ============================================================
# MÓDULO 8 — CÁLCULO DE VARIABLES POR REGIÓN FUNCIONAL
# Calcula promedios de ACC y ENT por región:
#   ANCHOR, SEED, BULGE, 3P, GLOBAL
# ============================================================

def calcular_variables_regionales(valores, region_anchor, region_seed,
                                   region_bulge, region_3p):
    """
    Calcula el promedio de una variable por región funcional.
    Si region_bulge == (0,0), retorna None para BULGE (región ausente).
    """

    def promedio(inicio, fin):
        if inicio == 0 and fin == 0:
            return None
        subset = valores[inicio-1 : fin]
        return sum(subset) / len(subset) if subset else 0.0

    return {
        "ANCHOR": promedio(*region_anchor),
        "SEED":   promedio(*region_seed),
        "BULGE":  promedio(*region_bulge),
        "3P":     promedio(*region_3p),
        "GLOBAL": sum(valores) / len(valores)
    }


# ============================================================
# MÓDULO 9 — REPORTE DE RESULTADOS
# ============================================================

def imprimir_reporte(id_var, secuencia, prob_pair, acc_pos, entropia,
                     region_anchor, region_seed, region_bulge, region_3p):

    n = len(secuencia)

    print("\n" + "="*75)
    print(f"  REPORTE OE2 — VAR_EST_SEC — {id_var}")
    print(f"  Secuencia: {secuencia}")
    print(f"  Longitud:  {n} nt")
    print("="*75)

    # ── Tabla posición a posición ──────────────────────────────────────────
    print(f"\n{'Pos':>4} {'Nt':>3} {'Region':>8} {'PROB_PAIR':>10} {'ACC_POS':>8} {'ENT_POS':>8}  Color (entropía)")
    print("-"*80)

    for i in range(1, n+1):
        nt     = secuencia[i-1]
        region = asignar_region(i, region_anchor, region_seed, region_bulge, region_3p)
        pp     = prob_pair[i-1]
        acc    = acc_pos[i-1]
        ent    = entropia[i-1]
        color, _ = interpretar_color_entropia(ent)

        print(f"{i:>4} {nt:>3} {region:>8} {pp:>10.6f} {acc:>8.6f} {ent:>8.6f}  {color}")

    # ── Variables por región ───────────────────────────────────────────────
    vars_acc = calcular_variables_regionales(
        acc_pos, region_anchor, region_seed, region_bulge, region_3p)
    vars_ent = calcular_variables_regionales(
        entropia, region_anchor, region_seed, region_bulge, region_3p)

    print("\n" + "="*75)
    print("  VARIABLES A REGISTRAR EN MATRIZ MAESTRA (Excel)")
    print("="*75)

    print(f"\n  ── Accesibilidad ──────────────────────────────────────────────────")
    print(f"  ACC_SITE   (global)   = {vars_acc['GLOBAL']:.6f}")
    print(f"  ACC_ANCHOR (t1)       = {vars_acc['ANCHOR']:.6f}")
    print(f"  ACC_SEED   (t2–t{REGION_SEED[1]})    = {vars_acc['SEED']:.6f}")
    bulge_acc = vars_acc['BULGE']
    print(f"  ACC_BULGE             = {'NA (ausente)' if bulge_acc is None else f'{bulge_acc:.6f}'}")
    print(f"  ACC_3P     (t{REGION_3P[0]}–t{REGION_3P[1]})  = {vars_acc['3P']:.6f}")

    print(f"\n  ── Entropía posicional ────────────────────────────────────────────")
    print(f"  ENT_SITE   (global)   = {vars_ent['GLOBAL']:.6f}")
    print(f"  ENT_ANCHOR (t1)       = {vars_ent['ANCHOR']:.6f}")
    print(f"  ENT_SEED   (t2–t{REGION_SEED[1]})    = {vars_ent['SEED']:.6f}")
    bulge_ent = vars_ent['BULGE']
    print(f"  ENT_BULGE             = {'NA (ausente)' if bulge_ent is None else f'{bulge_ent:.6f}'}")
    print(f"  ENT_3P     (t{REGION_3P[0]}–t{REGION_3P[1]})  = {vars_ent['3P']:.6f}")

    # ── Outliers entropía ──────────────────────────────────────────────────
    print(f"\n  ── Posiciones con ENT_POS > 0.30 (ambigüedad conformacional) ──────")
    outliers = [(i+1, secuencia[i], entropia[i]) for i in range(n) if entropia[i] > 0.30]
    if outliers:
        for pos, nt, ent in sorted(outliers, key=lambda x: -x[2]):
            color, interp = interpretar_color_entropia(ent)
            print(f"  Posición {pos:2d} ({nt}): ENT = {ent:.6f} → {color}")
    else:
        print("  Ninguna posición supera ENT = 0.30 — sitio completamente definido")

    # ── Bitácora ───────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("  BITÁCORA — Parámetros de ejecución")
    print("="*75)
    print(f"  ID_VAR:        {id_var}")
    print(f"  HERRAMIENTA:   RNAfold WebServer (ViennaRNA)")
    print(f"  VERSION:       RNAfold 2.6.3")
    print(f"  PARAMETROS:    -p -d2 --noLP (default del servidor)")
    print(f"  ARCHIVO_EPS:   {ARCHIVO_EPS}")
    print(f"  ENERGIA_PARAMS: Turner 2004 (Mathews DH et al. PNAS 101:7287)")
    print("="*75 + "\n")


# ============================================================
# MÓDULO 10 — EXPORTAR CSV POSICIONES (por nucleótido)
# Salida: OE2_SEC_CYR_M1_posiciones.csv
# Columnas: ID_VAR, Posicion, Nucleotido, Region,
#           PROB_PAIR, ACC_POS, ENT_POS
# ============================================================

def exportar_csv_posiciones(id_var, secuencia, prob_pair, acc_pos, entropia,
                            region_anchor, region_seed, region_bulge, region_3p):
    """
    CSV 1 — Por nucleótido.
    Columnas: ID_VAR, Posicion, Nucleotido, Region, PROB_PAIR, ACC_POS, ENT_POS
    """
    os.makedirs(CARPETA_OUTPUT, exist_ok=True)
    ruta = os.path.join(CARPETA_OUTPUT, f"OE2_SEC_{id_var}_posiciones.csv")
    n    = len(secuencia)

    with open(ruta, 'w', encoding='utf-8') as f:
        f.write("ID_VAR,Posicion,Nucleotido,Region,PROB_PAIR,ACC_POS,ENT_POS\n")
        for i in range(1, n+1):
            nt     = secuencia[i-1]
            region = asignar_region(i, region_anchor, region_seed, region_bulge, region_3p)
            f.write(f"{id_var},{i},{nt},{region},"
                    f"{prob_pair[i-1]:.6f},{acc_pos[i-1]:.6f},{entropia[i-1]:.6f}\n")

    print(f"[✓] CSV posiciones   : {ruta}")
    return ruta


# ============================================================
# MÓDULO 11 — EXPORTAR CSV REGIONES
# CSV 2 — Promedios por región funcional (ANCHOR/SEED/BULGE/3P/GLOBAL)
# Columnas: ID_VAR, Region, Nt_inicio, Nt_fin, N_nt,
#           ACC_REGION, ENT_REGION
# ============================================================

def exportar_csv_regiones(id_var, acc_pos, entropia,
                          region_anchor, region_seed, region_bulge, region_3p):
    """
    CSV 2 — Promedios de ACC y ENT por región estructural.
    """
    os.makedirs(CARPETA_OUTPUT, exist_ok=True)
    ruta = os.path.join(CARPETA_OUTPUT, f"OE2_SEC_{id_var}_regiones.csv")

    vars_acc = calcular_variables_regionales(acc_pos,   region_anchor, region_seed, region_bulge, region_3p)
    vars_ent = calcular_variables_regionales(entropia,  region_anchor, region_seed, region_bulge, region_3p)

    regiones = [
        ("ANCHOR", region_anchor),
        ("SEED",   region_seed),
        ("BULGE",  region_bulge),
        ("3P",     region_3p),
        ("GLOBAL", (1, len(acc_pos))),
    ]

    with open(ruta, 'w', encoding='utf-8') as f:
        f.write("ID_VAR,Region,Nt_inicio,Nt_fin,N_nt,ACC_REGION,ENT_REGION\n")
        for nombre, (ini, fin) in regiones:
            if ini == 0 and fin == 0:
                f.write(f"{id_var},{nombre},NA,NA,NA,NA,NA\n")
            else:
                n_nt = fin - ini + 1
                acc_val = vars_acc[nombre]
                ent_val = vars_ent[nombre]
                acc_str = f"{acc_val:.6f}" if acc_val is not None else "NA"
                ent_str = f"{ent_val:.6f}" if ent_val is not None else "NA"
                f.write(f"{id_var},{nombre},{ini},{fin},{n_nt},{acc_str},{ent_str}\n")

    print(f"[✓] CSV regiones     : {ruta}")
    return ruta


# ============================================================
# MÓDULO 12 — EXPORTAR CSV HEBRA GLOBAL
# CSV 3 — Variables termodinámicas globales de la hebra
# Columnas: ID_VAR, VARIABLE, VALOR, UNIDAD, FUENTE
# ============================================================

def exportar_csv_hebra(id_var):
    """
    CSV 3 — Variables globales de la hebra completa desde RNAfold.
    Incluye MFE, FREQ_MFE, ENSEMBLE_DIVERSITY, DOT_BRACKET.
    """
    os.makedirs(CARPETA_OUTPUT, exist_ok=True)
    ruta = os.path.join(CARPETA_OUTPUT, f"OE2_SEC_{id_var}_hebra.csv")

    filas = [
        ("MFE",                MFE_KCAL,           "kcal/mol", "RNAfold output"),
        ("DG_ENSEMBLE",        DG_ENSEMBLE,         "kcal/mol", "RNAfold output"),
        ("FREQ_MFE",           FREQ_MFE,            "%",        "RNAfold output"),
        ("ENSEMBLE_DIVERSITY", ENSEMBLE_DIVERSITY,  "adim",     "RNAfold output"),
        ("DOT_BRACKET",        DOT_BRACKET,         "—",        "RNAfold output"),
        ("LONGITUD",           len(SECUENCIA),      "nt",       "Secuencia input"),
        ("SECUENCIA",          SECUENCIA,           "—",        "Kleaveland 2018"),
    ]

    with open(ruta, 'w', encoding='utf-8') as f:
        f.write("ID_VAR,VARIABLE,VALOR,UNIDAD,FUENTE\n")
        for variable, valor, unidad, fuente in filas:
            f.write(f"{id_var},{variable},{valor},{unidad},{fuente}\n")

    print(f"[✓] CSV hebra        : {ruta}")
    return ruta


# ============================================================
# MÓDULO 13 — EXPORTAR CSV COLORES
# CSV 4 — Por nucleótido: ENT_POS + color RNAfold + interpretación
# Columnas: ID_VAR, Posicion, Nucleotido, Region, ENT_POS,
#           COLOR_RNAFOLD, INTERPRETACION
# ============================================================

def exportar_csv_colores(id_var, secuencia, entropia,
                         region_anchor, region_seed, region_bulge, region_3p):
    """
    CSV 4 — Tabla de colores e interpretación conformacional por nucleótido.
    """
    os.makedirs(CARPETA_OUTPUT, exist_ok=True)
    ruta = os.path.join(CARPETA_OUTPUT, f"OE2_SEC_{id_var}_colores.csv")
    n    = len(secuencia)

    with open(ruta, 'w', encoding='utf-8') as f:
        f.write("ID_VAR,Posicion,Nucleotido,Region,ENT_POS,COLOR_RNAFOLD,INTERPRETACION\n")
        for i in range(1, n+1):
            nt     = secuencia[i-1]
            region = asignar_region(i, region_anchor, region_seed, region_bulge, region_3p)
            ent    = entropia[i-1]
            color, interp = interpretar_color_entropia(ent)
            f.write(f"{id_var},{i},{nt},{region},{ent:.6f},{color},\"{interp}\"\n")

    print(f"[✓] CSV colores      : {ruta}")
    return ruta


# ============================================================
# MÓDULO 14 — EXPORTAR REPORTE TXT
# Formato bitácora OE2 — estructura análoga al reporte FR3D
# ============================================================

def exportar_reporte_txt(id_var, secuencia, prob_pair, acc_pos, entropia,
                         region_anchor, region_seed, region_bulge, region_3p):
    """
    TXT — Reporte completo de resultados en formato bitácora.
    """
    from datetime import date

    os.makedirs(CARPETA_OUTPUT, exist_ok=True)
    ruta = os.path.join(CARPETA_OUTPUT, f"OE2_SEC_{id_var}_reporte.txt")
    n    = len(secuencia)
    hoy  = date.today().strftime("%d/%m/%Y")

    vars_acc = calcular_variables_regionales(acc_pos,  region_anchor, region_seed, region_bulge, region_3p)
    vars_ent = calcular_variables_regionales(entropia, region_anchor, region_seed, region_bulge, region_3p)

    lineas = []
    sep    = "=" * 70

    lineas += [
        sep,
        f"REPORTE DE ANÁLISIS ESTRUCTURAL SECUNDARIO — {id_var}",
        sep,
        "",
        f"HERRAMIENTA      : RNAfold WebServer (ViennaRNA Package)",
        f"VERSION          : RNAfold 2.6.3",
        f"FECHA            : {hoy}",
        f"INPUT            : {secuencia}  ({n} nt)",
        f"PARAMETROS       : -p -d2 --noLP (default del servidor WebServer)",
        f"OUTPUT_ARCHIVO   : {os.path.basename(ARCHIVO_EPS)}",
        "",
        "-" * 70,
        "VARIABLES GLOBALES DE LA HEBRA",
        "-" * 70,
        f"  VARIABLE_EXTRAIDA : MFE",
        f"  VALOR             : {MFE_KCAL} kcal/mol",
        f"  NOTAS             : Energía libre mínima — estructura completamente desapareada",
        "",
        f"  VARIABLE_EXTRAIDA : DG_ENSEMBLE",
        f"  VALOR             : {DG_ENSEMBLE:.2f} kcal/mol",
        f"  NOTAS             : El conjunto de conformaciones posibles presentan una estructura desapareada",
        "",
        f"  VARIABLE_EXTRAIDA : FREQ_MFE",
        f"  VALOR             : {FREQ_MFE} %",
        f"  NOTAS             : Frecuencia de la estructura MFE en el ensemble termodinámico",
        "",
        f"  VARIABLE_EXTRAIDA : ENSEMBLE_DIVERSITY",
        f"  VALOR             : {ENSEMBLE_DIVERSITY}",
        f"  NOTAS             : Diversidad del ensemble — valor cercano a 0 indica ensemble homogéneo",
        "",
        f"  VARIABLE_EXTRAIDA : DOT_BRACKET",
        f"  VALOR             : {DOT_BRACKET}",
        f"  NOTAS             : Todos los nucleótidos desapareados (solo puntos) — ssRNA puro",
        "",
        "-" * 70,
        "VARIABLES EXTRAIDAS — POR REGIÓN ESTRUCTURAL",
        "-" * 70,
    ]

    regiones_info = [
        ("ANCHOR", region_anchor, f"t{region_anchor[0]} — contacto AGO"),
        ("SEED",   region_seed,   f"t{region_seed[0]}–t{region_seed[1]} — complementaria seed miR-7-5p → P1"),
        ("BULGE",  region_bulge,  "AUSENTE en esta variante — placeholder (0,0)"),
        ("3P",     region_3p,     f"t{region_3p[0]}–t{region_3p[1]} — complementaria región 3' miRNA → P2"),
        ("GLOBAL", (1, n),        "hebra completa"),
    ]

    for nombre, (ini, fin), desc in regiones_info:
        if ini == 0 and fin == 0:
            lineas += [
                "",
                f"  REGIÓN            : {nombre}  ({desc})",
                f"  VARIABLE_EXTRAIDA : ACC_{nombre}",
                f"  VALOR             : NA",
                f"  VARIABLE_EXTRAIDA : ENT_{nombre}",
                f"  VALOR             : NA",
            ]
        else:
            acc_val = vars_acc[nombre]
            ent_val = vars_ent[nombre]
            acc_str = f"{acc_val:.6f}" if acc_val is not None else "NA"
            ent_str = f"{ent_val:.6f}" if ent_val is not None else "NA"
            lineas += [
                "",
                f"  REGIÓN            : {nombre}  (nt {ini}–{fin}, {desc})",
                f"  VARIABLE_EXTRAIDA : ACC_{nombre}",
                f"  VALOR             : {acc_str}",
                f"  VARIABLE_EXTRAIDA : ENT_{nombre}",
                f"  VALOR             : {ent_str}",
            ]

    lineas += [
        "",
        "-" * 70,
        "VARIABLES EXTRAIDAS — POR NUCLEÓTIDO",
        "-" * 70,
        f"  {'Pos':>4}  {'Nt':>2}  {'Region':>8}  {'PROB_PAIR':>10}  {'ACC_POS':>8}  {'ENT_POS':>8}  COLOR",
        "  " + "-" * 68,
    ]

    for i in range(1, n+1):
        nt     = secuencia[i-1]
        region = asignar_region(i, region_anchor, region_seed, region_bulge, region_3p)
        pp     = prob_pair[i-1]
        acc    = acc_pos[i-1]
        ent    = entropia[i-1]
        color, _ = interpretar_color_entropia(ent)
        lineas.append(f"  {i:>4}  {nt:>2}  {region:>8}  {pp:>10.6f}  {acc:>8.6f}  {ent:>8.6f}  {color}")

    outliers = [(i+1, secuencia[i], entropia[i]) for i in range(n) if entropia[i] > 0.30]
    lineas += [
        "",
        "-" * 70,
        "POSICIONES CON ENT_POS > 0.30 (ambigüedad conformacional)",
        "-" * 70,
    ]
    if outliers:
        for pos, nt, ent in sorted(outliers, key=lambda x: -x[2]):
            color, interp = interpretar_color_entropia(ent)
            lineas.append(f"  Posición {pos:2d} ({nt}): ENT = {ent:.6f} → {color} — {interp}")
    else:
        lineas.append("  Ninguna posición supera ENT = 0.30 — sitio estructuralmente definido")

    lineas += [
        "",
        "-" * 70,
        "ARCHIVOS DE SALIDA",
        "-" * 70,
        f"  CSV posiciones : OE2_SEC_{id_var}_posiciones.csv",
        f"  CSV regiones   : OE2_SEC_{id_var}_regiones.csv",
        f"  CSV hebra      : OE2_SEC_{id_var}_hebra.csv",
        f"  CSV colores    : OE2_SEC_{id_var}_colores.csv",
        f"  TXT reporte    : OE2_SEC_{id_var}_reporte.txt",
        "",
        "-" * 70,
        "NOTAS GENERALES",
        "-" * 70,
        "  - Probabilidades extraídas de la matriz dot plot (_dp.eps) via función de partición McCaskill.",
        "  - PROB_PAIR(i) = Σⱼ P(i,j); ACC_POS(i) = 1 − PROB_PAIR(i).",
        "  - ENT_POS calculada con fórmula de Shannon en bits (log₂), rango [0,1].",
        "  - Parámetros energéticos: Turner 2004 (Mathews DH et al. PNAS 101:7287).",
        "  - ANCHOR definida como región independiente desde variante M1 en adelante.",
        "",
        sep,
    ]

    with open(ruta, 'w', encoding='utf-8') as f:
        f.write("\n".join(lineas))

    print(f"[✓] TXT reporte      : {ruta}")
    return ruta


# ============================================================
# MÓDULO 15 — EJECUCIÓN PRINCIPAL
# Flujo: leer .eps → calcular → imprimir → exportar (4 CSV + 1 TXT)
# ============================================================

if __name__ == "__main__":

    N = len(SECUENCIA)

    # Paso 1: Leer probabilidades de pares desde el archivo .eps
    prob_matrix = leer_probabilidades_pares(ARCHIVO_EPS, N)

    # Paso 2: Calcular PROB_PAIR y ACC_POS por posición
    prob_pair, acc_pos = calcular_prob_pair_y_acc(prob_matrix, N)

    # Paso 3: Calcular entropía posicional por posición
    entropia = calcular_entropia_posicional(prob_matrix, N, SECUENCIA)

    # Paso 4: Imprimir reporte en consola
    imprimir_reporte(
        ID_VAR, SECUENCIA,
        prob_pair, acc_pos, entropia,
        REGION_ANCHOR, REGION_SEED, REGION_BULGE, REGION_3P
    )

    print("\n[→] Exportando archivos de salida...\n")

    # Paso 5: CSV 1 — por nucleótido
    exportar_csv_posiciones(
        ID_VAR, SECUENCIA, prob_pair, acc_pos, entropia,
        REGION_ANCHOR, REGION_SEED, REGION_BULGE, REGION_3P
    )

    # Paso 6: CSV 2 — promedios por región
    exportar_csv_regiones(
        ID_VAR, acc_pos, entropia,
        REGION_ANCHOR, REGION_SEED, REGION_BULGE, REGION_3P
    )

    # Paso 7: CSV 3 — variables globales de la hebra
    exportar_csv_hebra(ID_VAR)

    # Paso 8: CSV 4 — colores e interpretación conformacional
    exportar_csv_colores(
        ID_VAR, SECUENCIA, entropia,
        REGION_ANCHOR, REGION_SEED, REGION_BULGE, REGION_3P
    )

    # Paso 9: TXT — reporte completo bitácora
    exportar_reporte_txt(
        ID_VAR, SECUENCIA, prob_pair, acc_pos, entropia,
        REGION_ANCHOR, REGION_SEED, REGION_BULGE, REGION_3P
    )

    print("\n[✓] Pipeline OE2 completado — todos los archivos generados en Py_output\n")
