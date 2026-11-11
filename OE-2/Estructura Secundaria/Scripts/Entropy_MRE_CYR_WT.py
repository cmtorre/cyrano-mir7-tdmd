#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import numpy as np
import math


# In[3]:


# DEFINIR DIRECCIONES
# DEFINICIÓN DE RUTA PARA DATOS "MRE_Cyrano_W_ent"
MRE_Cyrano_W_ent=r"C:\Users\cmart\OneDrive\Escritorio\Pruebas_EPS_RNAfold\MRE_CYR_WT\MRE_Cyrano_W_ent.eps"


# In[4]:


# Leer el archivo .eps
with open('MRE_Cyrano_W_ent.eps', 'r', encoding='latin-1') as f:
    content = f.read()


# In[9]:


# ============================================================
# OE2 — Extracción de entropía posicional desde RNAfold
# Proyecto: Análisis computacional Cyrano / miR-7-5p / TDMD
# Autor: Martin
# Fecha: 14/04/2026
# Herramienta fuente: RNAfold 2.6.3 (ViennaRNA)
# Referencia: Lorenz R et al. Algorithms Mol. Biol. 2011, 6:26
# ============================================================


# ============================================================
# MÓDULO 1 — IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import re          # para parsear el archivo .eps con expresiones regulares
import math        # para calcular logaritmos en la entropía de Shannon
import numpy as np # para operaciones matriciales
import os          # para verificar que el archivo existe


# ============================================================
# MÓDULO 2 — PARÁMETROS DEL ANÁLISIS
# Define aquí la variante que estás analizando.
# Cambia estos valores para cada variante del corpus.
# ============================================================

# Identificador de la variante (para bitácora y nombres de archivo)
ID_VAR = "CYR_WT"

# Secuencia real de la variante (SEQ_REAL de OE1, sin gaps)
SECUENCIA = "AACAACAAAAUCACCAAUGUCUUCCA"

# Nombre del archivo .eps descargado del servidor RNAfold
# Usa el archivo MRE_[nombre]_dp.eps (dot plot = fuente de probabilidades de pares)
ARCHIVO_EPS = "MRE_Cyrano_W_dp.eps"

# Definición de regiones funcionales del dúplex bipartito
# Basado en Sheu-Gruttadauria et al. Mol. Cell 2019
# Posiciones 1-based sobre SEQ_REAL
REGION_SEED  = (1, 8)   # t1–t8: complementaria al seed de miR-7-5p → forma P1
REGION_BULGE = (9, 12)  # t9–t12: región del bulge central J
REGION_3P    = (13, 26) # t13–t26: complementaria a región 3' del miRNA → forma P2


# ============================================================
# MÓDULO 3 — LECTURA Y PARSEO DEL ARCHIVO .eps
# El archivo dot plot de RNAfold contiene las probabilidades
# de pares de bases en formato:
#   i  j  sqrt(prob_ij)  ubox
# donde prob_ij es la probabilidad de que las posiciones
# i y j estén apareadas en el ensemble termodinámico.
# ============================================================

def leer_probabilidades_pares(archivo_eps, n_seq):
    """
    Lee el archivo .eps del dot plot de RNAfold y extrae
    la matriz de probabilidades de pares de bases.

    Parámetros:
        archivo_eps (str): ruta al archivo .eps
        n_seq (int): longitud de la secuencia

    Retorna:
        prob_matrix (np.ndarray): matriz n x n de probabilidades de pares
                                  prob_matrix[i][j] = P(par i,j)
                                  indexada desde 1 (posición 0 no se usa)
    """

    # Verificar que el archivo existe
    if not os.path.exists(archivo_eps):
        raise FileNotFoundError(
            f"No se encontró el archivo: {archivo_eps}\n"
            f"Asegúrate de haber descargado el archivo dp.eps del servidor RNAfold."
        )

    # Leer el contenido del archivo
    with open(archivo_eps, 'r', encoding='latin-1') as f:
        content = f.read()

    # Parsear líneas con formato: i j sqrt(prob) ubox
    # La probabilidad real = sqrt(prob)^2 = valor^2
    patron = r'(\d+)\s+(\d+)\s+([\d.eE+\-]+)\s+ubox'
    pares = re.findall(patron, content)

    if not pares:
        raise ValueError(
            f"No se encontraron datos de probabilidades en {archivo_eps}.\n"
            f"Verifica que el archivo sea el dot plot (.dp.eps) de RNAfold."
        )

    # Construir la matriz de probabilidades (indexada desde 1)
    # Tamaño n+1 x n+1 para usar índices 1-based directamente
    prob_matrix = np.zeros((n_seq + 1, n_seq + 1))

    for i_str, j_str, sqrt_prob_str in pares:
        i = int(i_str)
        j = int(j_str)
        # Elevar al cuadrado para obtener probabilidad real
        prob = float(sqrt_prob_str) ** 2
        # La matriz es simétrica: P(i,j) = P(j,i)
        prob_matrix[i][j] = prob
        prob_matrix[j][i] = prob

    print(f"[✓] Archivo leído: {archivo_eps}")
    print(f"[✓] Pares de bases encontrados en el ensemble: {len(pares)}")

    return prob_matrix


# ============================================================
# MÓDULO 4 — CÁLCULO DE PROB_PAIR Y ACC_POS POR POSICIÓN
# Para cada nucleótido i:
#   prob_pair(i) = suma de P(i,j) para toda j ≠ i
#   ACC_POS(i)   = 1 − prob_pair(i)
# ============================================================

def calcular_prob_pair_y_acc(prob_matrix, n_seq):
    """
    Calcula la probabilidad de apareamiento total y la
    accesibilidad para cada nucleótido de la secuencia.

    Parámetros:
        prob_matrix (np.ndarray): matriz de probabilidades de pares
        n_seq (int): longitud de la secuencia

    Retorna:
        prob_pair (list): probabilidad de apareamiento por posición (1-based)
        acc_pos   (list): accesibilidad por posición (1-based)
    """

    prob_pair = []
    acc_pos   = []

    for i in range(1, n_seq + 1):
        # Suma de probabilidades de apareamiento con cualquier partner j
        p_paired = sum(prob_matrix[i][j] for j in range(1, n_seq + 1) if j != i)
        # Limitar a 1.0 por posibles errores de redondeo numérico
        p_paired = min(p_paired, 1.0)
        p_unpaired = 1.0 - p_paired

        prob_pair.append(p_paired)
        acc_pos.append(p_unpaired)

    return prob_pair, acc_pos


# ============================================================
# MÓDULO 5 — CÁLCULO DE ENTROPÍA POSICIONAL
# La entropía posicional de Shannon para la posición i es:
#
#   S(i) = − Σⱼ P(i,j) · log₂[P(i,j)] − P_unpaired(i) · log₂[P_unpaired(i)]
#
# Mide la incertidumbre conformacional de cada nucleótido
# en el ensemble termodinámico completo.
# Rango: 0 (posición perfectamente definida) a >1 (muy ambigua)
#
# Referencia: Lorenz R et al. Algorithms Mol. Biol. 2011, 6:26
# ============================================================

def calcular_entropia_posicional(prob_matrix, n_seq, secuencia):
    """
    Calcula la entropía posicional de Shannon para cada
    nucleótido de la secuencia.

    Parámetros:
        prob_matrix (np.ndarray): matriz de probabilidades de pares
        n_seq (int): longitud de la secuencia
        secuencia (str): secuencia de nucleótidos (para etiquetas)

    Retorna:
        entropia (list): entropía posicional por posición (1-based)
    """

    entropia = []

    for i in range(1, n_seq + 1):
        S = 0.0

        # Contribución de cada par posible (i,j)
        for j in range(1, n_seq + 1):
            if j == i:
                continue
            p_ij = prob_matrix[i][j]
            if p_ij > 0:
                S -= p_ij * math.log2(p_ij)

        # Contribución de estar no apareado
        p_paired_total = sum(prob_matrix[i][j] for j in range(1, n_seq + 1) if j != i)
        p_paired_total = min(p_paired_total, 1.0)
        p_unpaired = 1.0 - p_paired_total

        if p_unpaired > 0:
            S -= p_unpaired * math.log2(p_unpaired)

        entropia.append(S)

    return entropia


# ============================================================
# MÓDULO 6 — INTERPRETACIÓN DE COLORES
# Convierte el valor numérico de entropía a la categoría
# de color equivalente en el gráfico circular de RNAfold.
# ============================================================

def interpretar_color_entropia(valor_entropia):
    """
    Convierte un valor de entropía posicional a su categoría
    de color en el gráfico circular de RNAfold y su interpretación.

    Parámetros:
        valor_entropia (float): valor de S(i)

    Retorna:
        color (str): color aproximado en el gráfico
        interpretacion (str): descripción del estado conformacional
    """

    if valor_entropia < 0.05:
        return "ROJO INTENSO", "Posición perfectamente definida — siempre no apareada en todo el ensemble"
    elif valor_entropia < 0.15:
        return "ROJO-NARANJA", "Posición bien definida — casi siempre no apareada"
    elif valor_entropia < 0.30:
        return "NARANJA", "Posición mayormente definida — baja ambigüedad"
    elif valor_entropia < 0.50:
        return "NARANJA-AMARILLO", "Posición moderadamente ambigua"
    elif valor_entropia < 0.65:
        return "AMARILLO-VERDE", "Posición ambigua — participa en varias estructuras"
    elif valor_entropia < 0.80:
        return "VERDE", "Posición muy ambigua — alta variabilidad conformacional"
    elif valor_entropia < 0.90:
        return "AZUL CLARO", "Posición extremadamente ambigua"
    else:
        return "AZUL OSCURO/MORADO", "Entropía máxima — participa en múltiples estructuras distintas"


# ============================================================
# MÓDULO 7 — CÁLCULO DE VARIABLES POR REGIÓN FUNCIONAL
# Calcula las variables agregadas por región:
#   ACC_SEED, ACC_BULGE, ACC_3P
#   ENT_SEED, ENT_BULGE, ENT_3P
# ============================================================

def calcular_variables_regionales(valores_por_posicion, region_seed, region_bulge, region_3p):
    """
    Calcula el promedio de una variable por región funcional del sitio MRE.

    Parámetros:
        valores_por_posicion (list): lista de valores 1-based por posición
        region_seed  (tuple): (inicio, fin) de la región seed-complementaria
        region_bulge (tuple): (inicio, fin) de la región del bulge J
        region_3p    (tuple): (inicio, fin) de la región 3'-complementaria

    Retorna:
        dict con promedios por región
    """

    def promedio_region(valores, inicio, fin):
        # Convertir a índice 0-based para la lista
        subset = valores[inicio-1 : fin]
        return sum(subset) / len(subset) if subset else 0.0

    return {
        "SEED":  promedio_region(valores_por_posicion, *region_seed),
        "BULGE": promedio_region(valores_por_posicion, *region_bulge),
        "3P":    promedio_region(valores_por_posicion, *region_3p),
        "GLOBAL": sum(valores_por_posicion) / len(valores_por_posicion)
    }


# ============================================================
# MÓDULO 8 — REPORTE DE RESULTADOS
# Imprime la tabla completa de resultados y los valores
# a registrar en la matriz maestra (Excel).
# ============================================================

def imprimir_reporte(id_var, secuencia, prob_pair, acc_pos, entropia,
                     region_seed, region_bulge, region_3p):
    """
    Imprime el reporte completo de resultados para la variante analizada.
    """

    n = len(secuencia)

    print("\n" + "="*70)
    print(f"  REPORTE OE2 — VAR_EST_SEC — {id_var}")
    print(f"  Secuencia: {secuencia}")
    print(f"  Longitud:  {n} nt")
    print("="*70)

    # ── Tabla posición a posición ──────────────────────────────────────────
    print(f"\n{'Pos':>4} {'Nt':>3} {'Region':>8} {'prob_pair':>10} {'ACC_POS':>8} {'ENT_POS':>8}  Color (entropía)")
    print("-"*75)

    for i in range(1, n+1):
        nt = secuencia[i-1]
        # Asignar región
        if region_seed[0] <= i <= region_seed[1]:
            region = "SEED"
        elif region_bulge[0] <= i <= region_bulge[1]:
            region = "BULGE"
        elif region_3p[0] <= i <= region_3p[1]:
            region = "3P"
        else:
            region = "—"

        pp   = prob_pair[i-1]
        acc  = acc_pos[i-1]
        ent  = entropia[i-1]
        color, _ = interpretar_color_entropia(ent)

        print(f"{i:>4} {nt:>3} {region:>8} {pp:>10.4f} {acc:>8.4f} {ent:>8.4f}  {color}")

    # ── Variables por región ───────────────────────────────────────────────
    vars_acc = calcular_variables_regionales(acc_pos,   region_seed, region_bulge, region_3p)
    vars_ent = calcular_variables_regionales(entropia,  region_seed, region_bulge, region_3p)

    print("\n" + "="*70)
    print("  VARIABLES A REGISTRAR EN MATRIZ MAESTRA (Excel)")
    print("="*70)
    print(f"\n  ── Accesibilidad ──────────────────────────────────────────────")
    print(f"  ACC_SITE  (global)  = {vars_acc['GLOBAL']:.4f}")
    print(f"  ACC_SEED  (t1–t8)   = {vars_acc['SEED']:.4f}")
    print(f"  ACC_BULGE (t9–t12)  = {vars_acc['BULGE']:.4f}")
    print(f"  ACC_3P    (t13–t26) = {vars_acc['3P']:.4f}")
    print(f"\n  ── Entropía posicional ────────────────────────────────────────")
    print(f"  ENT_SITE  (global)  = {vars_ent['GLOBAL']:.4f}")
    print(f"  ENT_SEED  (t1–t8)   = {vars_ent['SEED']:.4f}")
    print(f"  ENT_BULGE (t9–t12)  = {vars_ent['BULGE']:.4f}")
    print(f"  ENT_3P    (t13–t26) = {vars_ent['3P']:.4f}")

    # ── Posiciones con mayor entropía (outliers) ───────────────────────────
    print(f"\n  ── Posiciones con entropía > 0.30 (ambigüedad conformacional) ──")
    outliers = [(i+1, secuencia[i], entropia[i]) for i in range(n) if entropia[i] > 0.30]
    if outliers:
        for pos, nt, ent in sorted(outliers, key=lambda x: -x[2]):
            color, interp = interpretar_color_entropia(ent)
            print(f"  Posición {pos:2d} ({nt}): S = {ent:.4f} → {color}")
    else:
        print("  Ninguna posición supera S = 0.30 — sitio completamente definido")

    print("\n" + "="*70)
    print("  BITÁCORA — Parámetros de ejecución")
    print("="*70)
    print(f"  ID_VAR:       {id_var}")
    print(f"  HERRAMIENTA:  RNAfold WebServer (ViennaRNA)")
    print(f"  VERSION:      RNAfold 2.6.3")
    print(f"  PARAMETROS:   -p -d2 --noLP (default del servidor)")
    print(f"  ARCHIVO_EPS:  {ARCHIVO_EPS}")
    print(f"  ENERGIA_PARAMS: Turner 2004 (Mathews DH et al. PNAS 101:7287)")
    print("="*70 + "\n")


# ============================================================
# MÓDULO 9 — EXPORTAR A CSV
# Guarda los resultados en un archivo .csv para importar
# directamente a Excel como hoja de la matriz maestra.
# ============================================================

def exportar_csv(id_var, secuencia, prob_pair, acc_pos, entropia,
                 region_seed, region_bulge, region_3p):
    """
    Exporta la tabla posición a posición a un archivo CSV.
    """

    nombre_csv = f"OE2_SEC_{id_var}_posiciones.csv"
    n = len(secuencia)

    with open(nombre_csv, 'w', encoding='utf-8') as f:
        # Encabezado
        f.write("ID_VAR,Posicion,Nucleotido,Region,prob_pair,ACC_POS,ENT_POS\n")

        for i in range(1, n+1):
            nt = secuencia[i-1]
            if region_seed[0] <= i <= region_seed[1]:
                region = "SEED"
            elif region_bulge[0] <= i <= region_bulge[1]:
                region = "BULGE"
            elif region_3p[0] <= i <= region_3p[1]:
                region = "3P"
            else:
                region = "otro"

            pp  = prob_pair[i-1]
            acc = acc_pos[i-1]
            ent = entropia[i-1]

            f.write(f"{id_var},{i},{nt},{region},{pp:.6f},{acc:.6f},{ent:.6f}\n")

    print(f"[✓] Resultados exportados a: {nombre_csv}")
    return nombre_csv


# ============================================================
# MÓDULO 10 — EJECUCIÓN PRINCIPAL
# Llama a todos los módulos en orden para una variante.
# Para procesar otra variante: cambia ID_VAR, SECUENCIA
# y ARCHIVO_EPS en el Módulo 2 y vuelve a ejecutar.
# ============================================================

if __name__ == "__main__":

    # Longitud de la secuencia
    N = len(SECUENCIA)

    # Paso 1: Leer probabilidades de pares desde el archivo .eps
    prob_matrix = leer_probabilidades_pares(ARCHIVO_EPS, N)

    # Paso 2: Calcular prob_pair y ACC_POS por posición
    prob_pair, acc_pos = calcular_prob_pair_y_acc(prob_matrix, N)

    # Paso 3: Calcular entropía posicional por posición
    entropia = calcular_entropia_posicional(prob_matrix, N, SECUENCIA)

    # Paso 4: Imprimir reporte completo
    imprimir_reporte(
        ID_VAR, SECUENCIA,
        prob_pair, acc_pos, entropia,
        REGION_SEED, REGION_BULGE, REGION_3P
    )

    # Paso 5: Exportar a CSV para importar en Excel
    exportar_csv(
        ID_VAR, SECUENCIA,
        prob_pair, acc_pos, entropia,
        REGION_SEED, REGION_BULGE, REGION_3P
    )


# In[ ]:




