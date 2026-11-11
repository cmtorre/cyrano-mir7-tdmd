# Cyrano–miR-7-5p | Análisis computacional de estructura–función y TDMD

**Título de la tesis:** Análisis computacional de la relación estructura–función en variantes del sitio MRE de Cyrano frente a miR-7-5p asociados a degradación dirigida por diana.

**Proyecto:** Maestría en Ciencias Bioquímicas, Universidad Nacional Autónoma de México (UNAM).

## Descripción

Este repositorio reúne los archivos de entrada, scripts, estructuras moleculares, matrices de descriptores y resultados del análisis computacional de siete variantes del sitio de reconocimiento de miRNA (MRE) de Cyrano (WT y M1–M6), en relación con la eficiencia experimental de degradación dirigida por diana (TDMD) sobre miR-7-5p. El estudio es exploratorio y descriptivo de relaciones estructura–función; no constituye un modelo predictivo validado.

## Objetivos de la investigación

1. Caracterizar las propiedades estructurales y termodinámicas de las hebras individuales de las variantes del MRE de Cyrano y de miR-7-5p.
2. Caracterizar las propiedades estructurales y termodinámicas de cada complejo dúplex variante de MRE–miR-7-5p.
3. Determinar la asociación entre los descriptores biofísicos y su eficiencia TDMD sobre miR-7-5p.

## Organización de archivos

- `OE-1/` a `OE-5/`: archivos organizados según las cinco etapas operativas originales de adquisición y caracterización molecular.
- `PDB/`: estructuras moleculares de las variantes del MRE, miR-7-5p y sus complejos.
- `Analisis/1. ETL/`: integración, homogeneización y documentación de las matrices de entrada.
- `Analisis/2. Embudo/`: iteraciones de exploración, control de calidad y reducción de redundancia entre descriptores.
- `Analisis/3. Asociacion/`: análisis de asociación estructura–función, con ramas CYRMIR y MEC.
- `Repositorio_1.docx`: descripción detallada de la organización del repositorio y consideraciones metodológicas.

**Nota sobre la numeración:** La estructura OE-1 a OE-5 corresponde a la organización operativa inicial. En la versión final de la tesis, esas actividades se agrupan en tres objetivos específicos. Se conservan los nombres originales para mantener la trazabilidad.

## Organización de cada fase

Las carpetas de análisis contienen scripts Python, entradas locales y salidas. Cuando corresponda:

- `00_DATASET/` o `00_DATA/`: archivos requeridos como entrada en la fase.
- `01_FIGURAS/`: figuras generadas en la fase.
- `02_TXT/` o `02_output_txt/`: resultados tabulares, registros de ejecución y archivos de seguimiento.

Algunos archivos aparecen tanto en los resultados de una fase como en las entradas de la siguiente. **Estas copias se conservan deliberadamente** para facilitar la ejecución independiente por etapas y el seguimiento de la procedencia de los datos.

## Reproducción de los análisis

1. Consultar `Repositorio_1.docx` para comprender la organización y la secuencia de procesamiento.
2. Revisar los archivos de entrada y el script correspondiente a la etapa deseada.
3. Configurar localmente las versiones de Python, bibliotecas y herramientas especificadas en la documentación metodológica del proyecto.
4. **Adaptar las rutas absolutas y directorios** que en algunos scripts conservan las ubicaciones del equipo donde se ejecutó originalmente el análisis.
5. Ejecutar la etapa y contrastar los resultados con los archivos tabulares, figuras y registros conservados en la carpeta correspondiente.

El repositorio conserva archivos intermedios para permitir la consulta y repetición de etapas concretas sin ejecutar necesariamente toda la cadena computacional desde el inicio. La presencia de scripts y resultados documentados no implica, por sí sola, que se haya verificado su reproducción integral en un segundo entorno independiente.

## Datos experimentales y alcance

La unidad biológica de comparación es cada variante del MRE (WT, M1–M6; **n = 7**). Los descriptores computacionales no representan observaciones biológicas independientes adicionales. Los resultados deben interpretarse como asociaciones exploratorias, sin atribución causal ni generalización predictiva.

**Referencia experimental de partida:** Kleaveland, B., Shi, C. Y., Stefano, J., & Bartel, D. P. (2018). A network of noncoding regulatory RNAs acts in the mammalian brain. *Cell, 174*(2), 350–362.e17. https://doi.org/10.1016/j.cell.2018.05.022

## Tesis y citación

Los resultados, la justificación de los descriptores seleccionados y su interpretación están desarrollados en el manuscrito de tesis. Una vez publicada una versión archivada con DOI, se incorporará aquí su referencia definitiva.

## Condiciones de reutilización

Los derechos de terceros (artículos, herramientas y recursos ajenos) siguen correspondiendo a sus titulares. La licencia de los scripts y materiales originales de este repositorio se definirá antes de su publicación.
