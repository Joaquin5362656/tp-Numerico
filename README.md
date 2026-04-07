# Proyecto de Modelación Numérica

Este repositorio contiene un trabajo práctico orientado a **Modelación / Métodos Numéricos**, con implementaciones en **Python** usando **NumPy** y **Matplotlib** para el procesamiento numérico y la visualización de resultados.

## ¿Qué abarca el proyecto?

En general, el proyecto reúne herramientas típicas de métodos numéricos para resolver problemas matemáticos de forma aproximada, incluyendo (según el caso):

- **Aproximación de soluciones** mediante métodos iterativos, buscando un buen balance entre precisión y costo computacional.
- **Resolución de ecuaciones (búsqueda de raíces)** y análisis de convergencia.
- **Sistemas de ecuaciones lineales** aplicados a modelos con múltiples incógnitas.
- **Interpolación y/o ajuste de curvas** para aproximar funciones a partir de datos.
- **Derivación e integración numérica** cuando no es práctico obtener una solución exacta.
- **Análisis de error y estabilidad** para evaluar la calidad de los resultados numéricos.

---

## Requisitos previos

- Tener **Python 3.10+** instalado.  
  Para verificarlo, ejecutar en la terminal:

```bash
python --version
```

## Clonar o copiar el proyecto

Ubicate en la carpeta donde querés guardar el proyecto y cloná el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd codigo
```

## Activar el entorno virtual (Git Bash)

En **Git Bash**, ejecutá:

```bash
source venv/Scripts/activate
```

## Instalar las dependencias

Con el entorno activado, instalá las librerías necesarias usando el archivo requirements.txt:

```bash
pip install -r requirements.txt
```

## Ejecutar el programa del proyecto

```bash
python procesamiento_datos.py
```