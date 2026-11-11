# Set Cover: método exacto vs. algoritmo genético

Proyecto de clase de **Investigación de Operaciones I** (UNAB, 2026).

Una empresa de telefonía quiere cubrir a **500 clientes** eligiendo entre **500 ubicaciones de antenas**, cada una con un costo distinto. El problema se modela como un **Set Cover con costos** y se resuelve de dos formas:

- **Método exacto:** programación lineal entera con **CBC** (vía PuLP) y una corrida larga con **HiGHS** (vía SciPy).
- **Algoritmo genético (GA):** codificación binaria, torneo, cruce uniforme, mutación bit-flip y un operador de reparación voraz.

Luego se comparan en calidad de la solución, tiempo de cómputo y comportamiento.

## Modelo

$$\min Z = \sum_{j=1}^{n} c_j x_j \quad \text{s.a.} \quad \sum_{j=1}^{n} a_{ij} x_j \ge 1 \;\; \forall i, \qquad x_j \in \{0,1\} \;\; \forall j$$

- $x_j = 1$ si se instala la antena $j$.
- $c_j$: costo de la antena $j$.
- $a_{ij} = 1$ si la antena $j$ cubre al cliente $i$.

## Resultados principales

| Método | Tiempo | Mejor costo | Cota inferior | ¿Óptimo probado? |
|---|---|---|---|---|
| CBC | 300 s | 51.509 | 30.005 | No |
| HiGHS | 3 h | 49.347 | 32.427 | No |
| **GA (30 corridas)** | **7,8 s** prom. | **48.878** (22 antenas) | — | — |

- El GA encontró la **mejor solución conocida** (48.878): un 5,1 % mejor que CBC y un 1,0 % mejor que HiGHS tras 3 horas.
- Las 30 corridas del GA superaron a CBC; promedio 50.363, desviación estándar 561.
- Ningún solver probó la optimalidad: el óptimo está en **[32.427, 48.878]**.

## Instalación

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

## Cómo ejecutar

| Paso | Comando | Qué hace | Tiempo aprox. |
|---|---|---|---|
| 0 | `python prueba.py` | Valida ambos métodos con un ejemplo pequeño de óptimo conocido | segundos |
| 1 | `python experimentos.py` | CBC con 10, 60 y 300 s + 30 corridas del GA | ~10 min |
| 2 | `python sensibilidad.py` | Sensibilidad de los parámetros del GA (30 corridas por configuración) | ~40 min |
| 3 | `python graficas.py` | Genera las figuras y tablas en `figuras/` | ~1 min |
| 4 | `python optimo.py 10800` | Corrida larga de HiGHS (3 h) para acotar el óptimo | 3 h |
| 5 | `streamlit run app.py` | Interfaz visual para ejecutar y comparar los métodos | — |

Para medir tiempos comparables, conviene correr `experimentos.py` sin otros procesos pesados en paralelo.

## Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `datos.py` | Lee la matriz de cobertura y los costos; valida la instancia |
| `exacto.py` | Modelo de programación entera con PuLP/CBC, relajación lineal y verificación |
| `ga.py` | Algoritmo genético y operador de reparación |
| `prueba.py` | Prueba de validación con un ejemplo pequeño |
| `experimentos.py` | Experimento principal (CBC + 30 corridas del GA) |
| `sensibilidad.py` | Análisis de sensibilidad de los parámetros del GA |
| `optimo.py` | Corrida larga con HiGHS y registro de la evolución de la cota |
| `graficas.py` | Figuras y tablas |
| `app.py` | Interfaz en Streamlit |
| `set_cover_500x500.csv` | Matriz de cobertura (filas = clientes, columnas = antenas) |
| `Costo_S.xlsx` | Costo de cada antena |

**Resultados guardados:** `resultados*.csv`, `corridas_ga.csv`, `sensibilidad_*.csv`, `evolucion_highs.csv`, `solucion_*.csv` (antenas numeradas desde 1), `conv.npz` (curvas de convergencia), `config_ga.txt`, `entorno.txt` y la carpeta `figuras/`.

## Parámetros del GA

Población 100 · máx. 1.000 generaciones · paciencia 200 · pc = 0,9 · pm = 3/n · torneo k = 3 · elitismo 2 · semillas 0–29.

## Autores

Moisés Colmenares y Miguel Vargas.
