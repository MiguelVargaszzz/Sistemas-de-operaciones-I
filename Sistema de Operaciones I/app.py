import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

from datos import cargar_datos
from exacto import resolver_exacto, cota_inferior_lp, verificar_cobertura
from ga import algoritmo_genetico

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#52514e"

st.set_page_config(page_title="Set Cover: exacto vs. algoritmo genético", layout="wide")


# ====================== Datos ======================
@st.cache_data(show_spinner=False)
def datos_profesor(ruta_csv, ruta_xlsx):
    return cargar_datos(ruta_csv, ruta_xlsx, verbose=False)


@st.cache_data(show_spinner=False)
def datos_aleatorios(m, n, densidad, seed):
    rng = np.random.default_rng(seed)
    A = (rng.random((m, n)) < densidad).astype(int)
    for i in range(m):                                   # cada cliente debe tener al menos una antena
        if A[i].sum() == 0:
            A[i, rng.integers(0, n)] = 1
    for j in range(n):                                   # sin antenas vacías
        if A[:, j].sum() == 0:
            A[rng.integers(0, m), j] = 1
    c = rng.integers(2000, 4000, size=n)
    return A, c


@st.cache_data(show_spinner=False)
def cota_lp(A, c):
    return cota_inferior_lp(A, c)


# ====================== Gráficas ======================
def fig_convergencia(ga, ex):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    g = np.arange(len(ga["hist_mejor"]))
    ax.plot(g, ga["hist_mejor"], color=AZUL, linewidth=2, label="GA: mejor individuo")
    ax.plot(g, ga["hist_prom"], color=GRIS, linewidth=1.3, linestyle="--", label="GA: promedio de la población")
    bajo = ga["hist_mejor"][-1]
    if ex is not None and ex["costo"] is not None:
        ax.axhline(ex["costo"], color=NARANJA, linewidth=1.5, linestyle=":", label=f"Solver CBC: {ex['costo']:,}")
        bajo = min(bajo, ex["costo"])
    ax.set_ylim(bajo * 0.97, ga["hist_mejor"][0] * 1.05)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Costo total")
    ax.set_title("Convergencia del algoritmo genético")
    ax.legend(frameon=False)
    ax.grid(color="#e5e4e0")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def fig_barras(ex, ga):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    nombres, colores = ["Exacto (CBC)", "GA"], [NARANJA, AZUL]
    for ax, valores, titulo, fmt in ((axes[0], [ex["costo"], ga["costo"]], "Costo total", "%.0f"),
                                     (axes[1], [ex["tiempo"], ga["tiempo"]], "Tiempo de cómputo (s)", "%.2f")):
        barras = ax.bar(nombres, valores, color=colores, width=0.5)
        ax.bar_label(barras, fmt=fmt, fontsize=9, color=GRIS)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylim(0, max(valores) * 1.15)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def fig_cobertura(A, sel, titulo, color):
    """Matriz clientes x antenas elegidas: cada fila (cliente) debe tener al menos una celda coloreada."""
    sub = A[:, sel]
    orden = np.argsort(sub.argmax(axis=1), kind="stable")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.imshow(sub[orden], aspect="auto", cmap=ListedColormap(["#f4f3f0", color]),
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("Antenas seleccionadas")
    ax.set_ylabel("Clientes (agrupados por antena que los cubre)")
    ax.set_title(titulo, fontsize=11)
    if len(sel) <= 30:
        ax.set_xticks(range(len(sel)))
        ax.set_xticklabels([j + 1 for j in sel], rotation=90, fontsize=7)
    fig.tight_layout()
    return fig


def mostrar(fig):
    st.pyplot(fig)
    plt.close(fig)


# ====================== Barra lateral ======================
st.sidebar.header("1. Instancia")
fuente = st.sidebar.radio("Datos", ["Datos del profesor (500 x 500)", "Instancia aleatoria (demo rápida)"])
if fuente.startswith("Datos"):
    ruta_csv = st.sidebar.text_input("Archivo CSV", "set_cover_500x500.csv")
    ruta_xlsx = st.sidebar.text_input("Archivo Excel de costos", "Costo_S.xlsx")
    try:
        A, c = datos_profesor(ruta_csv, ruta_xlsx)
    except Exception as e:
        st.error(f"No se pudieron cargar los datos: {e}. "
                 "Ejecuta la app desde la carpeta donde están el CSV y el Excel.")
        st.stop()
    clave_inst = f"prof|{ruta_csv}|{ruta_xlsx}"
else:
    m_ = st.sidebar.slider("Clientes (m)", 10, 150, 40)
    n_ = st.sidebar.slider("Antenas (n)", 10, 150, 40)
    dens = st.sidebar.slider("Densidad de cobertura", 0.05, 0.40, 0.15)
    semilla_inst = st.sidebar.number_input("Semilla de la instancia", 0, 9999, 42)
    A, c = datos_aleatorios(int(m_), int(n_), float(dens), int(semilla_inst))
    clave_inst = f"alea|{m_}|{n_}|{dens}|{semilla_inst}"

st.sidebar.header("2. Método exacto (CBC)")
limite = st.sidebar.number_input("Límite de tiempo (s)", 5, 900, 30)

st.sidebar.header("3. Algoritmo genético")
tam_pob = st.sidebar.number_input("Tamaño de la población", 10, 500, 100, step=10)
generaciones = st.sidebar.number_input("Máximo de generaciones", 50, 3000, 1000, step=50)
pc = st.sidebar.slider("Probabilidad de cruce (pc)", 0.0, 1.0, 0.9)
pm_factor = st.sidebar.number_input("Mutación: pm = factor / n", 0.1, 10.0, 3.0, step=0.1)
k_torneo = st.sidebar.number_input("Tamaño del torneo", 2, 10, 3)
elitismo = st.sidebar.number_input("Elitismo (individuos)", 0, 10, 2)
paciencia = st.sidebar.number_input("Paciencia (generaciones sin mejora)", 10, 1000, 200, step=10)
semilla_ga = st.sidebar.number_input("Semilla del GA", 0, 9999, 0)

m, n = A.shape
if st.session_state.get("clave_inst") != clave_inst:     # nueva instancia: se borran los resultados
    st.session_state["clave_inst"] = clave_inst
    st.session_state["ex"] = None
    st.session_state["ga"] = None


def correr_exacto():
    with st.spinner(f"Resolviendo con CBC (hasta {int(limite)} s)..."):
        st.session_state["ex"] = resolver_exacto(A, c, tiempo_limite=int(limite))


def correr_ga():
    with st.spinner("Ejecutando el algoritmo genético..."):
        st.session_state["ga"] = algoritmo_genetico(
            A, c, tam_pob=int(tam_pob), generaciones=int(generaciones), pc=float(pc),
            pm=float(pm_factor) / n, k_torneo=int(k_torneo), elitismo=int(elitismo),
            paciencia=int(paciencia), seed=int(semilla_ga))


# ====================== Cuerpo ======================
st.title("Set Cover con costos: método exacto vs. algoritmo genético")
cota = cota_lp(A, c)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Clientes", m)
k2.metric("Antenas", n)
k3.metric("Densidad de cobertura", f"{A.mean():.3f}")
k4.metric("Cota inferior (relajación LP)", f"{cota:,.0f}")

tab_modelo, tab_res, tab_sol, tab_exp = st.tabs(
    ["Modelo", "Resolver y comparar", "Soluciones", "Experimento (30 corridas)"])

# ---------- Modelo ----------
with tab_modelo:
    st.subheader("Formulación matemática")
    st.markdown("**Variables de decisión:** $x_j = 1$ si se instala la antena $j$, y $0$ en caso contrario. "
                "**Parámetros:** $c_j$ costo de la antena $j$; $a_{ij}=1$ si la antena $j$ cubre al cliente $i$.")
    st.latex(r"\min \; Z = \sum_{j=1}^{n} c_j\, x_j")
    st.latex(r"\text{s.a.} \quad \sum_{j=1}^{n} a_{ij}\, x_j \;\ge\; 1 \qquad \forall\, i = 1,\dots,m")
    st.latex(r"x_j \in \{0,1\} \qquad \forall\, j = 1,\dots,n")
    st.markdown("La función objetivo suma el costo de las antenas instaladas. La restricción de cobertura obliga a que "
                "cada cliente quede cubierto por al menos una antena instalada.")
    st.subheader("Diseño del algoritmo genético")
    st.dataframe(pd.DataFrame({
        "Componente": ["Codificación", "Fitness", "Inicialización", "Selección", "Cruce", "Mutación",
                       "Reparación", "Elitismo", "Parada"],
        "Decisión": ["Vector binario de longitud n (gen j = 1: antena j instalada)",
                     "Costo total sum(c_j x_j), se minimiza",
                     "Cada gen = 1 con probabilidad 0.5, luego reparación",
                     "Torneo de tamaño k",
                     "Uniforme, con probabilidad pc",
                     "Bit-flip con probabilidad pm = factor/n por gen",
                     "Agrega la antena de menor costo por cliente nuevo cubierto; luego quita redundantes "
                     "(primero las más caras). Toda solución queda factible",
                     "Los mejores individuos pasan intactos",
                     "Máximo de generaciones o 'paciencia' generaciones sin mejora"]}))

# ---------- Resolver y comparar ----------
with tab_res:
    b1, b2, b3 = st.columns(3)
    if b1.button("Resolver con método exacto (CBC)"):
        correr_exacto()
    if b2.button("Resolver con algoritmo genético"):
        correr_ga()
    if b3.button("Ejecutar ambos"):
        correr_exacto()
        correr_ga()

    ex, ga = st.session_state.get("ex"), st.session_state.get("ga")
    if ex is None and ga is None:
        st.info("Elige un método en la barra lateral y presiona un botón. En la instancia de 500 x 500 el solver "
                "no suele probar el óptimo: se detiene en el límite de tiempo y entrega su mejor solución.")

    col_ex, col_ga = st.columns(2)
    with col_ex:
        st.subheader("Método exacto (CBC)")
        if ex is None:
            st.caption("Aún no se ha ejecutado.")
        elif ex["costo"] is None:
            st.warning("CBC no encontró ninguna solución factible dentro del límite de tiempo.")
        else:
            a, b = st.columns(2)
            a.metric("Costo total", f"{ex['costo']:,}")
            b.metric("Tiempo (s)", f"{ex['tiempo']:.2f}")
            a.metric("Antenas instaladas", len(ex["seleccion"]))
            b.metric("Cobertura verificada", "Sí" if verificar_cobertura(A, ex["seleccion"]) else "No")
            if ex["optimo_probado"]:
                st.success("Óptimo probado.")
            else:
                st.warning("Se agotó el tiempo: es la mejor solución hallada, sin prueba de optimalidad.")
                if ex.get("cota_cbc"):
                    st.caption(f"Cota inferior de CBC: {ex['cota_cbc']:,.0f}")
    with col_ga:
        st.subheader("Algoritmo genético")
        if ga is None:
            st.caption("Aún no se ha ejecutado.")
        else:
            a, b = st.columns(2)
            a.metric("Costo total", f"{ga['costo']:,}")
            b.metric("Tiempo total (s)", f"{ga['tiempo']:.2f}")
            a.metric("Antenas instaladas", len(ga["seleccion"]))
            b.metric("Cobertura verificada", "Sí" if verificar_cobertura(A, ga["seleccion"]) else "No")
            st.caption(f"Generaciones: {ga['generaciones']} | mejor solución hallada en la generación "
                       f"{ga['gen_mejor']} (a los {ga['t_mejor']:.2f} s)")

    if ga is not None:
        st.subheader("Comportamiento del GA")
        mostrar(fig_convergencia(ga, ex))

    if ex is not None and ex["costo"] is not None and ga is not None:
        st.subheader("Comparación")
        dif = 100 * (ga["costo"] - ex["costo"]) / ex["costo"]
        gap_cota = 100 * (ga["costo"] - cota) / cota
        st.dataframe(pd.DataFrame({
            "Métrica": ["Costo total", "Antenas instaladas", "Tiempo (s)", "Gap del GA vs. solver (%)",
                        "Gap del GA vs. cota inferior (%)"],
            "Exacto (CBC)": [f"{ex['costo']:,}", str(len(ex["seleccion"])), f"{ex['tiempo']:.2f}", "-", "-"],
            "GA": [f"{ga['costo']:,}", str(len(ga["seleccion"])), f"{ga['tiempo']:.2f}",
                   f"{dif:+.2f}", f"{gap_cota:.2f}"]}))
        if ga["costo"] < ex["costo"]:
            st.success("El GA encontró una solución más barata que la del solver dentro de su límite de tiempo.")
        elif ga["costo"] == ex["costo"]:
            st.success("El GA igualó el costo de la solución del solver.")
        else:
            st.info(f"El GA quedó {dif:.2f} % por encima de la solución del solver.")
        if not ex["optimo_probado"]:
            st.caption("Como el solver no probó optimalidad, el óptimo real está entre la cota inferior y la "
                       "mejor solución conocida; el gap contra la cota es una cota superior de qué tan lejos "
                       "del óptimo está la solución.")
        mostrar(fig_barras(ex, ga))

# ---------- Soluciones ----------
with tab_sol:
    ex, ga = st.session_state.get("ex"), st.session_state.get("ga")
    if (ex is None or ex["costo"] is None) and ga is None:
        st.info("Ejecuta al menos un método para ver su solución.")
    s1, s2 = st.columns(2)
    with s1:
        if ex is not None and ex["costo"] is not None:
            st.subheader("Solución exacta")
            mostrar(fig_cobertura(A, ex["seleccion"], f"Exacto: {len(ex['seleccion'])} antenas", NARANJA))
            st.dataframe(pd.DataFrame({"Antena": [j + 1 for j in ex["seleccion"]],
                                       "Costo": [int(c[j]) for j in ex["seleccion"]]}))
    with s2:
        if ga is not None:
            st.subheader("Solución del GA")
            mostrar(fig_cobertura(A, ga["seleccion"], f"GA: {len(ga['seleccion'])} antenas", AZUL))
            st.dataframe(pd.DataFrame({"Antena": [j + 1 for j in ga["seleccion"]],
                                       "Costo": [int(c[j]) for j in ga["seleccion"]]}))
    if ex is not None and ex["costo"] is not None and ga is not None:
        comunes = set(ex["seleccion"]) & set(ga["seleccion"])
        st.caption(f"Antenas en común entre ambas soluciones: {len(comunes)}")
    st.caption("Cada fila de la matriz es un cliente y cada columna una antena instalada: "
               "que toda fila tenga al menos una celda coloreada significa que todos los clientes están cubiertos.")

# ---------- Experimento guardado ----------
with tab_exp:
    st.subheader("Resultados guardados de experimentos.py (datos del profesor)")
    if not os.path.exists("resultados.csv"):
        st.info("Aún no hay resultados guardados. Corre primero experimentos.py en esta carpeta.")
    else:
        res = pd.read_csv("resultados.csv").iloc[0]
        st.dataframe(pd.DataFrame({"Métrica": list(res.index), "Valor": [str(v) for v in res.values]}))
        if os.path.exists("resultados_solver.csv"):
            st.markdown("**Método exacto con distintos límites de tiempo**")
            st.dataframe(pd.read_csv("resultados_solver.csv").astype(str))
        imagenes = [("figuras/convergencia.png", "Convergencia del GA"),
                    ("figuras/calidad_vs_tiempo.png", "Calidad frente al tiempo"),
                    ("figuras/distribucion_ga.png", "Distribución del costo del GA"),
                    ("sensibilidad.png", "Sensibilidad a los parámetros")]
        for i in range(0, len(imagenes), 2):
            cols = st.columns(2)
            for col, (ruta, titulo) in zip(cols, imagenes[i:i + 2]):
                if os.path.exists(ruta):
                    col.image(ruta, caption=titulo)
        for archivo, titulo in (("solucion_solver.csv", "Antenas elegidas por el solver"),
                                ("solucion_ga.csv", "Antenas elegidas por el mejor GA")):
            if os.path.exists(archivo):
                with st.expander(titulo):
                    st.dataframe(pd.read_csv(archivo))