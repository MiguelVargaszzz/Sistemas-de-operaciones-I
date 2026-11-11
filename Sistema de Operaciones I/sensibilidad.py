import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datos import cargar_datos
from ga import algoritmo_genetico

RUTA_CSV = "set_cover_500x500.csv"
RUTA_XLSX = "Costo_S.xlsx"
N_CORRIDAS = 30                       # corridas por configuración (súbelo a 30 si el tiempo lo permite)

# Valores base = configuración final del GA (pm = 3/n). Se varía un parámetro a la vez.
BASE = {"tam_pob": 100, "pc": 0.9, "pm_factor": 3.0, "k_torneo": 3}
VARIANTES = {
    "tam_pob":   [50, 100, 200],
    "pm_factor": [1.0, 3.0, 5.0],      # pm = pm_factor / n
    "pc":        [0.6, 0.9, 1.0],
    "k_torneo":  [2, 3, 5],
}

A, c = cargar_datos(RUTA_CSV, RUTA_XLSX)
n = A.shape[1]

configs = {}                           # configuraciones únicas (la base aparece una sola vez)
for param, valores in VARIANTES.items():
    for v in valores:
        cfg = dict(BASE)
        cfg[param] = v
        configs[(param, v)] = cfg
vistas, unicas = set(), []
for (param, v), cfg in configs.items():
    clave = tuple(sorted(cfg.items()))
    if clave not in vistas:
        vistas.add(clave)
        unicas.append(((param, v), cfg))

crudo = []
t0 = time.perf_counter()
for i, ((param, v), cfg) in enumerate(unicas, 1):
    for s in range(N_CORRIDAS):
        r = algoritmo_genetico(A, c, tam_pob=int(cfg["tam_pob"]), pc=cfg["pc"], pm=cfg["pm_factor"] / n,
                               k_torneo=int(cfg["k_torneo"]), seed=s)
        crudo.append({"Parametro": param, "Valor": v, "Es_base": cfg == BASE, "Semilla": s,
                      "Costo": r["costo"], "Tiempo_s": r["tiempo"], "Gen_hasta_mejor": r["gen_mejor"],
                      "Generaciones": r["generaciones"]})
        if i == 1 and s == 0:
            print(f"1 corrida ≈ {r['tiempo']:.1f} s -> ~{r['tiempo'] * N_CORRIDAS * len(unicas) / 60:.0f} min "
                  f"para {len(unicas)} configuraciones (estimado; tam_pob=200 tarda más)", flush=True)
    pd.DataFrame(crudo).to_csv("sensibilidad_crudo.csv", index=False, encoding="utf-8-sig")
    print(f"[{i}/{len(unicas)}] {param}={v} lista ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

df = pd.DataFrame(crudo)
# La configuración base se reutiliza en cada parámetro para poder comparar contra ella
base_df = df[df["Es_base"]]
filas = []
for param, valores in VARIANTES.items():
    for v in valores:
        sub = base_df if v == BASE[param] else df[(df["Parametro"] == param) & (df["Valor"] == v)]
        filas.append({"Parametro": param, "Valor": v, "Es_base": v == BASE[param],
                      "Costo_prom": sub["Costo"].mean(), "Costo_desv": sub["Costo"].std(),
                      "Costo_mejor": sub["Costo"].min(), "Costo_peor": sub["Costo"].max(),
                      "Tiempo_prom_s": sub["Tiempo_s"].mean(), "Gen_hasta_mejor": sub["Gen_hasta_mejor"].mean()})
res = pd.DataFrame(filas).round(2)
res.to_csv("sensibilidad_resumen.csv", index=False, encoding="utf-8-sig")
print(res.to_string(index=False))

# Gráfica: un panel por parámetro (cajas con todas las corridas)
AZUL, GRIS = "#2a78d6", "#52514e"
fig, axes = plt.subplots(1, len(VARIANTES), figsize=(15, 4.6), sharey=True)
for ax, (param, valores) in zip(axes, VARIANTES.items()):
    datos = []
    for v in valores:
        sub = base_df if v == BASE[param] else df[(df["Parametro"] == param) & (df["Valor"] == v)]
        datos.append(sub["Costo"].values)
    bp = ax.boxplot(datos, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=1.8),
                    boxprops=dict(facecolor=AZUL, edgecolor=AZUL),
                    whiskerprops=dict(color=GRIS), capprops=dict(color=GRIS),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=GRIS, markeredgecolor=GRIS))
    ax.set_xticks(range(1, len(valores) + 1))
    ax.set_xticklabels([f"{v}" + (" (base)" if v == BASE[param] else "") for v in valores], fontsize=9)
    ax.set_title(param, fontsize=11, color=GRIS)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Costo de la mejor solución")
fig.suptitle("Sensibilidad del algoritmo genético a sus parámetros", fontsize=13)
fig.tight_layout()
fig.savefig("sensibilidad.png", dpi=200)
print("\nFigura guardada: sensibilidad.png")