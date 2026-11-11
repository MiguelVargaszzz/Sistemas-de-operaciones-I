import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, NARANJA, GRIS, GRIS_CLARO = "#2a78d6", "#eb6834", "#52514e", "#e5e4e0"
os.makedirs("figuras", exist_ok=True)


def estilo(ax):
    ax.grid(color=GRIS_CLARO, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9b8b2")
    ax.tick_params(colors=GRIS)


res = pd.read_csv("resultados.csv").iloc[0]
solver = pd.read_csv("resultados_solver.csv")
corr = pd.read_csv("corridas_ga.csv")
conv = np.load("conv.npz")
costos = corr["Costo"].values
mejor_solver = res["Solver: mejor costo"]
costos_ok = solver.dropna(subset=["Costo"])

# 1) Convergencia del GA
mejor, prom = conv["mejor"], conv["prom"]
g = np.arange(mejor.shape[1])
fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(g, mejor.min(axis=0), mejor.max(axis=0), color=AZUL, alpha=0.18, linewidth=0)
ax.plot(g, mejor.mean(axis=0), color=AZUL, linewidth=2, label="Mejor individuo (media de las corridas)")
ax.plot(g, prom.mean(axis=0), color=GRIS, linewidth=1.5, linestyle="--", label="Promedio de la población")
ax.axhline(mejor_solver, color=NARANJA, linewidth=1.5, linestyle=":",
           label=f"Mejor solución del solver (CBC): {int(mejor_solver):,}")
ax.set_ylim(min(costos.min(), mejor_solver) * 0.97, np.max(mejor[:, 0]) * 1.03)
ax.set_xlabel("Generación")
ax.set_ylabel("Costo total")
ax.set_title("Convergencia del algoritmo genético", fontsize=13)
ax.legend(frameon=False)
estilo(ax)
fig.tight_layout()
fig.savefig("figuras/convergencia.png", dpi=200)
plt.close(fig)

# 2) Distribución del costo del GA (todas las corridas)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.boxplot([costos], widths=0.35, patch_artist=True, showfliers=False,
           boxprops=dict(facecolor=AZUL, alpha=0.35, edgecolor=AZUL),
           medianprops=dict(color=AZUL, linewidth=2), whiskerprops=dict(color=GRIS), capprops=dict(color=GRIS))
rng = np.random.default_rng(0)
ax.scatter(1 + rng.uniform(-0.08, 0.08, len(costos)), costos, s=22, color=AZUL, edgecolor="white", linewidth=0.8, zorder=3)
ax.axhline(mejor_solver, color=NARANJA, linewidth=1.5, linestyle=":")
ax.annotate(f"Mejor solver: {int(mejor_solver):,}", (1.22, mejor_solver), textcoords="offset points", xytext=(0, 6), color=GRIS, ha="right", fontsize=9)
ax.set_xticks([1])
ax.set_xticklabels([f"GA ({len(costos)} corridas)"])
ax.set_xlim(0.6, 1.25)
ax.set_ylabel("Costo total")
ax.set_title("Calidad del GA en todas las corridas", fontsize=13)
estilo(ax)
fig.tight_layout()
fig.savefig("figuras/distribucion_ga.png", dpi=200)
plt.close(fig)

# 3) Calidad vs tiempo (escala logarítmica en el tiempo)
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(costos_ok["Tiempo_s"], costos_ok["Costo"], color=NARANJA, marker="o", markersize=8, linewidth=2,
        label="Solver CBC (mejor solución con cada límite de tiempo)")
for _, f in costos_ok.iterrows():
    ax.annotate(f"{int(f['Costo']):,}", (f["Tiempo_s"], f["Costo"]), textcoords="offset points", xytext=(6, 8),
                fontsize=9, color=GRIS)
ax.errorbar(corr["Tiempo_s"].mean(), costos.mean(), yerr=costos.std(), color=AZUL, marker="s", markersize=9,
            capsize=4, linewidth=2, linestyle="none", label="GA (promedio ± desv. est.)")
ax.scatter(corr["Tiempo_s"], costos, s=14, color=AZUL, alpha=0.35, zorder=2)
ax.set_xscale("log")
ax.set_xlabel("Tiempo de cómputo (s, escala log)")
ax.set_ylabel("Costo total")
ax.set_title("Calidad de la solución frente al tiempo", fontsize=13)
ax.legend(frameon=False, loc="best")
estilo(ax)
fig.tight_layout()
fig.savefig("figuras/calidad_vs_tiempo.png", dpi=200)
plt.close(fig)

# 4) Tablas para el informe
tabla_ga = pd.DataFrame({
    "Métrica": ["Mejor", "Promedio", "Mediana", "Peor", "Desv. est.", "Tiempo promedio (s)",
                "Tiempo hasta su mejor (s)", "Generaciones promedio"],
    "GA": [int(costos.min()), round(costos.mean(), 1), float(np.median(costos)), int(costos.max()),
           round(costos.std(), 1), round(corr["Tiempo_s"].mean(), 2),
           round(corr["T_hasta_mejor_s"].mean(), 2), round(corr["Generaciones"].mean(), 1)]})
tabla_ga.to_csv("figuras/tabla_ga.csv", index=False, encoding="utf-8-sig")
solver.to_csv("figuras/tabla_solver.csv", index=False, encoding="utf-8-sig")
try:
    tabla_ga.to_latex("figuras/tabla_ga.tex", index=False, float_format="%.2f")
    solver.to_latex("figuras/tabla_solver.tex", index=False, float_format="%.2f", na_rep="--")
except ImportError:
    print("Para exportar tablas a LaTeX: python -m pip install jinja2")

print(tabla_ga.to_string(index=False))
print()
print(solver.to_string(index=False))
print("\nFiguras en la carpeta 'figuras/'")