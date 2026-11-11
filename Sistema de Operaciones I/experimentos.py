import inspect
import platform
import sys
import time
import numpy as np
import pandas as pd
import pulp
from datos import cargar_datos
from exacto import resolver_exacto, cota_inferior_lp, verificar_cobertura
from ga import algoritmo_genetico

RUTA_CSV = "set_cover_500x500.csv"
RUTA_XLSX = "Costo_S.xlsx"
N_CORRIDAS = 30                 # corridas independientes del GA (semillas 0..29)
LIMITES_SOLVER = [10, 60, 300]  # segundos. CBC se corre una vez por límite y se detiene si prueba optimalidad
PM_FACTOR = 3.0                 # mutación pm = PM_FACTOR / n (antes 1/n; la sensibilidad mostró que 3/n es mejor)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

A, c = cargar_datos(RUTA_CSV, RUTA_XLSX)
m, n = A.shape
PARAMS_GA = {"pm": PM_FACTOR / n}   # el resto de parámetros son los de ga.py

# ---------- Entorno (para el informe) ----------
firma = inspect.signature(algoritmo_genetico).parameters
params = {k: v.default for k, v in firma.items() if v.default is not inspect.Parameter.empty and k != "seed"}
params.update(PARAMS_GA)
params["pm"] = f"{PM_FACTOR:g}/n = {PM_FACTOR / n:.5f}"
with open("entorno.txt", "w", encoding="utf-8") as f:
    f.write(f"Procesador: {platform.processor()}\nSistema: {platform.platform()}\n"
            f"Python: {sys.version.split()[0]}\nPuLP: {pulp.__version__}\nNumPy: {np.__version__}\n"
            f"Pandas: {pd.__version__}\n(Agrega la RAM del equipo a mano)\n")
with open("config_ga.txt", "w", encoding="utf-8") as f:
    f.write("Parámetros del GA:\n" + "\n".join(f"  {k} = {v}" for k, v in params.items()) + "\n")
print(open("entorno.txt", encoding="utf-8").read())

t_total = time.perf_counter()

# ---------- Cota inferior (relajación lineal) ----------
cota_lp = cota_inferior_lp(A, c)
print(f"Cota inferior (relajación LP): {cota_lp:.1f}\n")

# ---------- Método exacto ----------
filas_solver = []
mejor_solver_costo, mejor_solver_sel = None, []
for lim in LIMITES_SOLVER:
    print(f"[Exacto] CBC con límite de {lim} s ...", flush=True)
    r = resolver_exacto(A, c, tiempo_limite=lim)
    if r["costo"] is not None and (mejor_solver_costo is None or r["costo"] < mejor_solver_costo):
        mejor_solver_costo, mejor_solver_sel = r["costo"], r["seleccion"]
    filas_solver.append({"Limite_s": lim, "Costo": r["costo"], "Tiempo_s": round(r["tiempo"], 2),
                         "Optimo_probado": r["optimo_probado"], "Cota_CBC": r["cota_cbc"],
                         "Factible": r["factible"]})
    print(f"          costo = {r['costo']} | {r['tiempo']:.1f} s | óptimo probado: {r['optimo_probado']}", flush=True)
    pd.DataFrame(filas_solver).to_csv("resultados_solver.csv", index=False, encoding="utf-8-sig")
    if r["optimo_probado"]:
        break
df_solver = pd.DataFrame(filas_solver)
if mejor_solver_sel:
    pd.DataFrame({"Antena": [j + 1 for j in mejor_solver_sel],
                  "Costo": [int(c[j]) for j in mejor_solver_sel]}
                 ).to_csv("solucion_solver.csv", index=False, encoding="utf-8-sig")
costos_solver = df_solver["Costo"].dropna()
mejor_solver = int(costos_solver.min()) if len(costos_solver) else None
cotas = [cota_lp] + [v for v in df_solver["Cota_CBC"].dropna()]
cota = max(cotas)
solver_probo = bool(df_solver["Optimo_probado"].any())

# ---------- Algoritmo genético ----------
print(f"\n[GA] {N_CORRIDAS} corridas independientes ...", flush=True)
corridas = []
for s in range(N_CORRIDAS):
    r = algoritmo_genetico(A, c, seed=s, **PARAMS_GA)
    corridas.append(r)
    if s == 0:
        print(f"     1 corrida = {r['tiempo']:.1f} s -> ~{r['tiempo'] * N_CORRIDAS / 60:.1f} min en total", flush=True)
    elif (s + 1) % 5 == 0:
        print(f"     {s + 1}/{N_CORRIDAS} corridas | mejor hasta ahora: {min(x['costo'] for x in corridas)}", flush=True)

costos = np.array([r["costo"] for r in corridas])
mejor_idx = int(np.argmin(costos))
mejor_sel = corridas[mejor_idx]["seleccion"]

# Verificación independiente de la mejor solución del GA
assert verificar_cobertura(A, mejor_sel), "La mejor solución del GA no cubre a todos los clientes"
assert int(c[mejor_sel].sum()) == int(costos.min()), "El costo reportado no coincide con el recalculado"

pd.DataFrame({"Corrida": range(N_CORRIDAS), "Semilla": range(N_CORRIDAS), "Costo": costos,
              "Tiempo_s": [r["tiempo"] for r in corridas],
              "T_hasta_mejor_s": [r["t_mejor"] for r in corridas],
              "Gen_hasta_mejor": [r["gen_mejor"] for r in corridas],
              "Generaciones": [r["generaciones"] for r in corridas]}
             ).to_csv("corridas_ga.csv", index=False, encoding="utf-8-sig")

L = max(len(r["hist_mejor"]) for r in corridas)
rell = lambda h: list(h) + [h[-1]] * (L - len(h))
np.savez("conv.npz", mejor=np.array([rell(r["hist_mejor"]) for r in corridas]),
         prom=np.array([rell(r["hist_prom"]) for r in corridas]))

pd.DataFrame({"Antena": [j + 1 for j in mejor_sel], "Costo": [int(c[j]) for j in mejor_sel]}
             ).to_csv("solucion_ga.csv", index=False, encoding="utf-8-sig")

# ---------- Resumen ----------
conocida = int(min(costos.min(), mejor_solver)) if mejor_solver is not None else int(costos.min())
gap = lambda v, ref: round(100 * (v - ref) / ref, 2)
resumen = {
    "Clientes (m)": m, "Antenas (n)": n,
    "Cota LP": round(cota_lp, 1), "Cota inferior usada": round(cota, 1),
    "Solver: mejor costo": mejor_solver, "Solver: óptimo probado": solver_probo,
    "Solver: tiempo máx. (s)": float(df_solver["Tiempo_s"].max()),
    "GA: mejor": int(costos.min()), "GA: promedio": round(costos.mean(), 1),
    "GA: mediana": float(np.median(costos)), "GA: peor": int(costos.max()),
    "GA: desv. est.": round(costos.std(), 1),
    "Mejor solución conocida": conocida,
    "GA mejor vs conocida (%)": gap(costos.min(), conocida),
    "GA promedio vs conocida (%)": gap(costos.mean(), conocida),
    "GA mejor vs cota (%)": gap(costos.min(), cota),
    "Corridas GA <= mejor del solver": (f"{int((costos <= mejor_solver).sum())}/{N_CORRIDAS}"
                                        if mejor_solver is not None else "n/d"),
    "GA: tiempo prom. (s)": round(float(np.mean([r['tiempo'] for r in corridas])), 2),
    "GA: tiempo hasta su mejor (s)": round(float(np.mean([r['t_mejor'] for r in corridas])), 2),
    "GA: generaciones prom.": round(float(np.mean([r['generaciones'] for r in corridas])), 1),
    "Antenas en la mejor solución GA": len(mejor_sel),
}
pd.DataFrame([resumen]).to_csv("resultados.csv", index=False, encoding="utf-8-sig")

print("\n===== SOLVER =====")
print(df_solver.to_string(index=False))
print("\n===== RESUMEN =====")
for k, v in resumen.items():
    print(f"{k:36s} {v}")
print(f"\nAntenas elegidas por el mejor GA (numeradas 1..{n}): {[j + 1 for j in mejor_sel]}")
print(f"Tiempo total: {(time.perf_counter() - t_total) / 60:.1f} min")