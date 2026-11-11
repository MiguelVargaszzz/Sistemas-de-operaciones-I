"""
optimo.py - Corrida larga del método EXACTO para acotar el óptimo del Set Cover (500 x 500).

Usa HiGHS (solver exacto de ramificación y acotamiento, viene dentro de SciPy) porque el CBC
que trae PuLP en Windows se queda colgado en corridas largas y no deja registro.

Por qué este script no pierde los datos aunque algo falle:
  - HiGHS corre en un proceso aparte y escribe su avance EN VIVO en highs_log.txt.
  - Este script lee ese registro cada pocos segundos y guarda la mejor cota que vaya apareciendo.
  - Si HiGHS se pasa 10 min de su límite, el script lo detiene solo y usa la última cota leída.
  - Aunque uses Ctrl+C, se guarda lo que se llevaba.

Uso (en la carpeta del proyecto):
  python optimo.py            -> 1 hora
  python optimo.py 10800      -> 3 horas

Guarda:
  resultados_optimo.csv  : resumen (mejor costo, cota inferior, gap, ¿óptimo probado?)
  solucion_optimo.csv    : antenas de la mejor solución (numeradas desde 1, igual que las otras)
  evolucion_highs.csv    : mejor solución y cota inferior a lo largo del tiempo (para una gráfica)
  highs_log.txt          : registro completo de HiGHS
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

LOG = "highs_log.txt"
RES = "highs_resultado.json"


# ====================================================================== proceso hijo: HiGHS
def trabajador(limite):
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
    from datos import cargar_datos

    A, c = cargar_datos(verbose=False)
    m, n = A.shape
    t0 = time.perf_counter()
    r = milp(c.astype(float),
             constraints=LinearConstraint(csr_matrix(A.astype(float)), lb=1, ub=np.inf),
             integrality=np.ones(n), bounds=Bounds(0, 1),
             options={"disp": True, "time_limit": float(limite), "mip_rel_gap": 0.0})
    sel = [] if r.x is None else [int(j) for j in np.flatnonzero(r.x > 0.5)]
    with open(RES, "w", encoding="utf-8") as f:
        json.dump({"status": int(r.status), "mensaje": str(r.message),
                   "costo": None if r.x is None else float(r.fun),
                   "cota": None if getattr(r, "mip_dual_bound", None) is None else float(r.mip_dual_bound),
                   "seleccion": sel, "tiempo": time.perf_counter() - t0}, f)


# ====================================================================== lectura del registro
def leer_linea(linea):
    """Extrae (tiempo_s, cota, mejor_sol) de una fila de la tabla de B&B de HiGHS, o None."""
    t = linea.split()
    if len(t) < 8 or not t[-1].endswith("s"):
        return None
    try:
        seg = float(t[-1][:-1])
        i = next(k for k, v in enumerate(t) if v.endswith("%"))      # columna "Expl." (porcentaje)
        cota, sol = t[i + 1], t[i + 2]
        cota = float(cota) if cota not in ("-inf", "inf") else None
        sol = float(sol) if sol not in ("-inf", "inf") else None
        return seg, cota, sol
    except (ValueError, StopIteration, IndexError):
        return None


# ====================================================================== proceso principal
def principal():
    import pandas as pd
    from datos import cargar_datos
    from exacto import verificar_cobertura, cota_inferior_lp
    from ga import algoritmo_genetico

    try:
        import scipy
        from scipy.optimize import milp  # noqa: F401
    except ImportError:
        sys.exit("Falta SciPy (o es muy viejo). Instálalo con:  pip install --upgrade scipy")

    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
    A, c = cargar_datos()
    m, n = A.shape

    # ---------- mejor solución conocida (GA) ----------
    print("\n[1] Mejor solución conocida ...", flush=True)
    candidatas = {}
    for nombre, p in {"GA pm=3/n semilla 12": dict(pm=3 / n, seed=12),
                      "GA pc=1.0 semilla 8": dict(pc=1.0, seed=8)}.items():
        candidatas[nombre] = algoritmo_genetico(A, c, **p)["seleccion"]
    for ruta in ("solucion_ga.csv", "solucion_optimo.csv"):
        if os.path.exists(ruta):
            sel = [int(a) - 1 for a in pd.read_csv(ruta, encoding="utf-8-sig")["Antena"]]
            if all(0 <= j < n for j in sel):
                candidatas[ruta] = sel
    validas = {k: s for k, s in candidatas.items() if verificar_cobertura(A, s)}
    origen, mejor_sel = min(validas.items(), key=lambda kv: c[kv[1]].sum())
    mejor_conocido = int(c[mejor_sel].sum())
    print(f"    {origen}: costo {mejor_conocido}", flush=True)
    cota_lp = cota_inferior_lp(A, c)
    print(f"    Cota de la relajación lineal: {cota_lp:.1f}", flush=True)

    # ---------- lanzar HiGHS en un proceso aparte ----------
    for f in (LOG, RES):
        if os.path.exists(f):
            os.remove(f)
    print(f"\n[2] HiGHS (SciPy {scipy.__version__}) | límite: {limite} s ({limite / 3600:.1f} h)", flush=True)
    print("    Cada 5 min se muestra la cota inferior y la mejor solución de HiGHS.\n", flush=True)

    salida = open(LOG, "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--trabajador", str(limite)],
                            stdout=salida, stderr=subprocess.STDOUT, cwd=os.getcwd())

    t0, pos, ultimo_aviso = time.perf_counter(), 0, 0.0
    evolucion, cota_vivo, sol_vivo = [], None, None
    estado = "terminó"

    def leer_nuevo():
        nonlocal pos, cota_vivo, sol_vivo
        with open(LOG, encoding="utf-8", errors="ignore") as f:
            f.seek(pos)
            texto = f.read()
            pos = f.tell()
        for linea in texto.splitlines():
            r = leer_linea(linea)
            if r:
                seg, cota, sol = r
                if cota is not None:
                    cota_vivo = cota if cota_vivo is None else max(cota_vivo, cota)
                if sol is not None:
                    sol_vivo = sol if sol_vivo is None else min(sol_vivo, sol)
                evolucion.append({"Tiempo_s": seg, "Cota_inferior": cota, "Mejor_solucion_HiGHS": sol})

    try:
        while proc.poll() is None:
            time.sleep(5)
            leer_nuevo()
            trans = time.perf_counter() - t0
            if trans - ultimo_aviso >= 300:
                ultimo_aviso = trans
                ref = mejor_conocido if sol_vivo is None else min(mejor_conocido, sol_vivo)
                txt = "—" if cota_vivo is None else f"{cota_vivo:.1f} (gap {100 * (ref - cota_vivo) / ref:.1f} %)"
                print(f"    {trans / 60:5.0f} min | cota inferior {txt} | mejor HiGHS "
                      f"{'—' if sol_vivo is None else int(sol_vivo)}", flush=True)
            if trans > limite + 600:
                print("    HiGHS pasó su límite por más de 10 min: se detiene y se usa lo registrado.", flush=True)
                proc.kill()
                estado = "detenido por el script"
                break
    except KeyboardInterrupt:
        estado = "detenido con Ctrl+C"
        print("\n    Detenido con Ctrl+C: se guarda lo registrado hasta ahora.", flush=True)
        try:
            proc.kill()
        except Exception:
            pass
    proc.wait()
    salida.close()
    tiempo = time.perf_counter() - t0
    leer_nuevo()

    # ---------- combinar resultados ----------
    res = None
    if os.path.exists(RES):
        with open(RES, encoding="utf-8") as f:
            res = json.load(f)
        os.remove(RES)

    cotas = [cota_lp] + [v for v in (cota_vivo, res and res["cota"]) if v is not None]
    cota = max(cotas)
    if res and res["seleccion"] and verificar_cobertura(A, res["seleccion"]):
        if c[res["seleccion"]].sum() < mejor_conocido:
            mejor_sel, origen = res["seleccion"], "HiGHS"
    mejor = int(c[mejor_sel].sum())
    optimo = bool(res and res["status"] == 0) or cota >= mejor - 0.5
    if optimo:
        cota = float(mejor)
    gap = 100 * (mejor - cota) / mejor

    resumen = {
        "Solver": f"HiGHS (SciPy {scipy.__version__})",
        "Límite (s)": limite,
        "Tiempo real (s)": round(tiempo, 1),
        "Estado": (res["mensaje"] if res else estado),
        "Mejor costo HiGHS": None if sol_vivo is None and not (res and res["costo"]) else
                             int(min(v for v in (sol_vivo, res and res["costo"]) if v)),
        "Mejor solución conocida": mejor,
        "Origen mejor solución": origen,
        "Antenas": len(mejor_sel),
        "Cota LP": round(cota_lp, 1),
        "Cota inferior final": round(cota, 1),
        "Gap (%)": round(gap, 2),
        "Óptimo probado": optimo,
    }
    pd.DataFrame([resumen]).to_csv("resultados_optimo.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Antena": [j + 1 for j in mejor_sel], "Costo": [int(c[j]) for j in mejor_sel]}
                 ).to_csv("solucion_optimo.csv", index=False, encoding="utf-8-sig")
    if evolucion:
        pd.DataFrame(evolucion).to_csv("evolucion_highs.csv", index=False, encoding="utf-8-sig")

    print("\n================ RESUMEN ================")
    for k, v in resumen.items():
        print(f"  {k:<26} {v}")
    if optimo:
        print(f"\n  Se PROBÓ que {mejor} es el ÓPTIMO.")
    else:
        print(f"\n  No se probó el óptimo. Lo seguro: el óptimo está entre {cota:.0f} y {mejor}.")
    print("  Archivos: resultados_optimo.csv, solucion_optimo.csv, evolucion_highs.csv, highs_log.txt")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--trabajador":
        trabajador(int(sys.argv[2]))
    else:
        principal()
