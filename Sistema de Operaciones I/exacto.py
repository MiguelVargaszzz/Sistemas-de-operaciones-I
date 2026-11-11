import os
import re
import time
import numpy as np
import pulp


def verificar_cobertura(A, seleccion):
    """True si las antenas seleccionadas cubren a todos los clientes."""
    if len(seleccion) == 0:
        return False
    return bool(np.all(A[:, seleccion].sum(axis=1) >= 1))


def resolver_exacto(A, c, tiempo_limite=None, mensajes=False, hilos=None):
    """
    Modelo de programación lineal entera del Set Cover resuelto con CBC (PuLP).
    tiempo_limite: segundos (None = hasta probar optimalidad).
    Devuelve el costo de la mejor solución hallada, si se PROBÓ que es óptima y la cota inferior de CBC.
    """
    m, n = A.shape
    prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(n)]
    prob += pulp.lpSum(int(c[j]) * x[j] for j in range(n)), "CostoTotal"
    for i in range(m):
        prob += pulp.lpSum(x[j] for j in range(n) if A[i, j] == 1) >= 1, f"cob_{i}"

    log = f"cbc_log_{os.getpid()}.txt"
    try:
        solver = pulp.PULP_CBC_CMD(msg=mensajes, timeLimit=tiempo_limite, threads=hilos, logPath=log)
    except TypeError:                              # versiones de PuLP sin logPath
        solver = pulp.PULP_CBC_CMD(msg=mensajes, timeLimit=tiempo_limite, threads=hilos)
    t0 = time.perf_counter()
    error = None
    try:
        prob.solve(solver)
    except Exception as e:                         # p. ej. CBC se detuvo sin ninguna solución entera
        error = str(e)
    tiempo = time.perf_counter() - t0

    seleccion = [j for j in range(n) if x[j].value() is not None and x[j].value() > 0.5]
    factible = verificar_cobertura(A, seleccion)
    costo = int(sum(c[j] for j in seleccion)) if factible else None

    # Con límite de tiempo, PuLP puede decir "Optimal" aunque CBC solo halló una solución factible.
    # Por eso se mira sol_status (y que no se haya agotado el tiempo).
    optimo = (error is None and prob.sol_status == pulp.LpSolutionOptimal
              and (tiempo_limite is None or tiempo < tiempo_limite - 1))

    cota_cbc = None                                # cota inferior que reporta CBC en su registro
    try:
        with open(log, encoding="utf-8", errors="ignore") as f:
            r = re.search(r"Lower bound:\s+([-+0-9.eE]+)", f.read())
        if r:
            cota_cbc = float(r.group(1))
        os.remove(log)
    except Exception:
        pass
    if optimo and costo is not None:
        cota_cbc = float(costo)

    return {"estado": pulp.LpStatus[prob.status], "costo": costo, "seleccion": seleccion,
            "tiempo": tiempo, "factible": factible, "optimo_probado": optimo, "cota_cbc": cota_cbc}


def cota_inferior_lp(A, c):
    """Relajación lineal (0 <= x <= 1): cota inferior válida del óptimo."""
    m, n = A.shape
    prob = pulp.LpProblem("SetCover_LP", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1) for j in range(n)]
    prob += pulp.lpSum(int(c[j]) * x[j] for j in range(n))
    for i in range(m):
        prob += pulp.lpSum(x[j] for j in range(n) if A[i, j] == 1) >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.value(prob.objective)