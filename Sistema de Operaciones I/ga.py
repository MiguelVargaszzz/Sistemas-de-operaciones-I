import time
import numpy as np


def _preparar(A, c):
    """Estructuras auxiliares para que la reparación sea rápida."""
    m, n = A.shape
    Af = A.astype(np.float32)
    cols = [np.flatnonzero(A[:, j]) for j in range(n)]     # clientes que cubre cada antena
    filas = [np.flatnonzero(A[i]) for i in range(m)]       # antenas que cubren a cada cliente
    return Af, cols, filas, np.asarray(c, dtype=np.float64)


def reparar(x, Af, cols, filas, c, rng, orden_drop="costo"):
    """
    Operador de reparación: convierte x en una solución factible y sin antenas redundantes.
      ADD : mientras haya clientes sin cubrir, toma uno al azar y agrega la antena que lo cubre
            con menor costo por cliente nuevo cubierto.
      DROP: recorre las antenas elegidas (de la más cara a la más barata) y quita las redundantes.
    """
    x = x.copy()
    cob = (Af @ x.astype(np.float32)).astype(np.int32)      # veces que está cubierto cada cliente
    while True:
        sin = np.flatnonzero(cob == 0)
        if sin.size == 0:
            break
        i = sin[rng.integers(sin.size)]
        cand = filas[i]
        nuevos = Af[sin][:, cand].sum(axis=0)                # clientes nuevos que cubriría cada candidata
        j = cand[np.argmin(c[cand] / nuevos)]
        x[j] = 1
        cob[cols[j]] += 1
    elegidas = np.flatnonzero(x)
    if orden_drop == "azar":
        elegidas = rng.permutation(elegidas)
    else:
        elegidas = elegidas[np.argsort(-c[elegidas])]
    for j in elegidas:
        idx = cols[j]
        if cob[idx].min() > 1:                               # todos sus clientes tienen otra antena
            x[j] = 0
            cob[idx] -= 1
    return x


def algoritmo_genetico(A, c, tam_pob=100, generaciones=1000, pc=0.9, pm=None, k_torneo=3,
                       elitismo=2, paciencia=200, p_inicial=0.5, orden_drop="costo",
                       evitar_duplicados=True, seed=None):
    """
    Algoritmo genético para el Set Cover con costos.
      Codificación : vector binario de longitud n (gen j = 1 si se instala la antena j).
      Fitness      : costo total sum(c_j * x_j) (se minimiza). Todas las soluciones son factibles
                     porque cada individuo pasa por el operador de reparación.
      Inicialización: cada gen = 1 con probabilidad p_inicial, luego reparación.
      Selección    : torneo de tamaño k_torneo.
      Cruce        : uniforme, con probabilidad pc.
      Mutación     : bit-flip con probabilidad pm por gen (por defecto 1/n), luego reparación.
      Elitismo     : los mejores individuos pasan intactos a la siguiente generación.
      Parada       : máximo de generaciones, o 'paciencia' generaciones seguidas sin mejora.
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    if pm is None:
        pm = 1.0 / n
    Af, cols, filas, cf = _preparar(A, c)

    t0 = time.perf_counter()
    pob = np.array([reparar((rng.random(n) < p_inicial).astype(np.int8), Af, cols, filas, cf, rng, orden_drop)
                    for _ in range(tam_pob)])
    fit = pob @ cf

    mejor_costo = fit.min()
    t_mejor, gen_mejor = time.perf_counter() - t0, 0
    hist_mejor, hist_prom = [mejor_costo], [fit.mean()]
    sin_mejora = 0

    for g in range(generaciones):
        orden = np.argsort(fit)
        nueva = [pob[i].copy() for i in orden[:elitismo]]
        vistos = {p.tobytes() for p in nueva}
        intentos, max_intentos = 0, 20 * tam_pob

        while len(nueva) < tam_pob:
            i1 = rng.integers(0, tam_pob, k_torneo)
            i2 = rng.integers(0, tam_pob, k_torneo)
            p1, p2 = pob[i1[np.argmin(fit[i1])]], pob[i2[np.argmin(fit[i2])]]

            if rng.random() < pc:                            # cruce uniforme
                h = np.where(rng.random(n) < 0.5, p1, p2)
            else:
                h = p1.copy()
            h = np.where(rng.random(n) < pm, 1 - h, h).astype(np.int8)   # mutación bit-flip
            h = reparar(h, Af, cols, filas, cf, rng, orden_drop)

            intentos += 1
            clave = h.tobytes()
            if evitar_duplicados and intentos < max_intentos and clave in vistos:
                continue                                      # evita clones para conservar diversidad
            vistos.add(clave)
            nueva.append(h)

        pob = np.array(nueva)
        fit = pob @ cf
        actual = fit.min()
        hist_mejor.append(actual)
        hist_prom.append(fit.mean())

        if actual < mejor_costo:
            mejor_costo, t_mejor, gen_mejor, sin_mejora = actual, time.perf_counter() - t0, g + 1, 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    tiempo = time.perf_counter() - t0
    mejor = pob[np.argmin(fit)]
    return {
        "costo": int(round(fit.min())),
        "seleccion": [int(j) for j in np.flatnonzero(mejor)],
        "tiempo": tiempo,
        "t_mejor": t_mejor,
        "gen_mejor": gen_mejor,
        "generaciones": len(hist_mejor) - 1,
        "hist_mejor": hist_mejor,
        "hist_prom": hist_prom,
    }