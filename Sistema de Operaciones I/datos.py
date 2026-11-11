import numpy as np


def cargar_datos(ruta_csv="set_cover_500x500.csv", ruta_xlsx="Costo_S.xlsx", verbose=True):
    """
    Lee los datos del profesor.
      - CSV : matriz A (filas = clientes, columnas = antenas). A[i, j] = 1 si la antena j cubre al cliente i.
      - XLSX: fila "Costo" con el costo de cada antena (500 valores).
    Devuelve A (m x n, enteros 0/1) y c (vector de n costos enteros).
    """
    # --- Matriz de cobertura (el CSV trae una primera fila de encabezado 0,1,2,...) ---
    primera = np.loadtxt(ruta_csv, delimiter=",", max_rows=1)
    saltar = 1 if np.max(primera) > 1 else 0
    A = np.loadtxt(ruta_csv, delimiter=",", skiprows=saltar, dtype=int, ndmin=2)

    # --- Costos (se lee el valor ya calculado de las celdas) ---
    import openpyxl
    hoja = openpyxl.load_workbook(ruta_xlsx, data_only=True).active
    fila_costo = None
    for fila in hoja.iter_rows(values_only=True):
        if fila and isinstance(fila[0], str) and fila[0].strip().lower().startswith("costo"):
            fila_costo = fila
            break
    if fila_costo is None:                      # plan B: la segunda fila
        fila_costo = next(hoja.iter_rows(min_row=2, max_row=2, values_only=True))
    c = np.array([v for v in fila_costo[1:] if isinstance(v, (int, float))], dtype=float)
    c = np.rint(c).astype(int)

    # --- Validaciones ---
    m, n = A.shape
    assert set(np.unique(A)) <= {0, 1}, "La matriz A debe contener solo 0 y 1"
    assert len(c) == n, f"Hay {n} columnas (antenas) pero {len(c)} costos"
    assert np.all(A.sum(axis=1) >= 1), "Hay clientes que ninguna antena cubre: instancia infactible"
    if verbose:
        print(f"Datos cargados: {m} clientes x {n} antenas | densidad {A.mean():.3f} | "
              f"costos entre {c.min()} y {c.max()} (promedio {c.mean():.0f})")
    return A, c