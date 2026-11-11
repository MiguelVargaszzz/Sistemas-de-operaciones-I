import numpy as np
from exacto import resolver_exacto
from ga import algoritmo_genetico
from datos import cargar_datos

# Ejemplo pequeño del Paso 1 (5 clientes, 4 antenas). Óptimo conocido: costo 6 con las antenas 0 y 2
A = np.array([
    [1, 0, 0, 1],   # cliente 1: antenas 1 y 4
    [1, 1, 0, 0],   # cliente 2: antenas 1 y 2
    [1, 0, 1, 0],   # cliente 3: antenas 1 y 3
    [0, 1, 1, 0],   # cliente 4: antenas 2 y 3
    [0, 0, 1, 1],   # cliente 5: antenas 3 y 4
])
c = np.array([3, 2, 3, 2])

print("=== 1) Solver exacto (esperado: costo 6, seleccion [0, 2], optimo_probado True) ===")
ex = resolver_exacto(A, c)
print(ex)

print("\n=== 2) Algoritmo genético (esperado: costo 6, seleccion [0, 2]) ===")
ga = algoritmo_genetico(A, c, tam_pob=20, seed=1)
print({k: ga[k] for k in ("costo", "seleccion", "generaciones", "tiempo")})

print("\n=== 3) Carga de los datos del profesor ===")
A2, c2 = cargar_datos("set_cover_500x500.csv", "Costo_S.xlsx")

ok = (ex["costo"] == 6 and ex["seleccion"] == [0, 2] and ex["optimo_probado"]
      and ga["costo"] == 6 and A2.shape == (500, 500) and len(c2) == 500)
print("\nTODO OK" if ok else "\nREVISAR: algún resultado no coincide con lo esperado")