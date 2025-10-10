import numpy as np


tiempos_te_6mm, alturas_te_6mm = np.loadtxt("./Mediciones/te_6mm.csv", delimiter=",", skiprows=1, unpack=True)
alturas_normalizadas_te_6mm = alturas_te_6mm / alturas_te_6mm[0]

exp_phi_1 = lambda t: 1
exp_phi_2 = lambda t: -t
exp_funciones_phi = [exp_phi_1, exp_phi_2]

#Matriz A
exp_A_array = np.zeros((2,2))
for i in range(len(exp_funciones_phi)):
  for j in range(len(exp_funciones_phi)):
    res = 0
    for t in tiempos_te_6mm:
      res += exp_funciones_phi[i](t) * exp_funciones_phi[j](t)
    exp_A_array[i][j] = res

#Matriz b
exp_b_array = np.zeros((2, 1))
for i in range(len(exp_funciones_phi)):
  res = 0
  for j in range(len(tiempos_te_6mm)):
    res += np.log(alturas_normalizadas_te_6mm[j]) * exp_funciones_phi[i](tiempos_te_6mm[j])
  exp_b_array[i][0] = res

#Matriz c
exp_c = np.linalg.solve(exp_A_array, exp_b_array) # Resuelve el sistema de ecuaciones A*c = b
print(f"a: {exp_c[0][0]} - b: {exp_c[1][0]}")

f_exp = lambda t: np.exp(exp_c[0][0] - exp_c[1][0]*t)


exp_alturas_calculadas_te_6mm = []
for t in tiempos_te_6mm:
  exp_alturas_calculadas_te_6mm.append(f_exp(t))


exp_alturas_calculadas_normalizadas_te_6mm = []
for altura in exp_alturas_calculadas_te_6mm:
  exp_alturas_calculadas_normalizadas_te_6mm.append(altura/exp_alturas_calculadas_te_6mm[0])


error_cuadratico = 0
m = len(tiempos_te_6mm)
for i in range(len(tiempos_te_6mm)):
  error_cuadratico += (alturas_normalizadas_te_6mm[i] - exp_alturas_calculadas_normalizadas_te_6mm[i])**2

ecm_exp_te = error_cuadratico / m
print(f"El Error cuadratico medio es {ecm_exp_te}")
print(f"Usando np.mean: el ECM es {np.mean((alturas_normalizadas_te_6mm - exp_alturas_calculadas_normalizadas_te_6mm)**2)}")