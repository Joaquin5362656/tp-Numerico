import numpy as np
import matplotlib.pyplot as plt

"""
Comparamos los coeficientes obtenidos usando la funcion Polifit de numpy
con los coeficientes obtenidos por un algoritmo basado en el esquema de resolucion 
del metodo lineal de cuadrados minimos expresado en forma matricial
"""


nombre_archivo = "te_4mm.csv"  #CAMBIAR A CUALQUIER ARCHIVO DE Mediciones
path_archivo = f"./Mediciones/{nombre_archivo}"


tiempos_te_6mm, alturas_te_6mm = np.loadtxt(path_archivo, delimiter=",", skiprows=1, unpack=True)
alturas_normalizadas_te_6mm = alturas_te_6mm / alturas_te_6mm[0]


# AJUSTE POR CUADRADOS MINIMOS SIN USAR POLYFIT

# FUNCIONES PHI
cuad_phi_1 = lambda t: 1
cuad_phi_2 = lambda t: t
cuad_phi_3 = lambda t: t**2
cuad_funciones_phi = [cuad_phi_1, cuad_phi_2, cuad_phi_3]

cub_phi_1 = lambda t: 1
cub_phi_2 = lambda t: t
cub_phi_3 = lambda t: t**2
cub_phi_4 = lambda t: t**3
cub_funciones_phi = [cub_phi_1, cub_phi_2, cub_phi_3, cub_phi_4]


#Ajuste Cuadratico
#Matriz A: producto interno entre funciones phi
cuad_A_array = np.zeros((3,3))
for i in range(len(cuad_funciones_phi)):
  for j in range(len(cuad_funciones_phi)):
    res = 0
    for t in tiempos_te_6mm:
      res += cuad_funciones_phi[i](t) * cuad_funciones_phi[j](t)
    cuad_A_array[i][j] = res

#Matriz b: producto interno entre funcion f y funciones phi
cuad_b_array = np.zeros((3, 1))
for i in range(len(cuad_funciones_phi)):
  res = 0
  for j in range(len(tiempos_te_6mm)):
    res += alturas_normalizadas_te_6mm[j] * cuad_funciones_phi[i](tiempos_te_6mm[j])
  cuad_b_array[i][0] = res

#Coeficientes obtenidos de A * c = b
cuad_c = np.linalg.solve(cuad_A_array, cuad_b_array)

f_cuadratico = lambda t: cuad_c[0][0] + cuad_c[1][0]*t + cuad_c[2][0]*t**2

cuad_alturas_calculadas_te_6mm = []
for t in tiempos_te_6mm:
  cuad_alturas_calculadas_te_6mm.append(f_cuadratico(t))

cuad_alturas_calculadas_normalizadas_te_6mm = []
for altura in cuad_alturas_calculadas_te_6mm:
  cuad_alturas_calculadas_normalizadas_te_6mm.append(altura/cuad_alturas_calculadas_te_6mm[0])


#Ajuste cubico
#Matriz A: producto interno entre funciones phi
cub_A_array = np.zeros((4,4))
for i in range(len(cub_funciones_phi)):
  for j in range(len(cub_funciones_phi)):
    res = 0
    for t in tiempos_te_6mm:
      res += cub_funciones_phi[i](t) * cub_funciones_phi[j](t)
    cub_A_array[i][j] = res

#Matriz b: producto interno entre funcion f y funciones phi
cub_b_array = np.zeros((4, 1))
for i in range(len(cub_funciones_phi)):
  res = 0
  for j in range(len(tiempos_te_6mm)):
    res += alturas_normalizadas_te_6mm[j] * cub_funciones_phi[i](tiempos_te_6mm[j])
  cub_b_array[i][0] = res

#Coeficientes obtenidos de A * c = b
cub_c = np.linalg.solve(cub_A_array, cub_b_array)
print(f"a: {cub_c[0][0]} - b: {cub_c[1][0]} - c: {cub_c[2][0]} - d: {cub_c[3][0]}")

f_cub = lambda t: cub_c[0][0] + cub_c[1][0]*t + cub_c[2][0]*t**2 + cub_c[3][0]*t**3

cub_alturas_calculadas_te_6mm = []
for t in tiempos_te_6mm:
  cub_alturas_calculadas_te_6mm.append(f_cub(t))

cub_alturas_calculadas_normalizadas_te_6mm = []
for altura in cub_alturas_calculadas_te_6mm:
  cub_alturas_calculadas_normalizadas_te_6mm.append(altura/cub_alturas_calculadas_te_6mm[0])



# AJUSTES POR CUADRADOS MINIMOS USANDO POLYFIT


# Ajuste cuadratico
# y = a + b*t + c*t^2
coef_quad = np.polyfit(tiempos_te_6mm, alturas_normalizadas_te_6mm, 2)
y_quad = np.polyval(coef_quad, tiempos_te_6mm)

# Ajuste cubico
# y = a + b*t + c*t^2 + d*t^3
coef_cub = np.polyfit(tiempos_te_6mm, alturas_normalizadas_te_6mm, 3)
y_cub = np.polyval(coef_cub, tiempos_te_6mm)



#COMPARACION DE COEFICIENTES OBTENIDOS

print("-------- Comparacion de coeficientes obtenidos --------")

print("Ajuste cuadratico:")
print(f"No usando Polyfit:  a = {cuad_c[0][0]:.10f} - b = {cuad_c[1][0]:.10f} - c = {cuad_c[2][0]:.10f}")
print(f"Usando Polyfit:     a = {coef_quad[2]:.10f} - b = {coef_quad[1]:.10f} - c = {coef_quad[0]:.10f}")

print("\nAjuste cubico")
print(f"No usando Polyfit: a = {cub_c[0][0]:.10f} - b = {cub_c[1][0]:.10f} - c = {cub_c[2][0]:.10f} - d = {cub_c[3][0]:.10f}")
print(f"Usando Polyfit:    a = {coef_cub[3]:.10f} - b = {coef_cub[2]:.10f} - c = {coef_cub[1]:.10f} - d = {coef_cub[0]:.10f}")

#GRAFICOS
plt.plot(tiempos_te_6mm, y_quad, '-r', label="Ajuste cuadratico usando Polyfit")
plt.plot(tiempos_te_6mm, cuad_alturas_calculadas_normalizadas_te_6mm, '-b', label="Ajuste cuadratico no usando Polyfit")
plt.plot(tiempos_te_6mm, y_cub, '-y', label="Ajuste cubico usando Polyfit")
plt.plot(tiempos_te_6mm, cub_alturas_calculadas_normalizadas_te_6mm, '-g', label="Ajuste cubico no usando Polyfit")

plt.xlabel("Tiempo [s]")
plt.ylabel("Altura normalizada h(t)/h0")
plt.title("Comparacion de Ajustes cuadraticos")
plt.legend()
plt.show()