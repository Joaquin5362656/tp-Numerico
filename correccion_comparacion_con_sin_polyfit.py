import numpy as np
import matplotlib.pyplot as plt

"""
Comparamos los coeficientes obtenidos usando la funcion Polifit de numpy
con los coeficientes obtenidos por un algoritmo basado en el esquema de resolucion 
del metodo lineal de cuadrados minimos expresado en forma matricial
"""
nombre_archivo = "te_5mm.csv"  #CAMBIAR A CUALQUIER ARCHIVO DE Mediciones
path_archivo = f"{nombre_archivo}"

tiempos, alturas = np.loadtxt(path_archivo, delimiter=",", skiprows=1, unpack=True)

cota_error = 0.9  # 0.9 mm de cota de calibracion
contador = 0

for h in alturas:
    if h >= cota_error * 5:   #Descartamos aquellos valores menores a tener un 20% de la cota de error
        contador += 1

alturas = alturas[0:contador]
tiempos = tiempos[0:contador]

alturas_normalizadas = alturas / alturas[0]


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

exp_phi_1 = lambda t: 1
exp_phi_2 = lambda t: -t
exp_funciones_phi = [exp_phi_1, exp_phi_2]

#Ajuste Cuadratico
#Matriz A: producto interno entre funciones phi
cuad_A_array = np.zeros((3,3))
for i in range(len(cuad_funciones_phi)):
  for j in range(len(cuad_funciones_phi)):
    res = 0
    for t in tiempos:
      res += cuad_funciones_phi[i](t) * cuad_funciones_phi[j](t)
    cuad_A_array[i][j] = res

#Matriz b: producto interno entre funcion f y funciones phi
cuad_b_array = np.zeros((3, 1))
for i in range(len(cuad_funciones_phi)):
  res = 0
  for j in range(len(tiempos)):
    res += alturas_normalizadas[j] * cuad_funciones_phi[i](tiempos[j])
  cuad_b_array[i][0] = res

#Coeficientes obtenidos de A * c = b
cuad_c = np.linalg.solve(cuad_A_array, cuad_b_array)

f_cuadratico = lambda t: cuad_c[0][0] + cuad_c[1][0]*t + cuad_c[2][0]*t**2

cuad_alturas_calculadas = []
for t in tiempos:
  cuad_alturas_calculadas.append(f_cuadratico(t))

cuad_alturas_calculadas_normalizadas = []
for altura in cuad_alturas_calculadas:
  cuad_alturas_calculadas_normalizadas.append(altura/cuad_alturas_calculadas[0])

ecm_cuad_calculado = np.mean((alturas_normalizadas - cuad_alturas_calculadas_normalizadas)**2)


#Ajuste cubico
#Matriz A: producto interno entre funciones phi
cub_A_array = np.zeros((4,4))
for i in range(len(cub_funciones_phi)):
  for j in range(len(cub_funciones_phi)):
    res = 0
    for t in tiempos:
      res += cub_funciones_phi[i](t) * cub_funciones_phi[j](t)
    cub_A_array[i][j] = res

#Matriz b: producto interno entre funcion f y funciones phi
cub_b_array = np.zeros((4, 1))
for i in range(len(cub_funciones_phi)):
  res = 0
  for j in range(len(tiempos)):
    res += alturas_normalizadas[j] * cub_funciones_phi[i](tiempos[j])
  cub_b_array[i][0] = res

#Coeficientes obtenidos de A * c = b
cub_c = np.linalg.solve(cub_A_array, cub_b_array)

f_cub = lambda t: cub_c[0][0] + cub_c[1][0]*t + cub_c[2][0]*t**2 + cub_c[3][0]*t**3

cub_alturas_calculadas = []
for t in tiempos:
  cub_alturas_calculadas.append(f_cub(t))

cub_alturas_calculadas_normalizadas = []
for altura in cub_alturas_calculadas:
  cub_alturas_calculadas_normalizadas.append(altura/cub_alturas_calculadas[0])

ecm_cub_calculado = np.mean((alturas_normalizadas - cub_alturas_calculadas_normalizadas)**2)

#AJUSTE EXPONENCIAL
#Matriz A: producto interno entre funciones phi
exp_A_array = np.zeros((2,2))
for i in range(len(exp_funciones_phi)):
  for j in range(len(exp_funciones_phi)):
    res = 0
    for t in tiempos:
      res += exp_funciones_phi[i](t) * exp_funciones_phi[j](t)
    exp_A_array[i][j] = res

#Matriz b: producto interno entre funcion f y funciones phi
exp_b_array = np.zeros((2, 1))
for i in range(len(exp_funciones_phi)):
  res = 0
  for j in range(len(tiempos)):
    res += np.log(alturas_normalizadas[j]) * exp_funciones_phi[i](tiempos[j])
  exp_b_array[i][0] = res

exp_c = np.linalg.solve(exp_A_array, exp_b_array)

f_exp = lambda t: np.exp(exp_c[0][0] - exp_c[1][0]*t)

exp_alturas_calculadas = []
for t in tiempos:
  exp_alturas_calculadas.append(f_exp(t))

exp_alturas_calculadas_normalizadas = []
for altura in exp_alturas_calculadas:
  exp_alturas_calculadas_normalizadas.append(altura/exp_alturas_calculadas[0])

ecm_exp_calculado = np.mean((alturas_normalizadas - exp_alturas_calculadas_normalizadas)**2)

# AJUSTES POR CUADRADOS MINIMOS USANDO POLYFIT

# Ajuste cuadratico
# y = a + b*t + c*t^2
coef_quad = np.polyfit(tiempos, alturas_normalizadas, 2)
y_quad = np.polyval(coef_quad, tiempos)
ecm_quad = np.mean((alturas_normalizadas - y_quad)**2)

# Ajuste cubico
# y = a + b*t + c*t^2 + d*t^3
coef_cub = np.polyfit(tiempos, alturas_normalizadas, 3)
y_cub = np.polyval(coef_cub, tiempos)
ecm_cub = np.mean((alturas_normalizadas - y_cub)**2)

# Ajuste exponencial
# y = exp(a - b*t)
# (metodo linealizado)
mask = alturas_normalizadas > 0 
coef_exp = np.polyfit(tiempos[mask], np.log(alturas_normalizadas[mask]), 1)
B, A = coef_exp
a_exp, b_exp = A, -B
y_exp = np.exp(a_exp - b_exp * tiempos)
ecm_exp = np.mean((alturas_normalizadas - y_exp)**2)


#COMPARACION DE COEFICIENTES OBTENIDOS

print("-------- Comparacion de coeficientes obtenidos --------")

print("Ajuste cuadratico:")
print(f"No usando Polyfit:  a = {cuad_c[0][0]:.10f} - b = {cuad_c[1][0]:.10f} - c = {cuad_c[2][0]:.10f}")
print(f"Usando Polyfit:     a = {coef_quad[2]:.10f} - b = {coef_quad[1]:.10f} - c = {coef_quad[0]:.10f}")

print("\nAjuste cubico")
print(f"No usando Polyfit: a = {cub_c[0][0]:.10f} - b = {cub_c[1][0]:.10f} - c = {cub_c[2][0]:.10f} - d = {cub_c[3][0]:.10f}")
print(f"Usando Polyfit:    a = {coef_cub[3]:.10f} - b = {coef_cub[2]:.10f} - c = {coef_cub[1]:.10f} - d = {coef_cub[0]:.10f}")

print("\nAjuste exponencial")
print(f"No usando Polyfit: a = {exp_c[0][0]:.10f} - b = {exp_c[1][0]:.10f}")
print(f"Usando Polyfit:    a = {a_exp:.10f} - b = {b_exp:.10f}")


print("-------- Comparacion de ECM obtenidos --------")

print("Ajuste cuadratico:")
print(f"No usando Polyfit:  ECM Cuadratico = {ecm_cuad_calculado:.10f}")
print(f"Usando Polyfit:     ECM Cuadratico = {ecm_quad:.10f}")

print("\nAjuste cubico")
print(f"No usando Polyfit:  ECM Cubico = {ecm_cub_calculado:.10f}")
print(f"Usando Polyfit:     ECM Cubico = {ecm_cub:.10f}")

print("\nAjuste exponencial")
print(f"No usando Polyfit:  ECM Exponencial = {ecm_exp_calculado:.10f}")
print(f"Usando Polyfit:     ECM Exponencial = {ecm_exp:.10f}")



#GRAFICOS
plt.plot(tiempos, y_quad, '-r', label="Ajuste cuadratico usando Polyfit")
plt.plot(tiempos, cuad_alturas_calculadas_normalizadas, '--r', label="Ajuste cuadratico no usando Polyfit")
plt.plot(tiempos, y_cub, '-y', label="Ajuste cubico usando Polyfit")
plt.plot(tiempos, cub_alturas_calculadas_normalizadas, '--y', label="Ajuste cubico no usando Polyfit")
plt.plot(tiempos, y_exp, '-b', label="Ajuste exponencial usando Polyfit")
plt.plot(tiempos, exp_alturas_calculadas_normalizadas, '--b', label="Ajuste exponencial no usando Polyfit")

plt.xlabel("Tiempo [s]")
plt.ylabel("Altura normalizada h(t)/h0")
plt.title("Comparacion de Ajustes cuadraticos")
plt.legend()
plt.show()