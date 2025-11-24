import numpy as np
import matplotlib.pyplot as plt


nombre_archivo = "aceite_6mm.csv"
path_archivo = f"./Mediciones/{nombre_archivo}"

# 1. CARGA DE DATOS
t, h = np.loadtxt(path_archivo, delimiter=",", skiprows=1, unpack=True)


# Procesamiento de datos
cota_error = 0.9  # 0.9 mm de cota de calibracion
contador = 0


for h1 in h:
    if h1 >= cota_error * 5:   #Descartamos aquellos valores menores a tener un 20% de la cota de error
        contador += 1

h = h[0:contador]
t = t[0:contador]


h0 = h[0]               # altura inicial
y = h / h0              # altura normalizada h(t)/h0



# 2. AJUSTES POR CUADRADOS MIN

# ---- Ajuste cuadratico ----
# y = a + b*t + c*t^2
coef_quad = np.polyfit(t, y, 2)
y_quad = np.polyval(coef_quad, t)
ecm_quad = np.mean((y - y_quad)**2)


# ---- Ajuste cubico ----
# y = a + b*t + c*t^2 + d*t^3
coef_cub = np.polyfit(t, y, 3)
y_cub = np.polyval(coef_cub, t)
ecm_cub = np.mean((y - y_cub)**2)


# ---- Ajuste exponencial ----
# y = exp(a - b*t)
# (metodo linealizado)
mask = y > 0 
coef_exp = np.polyfit(t[mask], np.log(y[mask]), 1)
B, A = coef_exp
a_exp, b_exp = A, -B
y_exp = np.exp(a_exp - b_exp * t)
ecm_exp = np.mean((y - y_exp)**2)



# 3. MODELO TEORICO
# h(t) = h0 * (1 - t/tf)^2
diam_tanque = 71.283
area_1 = 3.1415 * (diam_tanque**2)/4


diam_4mm = 4.069
diam_5mm = 5.071
diam_6mm = 5.982

diam_orificio_usado = diam_4mm

if "5mm" in nombre_archivo:
    diam_orificio_usado = diam_5mm
elif "6mm" in nombre_archivo:
    diam_orificio_usado = diam_6mm

area_2 = 3.1415 * (diam_orificio_usado**2)/4

g = 9800 #en mm/s^2

tf = np.sqrt(2 * (h0/g) * ((area_1**2 / area_2**2) - 1))
t_teo = t[t <= tf]
y_aux = h0 * (1 - t_teo/tf)**2
y0 = y_aux[0]
y_teo = y_aux / y0




# 4. GRAFICAR RESULTADOS
plt.figure(figsize=(8, 5))
plt.plot(t, y, 'xr', label="Datos experimentales")       
plt.plot(t, y_quad, '-b', label="Ajuste cuadratico")      
plt.plot(t, y_cub, ':m', label="Ajuste cubico")             
plt.plot(t, y_exp, '--g', label="Ajuste exponencial")      
plt.plot(t_teo, y_teo, 'k>', label="Modelo teorico") 

plt.xlabel("Tiempo [s]")
plt.ylabel("Altura normalizada h(t)/h0")
plt.title("Vaciado del tanque – Ajustes por cuadrados minimos")
plt.legend()
plt.grid(True)
plt.tight_layout()
# captura del grafico
plt.savefig("ajuste.png")
plt.show()



# 5. MOSTRAR RESULTADOS NUMERICOS
print("=== Resultados de los ajustes ===")
print(f"Cuadratico: a={coef_quad[2]:.6f}, b={coef_quad[1]:.6f}, c={coef_quad[0]:.6f}")
print(f"ECM Cuadratico = {ecm_quad:.6e}")

print(f"\nCubico: a={coef_cub[3]:.6f}, b={coef_cub[2]:.6f}, c={coef_cub[1]:.6f}, d={coef_cub[0]:.6f}")
print(f"ECM Cubico = {ecm_cub:.6e}")

print(f"\nExponencial: a_exp={a_exp:.6f}, b_exp={b_exp:.6f}")
print(f"ECM Exponencial = {ecm_exp:.6e}")



# 6. TABLA RESUMEN
print("\n=== Comparacion de errores (ECM) ===")
print(f"{'Modelo':<15}{'ECM':>15}")
print("-" * 30)
print(f"{'Cuadratico':<15}{ecm_quad:>15.6e}")
print(f"{'Cubico':<15}{ecm_cub:>15.6e}")
print(f"{'Exponencial':<15}{ecm_exp:>15.6e}")