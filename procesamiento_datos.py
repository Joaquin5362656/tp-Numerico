import numpy as np
import matplotlib.pyplot as plt


# 1. CARGA DE DATOS
t, h = np.loadtxt("aceite_5mm.csv", delimiter=",", skiprows=1, unpack=True)

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
tf = t[-1]  # tiempo tot (cuando el tanque se vacia)
y_teo = (1 - t/tf)**2



# 4. GRAFICAR RESULTADOS
plt.figure(figsize=(8, 5))
plt.plot(t, y, 'xr', label="Datos experimentales")       
plt.plot(t, y_quad, '-b', label="Ajuste cuadratico")      
plt.plot(t, y_cub, ':m', label="Ajuste cubico")             
plt.plot(t, y_exp, '--g', label="Ajuste exponencial")      
plt.plot(t, y_teo, 'k>', label="Modelo teorico") 

plt.xlabel("Tiempo [s]")
plt.ylabel("Altura normalizada h(t)/h0")
plt.title("Vaciado del tanque – Ajustes por cuadrados minimos")
plt.legend()
plt.grid(True)
plt.tight_layout()
# captura del grafico
plt.savefig("ajuste_aceite_5mm.png")
plt.show()



# 5. MOSTRAR RESULTADOS NUMERICOS
print("=== Resultados de los ajustes ===")
print(f"Cuadratico: a={coef_quad[2]:.6f}, b={coef_quad[1]:.6f}, c={coef_quad[0]:.6f}")
print(f"ECM Cuadratico = {ecm_quad:.6e}")

print(f"\nCCubico: a={coef_cub[3]:.6f}, b={coef_cub[2]:.6f}, c={coef_cub[1]:.6f}, d={coef_cub[0]:.6f}")
print(f"ECM CCubico = {ecm_cub:.6e}")

print(f"\nExponencial: a_exp={a_exp:.6f}, b_exp={b_exp:.6f}")
print(f"ECM Exponencial = {ecm_exp:.6e}")



# 6. TABLA RESUMEN
print("\n=== Comparacion de errores (ECM) ===")
print(f"{'Modelo':<15}{'ECM':>15}")
print("-" * 30)
print(f"{'Cuadratico':<15}{ecm_quad:>15.6e}")
print(f"{'Cubico':<15}{ecm_cub:>15.6e}")
print(f"{'Exponencial':<15}{ecm_exp:>15.6e}")