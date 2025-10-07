import numpy as np
import matplotlib.pyplot as plt
import math

# 1. CARGA DE DATOS
t, h = np.loadtxt("te_4mm.csv", delimiter=",", skiprows=1, unpack=True)

h0 = h[0]
y = h / h0  # altura normalizada



# 2. AJUSTE CUBICO (reutilizando el del punto 3)
coef_cub = np.polyfit(t, y, 3)
y_cub = np.polyval(coef_cub, t)



# 3. MODELO TEORICO (Torricelli)
tf = t[-1]
y_teo = (1 - t/tf)**2



# 4. FUNCIONES AUXILIARES
def tiempo_cubico(coef, proporcion, t_max):
    """Resuelve el tiempo donde el ajuste cubico alcanza h/h0 = proporcion."""
    d, c, b, a = coef
    roots = np.roots([d, c, b, a - proporcion])
    reales = [r.real for r in roots if abs(r.imag) < 1e-6 and 0 <= r.real <= t_max]
    return min(reales) if reales else np.nan

def tiempo_teorico(tf, proporcion):
    """Tiempo teorico de vaciado según ecuacion (1): h/h0 = (1 - t/tf)^2"""
    return tf * (1 - math.sqrt(proporcion))



# 5. PARAMETROS DE INCERTIDUMBRE Y FPS
fps = 30.0
dt_frame = 1 / fps  # 1 frame ≈ 0.033 s



# 6. CALCULO DE TIEMPOS PARA 50% Y 90%
puntos = {
    "Mitad del tanque (50%)": 0.5,
    "90% del tanque vaciado": 0.1
}

resultados = {}
for descripcion, p in puntos.items():
    t_teo = tiempo_teorico(tf, p)
    dt_teo = (1 - math.sqrt(p)) * dt_frame
    t_cub = tiempo_cubico(coef_cub, p, tf)
    dt_cub = dt_frame

    resultados[descripcion] = {
        "p": p,
        "t_teo": t_teo,
        "dt_teo": dt_teo,
        "t_cub": t_cub,
        "dt_cub": dt_cub
    }



# 7. IMPRESION DE RESULTADOS
print("=== Estimacion de tiempos teoricos vs ajuste cubico ===")
print(f"{'Caso':<30}{'t_teo [s]':>12}{'±dt_teo [s]':>15}{'t_cub [s]':>12}{'±dt_cub [s]':>15}")
print("-" * 84)
for desc, datos in resultados.items():
    print(f"{desc:<30}{datos['t_teo']:>12.3f}{datos['dt_teo']:>15.3f}{datos['t_cub']:>12.3f}{datos['dt_cub']:>15.3f}")

print("\n=== Analisis comparativo ===")
for desc, datos in resultados.items():
    diferencia = abs(datos["t_teo"] - datos["t_cub"])
    print(f"→ {desc}: diferencia = {diferencia:.3f} s entre teoria y ajuste cubico.")
    if diferencia <= (datos["dt_teo"] + datos["dt_cub"]):
        print("Coinciden dentro del margen de incertidumbre.")
    else:
        print("Diferencia mayor a la incertidumbre: posibles perdidas o errores experimentales.")



# 8. GRAFICAR RESULTADOS Y TIEMPOS CLAVE
plt.figure(figsize=(9, 6))

# Datos experimentales
plt.plot(t, y, 'xr', label="Datos experimentales")

# Ajuste cubico y modelo teorico
plt.plot(t, y_cub, ':m', label="Ajuste cubico")
plt.plot(t, y_teo, '--g', label="Modelo teorico")

# Lineas verticales y puntos en t50 y t90
for desc, datos in resultados.items():
    p = datos["p"]
    # Lineas teoricas
    plt.axvline(x=datos["t_teo"], color='g', linestyle='--', alpha=0.6)
    plt.scatter(datos["t_teo"], p, color='g', marker='o', label=f"{desc} teorico")
    # Lineas cubicas
    plt.axvline(x=datos["t_cub"], color='m', linestyle=':', alpha=0.6)
    plt.scatter(datos["t_cub"], p, color='m', marker='s', label=f"{desc} cubico")

plt.xlabel("Tiempo [s]")
plt.ylabel("Altura normalizada h(t)/h0")
plt.title("Vaciado del tanque – Comparacion entre modelo teorico y ajuste cubico")
plt.grid(True)
plt.legend(loc="best", fontsize=8)
plt.tight_layout()


# Guardar y mosttrar grafico
plt.savefig("tiempos_vaciado_te_4mm.png", dpi=300)
plt.show()
