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

h0 = h[0]
y = h / h0  # altura normalizada



# 2. AJUSTE CUBICO (reutilizando el del punto 3)
coef_cub = np.polyfit(t, y, 3)
y_cub = np.polyval(coef_cub, t)
tf_cub = t[-1]



# 3. MODELO TEORICO (Torricelli)
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
array_t_teo = t[t <= tf]
y_aux = h0 * (1 - array_t_teo/tf)**2
y0 = y_aux[0]
y_teo = y_aux / y0



# 4. FUNCIONES AUXILIARES
def tiempo_cubico(coef, proporcion, t_max):
    """Resuelve el tiempo donde el ajuste cubico alcanza h/h0 = proporcion."""
    
    d, c, b, a = coef
    f = lambda t: a - proporcion + b*t + c * t**2 + d * t**3
    
    #METODO DE BISECCION
    a0 = ai = 0.0
    b0 = bi = t_max
    m = (ai + bi) / 2
    iteraciones = i_totales = 16 # 16 para tener una cota menor a 0.0005 --> (tres decimales)
    while iteraciones >= 0:
        if f(ai) * f(m) < 0:
            bi = m
            m = (ai + bi) / 2
        elif f(ai) * f(m) > 0:
            ai = m 
            m = (ai + bi) / 2
        iteraciones -= 1

    cota = (b0 - a0) / (2**(i_totales+1))
    cota_objetivo = 5
    while cota_objetivo >= cota:
        cota_objetivo /= 10

    return m, cota_objetivo * 10
    
def tiempo_teorico(tf, proporcion):
    """Tiempo teorico de vaciado según ecuacion (1): h/h0 = (1 - t/tf)^2"""
    return tf * (1 - np.sqrt(proporcion))


def dt_tiempo_teorico(proporcion):
    cota_pi = 0.00001
    cota_area = lambda d: np.abs(d**2 / 4) * cota_pi + np.abs(3.1415 * d / 2) * cota_error  
    tf_prop_error_h0 = np.abs((1/(2*np.sqrt(h0))) * np.sqrt(2*(h0/g)*((area_1**2 / area_2**2) - 1))) * cota_error
    tf_prop_error_area_1 = np.abs((area_1 / area_2**2) * np.sqrt((2*h0/g)/((area_1**2/area_2**2) - 1))) * cota_area(diam_tanque)
    tf_prop_error_area_2 = np.abs((-area_1**2 / area_2**3) * np.sqrt((2*h0/g) / ((area_1**2 / area_2**2) - 1))) * cota_area(diam_orificio_usado)

    cota_tf = tf_prop_error_h0 + tf_prop_error_area_1 + tf_prop_error_area_2
    t_prop_error_tf = np.abs(1 - np.sqrt(proporcion)) * cota_tf
    t_prop_error_h0 = np.abs(2*tf*np.sqrt(proporcion*h0) / (2*h0**(3/2))) * cota_error
    return t_prop_error_tf + t_prop_error_h0

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
    dt_teo = (1 - np.sqrt(p)) * dt_frame + dt_tiempo_teorico(p)
    t_cub, cota = tiempo_cubico(coef_cub, p, tf_cub)
    dt_cub = dt_frame

    resultados[descripcion] = {
        "p": p,
        "t_teo": t_teo,
        "dt_teo": dt_teo,
        "t_cub": t_cub,
        "t_cota": cota,
        "dt_cub": dt_cub
    }



# 7. IMPRESION DE RESULTADOS
print("=== Estimacion de tiempos teoricos vs ajuste cubico ===")
print(f"{'Caso':<30}{'t_teo [s]':>12}{'±dt_teo [s]':>15}{'t_cub [s]':>12}{'±dt_cub [s]':>15}{'±cota [s]':>15}")
print("-" * 100)
for desc, datos in resultados.items():
    print(f"{desc:<30}{datos['t_teo']:>12.3f}{datos['dt_teo']:>15.3f}{datos['t_cub']:>12.3f}{datos['dt_cub']:>15.3f}{datos['t_cota']:>15}")

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
plt.plot(array_t_teo, y_teo, '--g', label="Modelo teorico")

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
plt.savefig("tiempos_vaciado.png", dpi=300)
plt.show()
