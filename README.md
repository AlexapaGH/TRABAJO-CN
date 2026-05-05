# Trabajo final de Cálculo Numérico: simulación de un látigo mediante la ecuación de ondas

Este proyecto desarrolla un trabajo final de la asignatura de Cálculo Numérico basado en la simulación de la propagación de una onda transversal en un látigo. La idea física es representar el látigo como una cuerda 1D sometida a una excitación en el extremo fijo o mango y con un extremo libre en la punta, de forma que se puedan estudiar la propagación, la reflexión y el efecto de las condiciones de contorno sobre la solución numérica.

El trabajo está pensado para responder al enunciado propuesto en clase, que permite elegir entre la ecuación del calor o la ecuación de ondas en 1D o 2D. En este caso se ha elegido la **ecuación de ondas 1D**, porque permite modelar de forma clara el comportamiento del látigo y analizar aspectos numéricos como la estabilidad, el error de semi-discretización espacial y la influencia del paso temporal.

## Finalidad del proyecto

La finalidad del trabajo es doble:

1. **Modelar físicamente** el látigo como un sistema gobernado por una ecuación en derivadas parciales.
2. **Estudiar numéricamente** el comportamiento de distintas discretizaciones temporales y espaciales, comparando soluciones estables e inestables y observando cómo cambian la energía y la forma de la onda.

Con ello se pretende demostrar no solo que la simulación reproduce el fenómeno, sino también que los métodos numéricos utilizados son coherentes con el análisis teórico de estabilidad y error pedido en el enunciado.

## Relación con el enunciado

El trabajo cubre especialmente los siguientes apartados solicitados:

- Explicación de la física del problema: propagación de ondas en un látigo.
- Descripción de los métodos numéricos empleados.
- Análisis de la semi-discretización espacial y de la discretización temporal.
- Estudio de la estabilidad asintótica de los esquemas.
- Casos de prueba y discusión de resultados.

## Modelo físico

El látigo se aproxima como una cuerda unidimensional de longitud `L` con desplazamiento transversal `u(x,t)`. La ecuación base es la ecuación de ondas:

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
$$

donde `c` es la velocidad de propagación.

Se imponen condiciones de contorno que representan el comportamiento físico del látigo:

- **Extremo fijo / mango**: se prescribe un impulso temporal en `x = 0`.
- **Extremo libre / punta**: se usa una condición de Neumann `\partial u / \partial x = 0` en `x = L`.

Estas condiciones permiten simular cómo una excitación inicial recorre el látigo y se refleja en la punta libre.

## Métodos numéricos usados

El proyecto incluye varios esquemas para estudiar la evolución de la solución:

- **Esquema explícito de segundo orden** tipo Leapfrog para la ecuación de ondas.
- **Esquema implícito Crank-Nicolson**, útil para comparar estabilidad y comportamiento energético.
- Cálculo de **energía discreta** para analizar la evolución numérica de la energía cinética y potencial.

Además, se usan distintos impulsos en el extremo fijo para comparar su efecto sobre la propagación:

- Impulso gaussiano.
- Impulso sinusoidal.
- Impulso triangular.

## Estructura de los archivos

- `utils.py`: contiene los parámetros físicos, los impulsos temporales y los solvers numéricos compartidos por todo el proyecto.
- `1_animacion_latigo.py`: genera la animación principal del látigo y varias figuras de apoyo.
- `2_estabilidad_latigo.py`: estudio de estabilidad del esquema numérico.
- `3_errores_semidisc_latigo.py`: análisis del error de semi-discretización espacial.
- `4_esquemas_temporales_latigo.py`: comparación entre distintos esquemas temporales.
- `5_energia_casos_latigo.py`: comparación de la energía en distintos casos y esquemas.
- `Imágenes/`: carpeta destinada a guardar figuras, capturas o material gráfico del trabajo.

## Qué muestra la simulación principal

El archivo `1_animacion_latigo.py` genera:

- snapshots de la solución en tiempos representativos,
- un diagrama espacio-tiempo `x-t`,
- una animación de la evolución de la onda en el látigo,
- una comparación visual entre distintos impulsos iniciales.

Estas salidas ayudan a explicar en la presentación cómo viaja la onda, cómo refleja en el extremo libre y cómo cambia la solución según el tipo de excitación aplicada en el mango.

## Objetivo didáctico

Más allá de obtener gráficas bonitas, el trabajo busca demostrar que:

- una EDP puede describir un problema físico real,
- la discretización numérica introduce restricciones de estabilidad,
- la elección del paso temporal `\Delta t` y del espacio `\Delta x` afecta directamente a la calidad de la solución,
- diferentes esquemas pueden producir resultados muy distintos aunque resuelvan el mismo modelo físico.

## Cómo ejecutar el proyecto

El script principal puede ejecutarse desde la carpeta del proyecto con:

```bash
python 1_animacion_latigo.py
(los demás módulos siguen el mismo estilo de ejecución, ej.: python 2_estabilidad_latigo.py)
```

Al ejecutarlo se generan las figuras y la animación asociadas al látigo. Los demás scripts están pensados para completar el estudio con análisis de estabilidad, error y energía.

## Bibliografía

Basada enteramente en lso apuntes de la asignatura de `Cálculo Numérico` así como documentación de `scipy` , `numpy` y `matplotlib`.
