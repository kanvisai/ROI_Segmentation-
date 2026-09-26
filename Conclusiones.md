# Conclusiones

Orden de las seis tecnologías, de mejor a peor, según los datos de
`salidas/comparativa/comparativa.csv`. La muestra son 258 filas: 43 clips por 6
tecnologías, todas con `ok=True`, ninguna con error y ninguna con código de
salida distinto de cero.

**Orden final: YOLO26-seg, Cutie, Track-Anything, HOIST-Former, SAM 3, VISOR.**

## Tabla de agregados

| # | backend | cobertura media | mediana | desapar./100 frames | identidad partida | victorias | puesto medio | fps | ms/frame | RAM pico | VRAM pico |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | YOLO26-seg | 75,0 % | 82,3 % | 0,85 | 3 | 15 | 2,42 | 2,89 | 374 | 6.745 MB | 193 MB |
| 2 | Cutie | 70,5 % | 79,0 % | 0,78 | 0 | 11 | 2,53 | 3,22 | 322 | 5.250 MB | 1.607 MB |
| 3 | Track-Anything | 70,8 % | 76,0 % | 0,87 | 1 | 13 | 2,72 | 2,86 | 363 | 4.941 MB | 12.002 MB |
| 4 | HOIST-Former | 60,2 % | 60,9 % | 2,60 | 8 | 2 | 3,35 | 0,93 | 1.106 | 9.674 MB | 16.640 MB |
| 5 | SAM 3 | 58,0 % | 58,4 % | 2,34 | 6 | 2 | 3,98 | 1,17 | 913 | 6.889 MB | 11.971 MB |
| 6 | VISOR | 0,0 % | 0,0 % | 0,00 | 38 | 0 | 6,00 | 13,22 | 76 | 3.470 MB | 1.370 MB |

Definición de cada columna derivada:

- **Cobertura**: `frames_con_objeto / frames`, media y mediana sobre los 43 clips.
  Mide en qué fracción del clip la tecnología mantuvo el objeto localizado.
- **Desapar./100 frames**: `desapariciones` normalizadas por longitud del clip,
  para que los clips largos no dominen la cifra.
- **Identidad partida**: número de clips en los que `objetos` salió distinto de 1,
  es decir donde el seguimiento rompió un único objeto en varias identidades.
- **Victorias** y **puesto medio**: enfrentamiento clip a clip. En cada uno de los
  43 clips se ordenan las seis tecnologías por cobertura; victorias son los clips
  en los que quedó primera y el puesto medio es su posición promedio de 1 a 6.
- **fps**, **ms/frame**, **RAM** y **VRAM**: medias de las columnas `fps`,
  `ms_frame`, `ram_pico_mb` y `vram_pico_mb`.

## 1. YOLO26-seg

La mejor de las seis y por margen claro. Tiene la cobertura media y mediana más
altas (75,0 % y 82,3 %), gana 15 de los 43 clips, es la que mejor puesto medio
saca del enfrentamiento directo (2,42 sobre 6) y pierde el objeto pocas veces,
0,85 desapariciones por cada 100 frames.

Su ventaja más grande está en el coste: 193 MB de VRAM pico, entre 60 y 85 veces
menos que HOIST-Former, SAM 3 o Track-Anything, con una velocidad de 2,89 fps que
está a la altura de los propagadores. Es la única que cabría con holgura en una
GPU modesta.

Su punto débil son algunos clips largos. En `4_clip_11`, de 741 frames, solo cubre
el 6 % mientras Cutie llega al 93 %; en `4_clip_7` se queda en el 25 % frente al
79 % de Cutie. Al redetectar cada frame sin memoria temporal, cuando el objeto deja
de ser visible para el detector no hay nada que rellene el hueco.

## 2. Cutie

El mejor de los propagadores y el más equilibrado del conjunto. Cobertura media
del 70,5 % con la mediana más alta de los tres seguidores (79,0 %), la menor tasa
de desapariciones de toda la tabla (0,78 por 100 frames) y el único caso en el que
nunca se partió la identidad del objeto: en los 43 clips terminó con `objetos = 1`.

Es además el más rápido de los que siguen el objeto (3,22 fps, 322 ms por frame) y
gasta 1.607 MB de VRAM, siete veces menos que Track-Anything con calidad
equivalente. Esa relación entre resultado y consumo lo coloca por delante.

Falla por completo en `4_clip_12`, donde no consiguió sembrar el objeto y la
cobertura es 0.

## 3. Track-Anything

Prácticamente empatado con Cutie en cobertura media (70,8 % frente a 70,5 %) y con
más victorias clip a clip (13 frente a 11), pero queda por detrás en mediana
(76,0 % frente a 79,0 %), en puesto medio (2,72 frente a 2,53), tiene un clip con
la identidad partida y consume 12 GB de VRAM contra los 1,6 GB de Cutie.

Donde sí es la mejor opción es en los clips largos: gana `4_clip_4` (96 %),
`4_clip_5` (66 %), `4_clip_8` (90 %) y `4_clip_10` (62 %). La memoria a largo plazo
de XMem se nota cuando hay muchos frames que sostener.

Comparte con Cutie el fallo total en `4_clip_12`.

## 4. HOIST-Former

Baja un escalón claro. Cubre el 60,2 % de los frames, pierde el objeto tres veces
más a menudo que los tres primeros (2,60 desapariciones por 100 frames) y fragmenta
la identidad en 8 de los 43 clips, el peor registro de las tecnologías que sí
detectan.

Es también la más cara de todas: 0,93 fps y 1.106 ms por frame la convierten en la
más lenta, con 16.640 MB de VRAM pico y una RAM media de 9.674 MB que en los clips
largos llega a 22.915 MB. Paga el coste más alto por el cuarto mejor resultado.

Aporta, eso sí, la persona, las manos y el ROI, que es el papel que cumple en la
tubería aunque no sea quien segmenta el objeto.

## 5. SAM 3

La peor de las que producen algo. Cobertura del 58,0 %, la más baja del grupo
funcional, 2,34 desapariciones por 100 frames, identidad partida en 6 clips y un
puesto medio de 3,98 sobre 6, lo que significa que en la mayoría de los clips queda
en la mitad baja de la comparación.

Cuesta 913 ms por frame y 11.971 MB de VRAM, un gasto propio de la primera posición
para un resultado de quinta. A esta distancia de cámara su prompt de texto no marca
el objeto en la mano, y lo que queda medido es sobre todo el seguimiento de la
persona.

## 6. VISOR

La peor sin discusión: 0 frames con objeto en los 43 clips, es decir cobertura nula
en toda la muestra. Las cifras de velocidad (13,22 fps, 76 ms por frame) y de
consumo (1.370 MB de VRAM) son las mejores de la tabla, pero son el coste de no
producir ninguna detección, así que no cuentan como mérito. El valor 38 en identidad
partida corresponde a clips donde `objetos` quedó en 0.

El motivo está documentado en el README: el modelo está entrenado sobre vídeo
egocéntrico de cocina de EPIC-KITCHENS y con estos clips en tercera persona ninguna
instancia supera el umbral. Con umbral 0.0 la probabilidad máxima de primer plano es
0,0101, así que no es un fallo de integración, es que el modelo no dispara en este
dominio.

## Cómo leer estas cifras

La cobertura mide persistencia, no acierto. Una tecnología que se queda agarrada a
una máscara equivocada puntúa igual que una que sigue bien al objeto, y los
propagadores (Cutie y Track-Anything) parten de una semilla y arrastran su estado,
de modo que tienden a inflar esa cifra frente a los que redetectan cada frame. Con
las columnas automáticas por sí solas no se puede distinguir un seguimiento correcto
de uno que mantiene un error, y para eso hacen falta las anotaciones manuales de
`Aceptable` y `Observaciones`, que este documento no usa.

Un detalle sobre las dos columnas de tiempo: `ms_frame` es el milisegundo medio por
fotograma del análisis completo del clip, mientras que `ms_segmentacion` es la
duración total en milisegundos del bucle de segmentación de ese mismo clip. Son
escalas distintas, así que la comparación de velocidad de este documento usa solo
`fps` y `ms_frame`.
