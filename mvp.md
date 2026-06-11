Objetivo del MVP
Demostrar que el sistema puede tomar video de un partido de fútbol sala, procesarlo automáticamente y presentar las métricas y el video del partido a cada jugador a través de una plataforma web accesible remotamente.

Alcance
El sistema recibe video del partido desde la Mini PC ubicada en el complejo deportivo, que lo captura, lo segmenta y lo envía al servidor. En el servidor se transforma el video para dejarlo apto para el análisis, se detectan y trackean los jugadores y la pelota, y sus posiciones se llevan a coordenadas reales del campo. Con esa información se calculan las métricas definidas: mapa de calor, posesión, distancia recorrida y eventos de gol. Los resultados y el video del partido quedan disponibles en la nube, donde cada jugador puede acceder a través de la plataforma web con su cuenta, consultar el historial de sus partidos, ver las estadísticas individuales de cada partido y sus estadísticas globales acumuladas.

Fuera del alcance del MVP
Cámaras simultáneas.
Procesamiento en tiempo real.
Detección de faltas o tarjetas.
Detección de atajadas y asistencias (puede incorporarse en versiones posteriores).

Caso de uso:
CU-01 — Iniciar sesión Actor: Jugador Precondición: El jugador tiene una cuenta registrada en el sistema. Flujo principal:
El jugador ingresa sus credenciales en la plataforma web.
El sistema valida la identidad y le da acceso a su perfil. Postcondición: El jugador accede al sistema con su cuenta.

CU-02 — Identificarse en un partido Actor: Jugador Precondición: El jugador inicia sesión. Existe un partido registrado en el sistema. Flujo principal:
El jugador selecciona un partido de la lista disponible y se identifica.
El sistema lo vincula como participante de ese partido. Postcondición: El partido queda asociado al perfil del jugador.

CU-03 — Ver historial de partidos Actor: Jugador Precondición: El jugador inició sesión y tiene al menos un partido registrado. Flujo principal:
El jugador accede a su historial desde la plataforma.
El sistema muestra la lista de partidos en los que participó. Postcondición: El jugador visualiza sus partidos disponibles.

CU-04 — Ver estadísticas de un partido Actor: Jugador Precondición: El jugador selecciona un partido del historial y las métricas fueron procesadas. Flujo principal:
El jugador selecciona un partido del historial.
El sistema muestra las estadísticas: mapa de calor, posesión, distancia recorrida y eventos de gol. Flujo alternativo: Si las métricas aún no fueron procesadas, el sistema informa que no están disponibles. Postcondición: El jugador visualiza sus estadísticas del partido.

CU-05 — Ver video del partido Actor: Jugador Precondición: El jugador selecciona un partido del historial y el video fue procesado. Flujo principal:
El jugador selecciona un partido del historial.
El sistema reproduce el video del partido. Flujo alternativo: Si el video aún no está disponible, el sistema informa que está en procesamiento. Postcondición: El jugador visualiza el video de su partido.

CU-06 — Ver estadísticas globales del perfil Actor: Jugador Precondición: El jugador tiene al menos un partido procesado. Flujo principal:
El jugador accede a su perfil.
El sistema muestra las estadísticas acumuladas de todos sus partidos. Postcondición: El jugador visualiza sus métricas globales.
