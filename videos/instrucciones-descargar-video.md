# Guía Técnica: Descarga de Streams HLS (.m3u8)

Este instructivo detalla el flujo de trabajo para capturar y reconstruir videos fragmentados mediante análisis de tráfico y herramientas de procesamiento de video.

## 1. Requisitos Previos (Setup en LTSC)

Dado que el entorno LTSC carece de Microsoft Store y `winget`, la gestión de dependencias se realiza vía Python/PIP. Ejecutá esto una única vez:

```powershell
# Install the core downloader and a standalone ffmpeg binary
pip install yt-dlp static-ffmpeg
```

## 2. Fase de Captura (Análisis de Tráfico)

1. Abrí DevTools: presioná `F12` en la pestaña del video.
2. En la pestaña Network, usá el filtro para buscar `m3u8` o `playlist`.
3. Nota crítica: el archivo no siempre termina en `.m3u8`. Buscá recursos cuyo `Content-Type` sea `application/x-mpegurl` (por ejemplo, scripts PHP que generan el manifiesto dinámicamente).
4. Extraé la URL: clic derecho sobre el recurso -> Copy -> Copy link address.

## 3. Fase de Descarga y Remuxing

Para evitar descargas lentas y archivos con duración corrupta, el comando debe incluir concurrencia y la ruta del binario de procesamiento.

Comando optimizado:

```powershell
# Execute download with 10 concurrent threads and direct ffmpeg pathing
yt-dlp --concurrent-fragments 10 `
  --ffmpeg-location "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python314\Lib\site-packages\static_ffmpeg\bin\win32\ffmpeg.exe" `
  -o "~/Downloads/%(title)s.%(ext)s" `
  "URL_COPIADA"
```

- `--concurrent-fragments 10`: abre múltiples conexiones para saltar el throttling del servidor.
- `--ffmpeg-location`: obliga a `yt-dlp` a usar el binario de `static-ffmpeg` para sellar el contenedor `.mp4` correctamente al finalizar.
- `-o`: direcciona la salida a la carpeta de Descargas del usuario.

## 4. Post-Procesamiento (Reparación de Emergencia)

Si el video ya se descargó pero la línea de tiempo está malformada (el video se corta o no permite adelantar), ejecutá un remuxing preventivo sin pérdida de calidad:

```powershell
# Rebuild the MP4 container and fix timestamps without re-encoding
static_ffmpeg -i "archivo_corrupto.mp4" -c copy -map 0 "video_final_arreglado.mp4"
```

## Diagnóstico de Problemas Comunes

- **Error 403 Forbidden**: el servidor requiere validación de origen. Agregá `--referer "https://dominio-del-sitio.com/"` al comando.
- **Velocidad nula**: algunos servidores bloquean el paralelismo extremo. Si falla, bajá `--concurrent-fragments` a `3` o `5`.
- **Comando no reconocido**: asegurate de que la carpeta `Scripts` de tu instalación de Python esté en el `PATH` del sistema, o ejecutá usando `python -m yt_dlp`.
