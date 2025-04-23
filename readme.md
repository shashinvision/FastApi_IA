# Empresa Service

Este servicio proporciona capacidades de recuperación de información y respuesta a preguntas basadas en un repositorio de documentos de empresa. Utiliza modelos de lenguaje grandes (LLM) para generar respuestas basadas en la información recuperada.

## Características

- **Recuperación de documentos:** Busca documentos relevantes en un repositorio de archivos (md y txt).
- **Generación de preguntas y respuestas:** Utiliza un LLM para responder preguntas basadas en los documentos recuperados.
- **Contexto de la pregunta:** La pregunta se utiliza como contexto para el LLM, lo que ayuda a generar respuestas más precisas.
- **Limitaciones:** El LLM solo utiliza la información proporcionada en los documentos recuperados. No inventa datos. Si no encuentra información sobre una empresa específica, indicará "No tengo información sobre esa empresa".
- **Soporte para Apple Silicon:** Utiliza la aceleración MPS (Metal Performance Shaders) si está disponible en tu Mac.
- **Almacenamiento vectorial:** Utiliza Chroma para almacenar y recuperar documentos de manera eficiente.

## Dependencias

- `transformers`
- `langchain`
- `chroma-db`
- `sentence-transformers`
- `torch`

## Instalación

1. Clona el repositorio.
2. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Configuración

- **Repositorio de documentos:** El servicio busca documentos en un directorio llamado `repository` en el mismo directorio que este script. Asegúrate de que este directorio contenga tus archivos de documentos (md y txt).
- **Modelo LLM:** Utiliza el modelo `google/flan-t5-small` por defecto.

## Uso

1. Asegúrate de que el directorio `repository` contenga tus archivos de documentos.
2. Ejecuta el script.
3. Haz preguntas sobre los documentos.

## Notas

- El rendimiento puede variar dependiendo del tamaño del repositorio de documentos y la complejidad de las preguntas.
- Considera optimizar el repositorio de documentos para mejorar el rendimiento.
- Para preguntas más complejas, considera proporcionar más contexto en la pregunta.
- Este es un proyecto en desarrollo y puede contener errores.
