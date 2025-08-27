import pandas as pd
import unicodedata

def normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    return "".join([c for c in text if not unicodedata.combining(c)])

print("Iniciando el procesamiento de la base de datos MINSAL...")

# URL de la fuente de datos
DATA_URL = "https://repositoriodeis.minsal.cl/ContenidoSitioWeb2020/uploads/2018/03/Lista-Tabular-CIE-10-1-1.xls"

try:
    # Leer el archivo Excel, saltando las primeras filas que no son datos
    print(f"Descargando y leyendo el archivo desde {DATA_URL}...")
    df_raw = pd.read_excel(DATA_URL, engine='openpyxl', header=None, skiprows=5)
    print("Archivo leído con éxito.")

    # Seleccionar y renombrar las columnas relevantes
    df = df_raw[[0, 1]].copy()
    df.columns = ['code', 'description']

    # Limpieza de la Lista Tabular
    df.dropna(subset=['code'], inplace=True)
    df['code'] = df['code'].astype(str)
    
    # FILTRO CLAVE: Mantener solo los códigos válidos
    df = df[~df['code'].str.contains('-') & (df['code'].str.len() >= 3)]
    df.reset_index(drop=True, inplace=True)
    print(f"Se encontraron {len(df)} códigos de diagnóstico válidos.")

    # Enriquecimiento de Datos
    print("Enriqueciendo datos con sinónimos, códigos de 4 dígitos y áreas clínicas...")
    df['code_4d'] = df['code'].str.replace('.', '', regex=False).apply(lambda x: x.ljust(4, 'X') if len(x) == 3 else x).str.slice(0, 4)
    df['search_field'] = df['description'].apply(normalize_text) # Normalizamos para la búsqueda

    # Guardar el archivo limpio en formato CSV
    output_filename = "cie10_minsal_clean.csv"
    df.to_csv(output_filename, index=False)
    
    print("-" * 50)
    print(f"¡Proceso completado! Se ha creado el archivo '{output_filename}'.")
    print("Ahora puedes subir este archivo a tu repositorio y modificar 'app.py' para que lo lea localmente.")
    print("-" * 50)

except Exception as e:
    print(f"\nError durante el procesamiento: {e}")
    print("Asegúrate de tener conexión a internet y las librerías 'pandas' y 'openpyxl' instaladas.")
