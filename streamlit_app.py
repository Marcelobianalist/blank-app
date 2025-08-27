import streamlit as st
import pandas as pd
from thefuzz import process, fuzz
import re

# --- Configuración de la Página ---
st.set_page_config(page_title="Diccionario CIE-10", page_icon="📚", layout="wide")

# --- Funciones de Carga y Procesamiento (Caché) ---

@st.cache_data
def load_and_prepare_data():
    DATA_URL = "https://raw.githubusercontent.com/verasativa/CIE-10/refs/heads/master/codes.json"
    with st.spinner("Cargando catálogo CIE-10..."):
        df = pd.read_json(DATA_URL).dropna(subset=['code', 'description'])
        df['description_lower'] = df['description'].str.lower()
        df['code_4d'] = df['code'].str.replace('.', '', regex=False).apply(lambda x: x.ljust(4, 'X') if len(x) == 3 else x).str.slice(0, 4)
    return df

# --- Función de Búsqueda Flexible ---
def fuzzy_search(query, choices, limit=10):
    results = process.extract(query, choices, limit=limit, scorer=fuzz.partial_ratio)
    return results

# --- Interfaz Principal ---
st.title("📚 Diccionario Interactivo CIE-10")
st.markdown("Busque códigos y descripciones rápidamente. Este diccionario no usa IA.")

# Cargar datos
df = load_and_prepare_data()

# --- Barra Lateral de Búsqueda ---
with st.sidebar:
    st.header("Opciones de Búsqueda")
    
    # Búsqueda por palabra clave
    keyword_query = st.text_input("Buscar en descripciones:", placeholder="Ej: neumonía")
    
    # Búsqueda por código exacto
    code_query = st.text_input("Buscar código exacto:", placeholder="Ej: J129")
    
    # Opciones de filtro
    chapter_filter = st.selectbox("Filtrar por capítulo (opcional):", options=["Todos"] + sorted(df['code'].str[0].unique().tolist()))

# --- Lógica de Filtrado ---
filtered_df = df.copy()

# Filtrar por capítulo
if chapter_filter != "Todos":
    filtered_df = filtered_df[filtered_df['code'].str.startswith(chapter_filter)]

# Filtrar por código exacto
if code_query:
    filtered_df = filtered_df[filtered_df['code_4d'].str.contains(code_query.upper(), na=False)]

# Filtrar por palabra clave
if keyword_query:
    results = fuzzy_search(keyword_query.lower(), filtered_df['description_lower'].tolist(), limit=50)
    matching_indices = [i for i, score in enumerate(filtered_df.index) if filtered_df['description_lower'].iloc[i] in [r[0] for r in results]]
    filtered_df = filtered_df.loc[matching_indices]

# --- Mostrar Resultados ---
st.header("Resultados")

if filtered_df.empty:
    st.info("No se encontraron coincidencias con los criterios de búsqueda.")
else:
    st.dataframe(filtered_df[['code_4d', 'description']], use_container_width=True)
