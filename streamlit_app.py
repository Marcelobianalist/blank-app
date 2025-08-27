import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Asistente de Codificación CIE-10",
    page_icon="🩺",
    layout="wide"
)

# --- Funciones de Carga y Procesamiento (con caché para eficiencia) ---

@st.cache_resource
def load_model():
    """Carga el modelo de embedding una sola vez."""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_and_prepare_data():
    """Carga los datos de la CIE-10 desde la nueva URL en formato JSON y los prepara."""
    # <--- CAMBIO 1: Nueva y definitiva URL del archivo JSON.
    DATA_URL = "https://raw.githubusercontent.com/verasativa/CIE-10/refs/heads/master/codes.json"
    
    with st.spinner("Cargando catálogo CIE-10 (JSON) desde la web..."):
        try:
            df = pd.read_json(DATA_URL)
        except Exception as e:
            st.error(f"Error al cargar los datos desde la fuente online: {e}")
            st.info("Por favor, revise su conexión a internet. Si el problema persiste, la fuente de datos puede estar temporalmente inaccesible.")
            st.stop()
    
    # <--- CAMBIO 2: ¡Ya no se necesita renombrar columnas! El formato es perfecto.
    # La línea df.rename(...) ha sido eliminada.
        
    # Limpieza básica
    df.dropna(subset=['code', 'description'], inplace=True)
    df = df[df['description'].str.strip() != '']

    # --- Creación del código de 4 dígitos (convención chilena) ---
    df['code_4d'] = df['code'].str.replace('.', '', regex=False)
    df['code_4d'] = df['code_4d'].apply(lambda x: x.ljust(4, 'X') if len(x) == 3 else x).str.slice(0, 4)
    
    return df

@st.cache_data
def create_embeddings(_model, descriptions):
    """Crea los embeddings para las descripciones de la CIE-10."""
    with st.spinner("Inicializando el motor de IA... (esto puede tardar un momento la primera vez)"):
        embeddings = _model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
    return embeddings.cpu().numpy()

# --- Función de Orientación para Codificación ---
def get_coding_guidance(code):
    """Proporciona orientación específica basada en el capítulo de la CIE-10."""
    if not code: return ""
    chapter = code[0].upper()
    guidance = []
    
    is_neoplasia_range = (
        chapter == 'D' and 
        len(code) > 2 and 
        code[1:3].isdigit() and 
        0 <= int(code[1:3]) <= 48
    )

    if chapter in ['A', 'B']: guidance.append("**Guía:** Para enfermedades infecciosas, considere codificar también el organismo causal si la CIE-10 lo indica.")
    elif chapter == 'C' or is_neoplasia_range: guidance.append("**Guía:** Para neoplasias, especifique el comportamiento (maligno, benigno, in situ) y la localización. Use códigos de la sección Z para historial personal de neoplasia.")
    elif chapter == 'F': guidance.append("**Guía:** Para trastornos mentales, sea lo más específico posible. Considere el estado (ej. en remisión), la severidad y si es un episodio único o recurrente.")
    elif chapter == 'I': guidance.append("**Guía:** Para enfermedades circulatorias, especifique la cronicidad (agudo vs. crónico). Para hipertensión, considere si hay relación causal con enfermedades renales o cardíacas.")
    elif chapter == 'J': guidance.append("**Guía:** Para enfermedades respiratorias, distinga entre agudo y crónico. Si hay una infección, codifique el organismo si es conocido.")
    elif chapter == 'M': guidance.append("**Guía:** Para enfermedades musculoesqueléticas, especifique la lateralidad (derecho/izquierdo) y la articulación o zona exacta si es posible.")
    elif chapter == 'R': guidance.append("**Guía:** Los códigos 'R' (síntomas y signos) son para casos no diagnosticados. Si se llega a un diagnóstico definitivo, debe reemplazarse por el código de esa enfermedad.")
    elif chapter in ['S', 'T']: guidance.append("**Guía:** Para lesiones y traumatismos, es crucial incluir la causa externa (códigos V, W, X, Y). Especifique si es un encuentro inicial, subsecuente o una secuela.")
    elif chapter == 'Z': guidance.append("**Guía:** Los códigos 'Z' no son enfermedades, sino factores que influyen en el estado de salud (ej. controles, historial). Úselos como código principal o secundario según corresponda.")
    else: guidance.append("**Guía General:** Revise la documentación clínica para asegurar que el código seleccionado refleje con la máxima precisión el diagnóstico. Considere si se necesitan códigos adicionales para manifestaciones o comorbilidades.")
    return "\n".join(guidance)

# --- Interfaz Principal de la Aplicación ---
st.title("🩺 Asistente Inteligente de Codificación CIE-10")
st.markdown("""
Esta herramienta utiliza un modelo de lenguaje para recomendar los códigos de diagnóstico CIE-10 más probables basados en una descripción clínica.
**Nota:** Es una ayuda y no reemplaza el juicio clínico profesional ni las normativas locales de codificación.
""")

model = load_model()
df = load_and_prepare_data()
embeddings = create_embeddings(model, df['description'].tolist())

with st.container(border=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_area("Ingrese la descripción clínica, síntoma o diagnóstico:", height=100, placeholder="Ej: Paciente con dolor de cabeza intenso y fiebre, sospecha de cefalea tensional...")
    with col2:
        num_results = st.slider("Resultados a mostrar:", min_value=1, max_value=10, value=5)
        search_button = st.button("Buscar Diagnósticos", type="primary", use_container_width=True)

if search_button and user_query:
    with st.spinner("Buscando diagnósticos relevantes..."):
        query_embedding = model.encode([user_query], convert_to_tensor=True).cpu().numpy()
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[-num_results:][::-1]
    st.subheader(f"Top {num_results} diagnósticos recomendados:")
    for i, idx in enumerate(top_indices):
        code = df.iloc[idx]['code_4d']
        description = df.iloc[idx]['description']
        similarity = similarities[idx]
        with st.container(border=True):
            st.markdown(f"#### **{i+1}. {code}** - {description}")
            st.progress(float(similarity), text=f"Similitud: {similarity:.2%}")
            guidance = get_coding_guidance(code)
            st.info(guidance)
elif search_button and not user_query:
    st.warning("Por favor, ingrese una descripción clínica para buscar.")

st.markdown("---")
st.markdown("Desarrollado como una herramienta de apoyo para profesionales de la salud.")
