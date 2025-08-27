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
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_and_prepare_data():
    DATA_URL = "https://raw.githubusercontent.com/verasativa/CIE-10/refs/heads/master/codes.json"
    with st.spinner("Cargando catálogo CIE-10 (JSON) desde la web..."):
        try:
            df = pd.read_json(DATA_URL)
        except Exception as e:
            st.error(f"Error al cargar los datos desde la fuente online: {e}")
            st.stop()
    
    df.dropna(subset=['code', 'description'], inplace=True)
    df = df[df['description'].str.strip() != '']
    # Pre-procesamiento para la búsqueda por palabras clave
    df['description_lower'] = df['description'].str.lower()
    df['code_4d'] = df['code'].str.replace('.', '', regex=False)
    df['code_4d'] = df['code_4d'].apply(lambda x: x.ljust(4, 'X') if len(x) == 3 else x).str.slice(0, 4)
    return df

@st.cache_data
def create_embeddings(_model, descriptions):
    with st.spinner("Inicializando el motor de IA... (esto puede tardar un momento la primera vez)"):
        embeddings = _model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
    return embeddings.cpu().numpy()

# --- Función de Alertas (sin cambios) ---
def show_complex_coding_alert(query):
    query_lower = query.lower()
    alerts = {
        ("suicidio", "autoinflingida", "autolesión", "auto inflingida"): """
        ### ⚠️ Alerta de Codificación Compleja: Lesión Autoinfligida
        Un **intento de suicidio** o **lesión autoinfligida** requiere al menos **DOS** códigos:
        1.  **Código de Lesión (Capítulo XIX: S00-T98):** Describe el daño físico (ej: `S610` para herida de muñeca).
        2.  **Código de Causa Externa (Capítulo XX: X60-X84):** Describe la intencionalidad y el método (ej: `X78X` para objeto cortante).
        """,
        ("accidente", "caída", "golpe", "atropello", "quemadura", "mordedura"): """
        ### ⚠️ Alerta de Codificación Compleja: Traumatismo y Lesiones
        Un **traumatismo** o **lesión accidental** requiere al menos **DOS** códigos:
        1.  **Código de Lesión (Capítulo XIX: S00-T98):** Describe el daño físico (ej: `S826` para fractura de peroné).
        2.  **Código de Causa Externa (Capítulo XX: V01-Y98):** Describe el evento que causó la lesión (ej: `W19X` para caída no especificada).
        """
    }
    for keywords, message in alerts.items():
        if any(keyword in query_lower for keyword in keywords):
            st.warning(message, icon="⚠️")
            return

# --- Función de Orientación (sin cambios) ---
def get_coding_guidance(code):
    if not code: return ""
    chapter = code[0].upper()
    guidance = []
    is_neoplasia_range = (chapter == 'D' and len(code) > 2 and code[1:3].isdigit() and 0 <= int(code[1:3]) <= 48)
    if chapter in ['A', 'B']: guidance.append("**Guía:** Para enfermedades infecciosas, considere codificar también el organismo causal si la CIE-10 lo indica.")
    elif chapter == 'C' or is_neoplasia_range: guidance.append("**Guía:** Para neoplasias, especifique el comportamiento (maligno, benigno, in situ) y la localización.")
    elif chapter in ['S', 'T']: guidance.append("**Guía de Lesión:** Este es un código de lesión. **Recuerde añadir un código de Causa Externa (V01-Y98)** para describir cómo y por qué ocurrió la lesión.")
    elif chapter in ['V', 'W', 'X', 'Y']: guidance.append("**Guía de Causa Externa:** Este es un código de causa externa. **Asegúrese de tener también un código de Lesión (S00-T98)** que describa el daño físico.")
    else: guidance.append("**Guía General:** Revise la documentación clínica para asegurar que el código seleccionado refleje con la máxima precisión el diagnóstico.")
    return "\n".join(guidance)
    
# --- Interfaz Principal de la Aplicación ---
st.title("🩺 Asistente Inteligente de Codificación CIE-10")
st.markdown("""
Esta herramienta utiliza una **búsqueda híbrida (semántica + palabras clave)** para recomendar los códigos CIE-10 más relevantes.
""")

model = load_model()
df = load_and_prepare_data()
embeddings = create_embeddings(model, df['description'].tolist())

with st.container(border=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_area("Ingrese la descripción clínica, síntoma o diagnóstico:", height=100, placeholder="Ej: intento de suicidio, cortes en muñeca")
    with col2:
        num_results = st.slider("Resultados a mostrar:", min_value=1, max_value=10, value=5)
        search_button = st.button("Buscar Diagnósticos", type="primary", use_container_width=True)

# --- LÓGICA DE BÚSQUEDA HÍBRIDA MEJORADA ---
if search_button and user_query:
    show_complex_coding_alert(user_query)

    with st.spinner("Realizando búsqueda híbrida inteligente..."):
        # --- 1. Búsqueda Semántica (como antes) ---
        query_embedding = model.encode([user_query], convert_to_tensor=True).cpu().numpy()
        semantic_similarities = cosine_similarity(query_embedding, embeddings)[0]

        # --- 2. Búsqueda por Palabras Clave ---
        # Dividir la consulta en palabras y buscar si *todas* están presentes
        query_keywords = [word for word in re.split(r'\W+', user_query.lower()) if word]
        keyword_mask = pd.Series(True, index=df.index)
        for keyword in query_keywords:
            keyword_mask &= df['description_lower'].str.contains(keyword, na=False)
        
        # --- 3. Combinación de Puntuaciones (Lógica Híbrida) ---
        KEYWORD_BOOST = 0.2  # Factor de impulso para las coincidencias de palabras clave
        
        # Copiamos las puntuaciones semánticas
        combined_scores = semantic_similarities.copy()
        
        # Aplicamos el impulso a las filas que coinciden con las palabras clave
        combined_scores[keyword_mask] += KEYWORD_BOOST
        
        # Nos aseguramos de que ninguna puntuación supere 1.0
        combined_scores = np.clip(combined_scores, 0, 1)

        # --- 4. Obtener los resultados con la nueva puntuación combinada ---
        top_indices = np.argsort(combined_scores)[-num_results:][::-1]

    st.subheader(f"Top {num_results} diagnósticos recomendados (Búsqueda Híbrida):")
    st.info("Resultados que contienen sus palabras clave exactas reciben un impulso en relevancia.")
    
    for i, idx in enumerate(top_indices):
        code = df.iloc[idx]['code_4d']
        description = df.iloc[idx]['description']
        final_score = combined_scores[idx]
        
        # Añadir indicador visual si hubo coincidencia de palabra clave
        is_keyword_match = keyword_mask.iloc[idx]
        
        with st.container(border=True):
            title = f"**{i+1}. {code}** - {description}"
            if is_keyword_match:
                st.markdown(f"#### {title} <span style='color: #28a745;'>✓ Coincidencia de palabra clave</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"#### {title}")
                
            st.progress(float(final_score), text=f"Relevancia: {final_score:.2%}")
            guidance = get_coding_guidance(code)
            if guidance:
                st.info(guidance)

elif search_button and not user_query:
    st.warning("Por favor, ingrese una descripción clínica para buscar.")

st.markdown("---")
st.markdown("Desarrollado como una herramienta de apoyo para profesionales de la salud.")
