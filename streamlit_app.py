import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process, fuzz
import re

# --- Configuración de la Página ---
st.set_page_config(page_title="Asistente CIE-10", page_icon="🩺", layout="wide")

# --- Funciones de Carga y Procesamiento (Caché) ---

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_and_prepare_data():
    DATA_URL = "https://raw.githubusercontent.com/verasativa/CIE-10/refs/heads/master/codes.json"
    with st.spinner("Cargando catálogo CIE-10..."):
        df = pd.read_json(DATA_URL).dropna(subset=['code', 'description'])
        df['description_lower'] = df['description'].str.lower()
        df['code_4d'] = df['code'].str.replace('.', '', regex=False).apply(lambda x: x.ljust(4, 'X') if len(x) == 3 else x).str.slice(0, 4)
    return df

@st.cache_data
def create_embeddings(_model, descriptions):
    with st.spinner("Inicializando motor de IA..."):
        return _model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()

# --- Lógica de Búsqueda Híbrida y Flexible ---
def hybrid_search(query, df, embeddings, model, num_results=10, boost=0.3):
    query_lower = query.lower()
    
    # 1. Búsqueda Semántica
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    semantic_scores = cosine_similarity(query_embedding, embeddings)[0]
    
    # 2. Búsqueda por Palabras Clave Flexible (Fuzzy Search)
    # Encuentra la mejor coincidencia para la consulta completa en las descripciones
    fuzzy_scores = df['description_lower'].apply(lambda x: fuzz.partial_ratio(query_lower, x) / 100.0)
    
    # 3. Combinación de puntuaciones
    combined_scores = (semantic_scores * 0.6) + (fuzzy_scores * 0.4)
    
    # 4. Impulso a coincidencias directas
    keywords = [word for word in re.split(r'\W+', query_lower) if word]
    keyword_mask = pd.Series(True, index=df.index)
    if keywords:
        for keyword in keywords:
            keyword_mask &= df['description_lower'].str.contains(keyword, na=False)
        combined_scores[keyword_mask] += boost
    
    top_indices = np.argsort(combined_scores)[-num_results:][::-1]
    return top_indices, combined_scores

# --- Lógica del Asistente de Codificación Guiada ---
def guided_coder_ui():
    st.header("🔍 Asistente de Codificación Guiada")
    flow_type = st.session_state.get('flow_type')

    if flow_type == 'autolesion':
        st.info("Detectamos un escenario de **Lesión Autoinfligida**. Por favor, complete los siguientes pasos para una codificación precisa.")
        
        # Opciones para el asistente
        metodos = {"No especificado": "X84X", "Objeto Cortante": "X78X", "Ahorcamiento": "X70X", "Envenenamiento (Drogas/Medicamentos)": "X64X", "Salto desde altura": "X80X", "Disparo de arma de fuego": "X74X"}
        zonas = ["No especificado", "Cabeza", "Cuello", "Tórax", "Abdomen/Espalda/Pelvis", "Hombro/Brazo", "Muñeca/Mano", "Cadera/Muslo", "Pierna/Tobillo/Pie"]

        # Paso 1: Método
        st.session_state.metodo = st.selectbox("Paso 1: Seleccione el método utilizado:", options=list(metodos.keys()), key="metodo_select")
        
        # Paso 2: Zona
        st.session_state.zona = st.selectbox("Paso 2: Seleccione la zona principal de la lesión:", options=zonas, key="zona_select")

        if st.button("Generar Códigos Recomendados", type="primary"):
            st.session_state.show_guided_results = True

    if st.session_state.get('show_guided_results'):
        st.subheader("Resultados de la Codificación Guiada")
        
        # Lógica para mostrar resultados
        st.success("Basado en sus selecciones, la codificación precisa requiere al menos dos códigos:")
        
        # Código de Causa Externa
        codigo_causa = metodos[st.session_state.metodo]
        desc_causa_df = df[df['code_4d'].str.startswith(codigo_causa[:3])]
        desc_causa = desc_causa_df['description'].iloc[0] if not desc_causa_df.empty else "Descripción no encontrada"
        st.markdown(f"#### 1. Causa Externa (Intencionalidad y Método)")
        with st.container(border=True):
            st.markdown(f"**Código:** `{codigo_causa}`")
            st.markdown(f"**Descripción:** {desc_causa}")

        # Código de Lesión
        st.markdown(f"#### 2. Lesión Física (Daño Corporal)")
        st.markdown(f"Use el buscador principal con términos como **'herida {st.session_state.zona}'** o **'fractura {st.session_state.zona}'** para encontrar el código de lesión más específico (Capítulos S y T).")
        with st.expander("Ver ejemplos de códigos de lesión para la zona seleccionada"):
            query_lesion = f"herida traumatismo {st.session_state.zona}".lower()
            if st.session_state.zona == "No especificado":
                query_lesion = "traumatismo de sitio no especificado"
            
            indices, scores = hybrid_search(query_lesion, df, embeddings, model, num_results=3)
            for idx in indices:
                st.write(f"`{df.iloc[idx]['code_4d']}` - {df.iloc[idx]['description']}")

    if st.button("↩️ Iniciar Nueva Búsqueda"):
        # Limpiar el estado para volver al buscador principal
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Inicialización y Flujo Principal de la App ---
if 'flow_type' not in st.session_state:
    st.session_state.flow_type = None

# Cargar datos y modelo
model = load_model()
df = load_and_prepare_data()
embeddings = create_embeddings(model, df['description'].tolist())

# Título
st.title("🩺 Asistente Inteligente de Codificación CIE-10")

# Decidir qué UI mostrar: el buscador principal o el asistente guiado
if st.session_state.flow_type:
    guided_coder_ui()
else:
    st.markdown("Use la búsqueda para encontrar códigos o active el **Asistente Guiado** con términos como `suicidio` o `accidente`.")
    with st.container(border=True):
        user_query = st.text_area("Ingrese la descripción clínica:", placeholder="Ej: intento de suicidio cortes en muñeca")
        search_button = st.button("Buscar Diagnósticos", type="primary")

    if search_button and user_query:
        query_lower = user_query.lower()
        
        # Detección de palabras clave para iniciar el Asistente Guiado
        if any(keyword in query_lower for keyword in ["suicidio", "autoinflingida", "autolesión"]):
            st.session_state.flow_type = 'autolesion'
            st.rerun() # Recargar la app para mostrar la UI del asistente
        # Aquí se pueden añadir más `elif` para otros flujos (ej: accidentes)
        
        else: # Búsqueda normal híbrida
            indices, scores = hybrid_search(user_query, df, embeddings, model, num_results=5)
            st.subheader("Resultados de la Búsqueda Híbrida")
            for idx in indices:
                with st.container(border=True):
                    st.markdown(f"**`{df.iloc[idx]['code_4d']}`** - {df.iloc[idx]['description']}")
                    st.progress(float(scores[idx]), text=f"Relevancia: {scores[idx]:.2%}")
