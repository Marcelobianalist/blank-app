import streamlit as st
import pandas as pd
from thefuzz import process

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Portal Interactivo CIE-10",
    page_icon="📖",
    layout="wide"
)

# --- Funciones de Lógica y Carga de Datos (con Caché para máxima velocidad) ---

@st.cache_data
def load_and_prepare_data():
    """Carga y pre-procesa el catálogo CIE-10 una sola vez."""
    DATA_URL = "https://raw.githubusercontent.com/verasativa/CIE-10/refs/heads/master/codes.json"
    with st.spinner("Cargando y optimizando el catálogo CIE-10..."):
        df = pd.read_json(DATA_URL).dropna(subset=['code', 'description'])
        
        # Crear la versión de 4 dígitos para cumplir la convención chilena
        df['code_4d'] = df['code'].str.replace('.', '', regex=False).apply(
            lambda x: x.ljust(4, 'X') if len(x) == 3 else x
        ).str.slice(0, 4)
        
        # Extraer información estructural para el explorador
        df['chapter_letter'] = df['code'].str[0]
        
        # Diccionario de capítulos para dar contexto (COMPLETO)
        chapters_dict = {
            'A': "I. Ciertas enfermedades infecciosas y parasitarias", 'B': "I. Ciertas enfermedades infecciosas y parasitarias",
            'C': "II. Neoplasias (tumores)", 'D': "III-IV. Enf. de la sangre / Enf. endocrinas, nutricionales y metabólicas",
            'E': "III-IV. Enf. de la sangre / Enf. endocrinas, nutricionales y metabólicas",
            'F': "V. Trastornos mentales y del comportamiento",
            'G': "VI. Enfermedades del sistema nervioso",
            'H': "VII-VIII. Enfermedades del ojo y del oído",
            'I': "IX. Enfermedades del sistema circulatorio",
            'J': "X. Enfermedades del sistema respiratorio",
            'K': "XI. Enfermedades del sistema digestivo",
            'L': "XII. Enfermedades de la piel y del tejido subcutáneo",
            'M': "XIII. Enfermedades del sistema musculoesquelético",
            'N': "XIV. Enfermedades del sistema genitourinario",
            'O': "XV. Embarazo, parto y puerperio",
            'P': "XVI. Ciertas afecciones originadas en el período perinatal",
            'Q': "XVII. Malformaciones congénitas y anomalías cromosómicas",
            'R': "XVIII. Síntomas, signos y hallazgos anormales",
            'S': "XIX. Traumatismos, envenenamientos y otras causas externas", 'T': "XIX. Traumatismos, envenenamientos y otras causas externas",
            'V': "XX. Causas externas de morbilidad", 'W': "XX. Causas externas de morbilidad", 'X': "XX. Causas externas de morbilidad", 'Y': "XX. Causas externas de morbilidad",
            'Z': "XXI. Factores que influyen en el estado de salud",
            'U': "XXII. Códigos para propósitos especiales"
        }
        df['chapter_desc'] = df['chapter_letter'].map(chapters_dict).fillna("Capítulo no especificado")
        
        return df

def get_coding_guidance(code):
    """Proporciona orientación específica y educativa basada en el código."""
    if not code: return ""
    chapter = code[0].upper()
    guidance = []

    if chapter in ['A', 'B']: guidance.append("💡 **Consejo:** Para enfermedades infecciosas, verifique si necesita un código adicional para el organismo causante.")
    elif chapter == 'C' or (chapter == 'D' and len(code) > 2 and code[1:3].isdigit() and int(code[1:3]) <= 48): guidance.append("💡 **Consejo:** Para tumores, es crucial especificar la localización y el comportamiento (maligno, benigno, etc.). Considere usar códigos 'Z' para historiales de cáncer.")
    elif chapter in ['S', 'T']: guidance.append("⚠️ **¡Atención!** Este es un código de **Lesión**. Es **obligatorio** añadir un segundo código del capítulo XX (V, W, X, Y) que describa la **Causa Externa** (ej: `W19X` para una caída, `X78X` para una autolesión).")
    elif chapter in ['V', 'W', 'X', 'Y']: guidance.append("⚠️ **¡Atención!** Este es un código de **Causa Externa**. Debe ser usado como código secundario junto a un código de **Lesión** (Capítulo XIX: S, T) que describa el daño físico.")
    elif chapter == 'R': guidance.append("💡 **Consejo:** Los códigos 'R' son para síntomas o hallazgos sin un diagnóstico definitivo. Si se confirma una enfermedad, este código debe ser reemplazado.")
    elif chapter == 'Z': guidance.append("💡 **Consejo:** Los códigos 'Z' no son enfermedades. Describen situaciones como controles, seguimientos, o factores de riesgo. Pueden ser el diagnóstico principal en consultas de control.")
    else: guidance.append("✅ **Guía General:** Asegúrese de que este código sea el más específico posible según la documentación clínica disponible.")
    
    return "\n".join(guidance)

# --- Inicialización de la Aplicación ---
st.title("📖 Portal Interactivo CIE-10")
st.markdown("Una herramienta para buscar, explorar y entender la codificación CIE-10 de forma intuitiva.")

df = load_and_prepare_data()

# Inicializar el estado de la sesión para guardar el código seleccionado
if 'selected_code_4d' not in st.session_state:
    st.session_state.selected_code_4d = None

# --- Interfaz de Pestañas: Búsqueda vs. Explorador ---
tab1, tab2 = st.tabs(["🔍 Búsqueda Rápida", "🗺️ Explorador Guiado"])

# --- PESTAÑA 1: BÚSQUEDA RÁPIDA ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Motor de Búsqueda")
        search_query = st.text_input("Buscar por descripción o palabra clave:", placeholder="Ej: fractura de clavicula, cefalea tensional...")
        
        # Lógica de búsqueda
        if search_query:
            # Usamos thefuzz para encontrar las mejores 20 coincidencias flexibles
            choices = df['description'].tolist()
            results = process.extract(search_query, choices, limit=20)
            
            # Obtener los índices de los resultados
            result_descriptions = [r[0] for r in results]
            result_df = df[df['description'].isin(result_descriptions)]
            
            st.write(f"**{len(result_df)} resultados encontrados para '{search_query}':**")
            
            # Mostrar resultados como botones para la selección
            for index, row in result_df.iterrows():
                if st.button(f"**{row['code_4d']}** - {row['description']}", key=f"btn_{row['code_4d']}", use_container_width=True):
                    st.session_state.selected_code_4d = row['code_4d']

    with col2:
        st.header("📋 Panel de Detalles")
        if st.session_state.selected_code_4d:
            selected_data = df[df['code_4d'] == st.session_state.selected_code_4d].iloc[0]
            with st.container(border=True):
                st.subheader(f"Código: {selected_data['code_4d']}")
                st.markdown(f"**Descripción:** {selected_data['description']}")
                st.divider()
                st.markdown(f"**Capítulo:** {selected_data['chapter_desc']} ({selected_data['chapter_letter']})")
                st.divider()
                st.subheader("Guía de Codificación")
                guidance = get_coding_guidance(selected_data['code_4d'])
                st.info(guidance)
        else:
            st.info("Haga clic en un resultado de la búsqueda para ver sus detalles aquí.")

# --- PESTAÑA 2: EXPLORADOR GUIADO ---
with tab2:
    st.header("Navegue la Estructura de la CIE-10")
    st.markdown("Ideal para aprender y encontrar códigos cuando no se conoce la descripción exacta.")
    
    # Paso 1: Seleccionar Capítulo
    chapter_list = sorted(df['chapter_desc'].unique().tolist())
    selected_chapter_desc = st.selectbox("**Paso 1: Elija un Capítulo**", options=chapter_list, index=None, placeholder="Seleccione una categoría general...")
    
    if selected_chapter_desc:
        df_chapter = df[df['chapter_desc'] == selected_chapter_desc]
        
        # Crear subcategorías basadas en los 3 primeros dígitos del código
        df_chapter['block'] = df_chapter['code'].str[:3]
        block_descriptions = df_chapter.groupby('block')['description'].first()
        block_options = {f"{block_code} - {block_descriptions[block_code][:50]}...": block_code for block_code in block_descriptions.index}

        # Paso 2: Seleccionar Bloque/Subcategoría
        selected_block_display = st.selectbox("**Paso 2: Elija una Subcategoría**", options=block_options.keys(), index=None, placeholder="Seleccione una subcategoría más específica...")
        
        if selected_block_display:
            selected_block_code = block_options[selected_block_display]
            df_block = df_chapter[df_chapter['block'] == selected_block_code]
            
            # Paso 3: Mostrar resultados
            st.subheader(f"Códigos en el bloque '{selected_block_code}'")
            st.dataframe(df_block[['code_4d', 'description']], use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("Desarrollado como una herramienta de apoyo y aprendizaje para la codificación clínica.")
