import streamlit as st
import pandas as pd
import unicodedata
from thefuzz import process

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Asistente de Codificación CIE-10 (MINSAL)",
    page_icon="🇨🇱",
    layout="wide"
)

# --- Funciones de Lógica y Carga de Datos (con Caché) ---
def normalize_text(text: str) -> str:
    """Limpia y normaliza texto para búsquedas robustas."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    return "".join([c for c in text if not unicodedata.combining(c)])

@st.cache_data
def load_and_prepare_data():
    """
    Carga y procesa la lista tabular oficial del MINSAL (.xls),
    limpiándola para obtener solo los códigos codificables.
    """
    # --- MEJORA: LECTURA DESDE LA FUENTE OFICIAL MINSAL ---
    DATA_URL = "https://repositoriodeis.minsal.cl/ContenidoSitioWeb2020/uploads/2018/03/Lista-Tabular-CIE-10-1-1.xls"
    
    with st.spinner("Cargando y procesando base de datos oficial MINSAL..."):
        # Leer el archivo Excel, saltando las primeras filas que no son datos
        df_raw = pd.read_excel(DATA_URL, engine='openpyxl', header=None, skiprows=5)
        
        # Seleccionar y renombrar las columnas relevantes (Código y Descripción)
        df = df_raw[[0, 1]].copy()
        df.columns = ['code', 'description']

        # --- Limpieza de la Lista Tabular ---
        # 1. Eliminar filas completamente vacías
        df.dropna(subset=['code'], inplace=True)
        # 2. Convertir códigos a string para poder manipularlos
        df['code'] = df['code'].astype(str)
        # 3. FILTRO CLAVE: Mantener solo los códigos válidos. Se eliminan los títulos de capítulos (ej: 'I') 
        #    y los rangos de bloques (ej: 'A00-A09'). Los códigos válidos tienen 3 o más caracteres y no contienen guiones.
        df = df[~df['code'].str.contains('-') & (df['code'].str.len() >= 3)]
        df.reset_index(drop=True, inplace=True)

        # --- Enriquecimiento de Datos (como en la versión anterior) ---
        synonym_map = {
            "dolor de cabeza": "cefalea", "infarto": "isquemia miocardio", "cancer": "neoplasia maligna tumor",
            "corazon": "cardiaco cardiaca", "riñon": "renal", "pulmon": "neumo respiratorio",
            "azucar": "diabetes mellitus", "presion alta": "hipertension",
            "ataque cerebral": "accidente cerebrovascular acv", "hueso roto": "fractura"
        }
        def expand_with_synonyms(text):
            normalized_text = normalize_text(text)
            for key, value in synonym_map.items():
                if key in normalized_text:
                    normalized_text += f" {value}"
            return normalized_text

        df['code_4d'] = df['code'].str.replace('.', '', regex=False).apply(lambda x: x.ljust(4, 'X') if len(x) == 3 else x).str.slice(0, 4)
        df['search_field'] = df['description'].apply(expand_with_synonyms)
        df['block_code'] = df['code'].str[:3]

        clinical_area_map = {
            "Infecciosas y Parasitarias": ['A', 'B'], "Oncología (Neoplasias)": ['C', 'D'],
            "Endocrinología y Metabolismo": ['E'], "Salud Mental y Comportamiento": ['F'], "Neurología": ['G'],
            "Oftalmología y Otorrinolaringología": ['H'], "Cardiología y Sist. Circulatorio": ['I'],
            "Neumología y Sist. Respiratorio": ['J'], "Gastroenterología y Sist. Digestivo": ['K'], "Dermatología": ['L'],
            "Traumatología y Sist. Musculoesquelético": ['M'], "Nefrología y Sist. Genitourinario": ['N'],
            "Ginecología y Obstetricia": ['O'], "Pediatría y Perinatología": ['P'], "Genética y Malformaciones": ['Q'],
            "Síntomas y Hallazgos Anormales": ['R'], "Traumatismos, Envenenamientos y Causas Externas": ['S', 'T', 'V', 'W', 'X', 'Y'],
            "Factores de Salud y Contacto con Servicios": ['Z'], "Códigos Especiales": ['U']
        }
        
        letter_to_area = {letter: area for area, letters in clinical_area_map.items() for letter in letters}
        df['chapter_letter'] = df['code'].str[0]
        df['clinical_area'] = df['chapter_letter'].map(letter_to_area).fillna("Área no especificada")
        
        block_descriptions = df.groupby('block_code')['description'].first()
        df['block_desc'] = df['block_code'].map(block_descriptions)
    return df

def get_coding_guidance(code):
    # (Función sin cambios)
    if not code: return "", "info"
    chapter = code[0].upper()
    if chapter in ['S', 'T']: return "⚠️ **¡Atención!** Este es un código de **Lesión**. Es obligatorio añadir un código de **Causa Externa (V-Y)** que describa cómo ocurrió.", "error"
    if chapter in ['V', 'W', 'X', 'Y']: return "⚠️ **¡Atención!** Este es un código de **Causa Externa**. Debe usarse como código secundario junto a un código de **Lesión (S, T)**.", "error"
    if chapter == 'R': return "💡 **Consejo:** Los códigos 'R' son para síntomas o hallazgos sin un diagnóstico definitivo. Si se confirma una enfermedad, este código debe ser reemplazado.", "warning"
    if chapter == 'Z': return "💡 **Consejo:** Los códigos 'Z' no son enfermedades. Describen situaciones como controles, seguimientos o factores de riesgo.", "info"
    if chapter == 'C' or (chapter == 'D' and len(code) > 2 and code[1:3].isdigit() and int(code[1:3]) <= 48): return "💡 **Consejo:** Para neoplasias, es crucial especificar la localización y el comportamiento (maligno, benigno, etc.).", "info"
    return "✅ **Guía General:** Asegúrese de que este código sea el más específico posible según la documentación clínica.", "success"

# --- Inicialización de la Aplicación ---
st.title("🇨🇱 Asistente de Codificación CIE-10 (Base MINSAL)")
st.markdown("Busque con sinónimos o explore por área clínica para encontrar el código correcto.")

df = load_and_prepare_data()

if 'selected_code_4d' not in st.session_state:
    st.session_state.selected_code_4d = None

# --- Panel de Detalles en la Barra Lateral ---
with st.sidebar:
    st.header("📋 Panel de Análisis")
    if st.session_state.selected_code_4d:
        row = df[df['code_4d'] == st.session_state.selected_code_4d].iloc[0]
        
        st.subheader(f"Código: {row['code_4d']}")
        st.markdown(f"**{row['description']}**")
        st.divider()

        st.markdown(f"**Área Clínica:** {row['clinical_area']}")
        st.markdown(f"**Bloque:** {row['block_code']} - {row['block_desc']}")
        st.divider()

        st.subheader("🧠 Guía de Codificación")
        guidance, level = get_coding_guidance(row['code_4d'])
        getattr(st, level)(guidance)
        
        related_codes_df = df[df['block_code'] == row['block_code']]
        if len(related_codes_df) > 1:
            with st.expander("Ver códigos relacionados para mayor precisión"):
                for _, related_row in related_codes_df.iterrows():
                    st.markdown(f"`{related_row['code_4d']}` – {related_row['description']}")
        
        if st.button("Limpiar selección", use_container_width=True):
            st.session_state.selected_code_4d = None
            st.rerun()
    else:
        st.info("Seleccione un código de la búsqueda o el explorador para ver sus detalles aquí.")

# --- Interfaz Principal de Pestañas ---
tab1, tab2 = st.tabs(["🔍 Búsqueda Inteligente", "🗺️ Explorador Clínico"])

with tab1:
    st.header("Búsqueda por Término, Sinónimo o Código")
    search_query = st.text_input("Escriba para buscar:", placeholder="Ej: dolor de cabeza, cáncer de mama, I10X...")
    
    if search_query:
        sq_norm = normalize_text(search_query)
        code_matches = df[df['code_4d'].str.startswith(sq_norm.upper())]
        
        if not code_matches.empty:
            result_df = code_matches.head(20)
        else:
            search_field_with_index = pd.Series(df['search_field'].values, index=df.index)
            results = process.extract(sq_norm, search_field_with_index, limit=20)
            result_indices = [r[2] for r in results]
            result_df = df.loc[result_indices]

        st.caption(f"Mostrando hasta 20 resultados para '{search_query}'. Haga clic en uno para analizarlo.")
        for _, row in result_df.iterrows():
            if st.button(f"**{row['code_4d']}** – {row['description']}", key=row['code_4d'], use_container_width=True):
                st.session_state.selected_code_4d = row['code_4d']
                st.rerun()

with tab2:
    st.header("Explorar por Estructura Clínica")
    
    area_list = sorted(df['clinical_area'].unique())
    selected_area = st.selectbox("**Paso 1: Elija un Área Clínica**", area_list, index=None, placeholder="Filtre por especialidad médica...")

    if selected_area:
        df_area = df[df['clinical_area'] == selected_area]
        block_options = {f"{code} – {desc[:70]}...": code for code, desc in df_area.groupby('block_code')['description'].first().items()}
        selected_block_display = st.selectbox("**Paso 2: Elija una Subcategoría (Bloque)**", block_options.keys(), index=None, placeholder="Seleccione un grupo de diagnósticos...")

        if selected_block_display:
            selected_block_code = block_options[selected_block_display]
            df_block = df_area[df_area['block_code'] == selected_block_code]
            
            st.subheader(f"Códigos en el bloque '{selected_block_code}'")
            st.caption("Haga clic en un código para analizarlo en el panel lateral.")
            for _, row in df_block.iterrows():
                if st.button(f"**{row['code_4d']}** – {row['description']}", key=f"exp_{row['code_4d']}", use_container_width=True):
                    st.session_state.selected_code_4d = row['code_4d']
                    st.rerun()
