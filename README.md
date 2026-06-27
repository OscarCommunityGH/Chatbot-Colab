# Chatbot-Colab

Open In Colab

# ============================================
# CHATBOT v2.1
# CHATBOT CON RANDOM FOREST + GRADIO
# Interfaz tipo ChatGPT + Soporte PDF
# Cambios: Corrección de palabras
# ============================================

# --------------------------------------------
# PASO 1: INSTALAR DEPENDENCIAS
# --------------------------------------------
!pip install gradio==6.19.0 pypdf2 -q

# --------------------------------------------
# PASO 2: IMPORTAR LIBRERÍAS
# --------------------------------------------
import gradio as gr
print("Versión de Gradio:", gr.__version__)
import PyPDF2
import io
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ============================================
# PASO 3: DATOS DE ENTRENAMIENTO (AMPLIADOS)
# ============================================

X_train = [
    # --- Relajante / Ansiolítico ---
    "tengo ansiedad", "no puedo dormir", "estres constante", "me siento nervioso",
    "insomnio severo", "me cuesta conciliar el sueño", "estoy muy tenso",
    "angustia y nerviosismo", "no logro relajarme", "tensión emocional",
    "me siento ansioso todo el dia", "nervios antes de dormir",
    "dan ataques de ansiedad", "inquietud", "alterado", "nervios de punta",
    "alterada", "crisis de pánico", "presión en el pecho por estrés",
    "no me puedo calmar", "no me puedo calmar", "muy acelerado",
    "ganas de llorar de la nada", "dolor de cabeza",
    "estoy desbordado por los problemas", "me cuesta respirar de los nervios",
    "siento un nudo en el estómago", "estoy hiperventilando",
    "me tiemblan las manos por nervios", "siento que voy a colapsar",
    "no puedo controlar mis pensamientos", "estoy irritable por todo",
    "tengo miedo sin razón", "me duele el cuello y la espalda por estrés",
    "dolor de cabeza por tensión", "estoy bajo mucha presión",
    "ya no aguanto el estrés", "siento el cuerpo muy rígido",
    "tensión muscular por nervios", "estoy agotado mentalmente", "ansioso",
    "ansiosa", "estresado", "estresada", "nerviosismo crónico", "angustia",
    "tranquilizante natural", "relajante para los nervios", "remedio para el estrés",
    "siento desespero", "estoy de mal humor por el estres", "mente saturada",
    "siento que me va a dar un ataque", "no me puedo concentrar de la tension",
    "pensamientos acelerados", "me dan tics nerviosos", "necesito calmarme",
    "infusion para relajar", "te para los nervios","no tengo paz mental",

    # --- Digestivo ---
    "dolor estomacal", "problemas digestivos", "me duele el estomago",
    "gases y distension", "nauseas despues de comer", "colicos abdominales",
    "indigestion", "mal aliento", "estomago inflamado", "flatulencias",
    "digestión lenta", "acidez estomacal", "parasitos intestinales",
    "tengo empacho", "me cayo pesada la comida", "ardor en el pecho despues de comer",
    "reflujo gastrico", "sensacion de lleno", "pesadez estomacal", "retortijones",
    "no puedo ir al bano", "estrenimiento", "infeccion en la panza", "plantas para la digestion",
    "tengo torzon", "vomito y asco", "mucha llenura", "agruras espantosas",
    "infeccion intestinal", "dolor de vientre bajo", "me gruñe la tripa", "gases atorados",
    "congestion estomacal", "pesadez despues del almuerzo", "no digiero bien", "ardor de estomago",
    "irritacion intestinal", "boca amarga", "empachado", "dolor de panza agudo",
    "colitis espasmodica", "flora intestinal danada", "deshacer los gases", "limpieza estomacal",
    "ascos matutinos", "inflamacion del colon", "gases con dolor", "pesadez estomacal cronica",
    "mucha acidez", "retorcijones fuertes", "alivio para el empacho", "mala digestion siempre",
    "estomago revuelto", "ganas de vomitar", "dolor de estomago fuerte", "inflamacion intestinal",
    "remedio para la acidez", "te para la digestion", "remedio para los parasitos", "limpiar el estomago",

    # --- Respiratorio ---
    "tengo tos", "congestion nasal", "dolor de garganta",
    "gripe y resfriado", "tos con flema", "pecho congestionado",
    "bronquitis leve", "garganta irritada", "mocos y congestion",
    "tos seca persistente", "resfriado con fiebre leve", "tengo la nariz tapada",
    "me cuesta respirar por la gripe", "siento resequedad en la garganta", "no dejo de estornudar",
    "flemas atoradas", "goteo nasal", "siento el pecho cerrado", "remedio para la tos",
    "te para el resfriado", "siento flemas en el pecho", "tengo mucha carraspera", "catarro fuerte",
    "ataque de tos", "no puedo respirar bien por moco", "irritacion en las vias respiratorias", "gripe estacional",
    "garganta inflamada", "resfriado comun", "congestion en los pulmones", "tos que no se quita",
    "ronquera", "me duele al pasar saliva", "afonia", "pecho cerrado por frio",
    "mucha mucosidad", "gripe fuerte", "tos con garganta seca", "respiracion ruidosa por flema",
    "dolor de pecho por toser", "nariz tapada por completo", "goteo nasal constante", "catarro mal cuidado",
    "garganta rasposa", "remedio para el catarro", "jarabe natural para tos", "te para la garganta",
    "desinflamar las amigdalas", "descongestionas las fosas nasales", "tos de perro", "siento ahogo por moco",
    "gripe invernal", "congestion respiratoria", "tos nocturna", "mucha flema verde",
    "irritacion de laringe", "resfriado con escalofrios", "bajar la congestion", "abrir los bronquios",
    "alivio para la gripe",

    # --- Energía ---
    "me siento cansado", "fatiga constante", "no tengo energia",
    "me falta vitalidad", "somnolencia durante el dia", "debilidad general",
    "despierto a mitad de la noche", "tengo el sueno muy ligero", "no descanso nada al dormir",
    "desvelo", "me siento sin fuerzas", "agotamiento fisico", "no puedo con el dia",
    "tengo mucha flojera", "apatia", "necesito algo que me despierte", "plantas estimulantes",
    "siento el cuerpo pesado", "flojera extrema", "cansancio mental", "no tengo ganas de hacer nada",
    "me falta fuerza", "agotamiento total", "sueño todo el dia", "me arrastro de cansancio",
    "bajas energias", "no tengo rendimiento", "fatiga muscular", "desgano total",
    "cuerpo sin pilas", "necesito un plus de energia", "me falta animo", "rendimiento bajo",
    "agotado desde la mañana", "siento debilidad en las piernas", "mucha pereza", "falta de vigor",
    "mente apagada", "cansancio cronico", "no rindo en la escuela", "no rindo en el trabajo",
    "cuerpo agotado", "necesito lucidez", "falta de concentracion por sueño", "sueño pesado de dia",
    "me falta motivacion", "bateria baja cuerpo", "siento letargo", "estado de somnolencia",
    "fatiga extrema", "despertar sin energia", "cansancio acumulado", "necesito reanimarme",
    "vitalidad baja", "cuerpo debil", "estimulante natural", "para quitar el sueño",
    "bebida energetica natural", "para el cansancio", "activar el cuerpo",

    # --- Dolor muscular / Inflamación ---
    "me duelen los musculos", "golpe e inflamacion", "dolor articular",
    "inflamacion en la rodilla", "inflamación", "torcedura y dolor", "tension muscular",
    "dolor en la espalda", "contractura muscular", "me dio un calambre",
    "dolor de reumas", "siento el cuerpo cortado", "tengo una contusion",
    "me duele todo el cuerpo", "inflamacion por un golpe", "dolor de huesos",
    "cuello torcido", "torticolis", "pomada para golpes", "dolor de articulaciones",
    "espalda baja lastimada", "dolor de hombro", "inflamacion de tendon", "tendinitis leve",
    "golpe muscular", "esguince leve", "muñeca abierta dolor", "dolor de rodillas cronico",
    "artritis leve", "dolor lumbar", "ciatica dolor", "cuello rigido",
    "espalda contracturada", "dolor por ejercicio", "musculos adoloridos", "dolor de cadera",
    "inflamacion de tobillo", "dolor por caida", "reumas en las manos", "dolor de ligamentos",
    "desinflamatorio muscular", "dolor de nuca", "cuerpo entumido por ejercicio", "desgarro muscular leve",
    "inflamacion articular", "dolor de pies por caminar", "huesos adoloridos por frio", "dolor muscular cronico",
    "tension en los hombros", "dolor de coxis", "inflamacion en los dedos", "pies hinchados y adoloridos",
    "punsadas en la espalda", "remedio para desinflamar", "hierba para los golpes", "te para el dolor de cabeza",
    "aliviar las reumas", "quitar contracturas", "desinflamar rodilla", "calmar dolor muscular",

    # --- Piel ---
    "quemadura leve", "irritacion en la piel", "herida superficial",
    "piel seca y deshidratada", "acne leve", "erupcion cutanea",
    "picadura de insecto", "piel irritada", "tengo una roncha", "me arde la piel",
    "raspadura leve", "granitos en la cara", "piel agrietada", "picazon en el cuerpo",
    "quemadura de sol", "cicatrizar una herida", "tengo eccema leve", "alergia en la piel",
    "comezon en los brazos", "piel descascarada", "ronchas rojas", "rozadura de piel",
    "manchas por el sol", "granos con pus", "espinillas en la nariz", "cutis graso con acne",
    "piel quemada por el frio", "resequedad extrema en manos", "talones agrietados", "picadura de mosquito",
    "cicatrizacion lenta", "pequeña cortada", "piel inflamada y roja", "urticaria leve",
    "comezon por hiedra", "escoriacion leve", "raspon con sangre", "piel sensible irritada",
    "ampolla en el pie", "puntos negros cara", "infeccion cutanea superficial", "remedio para las quemaduras",
    "crema natural para rozaduras", "quitar la picazon", "desinflamar roncha", "cicatrizante natural",
    "limpiar el acne", "suavizar piel seca", "alivio para picaduras", "regenerar la piel",
    "quitar manchas de piel", "piel escamosa", "irritacion por afeitado", "quemadura con aceite caliente",
    "ronchas por calor", "comezon nocturna en piel", "remedio para granitos", "limpieza cutanea",
    "proteger la dermis", "hidratar la piel",

    # --- Circulación ---
    "mala circulacion", "retencion de liquidos", "piernas hinchadas",
    "varices leves", "presion baja", "falta de circulacion en piernas",
    "siento las piernas pesadas", "hormigueo en las manos", "pies frios todo el tiempo",
    "entumecimiento de extremidades", "aranitas vasculares", "tobillos inflamados",
    "plantas para activar la sangre", "hormigueo en los pies", "manos dormidas",
    "circulacion lenta", "pesadez en los muslos", "venas saltadas dolor", "pies hinchados al caminar",
    "mala circulacion en los brazos", "siento frio el cuerpo por circulacion", "retencion de agua en piernas",
    "dolor de varices", "piernas cansadas al final del dia", "entumecimiento de dedos", "calambres por mala circulacion",
    "presion arterial baja", "mareo por presion baja", "falta de riego sanguineo", "tobillos muy hinchados",
    "manos frias constantemente", "sentir pinchazos en las piernas", "sangre pesada", "venas inflamadas",
    "acumulacion de liquidos", "piernas inflamadas por calor", "hormigueo constante", "dedos de los pies morados",
    "mala circulacion general", "activar el flujo sanguineo", "estimular la circulacion", "remedio para las varices",
    "desinflamar las piernas", "eliminar liquidos retenidos", "plantas diureticas", "te para la circulacion",
    "evitar la retencion de liquidos", "mejorar flujo de sangre", "pies adormecidos", "pantorrillas hinchadas",
    "dolor sordo en las piernas", "venitas rojas en la piel", "sensacion de opresion en piernas", "brazo entumecido",
    "circulacion deficiente", "bajar la hinchazon de pies", "remedio para presion baja", "plantas venotonicas",
    "alivio para piernas pesadas", "purificar la sangre",

    # --- Antiviral / Antimicrobiano ---
    "infeccion leve", "bacteria en piel", "herida infectada leve",
    "defensas bajas", "gripe con fiebre", "virus estomacal", "subir el sistema inmune",
    "remedio antiseptico", "para combatir bacterias", "desinfectar una cortada",
    "prevenir contagios", "antibiotico natural", "fortalecer las defensas", "subir plaquetas",
    "combatir un virus", "infeccion de garganta leve", "matar microbios", "evitar infecciones",
    "remedio antibacterial", "infeccion por hongo leve", "desinfectante natural", "protección contra virus",
    "subir los globulos blancos", "limpiar bacterias", "infección en raspadura", "antiseptico para heridas",
    "fiebre por infeccion leve", "fortalecer sistema inmunologico", "hongos en los pies", "contra las bacterias",
    "evitar la propagacion de virus", "gripe viral", "infeccion cutanea por microbios", "depurar el organismo de virus",
    "antimicrobiano natural", "remedio antimicrobiano", "combatir infecciones leves", "subir defensas del cuerpo",
    "evitar que se infecte", "lavado antiseptico", "eliminar germenes", "gripe con calentura",
    "virus de temporada", "cuerpo cortado por virus", "infeccion leve de vias urinarias", "plantas antibioticas",
    "te para subir defensas", "jarabe antimicrobiano", "desinfectar por dentro", "eliminar toxinas bacterianas",
    "evitar hongos", "fortalecedor inmunologico", "para no enfermarme", "remedio contra parasitos y bacterias",
    "infeccion leve en los ojos", "limpieza de microbios", "proteger contra bacterias", "eliminar celulas infecciosas",
    "subir defensas rapido", "antiviral natural",
]

# ============================================
# PASO 3.1: ETIQUETAS — generadas automáticamente
# ============================================

# Define cuántas muestras tiene cada categoría EN EL MISMO ORDEN que X_train
etiquetas_conteo = [
    ("relajante",      60),
    ("digestivo",      60),
    ("respiratorio",   60),
    ("energia",        60),
    ("dolor_muscular", 60),
    ("piel",           60),
    ("circulacion",    60),
    ("antimicrobiano", 60),
]

y_train = []
for etiqueta, cantidad in etiquetas_conteo:
    y_train.extend([etiqueta] * cantidad)

# Verificación automática
assert len(X_train) == len(y_train), (
    f"❌ Desalineado: X_train={len(X_train)}, y_train={len(y_train)}\n"
    f"   Diferencia: {len(X_train) - len(y_train)} etiquetas faltantes"
)
print(f"✅ Datos alineados: {len(X_train)} muestras")

# ============================================
# PASO 4: VECTORIZAR Y ENTRENAR
# ============================================

vectorizador = TfidfVectorizer()
X_vect = vectorizador.fit_transform(X_train)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_vect, y_train)

# ============================================
# PASO 5: BASE DE PLANTAS AMPLIADA
# ============================================

mapa_plantas = {
    "relajante":      ["Valeriana", "Lavanda", "Tila", "Pasiflora", "Cedrón", "Manzanilla", "Ruda"],
    "digestivo":      ["Manzanilla", "Hierbabuena", "Menta", "Anís", "Hinojo", "Epazote",
                       "Albahaca", "Romero", "Orégano", "Nopal"],
    "respiratorio":   ["Eucalipto", "Bugambilia", "Tomillo", "Jengibre", "Limón", "Cebolla",
                       "Canela", "Ajo"],
    "energia":        ["Jengibre", "Romero", "Canela"],
    "dolor_muscular": ["Árnica", "Romero", "Menta", "Manzanilla", "jengibre", "Lavanda"],
    "piel":           ["Sábila", "Caléndula", "Neem", "Árnica"],
    "circulacion":    ["Romero", "Diente de León", "Perejil", "Nopal"],
    "antimicrobiano": ["Ajo", "Neem", "Orégano", "Tomillo", "Limón"],
}

# Preparación y frecuencia de uso por planta
preparacion_plantas = {
    "Valeriana":      "🫖 Hierve 1 taza de agua, agrega 1 cdta de raíz seca. Reposa 10 min tapado.\n⏱️ Tomar 1 taza por la noche, antes de dormir.",
    "Lavanda":        "🫖 Vierte agua caliente (no hirviendo) sobre 1 cdta de flores secas. Reposa 5–7 min.\n⏱️ Hasta 2 tazas al día; ideal una antes de dormir.",
    "Manzanilla":     "🫖 Hierve 1 taza de agua, añade 1 cdta de flores. Reposa 5 min tapado.\n⏱️ 1–2 tazas al día, después de cada comida.",
    "Hierbabuena":    "🫖 Infusiona 5–6 hojas frescas (o 1 cdta secas) en agua caliente 5 min.\n⏱️ 2–3 veces al día, especialmente después de comidas pesadas.",
    "Sábila":         "🌿 Extrae el gel del interior de la hoja. Aplica directamente sobre la zona afectada.\n⏱️ 2–3 veces al día sobre piel limpia y seca.",
    "Eucalipto":      "🫖 Hierve 3 hojas en 1 taza de agua 5 min. Reposa 3 min. También puedes inhalar el vapor.\n⏱️ 2–3 tazas al día. No ingerir aceite esencial puro.",
    "Árnica":         "💊 Aplica pomada o compresa con infusión sobre la zona afectada.\n⏱️ 2–3 veces al día. Solo uso externo.",
    "Romero":         "🫖 Hierve 1 cdta de hojas en 1 taza de agua 5 min. Reposa y cuela.\n⏱️ 1–2 tazas al día. Evitar en exceso si hay hipertensión.",
    "Jengibre":       "🫖 Hierve 2–3 rodajas de raíz fresca 10 min. Agrega miel o limón al gusto.\n⏱️ 1–2 tazas al día. Consumir con moderación.",
    "Bugambilia":     "🫖 Hierve 5–6 brácteas (las hojas de color) en 1 taza de agua 5 min.\n⏱️ 2–3 tazas al día. Mezclar con miel para la tos.",
    "Menta":          "🫖 Infusiona 1 cdta de hojas secas en agua caliente 5 min.\n⏱️ 2 tazas al día. No consumir en exceso.",
    "Tila":           "🫖 Vierte agua caliente sobre 1 cdta de flores de tila. Reposa 7 min tapado.\n⏱️ 1–2 tazas al día, ideal 30 min antes de dormir.",
    "Canela":         "🫖 Hierve 1 raja de canela en 1 taza de agua 10 min.\n⏱️ 1–2 tazas al día. Evitar en exceso durante el embarazo.",
    "Limón":          "🍋 Exprimir medio limón en agua tibia. Opcionalmente agregar miel.\n⏱️ 2–3 veces al día. Consumir fresco.",
    "Ajo":            "🧄 Consumir 1–2 dientes crudos o en infusión con agua tibia.\n⏱️ Una vez al día, preferentemente en ayunas.",
    "Cebolla":        "🫖 Prepara jarabe: hierve 1 cebolla picada con miel 15 min. Cuela y guarda.\n⏱️ 1 cdta cada 4–6 horas para la tos.",
    "Diente de León": "🫖 Hierve 1 cdta de hojas o raíz seca en 1 taza de agua 5 min.\n⏱️ 2–3 tazas al día. Mantener buena hidratación.",
    "Caléndula":      "💊 Aplica pomada o infusión fría sobre la zona afectada.\n⏱️ 2–3 veces al día en uso externo. Consumir con moderación.",
    "Ruda":           "🫖 Hierve 2–3 hojas en 1 taza de agua 5 min. No concentrar demasiado.\n⏱️ 1 taza al día máximo. No recomendada en embarazo.",
    "Anís":           "🫖 Hierve 1 cdta de semillas de anís en 1 taza de agua 5 min.\n⏱️ 1–2 tazas al día, después de comer.",
    "Tomillo":        "🫖 Infusiona 1 cdta de tomillo seco en agua caliente 7 min.\n⏱️ 2–3 tazas al día para resfriados leves.",
    "Orégano":        "🫖 Hierve 1 cdta de orégano seco en 1 taza de agua 5 min.\n⏱️ 2 tazas al día. No consumir en exceso.",
    "Perejil":        "🫖 Infusiona 1 cdta de perejil fresco en agua caliente 5 min.\n⏱️ 2 tazas al día. También consumir fresco en ensaladas.",
    "Albahaca":       "🫖 Hierve 5–6 hojas en 1 taza de agua 5 min.\n⏱️ 1–2 tazas al día. Guardar en lugares frescos.",
    "Cedrón":         "🫖 Vierte agua caliente sobre 1 cdta de hojas de cedrón. Reposa 7 min.\n⏱️ 1–2 tazas al día, tomadas tibias.",
    "Pasiflora":      "🫖 Hierve 1 cdta de hojas o flores secas en 1 taza de agua 10 min.\n⏱️ 1 taza por la noche. No combinar con sedantes.",
    "Neem":           "💊 Aplica pomada de neem sobre la zona afectada o infusiona 1 cdta en agua.\n⏱️ 1–2 veces al día. Usar con moderación.",
    "Epazote":        "🫖 Hierve 3–4 ramitas en 1 taza de agua 5 min.\n⏱️ 1 taza al día. Consumir en pequeñas cantidades.",
    "Hinojo":         "🫖 Hierve 1 cdta de semillas de hinojo en 1 taza de agua 7 min.\n⏱️ 1–2 tazas al día, después de comer.",
    "Nopal":          "🥤 Licua 1 penca limpia con agua o agua de jamaica. También consumir cocido.\n⏱️ 1 vaso al día en ayunas para mejor efecto.",
}

# ============================================
# MUNICIPIO TLAXCALA → PLANTAS DISPONIBLES (fuente: catálogo PDF)
# ============================================

MUNICIPIOS_PLANTAS = {
    "Amaxac": {
        "Manzanilla", "Árnica", "Tila", "Zacate limón", "Cola de caballo",
        "Pasiflora", "Nopal", "Higuera", "Estevia", "Aguacate", "Cedrón",
    },
    "Huamantla": {
        "Guayaba", "Limón", "Granada", "Durazno", "Bugambilia",
        "Pirul", "Tejocote", "Capulín", "Tártago", "Siempreviva", "Cempasúchil",
    },
    "Ixtenco": {
        "Epazote", "Estafiate", "Hierbabuena", "Gordolobo", "Malva",
        "Menta", "Anís", "Hinojo", "Ajenjo", "Pericón", "Santa María",
        "Té de monte",
    },
    "Tlaxco": {
        "Eucalipto", "Sábila", "Valeriana", "Marrubio", "Sauco",
        "Maguey", "Palo azul", "Lavanda", "Cuachalalate", "Camote silvestre",
    },
    "Nativitas": {
        "Romero", "Ruda", "Mejorana", "Poleo",
    },
    "Terrenate": {
        "Orégano", "Albahaca", "Caléndula", "Diente de León", "Muicle",
        "Chaya", "Hoja santa", "Árbol del trueno", "Hierba del sapo",
    },
    "Tlaxcala (ciudad)": {
        "Jengibre", "Tomillo",
    },
    "Todo Tlaxcala / No especificar": None,   # None = sin filtro
}

LISTA_MUNICIPIOS = list(MUNICIPIOS_PLANTAS.keys())
# ============================================
# PASO 6: PALABRAS DE RIESGO Y RESPUESTAS
# ============================================

palabras_peligro = [
    "pecho", "desmayo", "sangrado", "convulsion", "infarto",
    "paralisis", "vomito sangre", "dificultad respirar", "perdida de consciencia", "ataque",
    "emergencia", "cáncer", "ronchas", "accidente", "mareado",
    "palpitaciones", "dificultad", "confusión", "desorientación", "temblores",
    "entumecimiento", "cansancio extremo", "dificultad para hablar", "ritmo cardíaco", "asfixia",
    "ahogamiento", "dolor agudo", "hemorragia", "envenenamiento", "intoxicación",
    "perdida de vista", "vision borrosa de repente", "paro cardiaco", "derrame cerebral", "no puedo moverme",
    "fiebre muy alta", "convulsiones", "delirio", "alucinaciones", "dolor de cabeza insoportable",
    "perdida de sensibilidad", "labios azules", "falta de oxigeno", "ahogo", "no reacciona",
    "se desplomo", "boca torcida", "dolor opresivo", "picadura venenosa", "mordedura de serpiente",
    "ingirio veneno", "fractura expuesta", "quemadura grave", "shock", "taquicardia severa",
    "orina con sangre", "tos con sangre", "asfixiandome", "sin pulso", "coma",
]

alertas_seguridad = [
    (
        "⚠️ **ATENCIÓN MÉDICA RECOMENDADA**\n\n"
        "El síntoma que describes podría requerir atención profesional. "
        "Este chatbot **no sustituye** una valoración médica. "
        "Por favor acude a un centro de salud."
    ),
    (
        "🚨 **CONSULTA MÉDICA URGENTE**\n\n"
        "Lo que describes puede ser una señal de alerta importante. "
        "Las plantas medicinales **no reemplazan** la atención de un médico. "
        "Te recomendamos visitar un profesional de salud a la brevedad."
    ),
    (
        "🏥 **SE RECOMIENDA VER A UN MÉDICO**\n\n"
        "Este síntoma está fuera del alcance de un chatbot de plantas medicinales. "
        "Por tu seguridad, **acude a urgencias o a tu médico de confianza** lo antes posible."
    ),
]

intro_respuestas = [
    "Según lo que describes, esto es lo que encontré:",
    "Basándome en tu síntoma, aquí va mi recomendación:",
    "Con base en lo que mencionas, te sugiero lo siguiente:",
    "Entendido, esto podría ayudarte:",
    "De acuerdo a lo que me comentas, aquí tienes una opción natural:",
    "Para ese malestar que mencionas, te recomiendo revisar esto:",
    "Tomando en cuenta tus síntomas, esta planta te puede servir:",
    "Perfecto, aquí tienes un remedio herbolario para eso:",
    "Revisando lo que me cuentas, encontré esta alternativa natural:",
    "Entiendo perfectamente, esto es lo ideal para ese caso:",
    "Basado en la sabiduría tradicional, te sugiero probar esto:",
    "De acuerdo con las propiedades medicinales, lo mejor sería esto:",
    "Para aliviar eso que describes, puedes intentar lo siguiente:",
    "Aquí tienes una recomendación natural que te vendrá muy bien:",
    "Según las propiedades de las plantas, esto te puede ayudar:"
]

respuestas_fuera_tema = [
    "🌿 Solo puedo ayudarte con síntomas y plantas medicinales. ¿Tienes algún malestar del que quieras saber más?",
    "🌱 Ese tema está fuera de mi área. Estoy aquí para orientarte sobre plantas medicinales y síntomas. ¿En qué puedo ayudarte?",
    "🤔 Parece que tu pregunta no está relacionada con salud o plantas medicinales. Cuéntame cómo te sientes y te oriento.",
    "💬 Mi especialidad son las plantas medicinales. Si tienes algún síntoma o malestar, con gusto te ayudo.",
    "🌾 Vaya, de eso no conozco mucho. Pero si me platicas sobre algún síntoma o dolor, te puedo recomendar una planta medicinal.",
    "🍃 Disculpa, por ahora solo sé de herbolaria y remedios naturales. ¿Te gustaría saber sobre alguna planta para la salud?",
    "🧐 Ese tema no lo tengo en mi base de datos. Recuerda que te puedo ayudar a buscar alternativas naturales para tus malestares.",
    "🍂 Prefiero no adivinar sobre ese tema. Mejor cuéntame, ¿tienes algún dolor o problema de salud en el que te pueda orientar?",
    "✨ Mi conocimiento se limita al maravilloso mundo de las plantas medicinales. ¿Hay algún síntoma físico que quieras revisar?",
    "🏡 Fuera de la herbolaria y el bienestar corporal no sé mucho. Si gustas, pregúntame por algún remedio para la salud.",
    "🌻 Lo siento, solo estoy programado para conversar sobre salud natural y plantas. ¿En qué te puedo asesorar hoy respecto a eso?",
    "🎒 Dejemos ese tema para después. Si me dices qué te duele o qué malestar tienes, con gusto te busco una opción natural.",
    "🌵 Me pierdo un poco con esos temas. Si tu duda es sobre plantas medicinales o aliviar un síntoma, soy todo oídos.",
    "🔮 Solo puedo guiarte en el uso de la medicina tradicional y plantas medicinales. ¿Tienes alguna molestia física o duda de salud?",
    "🌸 Interesante, pero mi fuerte es la salud y la herbolaria. Cuéntame cómo te sientes hoy y vemos qué planta te puede servir."
]

aviso_legal = (
    "\n\n---\n"
    "⚠️ *Este chatbot es solo orientativo. "
    "Consulta a un médico o farmacéutico antes de usar cualquier planta medicinal, "
    "especialmente si tomas medicamentos o tienes condiciones de salud previas.*"
)

conocimiento_pdf = {"texto": ""}

# ============================================
# PASO 6.1: SALUDOS
# ============================================

palabras_saludo = [
    "hola", "buenos dias", "buenas tardes", "buenas noches", "buenas",
    "hey", "que tal", "como estas", "saludos", "buen dia", "ey", "hi",
    "hello", "buen dia", "que hay", "que onda", "como te va", "q pd",
    "q onda", "q tal", "q dices", "oye", "carnal",
]

respuestas_saludo = [
    "👋 ¡Hola! Soy tu asistente de plantas medicinales. Cuéntame cómo te sientes y te ayudo a encontrar el remedio natural adecuado. 🌿",
    "😊 ¡Buenas! Estoy aquí para orientarte sobre plantas medicinales. ¿Qué síntoma o malestar tienes hoy?",
    "🌱 ¡Hola! ¿En qué puedo ayudarte? Descríbeme tu síntoma y te recomendaré las mejores plantas medicinales.",
    "👋 ¡Bienvenido/a! Soy un asistente especializado en plantas medicinales. Cuéntame cómo te sientes. 🌿",
    "✨ ¡Hola! Qué gusto saludarte. Platícame si tienes algún dolor o malestar y buscamos una alternativa natural para ti. 🍃",
    "🌻 ¡Hola, bienvenido! Estoy listo para ayudarte a descubrir el poder de las plantas medicinales. ¿Cómo te sientes hoy?",
    "🌿 ¡Hola! Soy tu guía en medicina tradicional y herbolaria. Cuéntame qué síntoma tienes y te recomiendo una buena opción.",
    "👋 ¡Buenas! Si buscas aliviar algún malestar con plantas medicinales, estás en el lugar correcto. ¿En qué te puedo ayudar hoy? 🌱",
    "🏡 ¡Hola! ¿Qué tal? Cuéntame cómo te has sentido físicamente y con gusto te sugiero qué plantas te pueden servir.",
    "🌸 ¡Hola! Estoy aquí para ayudarte a cuidar tu salud de forma natural. Dime qué molestia tienes y lo revisamos juntos.",
    "💬 ¡Hola! Soy tu asistente herbolario. Descríbeme tu síntoma (como dolor de panza, estrés o tos) y te daré opciones de plantas.",
    "🍂 ¡Bienvenido/a! ¿Tienes algún malestar físico o problema de salud leve? Platícame y te digo qué remedio natural te ayuda.",
    "☀️ ¡Buen día! Estoy listo para orientarte sobre el uso seguro de plantas medicinales. ¿Qué síntoma te gustaría revisar hoy?",
    "🔮 ¡Hola! Qué tal. Cuéntame cómo te sientes hoy o qué planta te interesa conocer, y con gusto te doy la información. 🌿",
    "🌾 ¡Hola! Soy tu asistente de bienestar natural. Dime qué te duele o qué te aqueja hoy para recomendarte la planta ideal."
]

def extraer_saludo_y_resto(mensaje):
    """
    Detecta si el mensaje EMPIEZA con un saludo y separa lo que viene después.
    Devuelve (tiene_saludo: bool, resto: str)
    """
    msg = mensaje.lower().strip()

    # Ordenamos de más largo a más corto para evitar cortes parciales
    for s in sorted(palabras_saludo, key=len, reverse=True):
        patron = r'^' + re.escape(s) + r'\b[\s,.!¿?¡\-]*'
        m = re.match(patron, msg)
        if m:
            resto = msg[m.end():].strip()
            return True, resto

    return False, mensaje

    # ============================================
# PASO 6.2: GROSERÍAS E INSULTOS
# ============================================

palabras_groseras = [
    "puta", "puto", "chinga", "chingada", "cabron", "cabrón", "pendejo",
    "pendeja", "idiota", "imbecil", "imbécil", "estupido", "estúpido",
    "mierda", "joder", "coño", "verga", "pinche", "culero", "culera",
    "mamadas", "gilipollas", "bastardo", "bastarda", "maldito", "maldita",
    "hdp", "wtf", "hijo de", "vete al", "asco", "inutil", "inútil",
    "kbron", "pndjo", "pndja", "mrda", "chngar", "shinga", "weon",
    "weona", "culiao", "maricon", "maricón", "baboso", "babosa",
    "p bere", "mierd", "gacho", "chingaderas", "pendejadas", "cbron",
    "vete a la", "alv", "lpm", "hijo de perra", "ctm", "cabrona", "mamon", "mamona"
]

respuestas_groseria = [
    "😊 Entiendo que quizás estés frustrado/a, pero estoy aquí para ayudarte. ¿Tienes algún síntoma del que quieras saber más?",
    "🌿 Prefiero mantener una conversación respetuosa. Cuéntame cómo te sientes y con gusto te oriento.",
    "😌 No te preocupes, estoy aquí para ayudarte. ¿Qué malestar o síntoma tienes hoy?",
    "💬 Por favor, usemos un lenguaje adecuado. Estoy aquí para guiarte en temas de herbolaria, ¿en qué te puedo ayudar?",
    "🌱 Mantengamos el respeto para poder ayudarte mejor. Cuéntame, ¿tienes alguna molestia física o duda de salud?",
    "🤫 Te pido amablemente que evitemos esas palabras. Si gustas, dime qué te duele y buscamos un remedio natural.",
    "🍃 Mi objetivo es ayudarte a sentirte mejor. Si me platicas qué síntoma tienes, con gusto te recomiendo una planta.",
    "🌻 Entiendo tu molestia, pero como asistente solo puedo responder a dudas de salud con respeto. ¿Qué malestar revisamos?",
    "🌾 Vamos a enfocarnos en tu bienestar. Cuéntame cómo te has sentido físicamente y dejamos de lado esas expresiones.",
    "🧐 Para darte una buena orientación, es necesario mantener un trato cordial. ¿Te interesa saber sobre alguna planta medicinal?",
    "✨ Estoy programado para ayudar con remedios naturales en un ambiente respetuoso. Dime qué te aqueja hoy y lo checamos.",
    "🏡 Dejemos los insultos atrás. Si tienes algún dolor de panza, estrés, tos o algún síntoma, con gusto te asesoro.",
    "🌵 Por favor, evita el lenguaje ofensivo. Si tienes una duda legítima sobre plantas medicinales, soy todo oídos.",
    "🌸 Te invito a comunicarte de forma pacífica. Cuéntame cómo te sientes hoy para poder recomendarte la planta ideal.",
    "Calmate wey, muy vrgs te pones como si me aguantaras un vrgzo kbron.",
    "🍁 Mi base de datos solo responde con respeto. Si necesitas un remedio casero para un malestar, platícame qué tienes.",
]

def tiene_groserías(mensaje):
    msg = mensaje.lower()
    msg_limpio = re.sub(r'[^\w\s]', '', msg)
    return any(g in msg_limpio.split() or g in msg_limpio for g in palabras_groseras)

# ============================================
# PASO 6.3: TEXTO SIN SENTIDO
# ============================================

respuestas_sin_sentido = [
    "🤔 No logré entender tu mensaje. ¿Podrías describirme un síntoma? Por ejemplo: *'tengo tos'* o *'me duele el estómago'*.",
    "🌿 Hmm, eso no parece un síntoma. Intenta contarme cómo te sientes con tus propias palabras.",
    "💬 No reconozco ese mensaje. ¿Tienes algún malestar del que quieras saber más?",
    "🌱 Disculpa, no comprendí bien. Recuerda que puedes pedirme remedios diciendo cosas como: *'estoy estresado'* o *'me duele la espalda'*.",
    "🧐 Me costó un poco entender tu texto. Si me dices qué parte del cuerpo te molesta, te busco una alternativa natural.",
    "🍃 Vaya, no capté la idea. Intenta explicarme tu malestar de otra forma para que pueda recomendarte la planta ideal.",
    "✨ Creo que nos confundimos un poco. Escribe tu síntoma de forma sencilla para poder ayudarte a buscar una infusión.",
    "🌻 Lo siento, no logré procesar esa frase. Prueba escribiendo algo breve sobre cómo te sientes físicamente hoy.",
    "🌾 Ups, esa expresión no está en mi manual. Si tienes alguna molestia de salud o quieres saber de herbolaria, platícame.",
    "🏡 No estoy seguro de qué significa eso. Cuéntame si tienes algún problema digestivo, respiratorio o muscular y lo checamos.",
    "🍂 Lo siento, mi sistema no entendió. Si me dices qué te aqueja con palabras simples, con gusto te oriento.",
    "🌵 ¿Me lo podrías repetir de otra manera? Recuerda que soy un asistente de plantas medicinales y busco ayudarte con tus síntomas.",
    "🌸 No alcancé a entender tu consulta. Intenta decirme concretamente qué te duele o qué malestar tienes.",
    "🔮 Creo que esa frase se sale de lo que puedo procesar. Si me describes tu dolor o malestar, con gusto te ayudo.",
    "🍁 Mis hojitas se confundieron con ese mensaje. Dime qué síntoma tienes o qué planta buscas y lo resolvemos juntos."
]

def es_texto_sin_sentido(mensaje):
    msg = mensaje.strip()

    # Muy corto (1-2 caracteres)
    if len(msg) <= 2:
        return True

    palabras = msg.split()

    # Detectar palabras con demasiadas consonantes seguidas (sin vocales = sin sentido)
    vocales = set("aeiouáéíóúü")
    def palabra_sin_sentido(palabra):
        palabra = palabra.lower()
        if len(palabra) < 4:
            return False
        # Contar consonantes consecutivas
        consonantes_seguidas = 0
        max_consonantes = 0
        for letra in palabra:
            if letra.isalpha() and letra not in vocales:
                consonantes_seguidas += 1
                max_consonantes = max(max_consonantes, consonantes_seguidas)
            else:
                consonantes_seguidas = 0
        # Si más del 80% son consonantes o hay 5+ consonantes seguidas
        total_letras = sum(1 for c in palabra if c.isalpha())
        total_consonantes = sum(1 for c in palabra if c.isalpha() and c not in vocales)
        if total_letras == 0:
            return False
        ratio_consonantes = total_consonantes / total_letras
        return max_consonantes >= 5 or ratio_consonantes > 0.80

    palabras_raras = sum(1 for p in palabras if palabra_sin_sentido(p))

    # Si la mayoría de las palabras son sin sentido
    return palabras_raras >= max(1, len(palabras) * 0.6)

# ============================================
# PASO 7: FUNCIÓN PARA CARGAR PDF
# ============================================

def cargar_pdf(archivo):
    if archivo is None:
        return "No se ha subido ningún archivo."
    try:
        reader = PyPDF2.PdfReader(archivo.name)
        texto = ""
        for pagina in reader.pages:
            texto += pagina.extract_text() or ""
        conocimiento_pdf["texto"] = texto.strip()
        palabras = len(texto.split())
        return f"✅ PDF cargado correctamente — {len(reader.pages)} página(s), ~{palabras} palabras extraídas."
    except Exception as e:
        return f"❌ Error al leer el PDF: {str(e)}"

# ============================================
# PASO 7.1: DETECTAR TEMA FUERA DE CONTEXTO
# ============================================

import re

STOPWORDS_ES = {
    "a","ante","bajo","con","contra","de","desde","durante","en","entre",
    "hacia","hasta","para","por","segun","según","sin","sobre","tras",
    "el","la","los","las","un","una","unos","unas","lo",
    "y","e","o","u","ni","que","como","cuando","donde","dónde","cual","cuál",
    "yo","tu","tú","él","ella","nosotros","ustedes","ellos","ellas",
    "me","te","se","nos","os","le","les","mi","mis","tus","su","sus",
    "este","esta","estos","estas","ese","esa","esos","esas",
    "soy","eres","es","somos","son","voy","vas","va","vamos","van",
    "ir","fui","fue","ser","estar","estoy","estamos","estan","están",
    "no","si","sí","muy","mas","más","pero","porque","tambien","también",
    "ya","aun","aún","solo","sólo","al","del"
}

UMBRAL_CONFIANZA = 0.55
UMBRAL_COBERTURA = 0.34

def es_fuera_de_tema(mensaje):
    msg = mensaje.lower()
    palabras = re.findall(r'\b[a-záéíóúñ]+\b', msg)

    # Quitamos los conectores genéricos: solo nos importan las palabras con contenido real
    palabras_clave = [p for p in palabras if p not in STOPWORDS_ES]

    if not palabras_clave:
        return True  # el mensaje no tiene ninguna palabra con significado propio

    vocabulario = vectorizador.vocabulary_
    conocidas = sum(1 for p in palabras_clave if p in vocabulario)
    cobertura = conocidas / len(palabras_clave)

    vect = vectorizador.transform([mensaje])
    if vect.nnz == 0:
        return True

    proba = modelo.predict_proba(vect)[0]
    confianza = max(proba)

    es_sintoma_valido = (cobertura >= UMBRAL_COBERTURA) and (confianza >= UMBRAL_CONFIANZA)
    return not es_sintoma_valido

# ============================================
# PASO 7.2: CORRECCIÓN AUTOMÁTICA DE ERRORES DE TECLEO
# ============================================

def _distancia_levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    fila_anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        fila_actual = [i + 1]
        for j, cb in enumerate(b):
            inserciones = fila_anterior[j + 1] + 1
            eliminaciones = fila_actual[j] + 1
            sustituciones = fila_anterior[j] + (ca != cb)
            fila_actual.append(min(inserciones, eliminaciones, sustituciones))
        fila_anterior = fila_actual
    return fila_anterior[-1]

MAX_EDICIONES_TYPO = 1  # solo corrige errores de 1 letra (insertar/borrar/cambiar)

def _mejor_candidato(palabra, candidatos):
    mejor, mejor_d = None, None
    for c in candidatos:
        if abs(len(c) - len(palabra)) > MAX_EDICIONES_TYPO:
            continue
        d = _distancia_levenshtein(palabra, c)
        if d <= MAX_EDICIONES_TYPO and (mejor_d is None or d < mejor_d):
            mejor_d, mejor = d, c
    return mejor

# Vocabulario de saludos sueltos (para corregir "bola"->"hola", "ola"->"hola", etc.)
_PALABRAS_SALUDO_SUELTAS = set()
for _frase in palabras_saludo:
    _PALABRAS_SALUDO_SUELTAS.update(_frase.split())
VOCABULARIO_SALUDOS = _PALABRAS_SALUDO_SUELTAS - STOPWORDS_ES

# Vocabulario de síntomas (tomado directamente de tu TfidfVectorizer ya entrenado)
VOCABULARIO_SINTOMAS = set(vectorizador.vocabulary_.keys()) - STOPWORDS_ES

def corregir_typos(mensaje):
    """
    Revisa palabra por palabra el mensaje del usuario y corrige errores de tecleo
    leves (1 letra de diferencia) comparando contra el vocabulario de saludos
    y de síntomas que ya conoce el chatbot.
    """
    palabras = mensaje.split()
    corregidas = []
    for palabra in palabras:
        limpia = re.sub(r'[^\wáéíóúñ]', '', palabra.lower())

        # Si la palabra ya es válida tal cual, o es un conector, no la tocamos
        if not limpia or limpia in VOCABULARIO_SALUDOS or limpia in VOCABULARIO_SINTOMAS or limpia in STOPWORDS_ES:
            corregidas.append(palabra)
            continue

        # 1) Probar primero contra saludos (vocabulario chico, alta confianza)
        match = _mejor_candidato(limpia, VOCABULARIO_SALUDOS)

        # 2) Si no hubo match de saludo, probar contra síntomas
        if not match and len(limpia) >= 4:
            match = _mejor_candidato(limpia, VOCABULARIO_SINTOMAS)

        corregidas.append(match if match else palabra)

    return " ".join(corregidas)

# ============================================
# PASO 8: DETECCIÓN Y SEPARACIÓN DE SÍNTOMAS
# ============================================

SEPARADORES = r'\b(y|e|además|tambien|también|también tengo|con|más|mas|sumado a|junto con|aparte de)\b|[,;]'

def separar_sintomas(mensaje):
    fragmentos = re.split(SEPARADORES, mensaje, flags=re.IGNORECASE)
    limpios = [f.strip() for f in fragmentos if f and len(f.strip()) >= 3
               and not re.fullmatch(r'y|e|además|tambien|también|con|más|mas|sumado a|junto con|aparte de',
                                    f.strip(), flags=re.IGNORECASE)]
    return limpios if len(limpios) >= 2 else [mensaje]

def clasificar_fragmento(fragmento):
    vect = vectorizador.transform([fragmento])
    return modelo.predict(vect)[0]
# ============================================
# PASO 9: FUNCIÓN PRINCIPAL DEL CHATBOT
# ============================================

def filtrar_plantas_por_municipio(plantas: list, municipio: str) -> list:
    """
    Filtra la lista de plantas recomendadas según el municipio del usuario.
    Si no hay plantas del municipio en esa categoría, devuelve todas (fallback).
    """
    disponibles = MUNICIPIOS_PLANTAS.get(municipio)
    if disponibles is None:                          # "Todo Tlaxcala"
        return plantas
    filtradas = [p for p in plantas if p in disponibles]
    return filtradas if filtradas else plantas        # fallback: todas

def recomendar(mensaje, municipio, historial):

    # -1. Corregir errores de tecleo antes de cualquier otra detección
    mensaje = corregir_typos(mensaje)

    # 0. Detectar saludo y separarlo del resto del mensaje
    tiene_saludo, resto = extraer_saludo_y_resto(mensaje)
    prefijo_saludo = ""

    if tiene_saludo:
        palabras_resto = re.findall(r'\b[a-záéíóúñ]+\b', resto)
        palabras_clave_resto = [p for p in palabras_resto if p not in STOPWORDS_ES]

        if not palabras_clave_resto:
            # Es SOLO un saludo, sin nada más
            respuesta = random.choice(respuestas_saludo)
            historial.append({"role": "user", "content": mensaje})
            historial.append({"role": "assistant", "content": respuesta})
            return "", historial
        else:
            # Hay saludo + síntoma: seguimos con el resto, pero guardamos un saludo breve
            mensaje = resto
            prefijo_saludo = "👋 ¡Hola! "

    # 0.1 Detectar groserías
    if tiene_groserías(mensaje):
        respuesta = random.choice(respuestas_groseria)
        historial.append({"role": "user", "content": mensaje})
        historial.append({"role": "assistant", "content": respuesta})
        return "", historial

    # 0.2 Detectar texto sin sentido
    if es_texto_sin_sentido(mensaje):
        respuesta = random.choice(respuestas_sin_sentido)
        historial.append({"role": "user", "content": mensaje})
        historial.append({"role": "assistant", "content": respuesta})
        return "", historial

    # 1. Detectar síntomas graves
    for palabra in palabras_peligro:
        if palabra in mensaje.lower():
            respuesta = random.choice(alertas_seguridad)
            historial.append({"role": "user", "content": mensaje})
            historial.append({"role": "assistant", "content": respuesta})
            return "", historial

    # 2. Detectar mensaje fuera de tema
    if es_fuera_de_tema(mensaje):
        respuesta = random.choice(respuestas_fuera_tema)
        historial.append({"role": "user", "content": mensaje})
        historial.append({"role": "assistant", "content": respuesta})
        return "", historial

    # 3. Buscar en PDF si hay contenido cargado
    contexto_pdf = ""
    if conocimiento_pdf["texto"]:
        texto = conocimiento_pdf["texto"].lower()
        palabras_consulta = mensaje.lower().split()
        coincidencias = [p for p in palabras_consulta if p in texto and len(p) > 3]
        if coincidencias:
            idx = texto.find(coincidencias[0])
            inicio = max(0, idx - 100)
            fin = min(len(texto), idx + 300)
            fragmento_pdf = conocimiento_pdf["texto"][inicio:fin].strip()
            contexto_pdf = f"\n\n📄 **Del PDF cargado:**\n_{fragmento_pdf}..._"

    # 4. Separar síntomas y clasificar cada uno
    fragmentos = separar_sintomas(mensaje)

    # Filtramos los fragmentos que no son síntomas reales (preguntas, relleno, etc.)
    fragmentos = [f for f in fragmentos if not es_fuera_de_tema(f)] or [mensaje]

    multisintoma = len(fragmentos) >= 2

    # detalle_categorias: lista de (fragmento, categoria)
    detalle_categorias = []
    categorias_vistas  = []

    for frag in fragmentos:
        cat = clasificar_fragmento(frag)
        detalle_categorias.append((frag.strip(), cat))
        if cat not in categorias_vistas:
            categorias_vistas.append(cat)

    # ─── Caso 1: UN solo síntoma ──────────────────────────────────────────────
    if not multisintoma:
        cat_unica = categorias_vistas[0]
        plantas = filtrar_plantas_por_municipio(mapa_plantas[cat_unica], municipio)           # todas las de esa categoría

        encabezado = (
            f"{random.choice(intro_respuestas)}\n\n"
            f"🔍 Categoría detectada: **{cat_unica.replace('_', ' ').title()}**\n\n"
            f"🌿 **Plantas recomendadas:**\n"
        )
        cuerpo = "".join(
            f"• **{p}**\n  {preparacion_plantas.get(p, '')}\n\n"
            for p in plantas
        )

        respuesta = prefijo_saludo + encabezado + cuerpo + contexto_pdf + aviso_legal

    # ─── Caso 2: MÚLTIPLES síntomas ──────────────────────────────────────────
    else:
        # Plantas únicas de cada categoría detectada
        plantas_por_cat = {}
        for cat in categorias_vistas:
            plantas_por_cat[cat] = filtrar_plantas_por_municipio(mapa_plantas[cat], municipio)   # copia completa

        # Plantas que aparecen en MÁS de una categoría → "sirven para todo"
        from collections import Counter
        conteo = Counter(p for lista in plantas_por_cat.values() for p in lista)
        plantas_compartidas = [p for p, n in conteo.items() if n > 1]

        # Encabezado con resumen de síntomas detectados
        lineas_sintomas = "\n".join(
            f"  {i+1}. '{frag}' → **{cat.replace('_',' ').title()}**"
            for i, (frag, cat) in enumerate(detalle_categorias)
        )

        # Sección de plantas compartidas (si las hay)
        seccion_compartidas = ""
        if plantas_compartidas:
            seccion_compartidas = (
                "✨ **Plantas que ayudan con varios de tus síntomas** *(recomendadas primero)*:\n\n" +
                "".join(
                    f"• **{p}** ✦\n  {preparacion_plantas.get(p, '')}\n\n"
                    for p in plantas_compartidas
                )
            )

        # Plantas específicas por categoría (excluyendo las compartidas)
        secciones_cat = ""
        for cat, plantas in plantas_por_cat.items():
            exclusivas = [p for p in plantas if p not in plantas_compartidas]
            if exclusivas:
                secciones_cat += (
                    f"🌿 **Para {cat.replace('_',' ').title()}:**\n\n" +
                    "".join(
                        f"• **{p}**\n  {preparacion_plantas.get(p, '')}\n\n"
                        for p in exclusivas
                    )
                )


        respuesta = (
        prefijo_saludo +
      (
        f"{random.choice(intro_respuestas)}\n\n"
        f"🔍 Detecté **{len(detalle_categorias)} síntoma(s)**:\n{lineas_sintomas}\n\n"
      ) +
        seccion_compartidas +
        secciones_cat +
        contexto_pdf +
        aviso_legal
      )

    historial.append({"role": "user", "content": mensaje})
    historial.append({"role": "assistant", "content": respuesta})
    return "", historial

# ============================================
# PASO 9: INTERFAZ GRADIO (CSS CORREGIDO)
# ============================================

CSS = """
.gradio-container {
    max-width: 760px !important;
    margin: 0 auto !important;
    background: #1a1a1a !important;
}

#titulo {
    text-align: center;
    padding: 24px 20px 12px;
}

/* Input */
#msg-input textarea {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 14px !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
    resize: none !important;
}
#msg-input textarea:focus {
    border-color: #0f9970 !important;
    box-shadow: 0 0 0 2px rgba(15,153,112,0.2) !important;
    outline: none !important;
}

/* Botón enviar */
#btn-enviar {
    background: #0f9970 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
#btn-enviar:hover { background: #0c7d5c !important; }

/* Botón limpiar */
#btn-limpiar {
    background: #2a2a2a !important;
    color: #aaa !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 10px !important;
}

/* Sección PDF */
#seccion-pdf {
    background: #242424;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 10px 0;
}
#estado-pdf { font-size: 0.8rem; color: #0f9970; margin-top: 6px; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 3px; }
"""

TEMA = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.neutral,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#1a1a1a",
    body_background_fill_dark="#1a1a1a",
    body_text_color="#ececec",
    body_text_color_dark="#ececec",
    block_background_fill="#242424",
    block_background_fill_dark="#242424",
    block_border_color="#333333",
    block_border_color_dark="#333333",
    input_background_fill="#2a2a2a",
    input_background_fill_dark="#2a2a2a",
    input_border_color="#3a3a3a",
    input_border_color_dark="#3a3a3a",
    button_primary_background_fill="#0f9970",
    button_primary_background_fill_dark="#0f9970",
    button_primary_text_color="#ffffff",
    chatbot_text_size="md",
)

with gr.Blocks(title="🌿 Chatbot Plantas Medicinales") as app:

    gr.HTML("""
    
      
        🌿
      
      
        Chatbot de Plantas Medicinales | TÉ VIDA
      
      
        Describe tu síntoma y te recomendaré el remedio natural adecuado
      
    
    """)

    municipio_selector = gr.Dropdown(
        choices=LISTA_MUNICIPIOS,
        value="Todo Tlaxcala / No especificar",
        label="📍 Tu municipio (Tlaxcala)",
        info="Filtra las plantas del catálogo local de tu zona",
        interactive=True,
    )

    with gr.Group(elem_id="seccion-pdf"):
        gr.Markdown("**📄 Base de conocimiento** · Sube un PDF adicional sobre plantas medicinales (opcional)")
        pdf_input  = gr.File(label="Subir PDF", file_types=[".pdf"], type="filepath")
        estado_pdf = gr.Markdown(elem_id="estado-pdf")

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        label="",
        show_label=False,
        height=450,
        render_markdown=True,
    )

    with gr.Row(elem_id="input-row"):
        msg = gr.Textbox(
            placeholder="Describe tu síntoma…  (ej: tengo ansiedad y no puedo dormir)",
            show_label=False,
            lines=1,
            elem_id="msg-input",
            scale=5,
        )
        btn_enviar  = gr.Button("Enviar ↑",  elem_id="btn-enviar",  scale=1, variant="primary")
        btn_limpiar = gr.Button("Limpiar",   elem_id="btn-limpiar", scale=1)

    pdf_input.change(fn=cargar_pdf, inputs=pdf_input, outputs=estado_pdf)
    msg.submit(fn=recomendar, inputs=[msg, municipio_selector, chatbot], outputs=[msg, chatbot])
    btn_enviar.click(fn=recomendar, inputs=[msg, chatbot], outputs=[msg, chatbot])
    btn_limpiar.click(fn=lambda: ([], ""), outputs=[chatbot, msg])

# ============================================
# PASO 10: EJECUTAR APP
# ============================================

app.launch(share=True, css=CSS, theme=TEMA)
     
Versión de Gradio: 6.19.0
✅ Datos alineados: 480 muestras
/usr/local/lib/python3.12/dist-packages/gradio/utils.py:1201: UserWarning: Expected 3 arguments for function <function recomendar at 0x7b0c31bdb100>, received 2.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/gradio/utils.py:1205: UserWarning: Expected at least 3 arguments for function <function recomendar at 0x7b0c31bdb100>, received 2.
  warnings.warn(
Colab notebook detected. To show errors in colab notebook, set debug=True in launch()
* Running on public URL: https://f4fd099d9be23b1489.gradio.live

This share link is temporary and will last for up to 1 week (best effort). For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
