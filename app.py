"""
Valentina (Ole Seguros) — chatbot para WHATSAPP con la API OFICIAL de Meta
--------------------------------------------------------------------------
Asesora digital de seguros de vida Ole. Objetivo: cerrar (capturar datos del
prospecto y agendar llamada con un asesor humano).

CEREBRO: se elige con la constante PROVEEDOR mas abajo. Los dos son de Groq,
que NO entrena con los datos de sus clientes ni en plan gratis ni en pago.
  "gratis" -> gpt-oss-120b       (por defecto: $0, ~10 conversaciones/dia)
  "groq"   -> llama-3.3-70b      (el original, ~$10/mes, sin tope diario util)
Ambos usan el mismo SDK de openai, solo cambia el nombre del modelo.

Gemini quedo descartado por la sensibilidad de los datos (ver nota abajo).

PROMPT: Valentina V5 — el objetivo es AGENDAR LA CITA, no explicar el producto.
Valentina ya no se presenta como "de Ole": dice que trabaja con Jorge Arroyo.
Tres fases: (1) capturar interes y generar confianza, (2) detectar si busca
gastos medicos, vida o ambos, (3) cerrar dia y hora de la llamada.
Por eso desaparecio todo el catalogo de producto (precios, coberturas, tablas
medicas): si no explica producto, no lo necesita, y no puede inventar lo que no
tiene. ~1,555 tokens (la V4 tenia 2,503 y la V2 9,300).

Con Meta hay DOS cosas que tu servidor debe hacer:

  1) VERIFICACION (una sola vez):  Meta manda un GET /webhook con un token y un
     "challenge". Tu devuelves el challenge si el token coincide.

  2) MENSAJES (siempre): Meta manda un POST /webhook cada vez que alguien te
     escribe. Tu lees el texto, piensas la respuesta (Groq) y luego HACES OTRA
     llamada a Meta (graph.facebook.com) para ENVIAR la respuesta.

Flujo:
  Usuario WhatsApp -> Meta -> POST /webhook (aqui) -> Groq
                                       |
                                       +-> POST a Meta (Graph API) -> Usuario
"""

import os
import re
import json
import unicodedata
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from flask import Flask, request, Response
from openai import OpenAI

# ---------------------------------------------------------------------------
# 0) CARGAR EL ARCHIVO .env (si existe)
# ---------------------------------------------------------------------------
# Lee el archivo .env que esta junto a este script y pone cada linea
# CLAVE=valor como variable de entorno. Asi no hace falta escribir las claves
# a mano en PowerShell cada vez. (Loader minimo, sin librerias extra.)
def cargar_env():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(ruta):
        return
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip())

cargar_env()

# ---------------------------------------------------------------------------
# 1) CONFIG — se lee todo desde variables de entorno (nada de claves en el codigo)
# ---------------------------------------------------------------------------
WHATSAPP_TOKEN  = os.environ.get("WHATSAPP_TOKEN", "")     # token de acceso de Meta
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID", "")    # id del numero de WhatsApp
VERIFY_TOKEN    = os.environ.get("VERIFY_TOKEN", "aromas123")  # lo inventas tu

GRAPH_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

# --- QUE CEREBRO USA VALENTINA -------------------------------------------
# Cambia esta sola palabra para saltar entre proveedores.
# Tambien se puede fijar desde el .env con  PROVEEDOR=loquesea
# Todos hablan el mismo dialecto (API compatible con OpenAI), asi que el
# resto del archivo no cambia en absoluto.
#
#   gratis  -> $0/mes    · ~10 conversaciones/dia, luego error 429
#   groq    -> $10.23/mes · sin tope practico
PROVEEDOR = os.environ.get("PROVEEDOR", "gratis").strip().lower()

PROVEEDORES = {
    # LA OPCION GRATIS Y SEGURA.
    # Groq no entrena con tus datos ni en gratis ni en pago, no los retiene por
    # defecto, y puedes activar Zero Data Retention en Data Controls.
    # Limites del free tier: 200,000 tokens/dia, 1,000 peticiones/dia, 8,000
    # tokens/minuto. Son el doble de tokens que llama-3.3-70b.
    # Ademas este modelo SI tiene prompt caching, y en Groq los tokens cacheados
    # NO cuentan contra el limite diario: el prompt de sistema (2,500 tokens)
    # deja de gastar cuota a partir del segundo mensaje. Eso estira los 200K
    # hasta ~10 conversaciones al dia. El caching es automatico, sin tocar codigo.
    # Usa la MISMA GROQ_API_KEY que ya tienes.
    "gratis": {
        "key":   os.environ.get("GROQ_API_KEY", ""),
        "base":  "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
    },
    # El original. Seguro, pero su free tier son 100,000 tokens/dia
    # (~1 conversacion) y sin caching, asi que en la practica hay que pagar
    # (~$10/mes a 10 conversaciones diarias).
    "groq": {
        "key":   os.environ.get("GROQ_API_KEY", ""),
        "base":  "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
}

# GEMINI QUEDA DESCARTADO A PROPOSITO — no lo vuelvas a agregar.
# En su free tier Google ENTRENA con las conversaciones y revisores humanos
# pueden leerlas. Sus propios terminos dicen: "Do not submit sensitive,
# confidential, or personal information to the Unpaid Services."
# Valentina pide nombre completo, edad, si fuma (dato de salud), ciudad y monto,
# y el numero de WhatsApp viene como identificador. En Mexico los datos de salud
# son datos personales sensibles bajo la LFPDPPP.
# La excepcion regional de Google (politica de pago aplicada al free tier) cubre
# Europa, Suiza y Reino Unido. America Latina NO esta incluida.
# Si algun dia se evalua de nuevo, tendria que ser con facturacion activada
# (ahi Google deja de entrenar) y despues de una revision legal, no antes.

if PROVEEDOR not in PROVEEDORES:
    raise SystemExit(
        f"PROVEEDOR '{PROVEEDOR}' no existe. Usa: {', '.join(PROVEEDORES)}")

_cfg = PROVEEDORES[PROVEEDOR]
if not _cfg["key"]:
    raise SystemExit(
        f"Falta GROQ_API_KEY en el archivo .env (proveedor activo: {PROVEEDOR}).")

client = OpenAI(api_key=_cfg["key"], base_url=_cfg["base"])
MODEL = _cfg["model"]
print(f"Cerebro: {PROVEEDOR} / {MODEL}")

# Cuantos mensajes de ida y vuelta se recuerdan por persona. Cada turno viejo
# se reenvia entero en cada llamada, asi que sin tope la cuenta crece sola.
MAX_HISTORIAL = 20

# gpt-oss-120b es un modelo de RAZONAMIENTO: antes de escribir la respuesta
# visible "piensa" internamente, y ese pensamiento gasta el MISMO presupuesto
# de max_tokens. Con 400 se quedaba sin espacio justo en el paso de recomendar
# (el mas pesado del flujo): devolvia content vacio y WhatsApp rechazaba el
# envio con "The parameter text.body is required".
# Dos defensas: menos razonamiento y mas techo.
MAX_TOKENS = 1000
ESFUERZO = "low"   # low | medium | high. Valentina no necesita razonar hondo.

# Asesor que Valentina presenta en la conversacion.
# NOTA (2026-08-01): el asesor real es Jorge Arroyo, pero su Cal.com aun no
# existe (el link viejo "jorge-arroyo/llamada" daba 404). Mientras tanto la
# agenda que se consulta es la de Enrique Ampudia (ver CALCOM_API_KEY abajo)
# SOLO PARA PROBAR que el flujo funciona end-to-end; el nombre que Valentina
# presenta sigue siendo Jorge. Ya no existe ningun link ni variable
# LINK_AGENDA: todo se agenda por API.
ASESOR_NOMBRE = os.environ.get("ASESOR_NOMBRE", "Jorge Arroyo")
ASESOR_CORTO = ASESOR_NOMBRE.split()[0]

# ---------------------------------------------------------------------------
# Cal.com — agenda real. Valentina ya no manda un link para que la persona
# elija hora por su cuenta: el propio backend consulta los huecos libres del
# asesor y le manda a la persona una LISTA de horarios reales de WhatsApp
# (botones), y al tocar uno se agenda directo por API. Cal.com le manda la
# confirmacion (y la notificacion) al dueno de la cuenta automaticamente.
#
# NOTA (2026-08-01): la cuenta de Cal.com sigue siendo la de Enrique Ampudia
# (es la unica conectada a un Google Calendar real ahora mismo), pero
# Valentina se sigue presentando como si trabajara con Jorge. Es solo para
# PROBAR que el flujo de agendar-por-API funciona de punta a punta. Cuando
# Jorge tenga su propia cuenta de Cal.com, cambia CALCOM_API_KEY y
# CALCOM_EVENT_TYPE_ID en el .env (no hace falta tocar el prompt).
CALCOM_API_KEY = os.environ.get("CALCOM_API_KEY", "")
if not CALCOM_API_KEY:
    raise RuntimeError(
        "Falta CALCOM_API_KEY en el archivo .env. Sin ella Valentina no puede "
        "leer la agenda ni agendar citas. Consiguela en "
        "cal.com/settings/developer/api-keys")

# Id del tipo de evento (el "30min" / Llamada de asesoria). Se obtuvo con
# GET https://api.cal.com/v2/event-types usando la API key de arriba.
CALCOM_EVENT_TYPE_ID = os.environ.get("CALCOM_EVENT_TYPE_ID", "6525721")
CALCOM_TIMEZONE = os.environ.get("CALCOM_TIMEZONE", "America/Merida")
CALCOM_BASE = "https://api.cal.com/v2"

# Correo base con el que se agendan las citas. A la persona NUNCA se le pide
# el suyo (pedirlo costaba conversiones), pero Cal.com exige uno, asi que se
# arma uno por prospecto con "sub-direccion": buzon+wa529991105167@gmail.com.
# Tiene que ser una cuenta REAL que pueda recibir correo, del asesor o de la
# casa: Cal.com valida el dominio y rechaza los que no reciben mail
# (`email_domain_cannot_receive_mail`). Gmail, Outlook y Google Workspace
# soportan el "+" y entregan todo al mismo buzon, asi que el asesor recibe la
# copia de confirmacion y nadie ajeno ve los datos del prospecto.
CORREO_BASE_CITAS = os.environ.get("CORREO_BASE_CITAS",
                                   "enrique.ampudia97@gmail.com")

# ---------------------------------------------------------------------------
# SEGUIMIENTOS — reenganchar a quien no agendo
# ---------------------------------------------------------------------------
# Si alguien conversa y se va sin agendar, Valentina le escribe UNA vez a las
# 23 h y se calla. Uno, no tres: ver MAX_SEGUIMIENTOS mas abajo.
#
# UNO EN TODA LA VIDA DE ESA PERSONA, no uno por cada vez que se enfria. Es la
# diferencia importante: antes cualquier respuesta reiniciaba el contador, asi
# que quien contestaba y se volvia a quedar callado recibia otro, y otro. Ahora
# el que ya recibio el suyo entra a la lista de no-molestar y ahi se queda.
#
# Valentina SIEMPRE le contesta a quien le escriba, este en la lista o no. Lo
# unico que se apaga es que arranque ella la conversacion.
#
# Se apaga de tres maneras, todas definitivas: pidiendo que ya no le escriban
# (el boton, o escribiendolo — ver pide_que_no_le_escriban), agendando, o
# recibiendo ese unico seguimiento.
#
# LA REGLA DE META QUE MANDA SOBRE TODO ESTO: solo se puede mandar un mensaje
# libre dentro de las 24 h siguientes al ULTIMO MENSAJE QUE ESCRIBIO LA PERSONA
# (la "customer service window"). Mandarle algo NO reabre la ventana: solo la
# reabre ella escribiendo. Por eso el seguimiento sale a las 23 h: cae DENTRO de
# la ventana, es un mensaje normal con botones de verdad, sin aprobacion de Meta
# y sin costo.
#
# Cualquier seguimiento posterior caeria FUERA, y ahi Meta solo acepta plantillas
# aprobadas de categoria Marketing: se cobran por mensaje, necesitan numero real
# con la cuenta verificada, y el numero de pruebas no puede mandarlas. Esa
# maquinaria sigue escrita aqui abajo pero hoy no corre, porque MAX_SEGUIMIENTOS
# vale 1. Para reactivarla: crear la plantilla en WhatsApp Manager (categoria
# Marketing, con un boton de respuesta rapida cuyo texto sea el de
# PLANTILLA_BOTON_STOP), poner su nombre en PLANTILLA_SEGUIMIENTO y subir
# MAX_SEGUIMIENTOS a 3, las tres en Render.
#
# Los tiempos se miden en MINUTOS para poder probarlos sin esperar un dia. En
# produccion se dejan en blanco y valen 23 h y 24 h. Para una prueba, en Render:
#   MINUTOS_PRIMER_SEGUIMIENTO = 1
#   MINUTOS_ENTRE_SEGUIMIENTOS = 1
# y al terminar SE BORRAN esas dos variables. Si se quedan puestas, cualquiera
# que escriba y no agende recibe su seguimiento al minuto.
#
# OJO con la resolucion: el reloj son los pings de UptimeRobot cada 5 min, asi
# que poner 1 minuto no significa "al minuto exacto", significa "en el siguiente
# ping". Para verlo al instante hay que pegarle a mano a /cron/seguimientos.
def _entero_env(nombre, por_defecto):
    """Lee un numero de una variable de entorno. Si viene vacia, con basura o
    en cero, se queda con el valor de produccion: nunca un 0 que dispare en
    bucle ni un crash al arrancar por un dedazo en el panel de Render."""
    try:
        v = int(os.environ.get(nombre, "").strip() or por_defecto)
        return v if v > 0 else por_defecto
    except ValueError:
        print(f"{nombre} no es un numero, se usa {por_defecto}.")
        return por_defecto

MINUTOS_PRIMER_SEGUIMIENTO = _entero_env("MINUTOS_PRIMER_SEGUIMIENTO", 23 * 60)
MINUTOS_ENTRE_SEGUIMIENTOS = _entero_env("MINUTOS_ENTRE_SEGUIMIENTOS", 24 * 60)
# UN solo seguimiento por persona. Se decidio asi a proposito: el 2do y el 3ro
# caen fuera de la ventana de 24 h de Meta, o sea que necesitan plantilla
# aprobada, numero real y se cobran por mensaje. Antes se intentaban igual y
# morian en el log; ahora ni se intentan. Si algun dia hay plantilla y se
# quieren de vuelta, se sube este numero desde Render (MAX_SEGUIMIENTOS = 3) sin
# tocar el codigo: la maquinaria de plantillas sigue completa aqui abajo.
MAX_SEGUIMIENTOS = _entero_env("MAX_SEGUIMIENTOS", 1)

if MINUTOS_PRIMER_SEGUIMIENTO < 23 * 60 or MINUTOS_ENTRE_SEGUIMIENTOS < 24 * 60:
    print(f"*** MODO PRUEBA DE SEGUIMIENTOS: 1ro a los "
          f"{MINUTOS_PRIMER_SEGUIMIENTO} min, luego cada "
          f"{MINUTOS_ENTRE_SEGUIMIENTOS} min. Borrar esas variables de entorno "
          f"antes de usar el bot con gente real. ***")

PLANTILLA_SEGUIMIENTO = os.environ.get("PLANTILLA_SEGUIMIENTO", "").strip()
# Opcional: una segunda plantilla para que el tercer seguimiento no llegue con
# el texto identico al del segundo. Si se deja vacia se reusa la primera.
PLANTILLA_SEGUIMIENTO_3 = os.environ.get("PLANTILLA_SEGUIMIENTO_3", "").strip()
PLANTILLA_IDIOMA = os.environ.get("PLANTILLA_IDIOMA", "es_MX").strip()
# El texto EXACTO del boton de la plantilla aprobada. Las plantillas no mandan
# un id propio: en el webhook llega el texto del boton, asi que tiene que
# coincidir letra por letra con el que se aprobo en Meta.
PLANTILLA_BOTON_STOP = os.environ.get("PLANTILLA_BOTON_STOP", "Ya no me escribas").strip()

# ---------------------------------------------------------------------------
# LISTA "NO MOLESTAR" — la unica memoria que sobrevive a un redeploy
# ---------------------------------------------------------------------------
# El disco de Render es efimero: cada deploy borra seguimientos.json. Eso hacia
# que alguien que ya habia pedido que no le escribieran volviera a la casilla de
# salida en cuanto se subia codigo nuevo. Por eso UNA sola cosa se guarda fuera:
# el conjunto de numeros a los que Valentina NO debe escribirle nunca por su
# cuenta. Entran por tres puertas distintas y salen por ninguna:
#
#   1. pidieron que no les escriban (boton o escrito)
#   2. agendaron la llamada
#   3. ya recibieron su unico seguimiento
#
# La lista SOLO CRECE. No hay codigo que saque a nadie, asi que no existe el bug
# de "se me desactivo el silencio sin querer".
#
# Se usa Upstash (Redis) porque su API es HTTP puro: no hace falta ninguna
# libreria nueva, requests ya estaba aqui para hablar con Meta. El plan gratis
# aguanta de sobra (esto escribe unas pocas veces al mes).
#
# En Render hay que poner dos variables: UPSTASH_URL y UPSTASH_TOKEN.
UPSTASH_URL = os.environ.get("UPSTASH_URL", "").strip().rstrip("/")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN", "").strip()
CLAVE_NO_MOLESTAR = os.environ.get("CLAVE_NO_MOLESTAR", "no_molestar").strip()

# --- ARCHIVO DE CONVERSACIONES (2026-09-03) -------------------------------
# Las conversaciones vivian SOLO en memoria (`historiales`), o sea que cada
# redeploy de Render las borraba y no quedaba nada que leer despues. Ahora cada
# conversacion se guarda tambien en Upstash, en la MISMA cuenta que la lista de
# no-molestar pero con claves aparte, para poder releerlas, clasificarlas y
# usarlas de retroalimentacion.
#
# Una clave por persona: "conv:5219991105167" -> un JSON con sus datos y todos
# sus mensajes con hora. Y un indice ("conversaciones") con los numeros que
# existen, para poder exportarlas todas sin usar SCAN.
#
# OJO: esto NO toca nada de no-molestar. Son claves distintas y funciones
# distintas a proposito; esa parte quedo exactamente igual.
PREFIJO_CONVERSACION = os.environ.get("PREFIJO_CONVERSACION", "conv:").strip()
CLAVE_INDICE_CONVERSACIONES = os.environ.get(
    "CLAVE_INDICE_CONVERSACIONES", "conversaciones").strip()
# Contrasena para bajar el archivo desde /exportar. Si se deja vacia, el
# endpoint queda APAGADO: son datos personales, no pueden quedar al aire.
CLAVE_EXPORT = os.environ.get("CLAVE_EXPORT", "").strip()

# Redes sociales de {ASESOR}: van en el primer seguimiento para que la persona
# pueda ver quien es antes de decidir. Es la unica parte del bot donde salen
# links, y NO los escribe el modelo: son texto fijo de aqui. Esa es la misma
# razon por la que se quito el link de Cal.com — en produccion el LLM tipeo mal
# una URL al regenerarla y mando a la gente a una pagina que no existe.
#
# Van con https:// completo a proposito: asi WhatsApp los vuelve tocables
# siempre, sin depender de que detecte el dominio suelto.
# Se puede sobreescribir desde el .env / Render con REDES_ASESOR en UNA linea.
# Si se pone vacia, el mensaje sale sin esta parte y no queda ningun hueco raro.
REDES_ASESOR = os.environ.get("REDES_ASESOR", "").strip() or (
    "Facebook: https://www.facebook.com/SegurosArroyoCampos/\n"
    "Instagram: https://instagram.com/segurosarroyocampos\n"
    "TikTok: https://tiktok.com/@jorgearroyodom\n"
    "Web: https://segurosarroyocampos.com"
)

# Cuantos seguimientos se mandan como maximo en un solo ping del despertador.
# Gunicorn corre con un worker: si se mandaran 200 de golpe, el webhook de
# WhatsApp se quedaria esperando y la gente veria a Valentina muda.
MAX_ENVIOS_POR_TICK = 10

# El estado vive en memoria y se copia a disco para aguantar un reinicio de
# Render. OJO: en Render el disco es efimero, un REDEPLOY si lo borra; los
# seguimientos pendientes de ese momento se pierden (no se manda de mas, se
# manda de menos). No vale la pena una base de datos por esto todavia.
ARCHIVO_SEGUIMIENTOS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "seguimientos.json")

SYSTEM_PROMPT = """
Eres Valentina. Trabajas con {ASESOR}, asesor de seguros. Atiendes por WhatsApp
a gente de America Latina que respondio un anuncio de PROTECCION DE INGRESOS.

## OBJETIVO UNICO: UNA LLAMADA DE 5 MINUTOS CON {ASESOR_CORTO_MAYUS}
Tu no explicas el producto ni cotizas: eso lo hace el en la llamada. Haces tres
cosas: que la persona se sienta escuchada, conocer su situacion, cerrar la
cita. Explicar de mas es peor que no explicar, porque le quitas a
{ASESOR_CORTO} la razon para llamar. Si dudas entre dar un dato o avanzar a la
cita, avanza a la cita.
Nunca digas que eres "de Ole" ni de una aseguradora: trabajas CON {ASESOR}. Si
preguntan de que compania, el trabaja con varias y en la llamada le muestra
las que apliquen a su caso.

## ENCUADRE (siempre el mismo)
Que si la persona faltara o no pudiera trabajar, su familia siga viviendo
igual que hoy. No hables de "seguro de vida", polizas, coberturas ni sumas
aseguradas: habla del ingreso que sostiene a su familia.

## ESTILO
- Espanol calido y tranquilo. Tutea. Nunca insistente, nunca vendedora.
- 2 a 4 lineas. UNA sola pregunta por mensaje, siempre.
- Usa su nombre en cuanto lo sepas, y los de su pareja e hijos.
- Reconoce o celebra lo que te acaban de decir ANTES de preguntar lo siguiente
  ("que bueno que...", "te felicito...", "excelente pregunta..."). Nunca
  preguntes en seco.
- 1 o 2 emojis maximo. Cero tecnicismos.
- Si te preguntan algo, contesta en una linea y devuelve con otra pregunta.
  Nunca cierres un turno sin pregunta o sin un siguiente paso.

## SECUENCIA (una pregunta a la vez, en este orden)
1. Primer mensaje, tal cual: "Hola! Soy Valentina, trabajo con {ASESOR} 😊
   Muchas gracias por tu interes. Como te llamas?"
2. Edad.
3. Dependientes: "Tienes dependientes economicos, como esposa, hijos o papas?"
   Si te dicen que si, pregunta sus nombres y edades.
4. Ocupacion: "A que te dedicas?"
5. LA PREGUNTA CLAVE: "Si hoy dejaras de recibir dinero, cuanto necesita tu
   familia al mes para seguir viviendo como viven?" No la suavices ni la
   brinques: casi nadie se la ha hecho nunca. Si la pareja tambien trabaja,
   pregunta despues que porcentaje aporta cada quien.
6. Proteccion actual, SOLO si ellos la mencionaron: "sabes cuanto dinero
   recibiria tu familia?" y "cuando termina?" Casi nunca lo saben, y que se
   den cuenta solos vale mas que cualquier argumento tuyo.
7. La cita.
No repitas un dato que ya te dieron. Con nombre, edad, dependientes, ocupacion
y gasto familiar YA TIENES SUFICIENTE: pide la cita. El diagnostico completo
es trabajo de {ASESOR_CORTO}.
Si preguntan por que preguntas tanto (pasa seguido, es buena senal):
"Podriamos ir directo, pero quiero que no termines contratando algo que no era
lo que tu familia necesita." Y sigue con la secuencia.

## LA CITA
1. Resume en una linea lo que entendiste de su situacion.
2. Pidela como 5 minutos, sin compromiso, para conocerse y que {ASESOR_CORTO}
   le de su punto de vista. Si dicen que tienen muchas dudas, se toman todo el
   tiempo que necesiten.
3. Si la decision es de pareja, ofrece que entren los dos: cierra mas citas
   que insistir.
4. TU NO OFRECES HORARIOS NI MANDAS LINKS. Nunca escribas una fecha, una hora,
   un dia de la semana ni una direccion de internet, aunque te la pidan: no
   conoces la agenda y te contradices con lo que sale despues. Prohibido "te
   acomoda hoy en la tarde?" o "te mando la liga". Di: "Dejame ver que dias
   tiene libres 😊"
5. Termina ese mensaje con la marca [MOSTRAR_HORARIOS] SOLA en la ultima
   linea, tal cual, sin nada alrededor:
   "Perfecto Carlos, con eso {ASESOR_CORTO} ya puede prepararte algo concreto.
   Dejame ver que dias tiene libres 😊
   [MOSTRAR_HORARIOS]"
   El sistema la intercepta (la persona nunca la ve), consulta la agenda real
   y le manda los dias libres; al tocar un dia salen las horas de ese dia, y
   al tocar una hora queda agendado. Siempre puede regresar o pedir mas
   opciones, asi que tu nunca ofreces alternativas.
6. Al tocar un horario el sistema agenda solo, sin pedirle ningun dato mas, y
   le manda un boton por si necesita cancelar. Si ves "[Cita agendada ...]":
   no vuelvas a proponer la cita ni pidas datos, sigue natural y despidete
   calido. Si ves "[Cita cancelada ...]": no la reganes ni insistas; si quiere
   otra hora, vuelve a poner [MOSTRAR_HORARIOS].
7. Si te pide cancelar por texto, tu no puedes hacerlo: pidele que toque el
   boton "Cancelar cita", o pon [MOSTRAR_HORARIOS] si quiere cambiar la hora.
No pidas correo ni telefono. Si ya agendo, agradece y cierra. Si dice "despues
te aviso": "Sin problema, yo te busco en unos dias. Aqui sigo cuando quieras."

## OBJECIONES
Regla de oro: jamas ataques lo que ya tiene ni lo que ya decidio. Reconocelo,
felicitalo si aplica, y haz UNA pregunta que lo lleve a darse cuenta solo.
Nunca argumentes largo. Toda objecion termina proponiendo la cita.
- Ya tengo seguro -> "Que bueno, eso habla muy bien de ti. Sabes cuanto
  recibiria tu familia, y cuando termina tu poliza?" (casi nunca lo saben)
- Me lo da mi empresa -> "Excelente prestacion. Sabes de cuanto es?" Luego,
  sin alarmarlo: esta ligada al empleo y cambia el dia que cambie de trabajo.
  Nunca digas que "no sirve" o "no alcanza".
- Prefiero invertir -> felicitalo en serio y pregunta en que invierte. JAMAS
  compares seguro contra inversion: se trata de proteger lo que ya construyo.
- Mandame informacion y costos -> no mandes folletos, videos ni precios: "para
  mandarte algo que te sirva y no un generico, ayudame con un par de
  preguntas." Y sigue la secuencia.
- Cuanto cuesta -> "Depende de tu edad y de lo que tu familia necesite, y no
  quiero darte un numero al aire. {ASESOR_CORTO} te lo calcula exacto."
- No tengo tiempo / no me gustan las llamadas -> respetalo, sigue por mensaje
  y vuelve a proponer la cita mas adelante. Nunca insistas dos mensajes
  seguidos.
- Lo he ido dejando -> "Te entiendo, estos temas son importantes pero nunca
  urgentes." Y propon la cita de inmediato.
- Lo consulto con mi pareja -> "Perfecto, es decision de los dos. Como se
  llama? Hacemos la llamada con ambos."
- No confio / no los conozco -> "Te entiendo. {ASESOR_CORTO} lleva anos
  asesorando familias en Latinoamerica y la llamada es sin compromiso."
- Ya tengo patrimonio, o cualquier otra -> no la contestes tu: validala en una
  linea y pasala a la llamada.

## EJEMPLO DE TU TONO (asi se ve bien hecho)
Persona: "Ya tengo un seguro de vida, no creo necesitar otro."
Tu: "Que bueno Andres, eso habla muy bien de ti 😊 Varios clientes de
{ASESOR_CORTO} estan igual. Sabes cuanto dinero recibiria tu familia?"
Persona: "La verdad no. Creo que son 5 millones, pero no estoy seguro."
Tu: "Es lo mas comun, no te preocupes, {ASESOR_CORTO} te lo revisa. Y cuanto
necesita tu familia al mes para vivir como viven hoy?"
Fijate: usaste su nombre, reconociste antes de preguntar, no discutiste su
seguro, no explicaste nada, y avanzaste con UNA pregunta.

## SEGUIMIENTOS (los manda el sistema, no tu)
Si no agendan, el sistema escribe UNA vez al dia siguiente y ahi para. Tu no
prometas fechas de reenganche. Si ves "[Seguimiento enviado: ...]" y despues
por fin contestan, NO les reproches el silencio: retoma calido y ve por la
cita. Si piden que no les escriban, el sistema lo apaga antes de que te
llegue; si aun asi te toca, confirmalo con amabilidad y despidete. Nunca
discutas esa decision.

## REGLAS ABSOLUTAS (mandan sobre todo lo anterior)
1. Nunca inventes productos, coberturas, precios ni condiciones: "eso te lo
   explica {ASESOR_CORTO}".
2. Nunca prometas un resultado antes de que el analice el caso. Prohibido
   "vas a estar feliz con lo que te voy a preparar" o "seguro te lo aprueban".
3. Nunca des un precio, ni rango, ni aproximado, ni "de ejemplo".
4. Nunca hables de enfermedades, exclusiones ni de a quien acepta o rechaza
   una aseguradora, ni siquiera para dar urgencia.
5. Nunca des asesoria medica, legal, fiscal ni de inversion.
6. SI puedes preguntar edad, ocupacion, dependientes y gasto familiar. NUNCA
   pidas identificacion, datos bancarios, tarjetas, contrasenas ni historial
   medico; si te mandan uno, no lo repitas.
7. Ante enfermedad grave o duelo, acompana primero; no propongas la cita en
   ese mismo mensaje.
8. Si es menor de edad, la contratacion es desde los 18: ofrece hablar con su
   papa o mama.
9. Quejas o problemas con una poliza existente: pasalos a {ASESOR_CORTO}.
10. Fuera de tema, redirige con amabilidad. Si piden hablar con una persona,
    conectalos de inmediato: es buena senal.

Contacto, solo si lo piden o quieren llamar ya:
Jorge Arroyo +52 999 949 2999 · Enrique Ampudia +52 990 310 0732
""".strip() \
    .replace("{ASESOR_CORTO_MAYUS}", ASESOR_CORTO.upper()) \
    .replace("{ASESOR_CORTO}", ASESOR_CORTO) \
    .replace("{ASESOR}", ASESOR_NOMBRE)

# Memoria por persona (se borra al reiniciar; en produccion iria a una base de datos)
historiales = {}

# Horarios que se le mostraron a cada persona en su ultima lista, para poder
# traducir el id del boton que toco ("slot_0") de vuelta a una hora real.
# Formato: {"modo": "dias", "dias": [...], "pagina_dias": 0} mientras eligen
# dia, y {"modo": "horas", "fecha": "2026-08-04", "slots": [...],
# "pagina_horas": 0} mientras eligen hora.
horarios_mostrados = {}

# Dias que la persona ya vio y descarto (entro al dia y toco "Otro dia").
# Dejan de aparecer en la lista para no hacerla dar vueltas sobre lo mismo.
# Los dias que nunca abrio SI siguen apareciendo. Formato: {remitente: {fecha}}
dias_descartados = {}

# WhatsApp permite maximo 10 filas por lista (sin importar en cuantas
# secciones se partan), asi que las tandas dejan lugar a las filas de
# navegacion: 9 dias + "Otras fechas", u 8 horas + "Mas tarde" + "Otro dia".
DIAS_POR_PAGINA = 9
HORAS_POR_PAGINA = 8

# Ultima cita agendada por persona, para poder cancelarla si toca el boton.
# Formato: {remitente: {"uid": "abc123", "start": iso}}
citas_agendadas = {}

# Estado de reenganche por persona. Formato:
#   {remitente: {"ultimo_mensaje": iso,   # ultima vez que ELLA escribio
#                "enviados": 0,            # seguimientos mandados EN TODA SU VIDA
#                "ultimo_envio": iso|None, # cuando se mando el ultimo
#                "agendo": False,          # tiene cita en pie -> no se le escribe
#                "detenido": False}}       # no se le escribe por su cuenta nunca mas
# Esto vive en disco y se pierde en cada deploy. No importa: lo unico que no se
# puede perder es la lista de no-molestar, que vive en Upstash (ver abajo).
seguimientos = {}

# Numeros a los que Valentina no debe escribirle por su cuenta nunca mas. Es la
# copia en memoria de lo que hay en Upstash; se llena al arrancar.
no_molestar = set()
# False mientras no se haya podido leer la lista. Si se queda en False NO se
# manda ningun seguimiento: es preferible mandar de menos (nadie se entera) que
# escribirle a alguien que pidio que lo dejaran en paz.
no_molestar_cargada = False

DIAS_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
MESES_CORTO = ["ene", "feb", "mar", "abr", "may", "jun",
               "jul", "ago", "sep", "oct", "nov", "dic"]

app = Flask(__name__)


# ---------------------------------------------------------------------------
# 2) VERIFICACION DEL WEBHOOK  (GET)  — Meta la llama una sola vez al configurar
# ---------------------------------------------------------------------------
@app.route("/webhook", methods=["GET"])
def verificar():
    modo      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if modo == "subscribe" and token == VERIFY_TOKEN:
        print("Webhook verificado por Meta.")
        return challenge, 200        # <- devolver el challenge tal cual
    return "Token invalido", 403


# ---------------------------------------------------------------------------
# 3) MENSAJES ENTRANTES  (POST)
# ---------------------------------------------------------------------------
@app.route("/webhook", methods=["POST"])
def recibir():
    data = request.get_json(force=True, silent=True) or {}

    try:
        value = data["entry"][0]["changes"][0]["value"]
        mensajes = value.get("messages")
        if not mensajes:
            # No es un mensaje (p. ej. una notificacion de estado): ignorar.
            return "ok", 200

        msg = mensajes[0]
        remitente = msg["from"]                       # numero de la persona

        # Cualquier cosa que llegue (texto o boton) reabre la ventana de 24 h
        # de Meta y reinicia la cuenta de seguimientos.
        registrar_mensaje_entrante(remitente)

        # Caso 0: boton de una PLANTILLA. Meta no manda un id propio en estos,
        # llega el texto del boton en msg["button"]["text"] — por eso se compara
        # contra PLANTILLA_BOTON_STOP, que debe decir exactamente lo mismo que
        # el boton aprobado en WhatsApp Manager.
        if msg.get("type") == "button":
            etiqueta = (msg.get("button") or {}).get("text", "").strip()
            if etiqueta.lower() == PLANTILLA_BOTON_STOP.lower():
                detener_seguimientos(remitente)
            else:
                mandar_lista_horarios(remitente)
            return "ok", 200

        # Caso 1: tocaron algo (un horario de la lista, o un boton). No pasa
        # por el modelo: es un evento estructurado, se maneja directo.
        interactivo = msg.get("interactive") or {}
        if msg.get("type") == "interactive":
            respuesta = (interactivo.get("list_reply")
                         or interactivo.get("button_reply") or {})
            id_boton = respuesta.get("id", "")
            # Lo que toco cuenta como algo que dijo: sin esto el expediente
            # tendria huecos justo en la parte que mas importa (el agendado).
            registrar_en_archivo(remitente, "prospecto",
                                 respuesta.get("title") or id_boton)
            if id_boton == "no_seguimiento":
                detener_seguimientos(remitente)
            elif id_boton == "ver_horarios":
                mandar_lista_horarios(remitente)
            elif id_boton.startswith("cancelar"):
                manejar_cancelacion(remitente, id_boton)
            else:
                manejar_seleccion_horario(remitente, id_boton, value)
            return "ok", 200

        texto = msg.get("text", {}).get("body", "")   # el texto que escribio
        registrar_en_archivo(remitente, "prospecto", texto)

        # Caso 1.5: pidio por ESCRITO que ya no le escriban. Tiene que atajarse
        # aqui, antes del modelo: mas arriba registrar_mensaje_entrante() ya le
        # reinicio la cuenta de seguimientos, asi que si esto no existiera,
        # pedir "ya no me escribas" terminaria provocando otro seguimiento.
        # Vale exactamente lo mismo que tocar el boton, porque la gente no lee
        # botones. El modelo no ve este mensaje: no hay nada que negociar.
        if pide_que_no_le_escriban(texto):
            print(f"{remitente} pidio por texto que no le escriban: {texto!r}")
            detener_seguimientos(remitente)
            historial = historiales.setdefault(remitente, [])
            historial.append({"role": "user", "content": texto})
            historial.append({"role": "assistant",
                              "content": "[La persona pidio que no le escriban "
                                         "mas. Seguimientos apagados.]"})
            del historial[:-MAX_HISTORIAL]
            return "ok", 200

        # Caso 2: conversacion normal, la lleva el modelo.
        historial = historiales.setdefault(remitente, [])
        historial.append({"role": "user", "content": texto})
        # Solo se mandan los ultimos turnos: lo viejo se reenvia en cada
        # llamada y encarece cada mensaje sin aportar nada.
        del historial[:-MAX_HISTORIAL]

        reply = pensar_respuesta(historial)
        if not reply:
            # Nunca dejar al prospecto en visto: mejor pedir que repita que
            # quedarse callado a media conversacion.
            reply = ("Perdon, se me trabo el sistema un momento 😅 "
                     "Me lo repites por favor?")

        # Si el modelo decidio que es momento de agendar, la marca viene
        # pegada a su mensaje: se manda el texto (si trae algo) y luego la
        # lista de horarios reales, en vez de la marca cruda.
        marca = re.search(r"\[\s*MOSTRAR_HORARIOS\s*\]", reply, re.IGNORECASE)
        if marca:
            texto_previo = (reply[:marca.start()] + reply[marca.end():]).strip()
            historial.append({"role": "assistant", "content": reply})
            if texto_previo:
                enviar_whatsapp(remitente, texto_previo)
            mandar_lista_horarios(remitente)
        else:
            historial.append({"role": "assistant", "content": reply})
            enviar_whatsapp(remitente, reply)
    except Exception as e:
        print("Error procesando el mensaje:", e)

    # A Meta siempre le respondemos 200 rapido para que no reintente.
    return "ok", 200


# ---------------------------------------------------------------------------
# 3b) PENSAR LA RESPUESTA (con red de seguridad contra respuestas vacias)
# ---------------------------------------------------------------------------
def pensar_respuesta(historial):
    """Devuelve el texto de Valentina, o None si Groq no logro producirlo.

    Si el modelo se queda sin tokens razonando, content vuelve vacio. En ese
    caso reintentamos UNA vez con el doble de techo antes de rendirnos.
    """
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}] + historial

    for intento, techo in enumerate([MAX_TOKENS, MAX_TOKENS * 2], start=1):
        r = client.chat.completions.create(
            model=MODEL,
            max_tokens=techo,
            messages=mensajes,
            extra_body={"reasoning_effort": ESFUERZO},
        )
        eleccion = r.choices[0]
        texto = (eleccion.message.content or "").strip()
        if texto:
            return texto
        print(f"Respuesta vacia (intento {intento}/2): "
              f"finish_reason={eleccion.finish_reason}, "
              f"{r.usage.completion_tokens} tokens gastados razonando.")

    return None


# ---------------------------------------------------------------------------
# 4) ENVIAR mensajes de vuelta por WhatsApp (llamadas a la Graph API de Meta)
# ---------------------------------------------------------------------------
def _post_whatsapp(payload):
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    r = requests.post(GRAPH_URL, headers=headers, json=payload, timeout=20)
    if r.status_code >= 400:
        print("Error al enviar a WhatsApp:", r.status_code, r.text)
    return r


def enviar_whatsapp(destino, texto):
    # Meta rechaza un body vacio con "The parameter text.body is required".
    # Ultima linea de defensa: aqui no deberia llegar nunca vacio.
    texto = (texto or "").strip()
    if not texto:
        print("Se intento enviar un mensaje vacio a WhatsApp. Cancelado.")
        return
    # Un solo punto de captura para TODO lo que sale por aqui (respuestas del
    # modelo, avisos del sistema y seguimientos): asi no hay que acordarse de
    # archivar en cada sitio que manda un mensaje.
    registrar_en_archivo(destino, "valentina", texto)
    _post_whatsapp({
        "messaging_product": "whatsapp",
        "to": destino,
        "type": "text",
        "text": {"body": texto},
    })


def enviar_lista_whatsapp(destino, cuerpo, boton, filas, titulo="Horarios disponibles"):
    """filas: lista de dicts {"id", "title", "description"} (maximo 10, es
    limite de WhatsApp: no importa en cuantas secciones se partan)."""
    _post_whatsapp({
        "messaging_product": "whatsapp",
        "to": destino,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": cuerpo},
            "action": {
                "button": boton,
                "sections": [{"title": titulo, "rows": filas}],
            },
        },
    })


def enviar_botones_whatsapp(destino, cuerpo, botones):
    """botones: lista de dicts {"id", "title"} (maximo 3, titulo <= 20)."""
    _post_whatsapp({
        "messaging_product": "whatsapp",
        "to": destino,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": cuerpo},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": b["id"], "title": b["title"]}}
                    for b in botones
                ],
            },
        },
    })


# ---------------------------------------------------------------------------
# 5) CAL.COM — consultar horarios reales y agendar por API
# ---------------------------------------------------------------------------
def _calcom_headers(version):
    return {
        "Authorization": f"Bearer {CALCOM_API_KEY}",
        "cal-api-version": version,
        "Content-Type": "application/json",
    }


def obtener_agenda(dias_adelante=30):
    """Devuelve la agenda libre real agrupada por dia:
    `{"2026-08-04": ["2026-08-04T09:00:00-05:00", ...], ...}`, en orden.
    Se vuelve a consultar en cada paso a proposito: entre que la persona ve
    los dias y toca una hora pudo ocuparse algo."""
    ahora = datetime.now(timezone.utc)
    inicio = ahora.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    fin = (ahora + timedelta(days=dias_adelante)).strftime("%Y-%m-%dT23:59:59.000Z")
    r = requests.get(
        f"{CALCOM_BASE}/slots",
        headers=_calcom_headers("2024-09-04"),
        params={
            "eventTypeId": CALCOM_EVENT_TYPE_ID,
            "start": inicio,
            "end": fin,
            "timeZone": CALCOM_TIMEZONE,
        },
        timeout=15,
    )
    r.raise_for_status()
    dias = r.json().get("data", {})
    agenda = {}
    for fecha in sorted(dias):
        horas = [s["start"] for s in dias[fecha]]
        if horas:
            agenda[fecha] = horas
    return agenda


def formatear_dia(fecha):
    """Titulo corto de la fila del dia, ej. 'Martes 4 ago'.
    WhatsApp corta los titulos en 24 caracteres."""
    dt = datetime.fromisoformat(fecha)
    return f"{DIAS_ES[dt.weekday()].capitalize()} {dt.day} {MESES_CORTO[dt.month - 1]}"


def formatear_dia_largo(fecha):
    """Ej. 'martes 4 de agosto'."""
    dt = datetime.fromisoformat(fecha)
    return f"{DIAS_ES[dt.weekday()]} {dt.day} de {MESES[dt.month - 1]}"


def formatear_hora(iso_str):
    """Solo la hora, ej. '9:00 a.m.' — el dia ya va en el texto del mensaje."""
    dt = datetime.fromisoformat(iso_str)
    hora12 = dt.hour % 12 or 12
    ampm = "a.m." if dt.hour < 12 else "p.m."
    return f"{hora12}:{dt.minute:02d} {ampm}"


def formatear_slot_largo(iso_str):
    """Version legible para texto normal, ej. 'jueves 7 de agosto a las 9:00 a.m.'"""
    dt = datetime.fromisoformat(iso_str)
    return (f"{DIAS_ES[dt.weekday()]} {dt.day} de {MESES[dt.month - 1]} "
            f"a las {formatear_hora(iso_str)}")


def mandar_lista_dias(remitente, pagina=0):
    """Paso 1: los dias que tienen huecos libres. La lista se arma siempre
    desde hoy, asi que volver aqui desde un dia nunca descarta los anteriores."""
    try:
        agenda = obtener_agenda()
    except Exception as e:
        print("Error consultando horarios en Cal.com:", e)
        enviar_whatsapp(remitente,
            "Se me trabo revisando la agenda, dame un momento y seguimos 😅")
        return

    if not agenda:
        enviar_whatsapp(remitente,
            f"Por ahora no veo huecos libres en los proximos dias, dejame "
            f"checar con {ASESOR_CORTO} y te aviso 🙏")
        return

    descartados = dias_descartados.get(remitente, set())
    fechas = [f for f in agenda if f not in descartados]
    if not fechas:
        # Ya le dio vuelta a todos los dias: se limpia el descarte y se
        # empieza otra vez, mejor que dejarla con una lista vacia.
        dias_descartados.pop(remitente, None)
        fechas = list(agenda)
        pagina = 0
        enviar_whatsapp(remitente,
            f"Ya vimos todos los dias que tiene libres {ASESOR_CORTO} 🙈 "
            f"Te los pongo de nuevo, y si de plano ninguno te acomoda dime "
            f"que dia y hora te quedarian mejor y lo checo con el.")

    # Si se pasaron del final (agenda mas corta que la ultima vez), se regresa
    # al principio en vez de dejar a la persona con una lista vacia.
    if pagina * DIAS_POR_PAGINA >= len(fechas):
        pagina = 0

    tanda = fechas[pagina * DIAS_POR_PAGINA:(pagina + 1) * DIAS_POR_PAGINA]
    hay_mas = len(fechas) > (pagina + 1) * DIAS_POR_PAGINA

    horarios_mostrados[remitente] = {
        "modo": "dias",
        "dias": tanda,
        "pagina_dias": pagina,
    }
    filas = []
    for i, fecha in enumerate(tanda):
        horas = agenda[fecha]
        filas.append({
            "id": f"dia_{i}",
            "title": formatear_dia(fecha),
            "description": (f"{len(horas)} horarios, desde las "
                            f"{formatear_hora(horas[0])}"),
        })
    if hay_mas:
        filas.append({
            "id": "mas_dias",
            "title": "Otras fechas",
            "description": "Ver mas dias mas adelante",
        })

    if pagina == 0:
        cuerpo = (f"Estos son los dias que {ASESOR_CORTO} tiene libres. "
                  f"Cual te queda mejor?")
    else:
        cuerpo = "Va, estas son las fechas que siguen. Cual te queda mejor?"

    enviar_lista_whatsapp(remitente, cuerpo, "Ver dias", filas,
                          titulo="Dias disponibles")


def mandar_lista_horas(remitente, fecha, pagina=0):
    """Paso 2: las horas libres de ese dia, mas las salidas "mas tarde ese
    dia" y "otro dia" para que nadie se quede sin opcion."""
    try:
        agenda = obtener_agenda()
    except Exception as e:
        print("Error consultando horarios en Cal.com:", e)
        enviar_whatsapp(remitente,
            "Se me trabo revisando la agenda, dame un momento y seguimos 😅")
        return

    horas = agenda.get(fecha)
    if not horas:
        enviar_whatsapp(remitente,
            f"Uy, {formatear_dia_largo(fecha)} se acaba de llenar 🙈 "
            f"Estos son los dias que siguen libres:")
        mandar_lista_dias(remitente, 0)
        return

    if pagina * HORAS_POR_PAGINA >= len(horas):
        pagina = 0

    tanda = horas[pagina * HORAS_POR_PAGINA:(pagina + 1) * HORAS_POR_PAGINA]
    hay_mas = len(horas) > (pagina + 1) * HORAS_POR_PAGINA

    horarios_mostrados[remitente] = {
        "modo": "horas",
        "fecha": fecha,
        "slots": tanda,
        "pagina_horas": pagina,
    }
    filas = [{"id": f"slot_{i}", "title": formatear_hora(h)}
             for i, h in enumerate(tanda)]
    if hay_mas:
        filas.append({
            "id": "mas_horas",
            "title": "Mas tarde ese dia",
            "description": f"Ver horarios mas tarde del {formatear_dia_largo(fecha)}",
        })
    filas.append({
        "id": "otro_dia",
        "title": "Otro dia",
        "description": "Ninguno me acomoda, ver otras fechas",
    })

    if pagina == 0:
        cuerpo = (f"Estas son las horas libres del {formatear_dia_largo(fecha)}. "
                  f"Cual te acomoda?")
    else:
        cuerpo = (f"Mas horarios del {formatear_dia_largo(fecha)}. "
                  f"Cual te acomoda?")

    enviar_lista_whatsapp(remitente, cuerpo, "Ver horarios", filas,
                          titulo=formatear_dia(fecha))


def mandar_lista_horarios(remitente):
    """Entrada del flujo de agendado: arranca por los dias y sin descartes,
    porque es un intento nuevo (lo que descarto hace rato pudo cambiar)."""
    dias_descartados.pop(remitente, None)
    mandar_lista_dias(remitente, 0)


def manejar_seleccion_horario(remitente, id_boton, value):
    """Un solo lugar para todo lo que se toca en las listas: dias, horas y las
    filas de navegacion."""
    estado = horarios_mostrados.get(remitente) or {}

    if id_boton == "otro_dia":
        # Ya vio las horas de ese dia y ninguna le sirvio: se saca de la lista
        # para que las siguientes fechas sean todas nuevas para ella.
        fecha = estado.get("fecha")
        if fecha:
            dias_descartados.setdefault(remitente, set()).add(fecha)
        mandar_lista_dias(remitente, 0)
        return

    if id_boton == "mas_dias":
        mandar_lista_dias(remitente, estado.get("pagina_dias", 0) + 1)
        return

    if id_boton == "mas_horas":
        fecha = estado.get("fecha")
        if not fecha:
            mandar_lista_dias(remitente, 0)
            return
        mandar_lista_horas(remitente, fecha, estado.get("pagina_horas", 0) + 1)
        return

    if id_boton.startswith("dia_"):
        dias = estado.get("dias") or []
        idx = _indice(id_boton)
        if idx is None or idx >= len(dias):
            mandar_lista_dias(remitente, 0)
            return
        mandar_lista_horas(remitente, dias[idx], 0)
        return

    horarios = estado.get("slots") or []
    idx = _indice(id_boton) if id_boton.startswith("slot_") else None
    if idx is None or idx >= len(horarios):
        # Casi siempre pasa porque el servicio se reinicio y se perdio el
        # estado en memoria: se vuelve a empezar en vez de dejarla trabada.
        enviar_whatsapp(remitente,
            "Perdon, perdi el hilo de los horarios 😅 Te los pongo de nuevo:")
        mandar_lista_dias(remitente, 0)
        return

    nombre = None
    try:
        nombre = value["contacts"][0]["profile"]["name"]
    except (KeyError, IndexError, TypeError):
        pass

    agendar(remitente, horarios[idx], nombre or "Prospecto")


def _indice(id_boton):
    """'slot_3' -> 3. None si viene algo raro."""
    try:
        return int(id_boton.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def normalizar_telefono(wa_id):
    """WhatsApp entrega el numero con el '1' viejo de movil (Mexico: 52 1 999...,
    Argentina: 54 9 11...). Cal.com valida con libphonenumber y esos formatos
    los rechaza con 'invalid_number', asi que hay que quitarles ese digito.
    Devuelve el numero en E.164, ej. '+529991105167'."""
    d = re.sub(r"\D", "", wa_id or "")
    if d.startswith("521") and len(d) == 13:      # Mexico movil (52 + 1 + 10)
        d = "52" + d[3:]
    elif d.startswith("549") and len(d) == 13:    # Argentina movil (54 + 9 + 10)
        d = "54" + d[3:]
    return f"+{d}"


def correo_de_whatsapp(telefono):
    """Cal.com exige un correo para agendar, pero a la persona no se le pide:
    ya la tenemos por WhatsApp y pedirlo costaba conversiones. Se arma uno
    derivado del numero, siempre el mismo, que ademas sirve para volver a
    encontrar sus citas en la API si se pierde el estado en memoria.

    OJO (2026-09-03): antes se usaba `wa<numero>@example.com`. Dejo de servir:
    example.com publica un MX nulo (RFC 7505), o sea que no puede recibir
    correo, y Cal.com empezo a rechazarlo con `email_domain_cannot_receive_mail`
    igual que a los dominios `.invalid`. Con eso NINGUNA cita se agendaba: el
    POST tronaba con 400 y la persona veia "ese horario se acaba de ocupar"
    para cualquier hora que tocara. Ahora se usa una sub-direccion del correo
    real del asesor (CORREO_BASE_CITAS), que si recibe mail, sigue siendo
    distinta por prospecto y no expone los datos a nadie de fuera."""
    digitos = re.sub(r"[^0-9]", "", telefono or "")
    buzon, _, dominio = CORREO_BASE_CITAS.partition("@")
    # Si el buzon ya traia un "+algo" se ignora, para no encadenar etiquetas.
    buzon = buzon.split("+")[0]
    return f"{buzon}+wa{digitos}@{dominio}"


# Cuanta conversacion cruda se pega en la descripcion del evento. Google
# Calendar aguanta ~8 KB ahi; con 3500 caracteres cabe una conversacion
# completa de Valentina y no hay riesgo de que Cal.com la recorte.
LIMITE_TRANSCRIPCION = 3500

PROMPT_RESUMEN_CITA = """Preparas la ficha de un prospecto para que un asesor de
seguros la lea en 10 segundos antes de llamarlo. Te doy su conversacion de
WhatsApp. Devuelve SOLO estos campos, uno por linea, sin introduccion:

Nombre:
Edad:
Dependientes:
Ocupacion:
Necesita al mes:
Proteccion actual:
Objeciones o dudas:
Por donde ir:

Reglas estrictas:
- Usa UNICAMENTE lo que la persona dijo. Si un dato no aparece, escribe "no lo dijo".
- Prohibido inventar, deducir o rellenar. Un campo vacio es mejor que uno inventado.
- Cada campo en una sola linea, corto y concreto.
- "Por donde ir" es una frase con lo que mas le importo a la persona."""


def resumen_para_calendario(remitente):
    """Arma lo que el asesor leera en la descripcion del evento: una ficha con
    los datos que dio el prospecto, mas la conversacion tal cual.

    Importa mas de lo que parece: los historiales viven en memoria y se borran
    en cada redeploy de Render, asi que el evento del calendario termina siendo
    el UNICO lugar donde queda lo que la persona conto. Si el resumen falla se
    manda la conversacion cruda igual: perder la ficha es molesto, perder la
    conversacion es perder todo el trabajo de Valentina."""
    salto = chr(10)
    historial = historiales.get(remitente) or []
    lineas = []
    for m in historial:
        contenido = (m.get("content") or "").strip()
        # Las marcas internas ([Cita agendada ...], [Seguimiento enviado: ...])
        # son recados para el modelo, no las dijo nadie: fuera de la ficha.
        if not contenido or contenido.startswith("["):
            continue
        quien = "Prospecto" if m.get("role") == "user" else "Valentina"
        lineas.append(f"{quien}: {contenido}")

    if not lineas:
        return "", ""

    transcripcion = salto.join(lineas)
    if len(transcripcion) > LIMITE_TRANSCRIPCION:
        # Se recorta por el principio: lo ultimo que hablaron es lo que le
        # sirve al asesor para arrancar la llamada.
        transcripcion = ("...(inicio recortado)" + salto
                         + transcripcion[-LIMITE_TRANSCRIPCION:])

    resumen = ""
    try:
        r = client.chat.completions.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "system", "content": PROMPT_RESUMEN_CITA},
                      {"role": "user", "content": transcripcion}],
            extra_body={"reasoning_effort": "low"},
        )
        resumen = (r.choices[0].message.content or "").strip()
    except Exception as e:
        # Nunca debe tumbar la reserva: la cita vale mas que la ficha.
        print("No se pudo resumir la conversacion para el calendario:", e)

    return resumen, transcripcion


def crear_reserva_calcom(start_iso, nombre, correo, telefono,
                         resumen="", transcripcion=""):
    dt = datetime.fromisoformat(start_iso)
    start_utc = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    telefono_e164 = normalizar_telefono(telefono)
    wa_id = re.sub(r"\D", "", telefono or "")
    # Estas notas son lo que ve el asesor en su Google Calendar y en el correo
    # de confirmacion: el numero para llamar y el link directo al chat.
    partes = [f"Prospecto de WhatsApp: {telefono_e164}",
              f"Abrir el chat: https://wa.me/{wa_id}"]
    if resumen:
        partes += ["", "--- FICHA DEL PROSPECTO ---", resumen]
    if transcripcion:
        partes += ["", "--- CONVERSACION CON VALENTINA ---", transcripcion]
    partes += ["", "Agendado automaticamente por Valentina (bot)."]
    notas = chr(10).join(partes)
    payload = {
        "start": start_utc,
        "eventTypeId": int(CALCOM_EVENT_TYPE_ID),
        "attendee": {
            "name": nombre,
            "email": correo,
            "timeZone": CALCOM_TIMEZONE,
            "phoneNumber": telefono_e164,
        },
        # attendeePhone = el asesor llama a ESTE numero. Cal.com lo pone solo
        # como "ubicacion" de la cita, no hay que llenarlo a mano.
        "location": {"type": "attendeePhone", "phone": telefono_e164},
        "bookingFieldsResponses": {"notes": notas},
        "metadata": {"whatsapp": wa_id[:50]},
    }
    r = requests.post(f"{CALCOM_BASE}/bookings",
                       headers=_calcom_headers("2024-08-13"),
                       json=payload, timeout=15)
    if r.status_code >= 400:
        return False, None, r.text
    uid = None
    try:
        uid = r.json()["data"]["uid"]
    except (ValueError, KeyError, TypeError):
        pass                       # la cita quedo; solo no podremos cancelarla
    return True, uid, r.text


def agendar(remitente, start_iso, nombre):
    """Cierra la cita en cuanto toca el horario: no se le pide ningun dato
    mas, todo lo que Cal.com necesita ya lo tenemos de WhatsApp."""
    # Se arma ANTES de reservar: si el resumen truena, truena aqui y no
    # despues de haberle dicho a la persona que su cita ya quedo.
    resumen, transcripcion = resumen_para_calendario(remitente)
    ok, uid, detalle = crear_reserva_calcom(
        start_iso, nombre, correo_de_whatsapp(remitente), remitente,
        resumen, transcripcion)

    horarios_mostrados.pop(remitente, None)

    if not ok:
        print("Error creando reserva en Cal.com:", detalle)
        # Antes CUALQUIER error se le contaba como "se ocupo el horario" y se
        # le volvia a mandar la lista. Si la falla no era del horario sino de
        # la peticion (pasaba con el correo invalido), la persona quedaba en un
        # bucle: elegia hora -> "se ocupo" -> lista -> elegia hora... sin salida
        # y sin que nadie se enterara. Solo se reintenta si de verdad se ocupo.
        texto = (detalle or "").lower()
        se_ocupo = any(p in texto for p in (
            "no_available_users", "already", "no longer available",
            "not available", "booking_seats_full", "slot", "conflict"))
        if se_ocupo:
            enviar_whatsapp(remitente,
                "Uy, ese horario se acaba de ocupar 🙈 Estos son los dias que "
                "siguen libres:")
            # Sin resetear los descartes: sigue siendo el mismo intento y los
            # dias que ya rechazo le siguen sin servir.
            mandar_lista_dias(remitente, 0)
            return
        # Falla tecnica: no se le vuelve a ofrecer la lista (tronaria igual).
        # Se le avisa de frente y se pasa a manos del asesor.
        enviar_whatsapp(remitente,
            f"Se me trabo el sistema al confirmar la cita 🙈 Ya le avise a "
            f"{ASESOR_CORTO} y el te contacta por aqui mismo para cerrarla. "
            f"Mil disculpas!")
        historiales.setdefault(remitente, []).append({
            "role": "assistant",
            "content": "[Fallo tecnico al agendar: no volver a ofrecer "
                       "horarios, el asesor lo contacta]",
        })
        return

    dias_descartados.pop(remitente, None)
    citas_agendadas[remitente] = {"uid": uid, "start": start_iso}
    # Ya agendo: se apagan los seguimientos mientras la cita siga en pie.
    marcar_agendado(remitente)
    # Se le avisa al modelo para que no vuelva a proponer la cita.
    historiales.setdefault(remitente, []).append({
        "role": "assistant",
        "content": f"[Cita agendada para {formatear_slot_largo(start_iso)} "
                   f"con {ASESOR_NOMBRE}]",
    })
    enviar_whatsapp(remitente,
        f"Listo! Quedo agendado {formatear_slot_largo(start_iso)} "
        f"{ASESOR_CORTO} ya quedo notificado y te va a marcar a este mismo "
        f"numero 😊")
    enviar_botones_whatsapp(remitente,
        "Si despues no puedes y necesitas cancelar, toca el boton. "
        "Si no lo tocas, la cita queda en pie.",
        [{"id": "cancelar_cita", "title": "Cancelar cita"}])


def buscar_cita_agendada(remitente):
    """Recupera la cita de la API cuando se perdio el estado en memoria
    (pasa en cada redeploy de Render). La busca por el correo derivado del
    numero, que siempre es el mismo para esa persona."""
    try:
        r = requests.get(f"{CALCOM_BASE}/bookings",
                         headers=_calcom_headers("2024-08-13"),
                         params={"attendeeEmail": correo_de_whatsapp(remitente),
                                 "status": "upcoming",
                                 "sortStart": "asc"},
                         timeout=15)
        r.raise_for_status()
        reservas = r.json().get("data") or []
    except Exception as e:
        print("Error buscando la cita en Cal.com:", e)
        return None
    if not reservas:
        return None
    return {"uid": reservas[0].get("uid"),
            "start": a_hora_local(reservas[0].get("start"))}


def a_hora_local(iso_str):
    """La API devuelve las horas en UTC ('...T15:00:00.000Z'); los horarios
    que mostramos van en la zona del asesor. Sin esto la cita recuperada de
    la API se le anunciaria a la persona con la hora corrida."""
    if not iso_str:
        return iso_str
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return iso_str
    return dt.astimezone(ZoneInfo(CALCOM_TIMEZONE)).isoformat()


def cancelar_reserva_calcom(uid):
    r = requests.post(f"{CALCOM_BASE}/bookings/{uid}/cancel",
                      headers=_calcom_headers("2024-08-13"),
                      json={"cancellationReason": "Cancelada por el prospecto "
                                                  "desde WhatsApp"},
                      timeout=15)
    return r.status_code < 400, r.text


def manejar_cancelacion(remitente, id_boton):
    """Cancelar siempre pide una confirmacion: un toque por error no puede
    tirar una cita ya ganada."""
    if id_boton == "cancelar_no":
        enviar_whatsapp(remitente,
            "Perfecto, tu cita sigue en pie 😊 Ahi te marca "
            f"{ASESOR_CORTO}.")
        return

    cita = citas_agendadas.get(remitente) or buscar_cita_agendada(remitente)
    if not cita or not cita.get("uid"):
        enviar_whatsapp(remitente,
            "No encuentro ninguna cita activa a tu nombre 🤔 Si quieres "
            "agendar una, dime y te paso los horarios.")
        return

    if id_boton == "cancelar_cita":
        citas_agendadas[remitente] = cita
        enviar_botones_whatsapp(remitente,
            f"Seguro que quieres cancelar tu cita del "
            f"{formatear_slot_largo(cita['start'])}?",
            [{"id": "cancelar_si", "title": "Si, cancelar"},
             {"id": "cancelar_no", "title": "No, dejarla"}])
        return

    # cancelar_si
    ok, detalle = cancelar_reserva_calcom(cita["uid"])
    if not ok:
        print("Error cancelando en Cal.com:", detalle)
        enviar_whatsapp(remitente,
            "Se me trabo cancelando 😅 Dame un momento y lo intento de nuevo.")
        return

    citas_agendadas.pop(remitente, None)
    # Vuelve a estar sin cita: la secuencia de seguimientos se rearma desde
    # cero, contando 23 h a partir de este momento.
    marcar_cita_cancelada(remitente)
    historiales.setdefault(remitente, []).append({
        "role": "assistant",
        "content": f"[Cita cancelada por la persona: "
                   f"{formatear_slot_largo(cita['start'])}]",
    })
    enviar_whatsapp(remitente,
        f"Listo, cancele tu cita del {formatear_slot_largo(cita['start'])} y "
        f"{ASESOR_CORTO} ya quedo enterado. Si luego quieres otra hora, "
        f"dime y te paso los horarios 😊")


# ---------------------------------------------------------------------------
# 6) SEGUIMIENTOS — reenganchar a quien se fue sin agendar
# ---------------------------------------------------------------------------
def _ahora():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat() if dt else None


def _desde_iso(texto):
    if not texto:
        return None
    try:
        dt = datetime.fromisoformat(texto)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _upstash(comando):
    """Manda un comando a Upstash por HTTP. `comando` es una lista, tal cual se
    escribiria en Redis: ["SADD", "no_molestar", "5215512345678"]."""
    if not (UPSTASH_URL and UPSTASH_TOKEN):
        return None
    r = requests.post(UPSTASH_URL,
                      headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
                      json=comando, timeout=8)
    r.raise_for_status()
    return r.json().get("result")


def cargar_no_molestar():
    """Trae de Upstash la lista de numeros a los que no hay que escribirles.
    Se llama al arrancar; si falla se reintenta en el siguiente ping del cron."""
    global no_molestar_cargada
    if not (UPSTASH_URL and UPSTASH_TOKEN):
        print("*** SIN UPSTASH: no hay lista de no-molestar que sobreviva a un "
              "redeploy, asi que NO se mandara ningun seguimiento. Poner "
              "UPSTASH_URL y UPSTASH_TOKEN en Render. ***")
        return
    try:
        resultado = _upstash(["SMEMBERS", CLAVE_NO_MOLESTAR]) or []
    except Exception as e:
        print("No se pudo leer la lista de no-molestar de Upstash:", e)
        return
    no_molestar.update(str(x) for x in resultado)
    no_molestar_cargada = True
    print(f"No-molestar recuperados de Upstash: {len(no_molestar)} numero(s).")


def agregar_no_molestar(remitente, motivo):
    """Apaga los mensajes por iniciativa propia hacia este numero, para siempre.
    Escribe primero en memoria (efecto inmediato aunque Upstash este caido) y
    despues afuera (para que aguante el proximo deploy)."""
    no_molestar.add(remitente)
    est = _estado(remitente)
    est["detenido"] = True
    guardar_seguimientos()
    try:
        _upstash(["SADD", CLAVE_NO_MOLESTAR, remitente])
        print(f"No-molestar: {remitente} ({motivo}).")
    except Exception as e:
        # Que falle Upstash no debe tumbar la conversacion. Queda apagado en
        # memoria; lo que se pierde es que aguante el proximo deploy.
        print(f"No se pudo guardar {remitente} en Upstash ({motivo}):", e)


# ---------------------------------------------------------------------------
# 5b) ARCHIVO DE CONVERSACIONES — para releerlas y aprender de ellas
# ---------------------------------------------------------------------------
# Nada de aqui puede tumbar una conversacion: si Upstash falla, se registra en
# memoria y se sigue. Perder el archivo es molesto; perder al prospecto no.

# Copia en memoria del archivo. {telefono: {...}}
conversaciones = {}


def _clasificar(remitente):
    """La etiqueta con la que se guarda la conversacion. Se calcula al vuelo
    LEYENDO el estado que ya existe: no se toca ni se duplica nada, sobre todo
    nada de no-molestar."""
    if remitente in no_molestar:
        return "no_molestar"
    if citas_agendadas.get(remitente):
        return "agendado"
    est = seguimientos.get(remitente) or {}
    if est.get("detenido"):
        return "no_molestar"
    if est.get("enviados", 0) > 0:
        return "sin_respuesta"      # se le mando el seguimiento y no volvio
    return "en_conversacion"


def registrar_en_archivo(remitente, de, texto):
    """Agrega un mensaje al expediente de esa persona y lo sube a Upstash.

    `de` es "prospecto", "valentina" o "sistema" (las marcas internas tipo
    "[Cita agendada ...]": se guardan porque cuentan que paso, y se distinguen
    para poder filtrarlas despues)."""
    texto = (texto or "").strip()
    if not texto:
        return

    ahora = _ahora().isoformat()
    exp = conversaciones.setdefault(remitente, {
        "telefono": remitente,
        "primer_mensaje": ahora,
        "mensajes": [],
    })
    exp["mensajes"].append({"t": ahora, "de": de, "texto": texto})
    exp["ultimo_mensaje"] = ahora
    exp["estado"] = _clasificar(remitente)
    exp["total_mensajes"] = len(exp["mensajes"])

    cita = citas_agendadas.get(remitente)
    if cita:
        exp["cita"] = cita.get("start")

    # El nombre es lo primero que se busca al revisar: se deja a mano en vez de
    # obligar a leer la conversacion entera.
    if de == "prospecto" and not exp.get("nombre") and len(exp["mensajes"]) <= 4:
        posible = texto.strip()
        if 1 < len(posible) <= 30 and " " not in posible.strip():
            exp["nombre"] = posible

    if not (UPSTASH_URL and UPSTASH_TOKEN):
        return
    try:
        _upstash(["SET", PREFIJO_CONVERSACION + remitente,
                  json.dumps(exp, ensure_ascii=False)])
        # SADD es idempotente: repetirlo no cuesta nada y evita llevar registro
        # aparte de quien ya estaba en el indice.
        _upstash(["SADD", CLAVE_INDICE_CONVERSACIONES, remitente])
    except Exception as e:
        print(f"No se pudo archivar la conversacion de {remitente}:", e)


def leer_conversaciones_archivadas():
    """Trae de Upstash todos los expedientes. Solo lo usa /exportar."""
    if not (UPSTASH_URL and UPSTASH_TOKEN):
        return []
    numeros = _upstash(["SMEMBERS", CLAVE_INDICE_CONVERSACIONES]) or []
    salida = []
    for numero in numeros:
        try:
            crudo = _upstash(["GET", PREFIJO_CONVERSACION + str(numero)])
            if crudo:
                salida.append(json.loads(crudo))
        except Exception as e:
            print(f"No se pudo leer el expediente de {numero}:", e)
    salida.sort(key=lambda c: c.get("ultimo_mensaje") or "", reverse=True)
    return salida


def cargar_seguimientos():
    """Recupera el estado del disco al arrancar. Un reinicio de Render no
    deberia hacer que la gente pendiente se quede sin su seguimiento."""
    try:
        with open(ARCHIVO_SEGUIMIENTOS, encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        return
    except Exception as e:
        print("No se pudo leer seguimientos.json:", e)
        return
    if isinstance(datos, dict):
        seguimientos.update(datos)
        print(f"Seguimientos recuperados: {len(seguimientos)} personas.")


def guardar_seguimientos():
    try:
        with open(ARCHIVO_SEGUIMIENTOS, "w", encoding="utf-8") as f:
            json.dump(seguimientos, f)
    except Exception as e:
        # Que no se pueda guardar no debe tumbar la conversacion en curso.
        print("No se pudo guardar seguimientos.json:", e)


def _estado(remitente):
    est = seguimientos.setdefault(remitente, {
        "ultimo_mensaje": None,
        "enviados": 0,
        "ultimo_envio": None,
        "intentos": 0,
        "agendo": False,
        "detenido": False,
    })
    # AQUI es donde el silencio sobrevive a un deploy. Despues de subir codigo
    # nuevo este diccionario arranca vacio, asi que la primera vez que la
    # persona escribe se le crea una ficha en blanco con detenido=False. Si su
    # numero esta en la lista de Upstash, se le vuelve a poner el freno antes de
    # que nadie lo lea.
    if not est.get("detenido") and remitente in no_molestar:
        est["detenido"] = True
    return est


def registrar_mensaje_entrante(remitente):
    """Cualquier cosa que llegue de la persona (texto o un boton) reabre la
    ventana de 24 h de Meta.

    OJO CON LO QUE **NO** HACE: ya no reinicia `enviados`. Antes si, y por eso
    el limite de "un seguimiento" era en realidad "un seguimiento cada vez que
    se enfria": la persona contestaba, el contador volvia a 0, se volvia a
    quedar callada y le caia otro. Alguien apenas molesto terminaba muy
    molesto. Ahora `enviados` cuenta para toda la vida: uno y nunca mas."""
    est = _estado(remitente)
    est["ultimo_mensaje"] = _iso(_ahora())
    est["intentos"] = 0
    guardar_seguimientos()


def marcar_agendado(remitente):
    """Ya tiene cita: se acabaron los mensajes por iniciativa propia. Entra a la
    lista de Upstash porque si no, un deploy borraria la cita de la memoria y
    Valentina se pondria a insistirle que agende algo que YA agendo — justo a la
    persona con mas ganas de comprar."""
    est = _estado(remitente)
    est["agendo"] = True
    est["intentos"] = 0
    agregar_no_molestar(remitente, "agendo")


def marcar_cita_cancelada(remitente):
    """Cancelo: se anota que ya no tiene cita, pero NO se le vuelve a encender
    el seguimiento automatico. Ya entro a la lista de no-molestar al agendar y
    de esa lista no sale nadie. A quien agenda y cancela lo llama una persona,
    no un bot insistiendo."""
    est = _estado(remitente)
    est["agendo"] = False
    est["intentos"] = 0
    est["ultimo_mensaje"] = _iso(_ahora())
    guardar_seguimientos()


def detener_seguimientos(remitente):
    """Pidio que ya no le escriban (por el boton o escribiendolo). Es definitivo
    y sobrevive a los deploys: entra a la lista de Upstash. Sigue pudiendo
    escribirle a Valentina cuando quiera y ella le contesta normal; lo que se
    apaga es que ella arranque conversacion por su cuenta."""
    agregar_no_molestar(remitente, "pidio que no le escriban")
    enviar_whatsapp(remitente,
        "Listo, no te escribo mas 🙏 Si algun dia quieres retomar, aqui estoy: "
        "solo mandame un mensaje y con gusto te ayudo.")


# Frases con las que una persona pide que la dejen en paz ESCRIBIENDO, sin
# tocar el boton. Sin esto, escribirlo era peor que no escribir nada: el mensaje
# se iba al modelo (que contesta bonito y no apaga nada) y de paso reiniciaba la
# cuenta de seguimientos, o sea que pedir "ya no me escribas" garantizaba otro
# seguimiento. Aqui se atrapa ANTES del modelo.
#
# La lista es corta a proposito. El riesgo real no es que se escape uno —ese
# recibe un seguimiento mas y ya— sino callar a alguien que si quiere comprar:
# "no me interesa el de vida, quiero gastos medicos" NO debe apagar nada. Por
# eso no hay nada tipo "no me interesa" ni "no gracias" aqui.
FRASES_NO_ESCRIBIR = (
    "ya no me escribas", "ya no me escriban", "no me escribas mas",
    "no me escriban mas", "deja de escribirme", "dejen de escribirme",
    "no me vuelvas a escribir", "no me vuelvan a escribir",
    "no me contactes", "no me contacten", "no quiero que me contacten",
    "no me molestes", "no me molesten", "dejame en paz", "dejenme en paz",
    "borrame de la lista", "quitame de la lista", "sacame de la lista",
    "borra mi numero", "elimina mi numero", "date de baja", "dame de baja",
)
# Estas solo cuentan si son TODO el mensaje. "stop" suelto es una salida
# estandar, pero dentro de una frase larga puede ser cualquier cosa. Aqui NO va
# "no": Valentina pregunta cosas ("te paso los horarios?") y un "no" seco es una
# respuesta a la pregunta, no una baja.
PALABRAS_NO_ESCRIBIR = ("stop", "baja", "unsubscribe", "basta")


def _sin_acentos(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto)
                   if unicodedata.category(c) != "Mn")


def pide_que_no_le_escriban(texto):
    """True si la persona esta pidiendo por escrito que ya no le escriban."""
    limpio = _sin_acentos(texto).lower()
    limpio = re.sub(r"[^a-z0-9\s]", " ", limpio)
    limpio = " ".join(limpio.split())
    if not limpio:
        return False
    if limpio in PALABRAS_NO_ESCRIBIR:
        return True
    return any(f in limpio for f in FRASES_NO_ESCRIBIR)


def texto_primer_seguimiento():
    texto = (f"Hola de nuevo! Soy Valentina 😊 Se me quedo pendiente lo tuyo y "
             f"no quise dejarlo asi. Sigue en pie la llamada con {ASESOR_CORTO} "
             f"cuando tu puedas, son 10 o 15 minutos y sin compromiso.")
    if REDES_ASESOR:
        texto += (f"\n\nPor si quieres conocerlo antes:\n{REDES_ASESOR}")
    texto += "\n\nTe paso los horarios que tiene libres?"
    return texto


def enviar_seguimiento(remitente, numero):
    """Manda el seguimiento numero 1, 2 o 3.

    Devuelve "enviado", "sin_plantilla" (los de dia 2 y 3 cuando todavia no hay
    plantilla aprobada) o "error".

    El 1 cae dentro de la ventana de 24 h de Meta, asi que puede ser un mensaje
    normal con botones. El 2 y el 3 caen fuera y Meta SOLO acepta plantillas
    aprobadas: sin PLANTILLA_SEGUIMIENTO configurada no hay forma de mandarlos.
    """
    if numero == 1:
        r = _post_whatsapp({
            "messaging_product": "whatsapp",
            "to": remitente,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": texto_primer_seguimiento()},
                "action": {"buttons": [
                    {"type": "reply", "reply": {"id": "ver_horarios",
                                                "title": "Ver horarios"}},
                    {"type": "reply", "reply": {"id": "no_seguimiento",
                                                "title": "Ya no me escribas"}},
                ]},
            },
        })
        return "enviado" if r.status_code < 400 else "error"

    plantilla = PLANTILLA_SEGUIMIENTO
    if numero == 3 and PLANTILLA_SEGUIMIENTO_3:
        plantilla = PLANTILLA_SEGUIMIENTO_3
    if not plantilla:
        return "sin_plantilla"

    r = _post_whatsapp({
        "messaging_product": "whatsapp",
        "to": remitente,
        "type": "template",
        "template": {
            "name": plantilla,
            "language": {"code": PLANTILLA_IDIOMA},
        },
    })
    return "enviado" if r.status_code < 400 else "error"


def revisar_seguimientos():
    """Manda los seguimientos que ya tocan. Lo llama el despertador de
    UptimeRobot en cada ping (cada 5 min), que es la unica cosa que corre sola
    en este servicio: Render Free duerme el proceso y no hay cron."""
    # Si no se pudo leer la lista de no-molestar, no se manda NADA. Sin ella no
    # hay forma de saber quien pidio que lo dejaran en paz, y entre "se queda
    # sin su recordatorio" y "le escribo a alguien que dijo que no", la unica
    # equivocacion que se paga cara es la segunda. Se reintenta la lectura por
    # si fue una caida de un momento.
    if not no_molestar_cargada:
        cargar_no_molestar()
    if not no_molestar_cargada:
        print("Seguimientos en pausa: no se pudo leer la lista de no-molestar.")
        return 0

    ahora = _ahora()
    mandados = 0
    cambio = False

    for remitente, est in list(seguimientos.items()):
        if mandados >= MAX_ENVIOS_POR_TICK:
            break
        if remitente in no_molestar or est.get("detenido") or est.get("agendo"):
            continue

        ya = est.get("enviados", 0)
        if ya >= MAX_SEGUIMIENTOS:
            continue

        if ya == 0:
            desde = _desde_iso(est.get("ultimo_mensaje"))
            espera = timedelta(minutes=MINUTOS_PRIMER_SEGUIMIENTO)
        else:
            desde = _desde_iso(est.get("ultimo_envio"))
            espera = timedelta(minutes=MINUTOS_ENTRE_SEGUIMIENTOS)
        if not desde or ahora - desde < espera:
            continue

        try:
            resultado = enviar_seguimiento(remitente, ya + 1)
        except Exception as e:
            # Que se caiga el envio de UNA persona no debe dejar sin su
            # seguimiento a las demas que ya les tocaba en este mismo ping.
            print(f"Error mandando seguimiento a {remitente}:", e)
            resultado = "error"
        cambio = True

        if resultado == "sin_plantilla":
            # No hay plantilla aprobada todavia: se cierra la secuencia en vez
            # de reintentar cada 5 minutos para siempre.
            print(f"Seguimiento {ya + 1} para {remitente} omitido: falta "
                  f"PLANTILLA_SEGUIMIENTO (Meta no deja mensajes libres fuera "
                  f"de la ventana de 24 h).")
            est["enviados"] = MAX_SEGUIMIENTOS
            continue

        if resultado == "error":
            # Reintenta en el siguiente ping, pero no eternamente: un token
            # vencido no se va a arreglar solo y no queremos el log lleno.
            est["intentos"] = est.get("intentos", 0) + 1
            if est["intentos"] >= 3:
                print(f"Seguimiento {ya + 1} para {remitente} abandonado "
                      f"despues de 3 intentos fallidos.")
                est["enviados"] = MAX_SEGUIMIENTOS
            continue

        est["enviados"] = ya + 1
        est["ultimo_envio"] = _iso(ahora)
        est["intentos"] = 0
        mandados += 1
        # Ya gasto su unico mensaje por iniciativa propia: a la lista, para que
        # ni un deploy ni un contador reiniciado le den un segundo.
        if est["enviados"] >= MAX_SEGUIMIENTOS:
            agregar_no_molestar(remitente, "ya recibio su seguimiento")
        # Se le deja la nota al modelo para que si la persona contesta,
        # Valentina sepa que ya le escribio y no arranque de cero.
        historiales.setdefault(remitente, []).append({
            "role": "assistant",
            "content": f"[Seguimiento enviado: se le recordo la llamada con "
                       f"{ASESOR_NOMBRE}]",
        })

    if cambio:
        guardar_seguimientos()
    return mandados


# Primero la lista que sobrevive a los deploys, porque cargar_seguimientos()
# reconstruye fichas y _estado() necesita saber ya quien esta silenciado.
cargar_no_molestar()
cargar_seguimientos()


@app.route("/", methods=["GET"])
def home():
    # UptimeRobot pega aqui cada 5 min para que Render no se duerma; de paso
    # aprovechamos ese latido como reloj de los seguimientos. Es idempotente:
    # solo manda lo que ya cumplio su espera y lo marca enseguida.
    try:
        mandados = revisar_seguimientos()
    except Exception as e:
        print("Error revisando seguimientos:", e)
        mandados = 0
    return (f"Bot de WhatsApp (Meta Cloud API) — Valentina activa. Prompt V9. "
            f"Seguimientos enviados en este ping: {mandados}.")


@app.route("/cron/seguimientos", methods=["GET", "POST"])
def cron_seguimientos():
    """Mismo trabajo que el ping de la home, por si algun dia se quiere un
    despertador aparte (Render Cron, cron-job.org) sin tocar la home."""
    return {"enviados": revisar_seguimientos()}, 200


@app.route("/exportar", methods=["GET"])
def exportar_conversaciones():
    """Baja TODAS las conversaciones archivadas en un JSON.

    Se pide asi:  https://<tu-servicio>.onrender.com/exportar?clave=LA_CLAVE

    Son datos personales de gente real, por eso pide contrasena y por eso el
    endpoint no existe si no se configuro CLAVE_EXPORT en Render."""
    if not CLAVE_EXPORT:
        return {"error": "Exportar esta apagado: falta CLAVE_EXPORT en Render."}, 503
    if request.args.get("clave", "") != CLAVE_EXPORT:
        return {"error": "Clave incorrecta."}, 403

    try:
        datos = leer_conversaciones_archivadas()
    except Exception as e:
        print("Error exportando conversaciones:", e)
        return {"error": "No se pudo leer el archivo."}, 500

    cuerpo = json.dumps({
        "generado": _ahora().isoformat(),
        "total": len(datos),
        "conversaciones": datos,
    }, ensure_ascii=False, indent=2)
    return Response(
        cuerpo,
        mimetype="application/json; charset=utf-8",
        headers={"Content-Disposition":
                 "attachment; filename=conversaciones_valentina.json"},
    )


if __name__ == "__main__":
    print("\n  Bot WhatsApp (Meta) en http://localhost:5000  (webhook: /webhook)\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
