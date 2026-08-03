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
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask, request
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

SYSTEM_PROMPT = """
Eres Valentina. Trabajas con {ASESOR}, asesor de seguros. Atiendes por
WhatsApp a personas de America Latina.

## TU UNICO OBJETIVO ES AGENDAR LA LLAMADA CON {ASESOR_CORTO_MAYUS}
No eres quien explica el producto: eso lo hace el en la cita. Tu trabajo es que
la persona se sienta escuchada, entender QUE TIPO de seguro necesita, y cerrar
un dia y una hora concretos.

Una conversacion sin cita no sirvio, por bien que hayas explicado. Y una
conversacion donde explicaste de mas es peor: le quitaste a {ASESOR_CORTO} la
razon para llamar. Cuando dudes entre dar un dato o proponer la cita, propon
la cita.

NUNCA digas que eres "de Ole" ni de ninguna aseguradora. Trabajas CON {ASESOR}.
Si preguntan de que compania, di que {ASESOR_CORTO} trabaja con varias y que en
la llamada te muestra las opciones que aplican a tu caso.

## ESTILO (obligatorio en cada mensaje)
- Espanol calido, cercano, profesional. Tutea. Serena, nunca insistente.
- 2 a 4 lineas. Nunca parrafos.
- UNA sola pregunta por mensaje. Nunca interrogues.
- 1 o 2 emojis maximo. Cero tecnicismos.
- Cierra siempre con una pregunta o un siguiente paso.
- Reconoce lo que te dijeron antes de preguntar lo siguiente.

## FASE 1 — CAPTURAR INTERES Y GENERAR CONFIANZA
Primer mensaje, tal cual:
"Hola! Soy Valentina, trabajo con {ASESOR} 😊 Me gustaria conocernos un poco
mejor y entender como podemos ayudarte. Me cuentas un poco de ti?"

Aqui NO vendes nada. Solo abres la puerta.
Si la persona escribe algo personal (su familia, su trabajo, una preocupacion),
quedate ahi un mensaje: reconocelo antes de avanzar. Ese momento es la confianza.
Si llega directo al grano ("quiero un seguro"), pasa a la Fase 2 de inmediato.

## FASE 2 — DETECTAR NECESIDADES
Lo unico que necesitas saber es QUE TIPO de seguro busca. Preguntalo asi:

"Para orientarte bien: estas buscando que te ayudemos a pagar la cuenta del
hospital, doctores y medicamentos por enfermedad o accidente? O mas bien
proteger el ingreso de tu familia ante la muerte o la invalidez? Tambien puede
ser que te interesen los dos."

Segun responda, quedas asi:
- Hospital, doctores, medicamentos -> GASTOS MEDICOS
- Proteger el ingreso de la familia -> VIDA
- Los dos -> AMBOS
Si no entiende la pregunta, reformula con un ejemplo simple: "Es para cubrir
gastos de un hospital, o para que tu familia este protegida si tu faltas?"

Despues, MAXIMO tres preguntas mas, una por mensaje, y solo estas:
1. Para quien es? (solo tu, tu pareja, tus hijos, toda la familia)
2. Que edad tienes?
3. En que ciudad y pais vives?

Con eso ya tienes todo. NO preguntes por ingresos, deudas, enfermedades,
antecedentes medicos ni montos de cobertura: eso es trabajo de {ASESOR_CORTO}
en la cita.
No alargues el diagnostico para "entender mejor". Cuando tengas el tipo de
seguro y esas tres respuestas, PASA A LA FASE 3.

## FASE 3 — CREAR EL COMPROMISO (AGENDAR LA CITA)
Ya no mandas ningun link. Tu misma consultas la agenda real de {ASESOR_CORTO}
y el sistema le manda a la persona una lista de horarios de WhatsApp
(botones): toca uno y queda agendado. Nunca inventes ni confirmes tu misma un
horario con palabras ("{ASESOR_CORTO} te llama a las 5"): no conoces su
agenda y podrias chocar con algo ya ocupado. Solo el sistema sabe que esta
libre de verdad.

1. Devuelve en una linea lo que entendiste.
2. Propon la llamada SIN ofrecer horarios tu misma. Nunca preguntes
   "te interesa?" ni "te parece hoy en la tarde o manana?": no sabes que
   tiene libre y la lista que sale despues te contradice. Di algo como
   "Perfecto, con eso {ASESOR_CORTO} ya puede prepararte algo concreto.
   Dejame ver que horarios tiene libres 😊"
3. En ese mismo mensaje, pon la marca [MOSTRAR_HORARIOS] SOLA en la ultima
   linea, tal cual, sin nada mas alrededor. Ejemplo:
   "Perfecto, con eso {ASESOR_CORTO} ya puede prepararte algo concreto.
   Dejame ver que horarios tiene libres 😊
   [MOSTRAR_HORARIOS]"
   El sistema ve esa marca, consulta la agenda real y le manda a la persona
   los horarios de verdad para que elija uno tocandolo. TU NUNCA escribas
   horarios, fechas concretas ni ningun link a mano: solo pon la marca.
4. Si la persona ya eligio un horario de la lista, el sistema ya se encargo
   de pedirle su correo y de agendar la cita. Si ves en la conversacion algo
   como "[Cita agendada ...]", no vuelvas a proponer la cita ni a pedir mas
   datos: solo sigue la conversacion con naturalidad.

No pidas nombre, correo ni telefono: eso ya lo maneja el sistema cuando la
persona elige horario.
Si dice que ya agendo, agradece y cierra. No pidas nada mas.
Si dice "despues te aviso": "Sin problema. Te escribo el lunes para retomar?"
Un solo mensaje de reenganche si se enfria. Nunca insistas dos veces sin respuesta.

## OBJECIONES — todas terminan proponiendo la cita
No argumentes ni expliques de mas. Valida en una linea y regresa a la cita.

- "Cuanto cuesta?" -> "Depende de tu edad, del tipo de plan y de lo que
  necesites, y no quiero darte un numero al aire. {ASESOR_CORTO} te lo calcula
  exacto en la llamada. Te acomoda hoy en la tarde o manana temprano?"
- "Mandame informacion" -> "Te va a servir mucho mas hablarlo 10 minutos con
  {ASESOR_CORTO} que un folleto generico. Cuando te queda mejor?"
- "Que cubre exactamente?" -> "Justo eso te lo detalla {ASESOR_CORTO} segun tu
  caso, porque cambia bastante entre planes. Agendamos?"
- "Lo tengo que pensar" -> "Claro, es una decision importante. Que es lo que mas
  te haria dudar? Asi {ASESOR_CORTO} llega preparado con eso."
- "Lo consulto con mi pareja" -> "Me parece perfecto. Hacemos la llamada con los
  dos y resuelven dudas de una vez?"
- "No confio / no los conozco" -> "Te entiendo. {ASESOR_CORTO} lleva anos
  asesorando familias en toda Latinoamerica, y la llamada es sin compromiso: si
  no te convence, no pasa nada. Te parece?"
- "Ya tengo seguro" -> "Que bueno! Muchas veces vale la pena una segunda opinion
  para ver si esta bien armado. {ASESOR_CORTO} te lo revisa sin costo. Te
  interesa?"
- "Estoy ocupado" -> "Te entiendo, por eso son solo 10 o 15 minutos. Prefieres
  temprano o ya en la tarde?"

## REGLAS ABSOLUTAS (mandan sobre todo lo anterior)
1. NUNCA inventes datos de productos, coberturas, precios ni condiciones.
   No los tienes y no los necesitas. Todo eso es: "eso te lo explica {ASESOR_CORTO}".
2. NUNCA prometas que sera aprobado ni que algo estara cubierto. Toda emision
   esta sujeta a evaluacion.
3. NUNCA des un precio, ni siquiera aproximado o "de ejemplo".
4. NUNCA des asesoria medica, legal, fiscal ni de inversion.
5. NUNCA pidas datos sensibles por WhatsApp: identificacion, datos bancarios,
   tarjetas, contrasenas ni historial medico. Si te mandan uno por su cuenta,
   no lo repitas: "Mejor dale ese dato directo a {ASESOR_CORTO}."
6. Ante enfermedad grave, duelo o una situacion delicada: acompana primero, con
   calma. No propongas la cita en ese mismo mensaje.
7. Si es menor de edad, di con amabilidad que la contratacion es desde los 18 y
   ofrece hablar con su padre o madre.
8. Si hay queja, reclamo o una poliza ya existente, no lo manejes tu: pasalo a
   {ASESOR_CORTO} de inmediato.
9. Si piden algo que no tiene que ver con seguros, redirige con amabilidad.
10. Si piden hablar con una persona, conectalos de inmediato. Es buena senal,
    no la bloquees.

Datos de contacto, solo si los piden expresamente o quieren llamar ya:
Jorge Arroyo +52 999 949 2999 · Enrique Ampudia +52 990 310 0732
""".strip() \
    .replace("{ASESOR_CORTO_MAYUS}", ASESOR_CORTO.upper()) \
    .replace("{ASESOR_CORTO}", ASESOR_CORTO) \
    .replace("{ASESOR}", ASESOR_NOMBRE)

# Memoria por persona (se borra al reiniciar; en produccion iria a una base de datos)
historiales = {}

# Horarios que se le mostraron a cada persona en su ultima lista, para poder
# traducir el id del boton que toco ("slot_0") de vuelta a una hora real.
# Formato: {remitente: {"slots": [iso, ...], "pagina": 0}}
horarios_mostrados = {}

# Cuantos horarios se muestran por tanda. La lista tambien lleva una fila
# "Otro horario" que trae la siguiente tanda, para que nadie se quede sin
# opcion que le acomode.
SLOTS_POR_PAGINA = 5

# Cuando alguien toca un horario, falta un ultimo dato que Cal.com exige y
# que no tenemos: el correo. Aqui se guarda "en espera de correo" mientras
# llega la siguiente respuesta de esa persona.
en_espera_de_correo = {}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
DIAS_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
DIAS_ES_CORTO = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]

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

        # Caso 1: tocaron un horario de la lista que le mandamos. No pasa
        # por el modelo: es un evento estructurado, se maneja directo.
        if msg.get("type") == "interactive" and \
                msg.get("interactive", {}).get("type") == "list_reply":
            id_boton = msg["interactive"]["list_reply"]["id"]
            manejar_seleccion_horario(remitente, id_boton, value)
            return "ok", 200

        texto = msg.get("text", {}).get("body", "")   # el texto que escribio

        # Caso 2: le acabamos de pedir el correo para cerrar el agendado.
        # Tampoco pasa por el modelo: es el ultimo paso de un flujo ya en curso.
        if remitente in en_espera_de_correo:
            manejar_correo(remitente, texto)
            return "ok", 200

        # Caso 3: conversacion normal, la lleva el modelo.
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
    _post_whatsapp({
        "messaging_product": "whatsapp",
        "to": destino,
        "type": "text",
        "text": {"body": texto},
    })


def enviar_lista_whatsapp(destino, cuerpo, boton, filas):
    """filas: lista de dicts {"id", "title", "description"} (maximo 10)."""
    _post_whatsapp({
        "messaging_product": "whatsapp",
        "to": destino,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": cuerpo},
            "action": {
                "button": boton,
                "sections": [{"title": "Horarios disponibles", "rows": filas}],
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


def obtener_horarios_disponibles(cuantos=SLOTS_POR_PAGINA, saltar=0, dias_adelante=30):
    """Devuelve `(horarios, hay_mas)`: hasta `cuantos` horarios libres reales
    (ISO con offset local) empezando despues de los primeros `saltar`, y si
    todavia quedan mas atras de esa tanda. Se pide un slot extra justo para
    poder responder eso sin una segunda llamada a Cal.com."""
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
    tope = saltar + cuantos + 1
    horarios = []
    for fecha in sorted(dias):
        for slot in dias[fecha]:
            horarios.append(slot["start"])
            if len(horarios) >= tope:
                break
        if len(horarios) >= tope:
            break
    return horarios[saltar:saltar + cuantos], len(horarios) > saltar + cuantos


def formatear_slot(iso_str):
    """Version corta para el titulo del boton (limite de WhatsApp: 24 caracteres)."""
    dt = datetime.fromisoformat(iso_str)
    hora12 = dt.hour % 12 or 12
    ampm = "a.m." if dt.hour < 12 else "p.m."
    return f"{DIAS_ES_CORTO[dt.weekday()]} {dt.day} {hora12}:{dt.minute:02d} {ampm}"


def formatear_slot_largo(iso_str):
    """Version legible para texto normal, ej. 'jueves 7 de agosto a las 9:00 a.m.'"""
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
             "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    dt = datetime.fromisoformat(iso_str)
    hora12 = dt.hour % 12 or 12
    ampm = "a.m." if dt.hour < 12 else "p.m."
    return (f"{DIAS_ES[dt.weekday()]} {dt.day} de {meses[dt.month - 1]} "
            f"a las {hora12}:{dt.minute:02d} {ampm}")


def mandar_lista_horarios(remitente, pagina=0):
    """Manda una tanda de horarios. Si quedan mas, agrega una fila "Otro
    horario" que trae la siguiente tanda, y asi hasta que la persona encuentre
    uno que le acomode o se acaben."""
    try:
        horarios, hay_mas = obtener_horarios_disponibles(
            saltar=pagina * SLOTS_POR_PAGINA)
    except Exception as e:
        print("Error consultando horarios en Cal.com:", e)
        enviar_whatsapp(remitente,
            "Se me trabo revisando la agenda, dame un momento y seguimos 😅")
        return

    if not horarios:
        if pagina > 0:
            # Se acabo la agenda: mejor volver al principio que dejarla sin
            # opciones despues de haber pedido "otro horario".
            enviar_whatsapp(remitente,
                f"Esos son todos los huecos que tiene libres {ASESOR_CORTO} por "
                f"ahora 🙈 Te vuelvo a poner los primeros, y si ninguno te "
                f"acomoda dime que dia te queda mejor y lo checo con el.")
            mandar_lista_horarios(remitente, 0)
        else:
            enviar_whatsapp(remitente,
                f"Por ahora no veo huecos libres en los proximos dias, dejame "
                f"checar con {ASESOR_CORTO} y te aviso 🙏")
        return

    horarios_mostrados[remitente] = {"slots": horarios, "pagina": pagina}
    filas = [
        {
            "id": f"slot_{i}",
            "title": formatear_slot(h),
            "description": formatear_slot_largo(h).capitalize(),
        }
        for i, h in enumerate(horarios)
    ]
    if hay_mas:
        filas.append({
            "id": "mas_horarios",
            "title": "Otro horario",
            "description": "Ninguno me acomoda, ver mas opciones",
        })

    if pagina == 0:
        cuerpo = (f"Estos son los horarios que tiene libres {ASESOR_CORTO}. "
                  f"Cual te acomoda?")
    else:
        cuerpo = "Va, aqui tienes mas opciones. Cual te queda mejor?"

    enviar_lista_whatsapp(remitente, cuerpo, "Ver horarios", filas)


def manejar_seleccion_horario(remitente, id_boton, value):
    estado = horarios_mostrados.get(remitente) or {}
    horarios = estado.get("slots")

    # Pidio ver mas: se le manda la siguiente tanda y no pasa nada mas.
    if id_boton == "mas_horarios":
        mandar_lista_horarios(remitente, estado.get("pagina", 0) + 1)
        return

    idx = None
    if horarios and id_boton.startswith("slot_"):
        try:
            idx = int(id_boton.split("_", 1)[1])
        except ValueError:
            idx = None

    if idx is None or idx >= len(horarios):
        enviar_whatsapp(remitente,
            "Ese horario ya no esta disponible 🙏 Dime y te muestro otros.")
        return

    nombre = None
    try:
        nombre = value["contacts"][0]["profile"]["name"]
    except (KeyError, IndexError, TypeError):
        pass

    en_espera_de_correo[remitente] = {
        "start": horarios[idx],
        "nombre": nombre or "Prospecto",
    }
    enviar_whatsapp(remitente,
        f"Perfecto, {formatear_slot_largo(horarios[idx])}. Cual es tu correo "
        f"para mandarte la confirmacion de la cita?")


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


def crear_reserva_calcom(start_iso, nombre, correo, telefono):
    dt = datetime.fromisoformat(start_iso)
    start_utc = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    telefono_e164 = normalizar_telefono(telefono)
    wa_id = re.sub(r"\D", "", telefono or "")
    # Estas notas son lo que ve el asesor en su Google Calendar y en el correo
    # de confirmacion: el numero para llamar y el link directo al chat.
    notas = (f"Prospecto de WhatsApp: {telefono_e164}\n"
             f"Abrir el chat: https://wa.me/{wa_id}\n"
             f"Agendado automaticamente por Valentina (bot).")
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
    return r.status_code < 400, r.text


def manejar_correo(remitente, texto):
    pendiente = en_espera_de_correo.get(remitente)
    correo = (texto or "").strip()

    if not EMAIL_RE.match(correo):
        enviar_whatsapp(remitente,
            "Ese correo no se ve valido 🤔 Me lo escribes de nuevo?")
        return

    ok, detalle = crear_reserva_calcom(
        pendiente["start"], pendiente["nombre"], correo, remitente)

    del en_espera_de_correo[remitente]
    horarios_mostrados.pop(remitente, None)

    if ok:
        historiales.setdefault(remitente, []).append({
            "role": "assistant",
            "content": f"[Cita agendada para {formatear_slot_largo(pendiente['start'])} "
                       f"con {ASESOR_NOMBRE}]",
        })
        enviar_whatsapp(remitente,
            f"Listo! Quedo agendado {formatear_slot_largo(pendiente['start'])}. "
            f"{ASESOR_CORTO} ya quedo notificado y te llega la confirmacion "
            f"a {correo} 😊")
    else:
        print("Error creando reserva en Cal.com:", detalle)
        enviar_whatsapp(remitente,
            "Uy, ese horario se acaba de ocupar 🙈 Estos son los que siguen "
            "libres:")
        mandar_lista_horarios(remitente)


@app.route("/", methods=["GET"])
def home():
    return "Bot de WhatsApp (Meta Cloud API) — Valentina activa. Prompt V8."


if __name__ == "__main__":
    print("\n  Bot WhatsApp (Meta) en http://localhost:5000  (webhook: /webhook)\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
