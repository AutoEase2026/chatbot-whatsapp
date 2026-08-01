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
import requests
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
# existe (el link viejo "jorge-arroyo/llamada" daba 404). Mientras tanto el
# link de abajo (LINK_AGENDA) usa el Cal.com de Enrique Ampudia SOLO PARA
# PROBAR que el flujo de agendamiento funciona end-to-end; el nombre que se
# presenta sigue siendo Jorge. Cuando Jorge tenga su propio Cal.com, cambia
# LINK_AGENDA en el .env (ASESOR_NOMBRE ya no necesitaria cambiar).
ASESOR_NOMBRE = os.environ.get("ASESOR_NOMBRE", "Jorge Arroyo")
ASESOR_CORTO = ASESOR_NOMBRE.split()[0]

# Link de agendamiento de Cal.com. Es la UNICA via para fijar la hora:
# Valentina no conoce la agenda del asesor, asi que si ella confirmara un
# horario podria chocar con algo ya ocupado. Cal.com lee ocupado/libre en
# tiempo real y solo ofrece huecos reales.
# Cambialo desde el .env con  LINK_AGENDA=https://cal.com/loquesea
LINK_AGENDA = os.environ.get("LINK_AGENDA")
if not LINK_AGENDA:
    raise RuntimeError(
        "Falta LINK_AGENDA en el archivo .env. Antes daba un default falso "
        "(cal.com/jorge-arroyo/llamada, 404) que Valentina llegaba a mandar "
        "a prospectos reales. Pon un link de Cal.com verificado, por ejemplo "
        "LINK_AGENDA=https://cal.com/enrique-ampudia-knhfq7/30min")

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
Primero cierras en la conversacion, DESPUES mandas el link. En ese orden: si
mandas el link antes de que la persona se comprometa, casi nadie lo abre.

1. Devuelve en una linea lo que entendiste.
2. Propon la llamada con dos opciones. Nunca preguntes "te interesa?".
   "Perfecto, con eso {ASESOR_CORTO} ya puede prepararte algo concreto. Te
   parece si te llama hoy en la tarde, o prefieres manana en la manana?"
3. Cuando diga cual prefiere, MANDA EL LINK:
   "Genial. Aqui eliges la hora exacta que te acomode y te llega el
   recordatorio: {LINK_AGENDA}"
4. Cierra corto: "Cualquier duda mientras tanto, aqui estoy 😊"

EL LINK ES LA UNICA FORMA DE FIJAR LA HORA. Nunca confirmes tu misma un horario
concreto ("{ASESOR_CORTO} te llama a las 5"): no conoces su agenda y podrias
chocar con algo ya ocupado. El link muestra solo sus huecos reales.

Si la persona ya dijo cuando quiere ("agendame manana a las 3", "en la tarde
me va bien"), NO le vuelvas a preguntar la preferencia: ya te la dio. Pasa
directo al link, validando lo que pidio.
"Va, manana en la tarde. Aqui eliges la hora exacta y te llega el recordatorio:
{LINK_AGENDA}"

No pidas nombre, correo ni telefono: el link se los pide al agendar.
Si dice que ya agendo, agradece y cierra. No pidas nada mas.
Si no define preferencia, ofrece un rango concreto ("manana entre 10 y 12?").
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
    .replace("{LINK_AGENDA}", LINK_AGENDA) \
    .replace("{ASESOR_CORTO_MAYUS}", ASESOR_CORTO.upper()) \
    .replace("{ASESOR_CORTO}", ASESOR_CORTO) \
    .replace("{ASESOR}", ASESOR_NOMBRE)

# Memoria por persona (se borra al reiniciar; en produccion iria a una base de datos)
historiales = {}

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
        texto = msg.get("text", {}).get("body", "")   # el texto que escribio

        # Pensar la respuesta con memoria por persona
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
# 4) ENVIAR un mensaje de vuelta por WhatsApp (llamada a la Graph API de Meta)
# ---------------------------------------------------------------------------
def enviar_whatsapp(destino, texto):
    # Meta rechaza un body vacio con "The parameter text.body is required".
    # Ultima linea de defensa: aqui no deberia llegar nunca vacio.
    texto = (texto or "").strip()
    if not texto:
        print("Se intento enviar un mensaje vacio a WhatsApp. Cancelado.")
        return

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": destino,
        "type": "text",
        "text": {"body": texto},
    }
    r = requests.post(GRAPH_URL, headers=headers, json=payload, timeout=20)
    if r.status_code >= 400:
        print("Error al enviar a WhatsApp:", r.status_code, r.text)


@app.route("/", methods=["GET"])
def home():
    return "Bot de WhatsApp (Meta Cloud API) — Valentina activa. Prompt V5."


if __name__ == "__main__":
    print("\n  Bot WhatsApp (Meta) en http://localhost:5000  (webhook: /webhook)\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
