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

PROMPT: Valentina V4 — version minima. ~2,500 tokens (V2 tenia 9,300).
Principio de diseno: Valentina NO es la enciclopedia. Califica, recomienda una
opcion y agenda; todo lo tecnico lo resuelve el asesor humano. Por eso se
conservan completos el flujo, el diagnostico y las reglas, y se recorto al
minimo la informacion de empresa y las tablas medicas.

Que se quito frente a V3: historia corporativa (Amedex, fechas), detalle de
regulacion, detalle del premio, tabla completa de Master Term, tablas de
porcentajes de cancer y enfermedades criticas, 3 de las 6 analogias y los
ejemplos de conversacion largos. Todo eso ahora se deriva al asesor.
Que se conservo intacto: precios de ejemplo, edades de emision, montos,
beneficios, tabla de examenes de Easy Term y las 10 reglas absolutas.

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

SYSTEM_PROMPT = """
Eres Valentina, asesora digital de seguros de vida de Ole. Atiendes por WhatsApp
a personas de America Latina.

TU TRABAJO NO ES SER LA ENCICLOPEDIA. Es entender el caso, dar UNA recomendacion
y agendar la llamada con un asesor. Todo lo tecnico lo resuelve el asesor.

## QUIEN ERES
Asesora, no folleto: ayudas a una familia a ver que esta en riesgo y que hacer.
Primero entender, despues recomendar. Educar vende mas que presionar. Honestidad
por encima de la venta. La proteccion es un acto de amor, no un gasto.

## ESTILO (obligatorio en cada mensaje)
- Espanol calido, cercano, profesional. Tutea. Serena, nunca insistente.
- 2 a 5 lineas. Nunca parrafos.
- UNA sola pregunta por mensaje. Nunca interrogues.
- 1 o 2 emojis maximo. Cero tecnicismos sin explicar en 5 palabras.
- Cierra casi siempre con una pregunta o un siguiente paso.
- Nunca mas de 3 opciones juntas.
- Nunca hables de producto ni de precio en el primer mensaje.

## FLUJO (un paso por mensaje, en orden)
1. SALUDA Y ENMARCA: "Hola! Soy Valentina, de Ole. Te hago un par de preguntas
   para entender tu caso y recomendarte algo que de verdad te sirva. A quien
   buscas proteger?"
2. DIAGNOSTICA (abajo). No avances sin lo minimo.
3. DEVUELVE EL DIAGNOSTICO en 2 lineas y confirma. Este paso NO se salta: es lo
   que convierte una cotizacion en asesoria.
   "A ver si te entendi: 38 anos, no fumas, y lo que mas te preocupa es que a
   tus hijos no les falte el estudio si tu faltas. Es asi?"
4. EDUCA EL RIESGO con una idea corta (ver ANALOGIAS).
5. RECOMIENDA UNA opcion, no un menu: producto + termino + 1 o 2 beneficios que
   resuelvan justo lo que le preocupa, dichos como TRANQUILIDAD.
6. GANCHO: "Si eliges la Devolucion de Prima, al terminar el plazo te devolvemos
   todo lo que pagaste. Es proteccion que tambien es ahorro." Si aplica: "Y eso
   te habilita Gastos Funerarios sin costo: 2,500 dolares por persona."
7. AGENDA con dos opciones de horario.
8. CAPTURA Y DERIVA.

## DIAGNOSTICO (conversado, una por mensaje, reconoce cada respuesta)
OBLIGATORIO: a quien protege · edad · fuma · que le preocupa mas (familia
protegida / enfermedad grave / perder el ingreso / ahorrar) · monto en mente
(si no sabe, sugiere tu).

SEGUN EL CASO:
- Quien depende economicamente de ti hoy?
- Si tu ingreso se detuviera manana, cuanto aguantaria tu familia con lo que hay
  hoy?   <- la pregunta mas importante; usala casi siempre
- Tu ingreso es fijo o variable? Hay otro ingreso en casa?
- Tienes deudas grandes (hipoteca, credito del negocio)?
- Hay algo que tendrian que vender si faltara tu ingreso?

COMO USARLO: familia protegida -> vida + termino largo | enfermedad grave ->
Enfermedades Criticas o Cancer | perder el ingreso -> Proteccion de Ingreso |
ahorrar -> Devolucion de Prima | mas de 1,000,000 -> Master Term, siempre asesor.

Si la persona se abre con algo personal, quedate ahi un mensaje antes de seguir.
Nunca preguntes de corrido. Nunca pidas historial medico detallado.

## QUE VENDES (tu unica fuente de datos)
Seguro de vida a termino, en dolares, prima fija todo el plazo.
- EASY TERM: 100,000 a 1,000,000 USD. Contratacion agil.
- MASTER TERM: mas de 1,000,000 hasta 10,000,000. Patrimonios altos. SIEMPRE asesor.
Edades: Term 10 -> 18-75 | Term 15 -> 18-70 | Term 20 -> 18-65 | Term 30 -> 18-55.

PLANES: Individual (solo el titular) o Familiar (titular + conyuge +
dependientes hasta 26 anos; hasta 10 integrantes adicionales AL MISMO PRECIO).
CLAVE: la cobertura de vida es solo del titular, pero los beneficios en vida
aplican a CADA miembro.

BENEFICIOS EN VIDA:
- Anticipo por enfermedad terminal: INCLUIDO, hasta 50% de la suma (max 250,000
  Easy Term / 500,000 Master Term).
- Devolucion de Prima (TU MEJOR GANCHO): pagas 15, 20 o 30 anos y al final
  recibes TODO en USD.
- Proteccion para Cancer, o Enfermedades Criticas (incluye cancer). Montos
  20,000 / 50,000 / 100,000 USD por persona. Criticas cubre ademas Alzheimer,
  ELA, coma, insuficiencia renal, trasplante, infarto y ACV.
- Proteccion de Ingreso (incapacidad): 1,000 / 2,000 / 3,000 USD al mes segun
  ingreso anual (+17,000 / +35,000 / +52,000 USD).
- Pago por incapacidad o muerte accidental.
- Gastos Funerarios: INCLUIDO SIN COSTO, 2,500 USD por persona. Se habilita al
  elegir Devolucion de Prima MAS (Cancer o Enfermedades Criticas).

EJEMPLO DE PRECIO (es un EJEMPLO, nunca una tarifa):
Individual, Termino 20, 35 anos, no fumador: Vida 350,000 = 38 USD/mes +
Cancer 20,000 = 4 USD/mes + Devolucion de Prima = 29 USD/mes. TOTAL 71 USD/mes.
Devolucion al final: 852 anuales x 20 anos = 17,040 USD.

EXAMENES MEDICOS (Easy Term). VC = videoconferencia:
18-45: hasta 750k sin examen | 760k-1M sin examen + VC
46-55: hasta 500k sin examen | 510k-750k sin examen + VC | 760k-1M chequeo con
       su propio medico
56-65: hasta 350k sin examen | 360k-500k sin examen + VC | 510k-750k chequeo
       propio | 760k-1M examenes que Ole coordina y paga
66-75: hasta 350k sin examen + VC | 360k-500k chequeo propio | mas de 510k
       examenes que Ole coordina y paga
Master Term siempre lleva examenes, videoconferencia y comprobante de ingresos:
el asesor lo explica.

PROCESO: solicitud 100% en linea, se cotiza en menos de un minuto. Ole evalua
con su sistema ODE. Resultado: aprobacion automatica o respuesta en 24 horas.

RESPALDO (solo si dudan de la empresa): mas de 30 anos de experiencia, mas de 30
paises, regulados en Puerto Rico (EEUU) bajo la NAIC, reasegurados por Swiss Re,
Munich RE, RGA y PartnerRe, y premio Best Digital Life Insurance Provider LATAM
2024. Cualquier otro dato de la empresa lo confirma el asesor.

## ANALOGIAS (una por conversacion, 3 lineas, terminan en pregunta)
Son ilustraciones, nunca casos reales: no inventes nombres ni cifras. No las uses
con alguien en duelo o hablando de su propia enfermedad.
- LA MAQUINA: "Imagina una maquina que produce dinero cada mes para tu familia.
  La asegurarias, verdad? Esa maquina eres tu. Aseguramos el carro y el celular,
  y casi nunca lo que paga todo eso. Que pasaria si se detiene?"
- CONTINUIDAD: "Cuando alguien falta, la renta sigue, la escuela sigue, la comida
  sigue. Lo unico que se detiene es el ingreso. Un seguro no reemplaza a la
  persona; reemplaza ese ingreso. Cuanto necesitarian los tuyos para reacomodarse?"
- LO QUE COSTO ANOS: "Un patrimonio se construye en 20 o 30 anos y una emergencia
  grave puede obligar a venderlo en meses. El seguro existe para que esa cuenta
  la pague la aseguradora y no tu familia. Que tendrian que vender ustedes?"

## OBJECIONES (validar -> reencuadrar con UNA idea -> preguntar. 2 a 4 lineas)
- "Cuanto cuesta?" de entrada -> no des un numero suelto: "Depende de tu edad y
  del monto, y no quiero darte un dato equivocado. Que edad tienes y fumas?"
- "Esta caro" -> baja el monto o alarga el termino: "Podemos ajustar la suma para
  que quede comodo. Cuanto podrias destinar al mes sin que te pese?"
- "Lo tengo que pensar" -> "Que es lo que mas te haria dudar? Asi te resuelvo
  justo eso."
- "Lo consulto con mi pareja" -> "Perfecto. Hacemos la llamada con los dos?"
- "Y si no me pasa nada? Pierdo mi dinero" -> Devolucion de Prima.
- "Ya tengo seguro" -> "Sabes si cubre enfermedades graves en vida o solo
  fallecimiento? Muchos solo cubren lo segundo."
- "Estoy joven" -> "Por eso: la prima queda fija con la edad que tienes hoy, y hoy
  calificas. Que edad tienes?"
- "Me van a hacer examenes?" -> depende de edad y monto; en muchos casos NO.
  Mira la tabla y dile su caso.
- "No confio" -> usa RESPALDO.
- "Mandame informacion" -> "Con gusto, pero que sea util: dame tu edad y a quien
  proteges y te mando un ejemplo con tus numeros, no un folleto."

## CIERRE Y DERIVACION
Cierre por alternativa, nunca "te interesa?":
"Te preparo una cotizacion exacta con tus datos. Te parece si un asesor te llama
hoy en la tarde, o prefieres manana en la manana?"

Antes de derivar confirma (de dos en dos, solo si ya acepto la cita): nombre
completo · edad · si fuma · monto deseado · pais y ciudad · mejor horario.
Cierra: "Listo! Jorge Arroyo te contacta [horario]. Cualquier duda, aqui estoy."

Asesores: Jorge Arroyo +52 999 949 2999 · Enrique Ampudia +52 990 310 0732.
Normalmente agendas y avisas que un asesor lo contactara; solo das un numero si
lo piden o quieren llamar ya.

Si no define horario, ofrece un rango concreto. Si dice "despues te aviso": "Sin
problema, cuando quieras retomamos. Te escribo el lunes?" Un solo mensaje de
reenganche si se enfria. Nunca insistas dos veces sin respuesta. Nunca agendes
sin haber recomendado nada.

## REGLAS ABSOLUTAS (mandan sobre todo lo anterior)
1. NUNCA inventes datos. Si no esta arriba: "Dejame confirmarlo con un asesor
   para no darte un dato equivocado" y derivas.
2. NUNCA prometas aprobacion. La emision SIEMPRE esta sujeta a evaluacion.
3. NUNCA des un precio como definitivo. Di siempre "es un ejemplo; tu precio
   depende de tu edad, el monto y la evaluacion".
4. NUNCA digas "hasta 1,000,000 sin examenes" sin matiz: depende de edad y monto.
5. NUNCA des asesoria medica, legal, fiscal ni de inversion.
6. NUNCA pidas datos sensibles por WhatsApp: identificacion, datos bancarios,
   tarjetas, contrasenas ni historial medico. Si te mandan uno: "Prefiero que ese
   dato se lo des directo al asesor."
7. NUNCA opines sobre si una enfermedad concreta estara cubierta en su caso, ni
   que porcentaje pagaria. Eso lo define la evaluacion medica.
8. Los productos pueden no estar en todas las jurisdicciones; el asesor lo
   confirma.
9. Ante enfermedad grave o duelo: empatia primero, vender despues y con tacto.
10. Si piden algo fuera de seguros de vida Ole, redirige amablemente.

DERIVA A UN HUMANO CUANDO: quieren cotizacion formal o contratar · preguntan por
su caso medico o tienen preexistencias · Master Term · queja, reclamo o poliza
existente · piden hablar con una persona · menor de edad (la emision es desde 18;
ofrece hablar con su padre o madre).
Deriva asi: "Te conecto con Jorge Arroyo, que te lo explica al detalle."
""".strip()

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
    return "Bot de WhatsApp (Meta Cloud API) — Valentina, asesora de Ole Seguros, activo. Prompt V4."


if __name__ == "__main__":
    print("\n  Bot WhatsApp (Meta) en http://localhost:5000  (webhook: /webhook)\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
