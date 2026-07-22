"""
Aromas del Valle — chatbot para WHATSAPP con la API OFICIAL de Meta (Cloud API)
------------------------------------------------------------------------------
Mismo cerebro que las versiones anteriores (Groq + Valentina). La diferencia es
que aqui hablamos DIRECTO con Meta (sin Twilio).

Con Meta hay DOS cosas que tu servidor debe hacer:

  1) VERIFICACION (una sola vez):  Meta manda un GET /webhook con un token y un
     "challenge". Tu devuelves el challenge si el token coincide. Asi Meta
     confirma que este servidor es tuyo.

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
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
WHATSAPP_TOKEN  = os.environ.get("WHATSAPP_TOKEN", "")     # token de acceso de Meta
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID", "")    # id del numero de prueba
VERIFY_TOKEN    = os.environ.get("VERIFY_TOKEN", "aromas123")  # lo inventas tu

GRAPH_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
Eres "Valentina", la asistente virtual de Aromas del Valle, una tostaduria de cafe
especializado que vende en linea. Eres amable, breve y entusiasta del cafe.

PRODUCTOS:
- Cafe Montana (250g) — $12 — tueste medio, notas de chocolate
- Cafe Volcan (250g) — $14 — tueste oscuro, intenso
- Cafe Amanecer (250g) — $13 — tueste claro, afrutado
- Sampler (3x100g) — $18 — uno de cada uno
- Suscripcion mensual — $30 — 2 bolsas + envio gratis

POLITICAS:
- Envio en 2 a 4 dias. Envio GRATIS en compras mayores a $40.
- Devoluciones dentro de 14 dias.
- Horario de atencion: lunes a viernes.
- Soporte humano: hola@aromasdelvalle.com

REGLAS:
- Solo respondes sobre Aromas del Valle. Si preguntan otra cosa, redirige con amabilidad.
- Si no sabes un dato, no lo inventes: ofrece el correo de soporte.
- Respuestas cortas (2 a 4 frases). Usa un emoji de cafe de vez en cuando.
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

        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + historial,
        )
        reply = resp.choices[0].message.content
        historial.append({"role": "assistant", "content": reply})

        enviar_whatsapp(remitente, reply)
    except Exception as e:
        print("Error procesando el mensaje:", e)

    # A Meta siempre le respondemos 200 rapido para que no reintente.
    return "ok", 200


# ---------------------------------------------------------------------------
# 4) ENVIAR un mensaje de vuelta por WhatsApp (llamada a la Graph API de Meta)
# ---------------------------------------------------------------------------
def enviar_whatsapp(destino, texto):
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
    return "Bot de WhatsApp (Meta Cloud API) de Aromas del Valle activo."


if __name__ == "__main__":
    print("\n  Bot WhatsApp (Meta) en http://localhost:5000  (webhook: /webhook)\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
