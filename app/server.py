"""Flask service — ARCH-02 §7. Routes only; no business logic here.

Routes: POST /api/chat, GET /healthz, GET /widget.js, GET /widget.html,
POST /telegram/webhook, admin endpoints (token-authenticated).
Run with gunicorn (1 worker on Render free): see render.yaml.
"""

import json
import os
import time

from flask import Flask, jsonify, request, send_from_directory

from app import store as store_mod
from app.chat import answer_question
from app.logdb import LogDB
from app.rate_limit import TokenBucket
from app.telegram import send_message, verify_webhook_secret

MAINTENANCE_AR = "الخدمة متوقفة مؤقتاً، يرجى التواصل مع المتجر مباشرة."
MAINTENANCE_EN = "The service is temporarily unavailable, please contact the store directly."

_CONFIG_CACHE = {}


def _load_config_yaml(path):
    """Minimal YAML subset reader (the template's config.yaml uses only
    simple key: value mappings and one nesting level). Avoids adding
    PyYAML as a 7th production dependency for one small file."""
    import re
    config = {}
    current_section = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            if stripped.startswith("- "):
                continue
            if ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # strip trailing inline comment (only when preceded by space)
            if " #" in value:
                value = value.split(" #")[0].strip().strip('"').strip("'")
            if indent == 0:
                current_section = None
                config[key] = _coerce(value)
                if value == "":
                    current_section = key
                    config[key] = {}
            else:
                if current_section is not None:
                    config[current_section][key] = _coerce(value)
    return config


def _coerce(value):
    if value == "":
        return ""
    if value in ("true", "false"):
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def create_app(config_path=None, client=None, collection=None):
    app = Flask(__name__)
    config_path = config_path or os.path.join(
        os.path.dirname(__file__), "..", "config.yaml")
    config = _load_config_yaml(config_path)
    app.config["RAG_CONFIG"] = config
    client = client or store_mod.get_client(config)
    collection = collection or store_mod.ensure_collection(client, config)
    app.config["QDRANT_CLIENT"] = client
    app.config["COLLECTION"] = collection
    db_path = os.environ.get("LOG_DB_PATH", os.path.join(
        os.path.dirname(__file__), "..", "data", "questions.db"))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    app.config["LOGDB"] = LogDB(db_path)
    buckets = {}

    def _rate_limited(ip):
        limit = int(config.get("limits", {}).get(
            "rate_limit_per_minute_per_ip", 30))
        bucket = buckets.setdefault(ip, TokenBucket(limit, 60))
        return not bucket.take()

    @app.get("/healthz")
    def healthz():
        ok = store_mod.health_check(client, collection)
        status = {"status": "ok" if ok else "degraded",
                  "qdrant": "ok" if ok else "unreachable",
                  "openai": "configured" if os.environ.get("OPENAI_API_KEY")
                          else "not_configured"}
        return jsonify(status), 200 if ok else 503

    @app.post("/api/chat")
    def api_chat():
        ip = request.remote_addr or "unknown"
        if _rate_limited(ip):
            return jsonify({"error": "rate limited",
                            "message": "Too many requests, please wait."}), \
                429, {"Retry-After": "60"}
        body = request.get_json(silent=True) or {}
        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "question is required"}), 400
        if len(question) > int(config.get("limits", {}).get(
                "max_question_chars", 1000)):
            return jsonify({"error": "question exceeds 1000 characters"}), 400
        lang = body.get("lang", "ar")
        if lang not in ("ar", "en"):
            lang = "ar"
        try:
            result = answer_question(
                question.strip(), client, collection, config, lang=lang,
                log_fn=lambda **kw: app.config["LOGDB"].log(
                    channel="widget", session_id=body.get("session_id"), **kw))
        except Exception:
            app.logger.exception("chat pipeline failed")
            # Log the failed question too (M15: the log records 100% of
            # questions, including ones the service could not answer).
            try:
                app.config["LOGDB"].log(
                    question=question.strip(), lang=lang, refused=True,
                    answer="maintenance", sources=[], latency_ms=0,
                    channel="widget", session_id=body.get("session_id"))
            except Exception:
                app.logger.exception("failed to log failed question")
            return jsonify({"answer": MAINTENANCE_AR if lang == "ar"
                            else MAINTENANCE_EN,
                            "sources": [], "refused": True,
                            "maintenance": True}), 503
        return jsonify(result)

    @app.get("/widget.html")
    def widget_html():
        return send_from_directory(os.path.join(
            os.path.dirname(__file__), "widget"), "widget.html")

    @app.get("/widget.js")
    def widget_js():
        return send_from_directory(os.path.join(
            os.path.dirname(__file__), "widget"), "widget.js")

    @app.post("/telegram/webhook")
    def telegram_webhook():
        secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
        if secret and not verify_webhook_secret(request, secret):
            return jsonify({"error": "forbidden"}), 403
        update = request.get_json(silent=True) or {}
        msg = update.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        text = msg.get("text", "")
        if not chat_id or not text:
            return jsonify({"ok": True})
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        try:
            result = answer_question(
                text[:1000], client, collection, config, lang="ar",
                log_fn=lambda **kw: app.config["LOGDB"].log(
                    channel="telegram", session_id=str(chat_id), **kw))
            reply = result["answer"]
            if result["sources"]:
                names = ", ".join(s["doc_title"] or s["doc_id"]
                                  for s in result["sources"])
                reply += f"\n\nالمصدر: {names}"
        except Exception:
            app.logger.exception("telegram pipeline failed")
            reply = MAINTENANCE_AR
        if token:
            send_message(token, chat_id, reply)
        return jsonify({"ok": True})

    def _admin_auth():
        token = os.environ.get("ADMIN_TOKEN", "")
        if not token:
            return False
        import hmac
        given = (request.headers.get("Authorization") or "").removeprefix("Bearer ")
        return hmac.compare_digest(given, token)

    @app.get("/api/admin/stats")
    def admin_stats():
        if not _admin_auth():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify(app.config["LOGDB"].stats(
            store_mod.count_points(client, collection)))

    @app.get("/api/admin/unanswered")
    def admin_unanswered():
        if not _admin_auth():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify(app.config["LOGDB"].unanswered())

    @app.get("/api/admin/log.csv")
    def admin_log_csv():
        if not _admin_auth():
            return jsonify({"error": "unauthorized"}), 401
        import io
        buf = io.StringIO()
        app.config["LOGDB"].export_csv(buf)
        return app.response_class(buf.getvalue(), mimetype="text/csv",
                                  headers={"Content-Disposition":
                                           "attachment; filename=questions.csv"})

    @app.post("/api/admin/reindex")
    def admin_reindex():
        if not _admin_auth():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify({"error": "reindex is a CLI operation (ingest/reindex.py); "
                                "this endpoint only confirms auth"}), 501

    return app


if __name__ == "__main__":
    # Local dev only. Production uses gunicorn (see render.yaml).
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
