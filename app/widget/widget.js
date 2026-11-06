/* Chat widget client — ARCH-02 §7.1. Minimal: POST /api/chat, render
   the answer with its sources. On refusal it shows the refusal text
   and a "contact the store" link. On a cold start (Render free sleeps
   after ~15 min idle) it shows a "waking up…" state — the 30–60 s
   delay is disclosed, not hidden. */
(function () {
  var log = document.getElementById("log");
  var input = document.getElementById("q");
  var status = document.getElementById("status");

  function addMsg(text, cls, sources) {
    var div = document.createElement("div");
    div.className = "msg " + cls;
    div.textContent = text;
    if (sources && sources.length) {
      var s = document.createElement("div");
      s.className = "sources";
      s.textContent = "المصدر: " + sources.map(function (x) {
        return x.doc_title || x.doc_id;
      }).join("، ");
      div.appendChild(s);
    }
    log.appendChild(div);
  }

  async function ask() {
    var question = input.value.trim();
    if (!question) return;
    input.value = "";
    addMsg(question, "user");
    status.textContent = "جارٍ البحث في مستندات المتجر…";
    status.className = "";
    var started = Date.now();
    try {
      var resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question, lang: "ar" })
      });
      var data = await resp.json();
      status.textContent = "";
      if (resp.status === 503) {
        status.textContent = "الخدمة تستيقظ الآن (الاستضافة المجانية تنام بعد فترة خمول) — أعد المحاولة خلال دقيقة.";
        status.className = "waking";
      }
      addMsg(data.answer, data.refused ? "refused" : "bot", data.sources);
    } catch (e) {
      status.textContent = "تعذّر الاتصال بالخدمة. يرجى التواصل مع المتجر مباشرة.";
    }
  }

  document.getElementById("send").addEventListener("click", ask);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") ask();
  });
})();
