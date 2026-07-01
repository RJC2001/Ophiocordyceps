async function api(path, options) {
  const headers = { "Content-Type": "application/json", ...(options?.headers || {}) };
  const r = await fetch(path, { headers, ...options });
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  return data;
}

document.addEventListener("DOMContentLoaded", async () => {
  const themeSelect = document.getElementById("themeSelect");
  const languageSelect = document.getElementById("languageSelect");

  const langData = await api("/api/language");
  languageSelect.innerHTML = "";
  for (const lang of (langData.available || [])) {
    const opt = document.createElement("option");
    opt.value = lang.key;
    opt.textContent = lang.label;
    opt.selected = lang.key === langData.language;
    languageSelect.appendChild(opt);
  }
  languageSelect.value = langData.language;

  let resp = await fetch(`/static/i18n/${langData.language}.json`);
  if (!resp.ok) resp = await fetch("/static/i18n/en.json");
  const translations = await resp.json();

  const themeData = await api("/api/theme");
  themeSelect.innerHTML = "";
  const themeLabels = translations.themes || {};
  for (const th of (themeData.available || [])) {
    const opt = document.createElement("option");
    opt.value = th.key;
    opt.textContent = themeLabels[th.key] || th.label || th.key;
    opt.selected = th.key === themeData.theme;
    themeSelect.appendChild(opt);
  }
  themeSelect.value = themeData.theme;

  document.querySelectorAll("[data-i18n]").forEach(node => {
    const key = node.getAttribute("data-i18n");
    const tag = node.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea") {
      node.placeholder = translations[key] ?? key;
    } else {
      node.textContent = translations[key] ?? key;
    }
  });

  themeSelect.addEventListener("change", async () => {
    await api("/api/theme", { method: "POST", body: JSON.stringify({ theme: themeSelect.value }) });
    window.location.reload();
  });
  languageSelect.addEventListener("change", async () => {
    await api("/api/language", { method: "POST", body: JSON.stringify({ language: languageSelect.value }) });
    window.location.reload();
  });
});
