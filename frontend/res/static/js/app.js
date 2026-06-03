const themeSelect = document.getElementById("themeSelect");
const languageSelect = document.getElementById("languageSelect");
const trainingForm = document.getElementById("trainingForm");
const outputLog = document.getElementById("outputLog");
const modelList = document.getElementById("modelList");
const refreshModels = document.getElementById("refreshModels");
const clearOutput = document.getElementById("clearOutput");

let translations = {};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });

  const text = await response.text();
  let data = null;

  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }

  if (!response.ok) {
    throw new Error((data && data.error) ? data.error : `HTTP ${response.status}`);
  }

  return data;
}

async function loadTranslations(language) {
  const response = await fetch(`/static/i18n/${language}.json`);
  if (response.ok) {
    translations = await response.json();
    return;
  }

  const fallback = await fetch("/static/i18n/en.json");
  translations = await fallback.json();
}

function t(key) {
  return translations[key] ?? key;
}

function applyTranslations() {
  document.documentElement.lang = languageSelect.value || "en";

  document.querySelectorAll("[data-i18n]").forEach(node => {
    const key = node.getAttribute("data-i18n");
    node.textContent = t(key);
  });
}

function fillForm(data) {
  trainingForm.elements.dataset.value = data.dataset || "";
  trainingForm.elements.model_name.value = data.model_name || "";
  trainingForm.elements.epochs.value = data.epochs ?? 25;
  trainingForm.elements.batch_size.value = data.batch_size ?? 16;
  trainingForm.elements.learning_rate.value = data.learning_rate ?? 0.0003;
}

function renderOutput(data) {
  const lines = Array.isArray(data.lines) ? data.lines : [];
  outputLog.textContent = lines.length ? lines.join("\n") : t("no_output");
}

function renderModels(models) {
  const items = Array.isArray(models) ? models : [];
  modelList.innerHTML = "";

  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "model-empty";
    empty.textContent = t("no_models");
    modelList.appendChild(empty);
    return;
  }

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "model-item";

    const meta = document.createElement("div");
    meta.className = "model-meta";

    const name = document.createElement("strong");
    name.textContent = item.name;

    const details = document.createElement("span");
    const size = Number(item.size || 0);
    const modified = item.modified ? new Date(item.modified * 1000).toLocaleString() : "";
    details.textContent = `${size} B${modified ? ` · ${modified}` : ""}`;

    meta.appendChild(name);
    meta.appendChild(details);

    const actions = document.createElement("div");
    actions.className = "model-actions";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "secondary";
    renameBtn.textContent = t("rename");
    renameBtn.addEventListener("click", async () => {
      const next = prompt(t("rename_prompt"), item.name);
      if (!next || next === item.name) {
        return;
      }

      await api("/api/models/rename", {
        method: "POST",
        body: JSON.stringify({
          old_name: item.name,
          new_name: next
        })
      });

      await loadModels();
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "danger";
    deleteBtn.textContent = t("delete");
    deleteBtn.addEventListener("click", async () => {
      const confirmed = confirm(t("delete_confirm").replace("{name}", item.name));
      if (!confirmed) {
        return;
      }

      await api("/api/models/delete", {
        method: "POST",
        body: JSON.stringify({ name: item.name })
      });

      await loadModels();
    });

    actions.appendChild(renameBtn);
    actions.appendChild(deleteBtn);

    li.appendChild(meta);
    li.appendChild(actions);
    modelList.appendChild(li);
  }
}

async function loadTheme() {
  const data = await api("/api/theme");
  const themes = Array.isArray(data.available) ? data.available : [];
  const themeLabels = translations.themes || {};

  themeSelect.innerHTML = "";

  for (const theme of themes) {
    const option = document.createElement("option");
    option.value = theme.key;
    option.textContent = themeLabels[theme.key] || theme.label || theme.key;
    option.selected = theme.key === data.theme;
    themeSelect.appendChild(option);
  }

  themeSelect.value = data.theme;
}

async function loadLanguage() {
  const data = await api("/api/language");
  const languages = Array.isArray(data.available) ? data.available : [];

  languageSelect.innerHTML = "";
  for (const language of languages) {
    const option = document.createElement("option");
    option.value = language.key;
    option.textContent = language.label;
    option.selected = language.key === data.language;
    languageSelect.appendChild(option);
  }

  languageSelect.value = data.language;
  await loadTranslations(data.language);
  applyTranslations();
  await loadTheme();
}

async function saveTheme(theme) {
  await api("/api/theme", {
    method: "POST",
    body: JSON.stringify({ theme })
  });
  window.location.reload();
}

async function saveLanguage(language) {
  await api("/api/language", {
    method: "POST",
    body: JSON.stringify({ language })
  });
  window.location.reload();
}

async function loadForm() {
  const data = await api("/api/form");
  fillForm(data);
}

async function saveForm(event) {
  event.preventDefault();

  const payload = {
    dataset: trainingForm.elements.dataset.value,
    model_name: trainingForm.elements.model_name.value,
    epochs: trainingForm.elements.epochs.value,
    batch_size: trainingForm.elements.batch_size.value,
    learning_rate: trainingForm.elements.learning_rate.value
  };

  await api("/api/form", {
    method: "POST",
    body: JSON.stringify(payload)
  });

  await api("/api/output/append", {
    method: "POST",
    body: JSON.stringify({ line: `Saved form for ${payload.model_name || "unnamed model"}` })
  });

  await loadOutput();
}

async function loadOutput() {
  const data = await api("/api/output");
  renderOutput(data);
}

async function loadModels() {
  const data = await api("/api/models");
  renderModels(data);
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadTheme();
    await loadLanguage();
    await loadForm();
    await loadOutput();
    await loadModels();
  } catch (error) {
    outputLog.textContent = String(error.message || error);
  }

  themeSelect.addEventListener("change", async () => {
    await saveTheme(themeSelect.value);
  });

  languageSelect.addEventListener("change", async () => {
    await saveLanguage(languageSelect.value);
  });

  trainingForm.addEventListener("submit", saveForm);

  refreshModels.addEventListener("click", loadModels);

  clearOutput.addEventListener("click", async () => {
    await api("/api/output/clear", {
      method: "POST",
      body: JSON.stringify({})
    });
    await loadOutput();
  });

  setInterval(loadOutput, 1000);
});