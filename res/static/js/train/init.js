document.addEventListener("DOMContentLoaded", async () => {
  try {
    disableResults();
    await loadConfig();
    await loadLanguage();

    await api("/api/dataset/clear", { method: "POST", body: "{}" });
    await api("/api/output/clear", { method: "POST", body: "{}" });
    await api("/api/image_train/clear", { method: "POST", body: "{}" });

    await loadAlgorithms();
    await loadImageAlgorithms();
    await loadModels();
    await loadImageModels();
    startModelRefresh();
    startImageModelRefresh();
    updateSplitBar();
    updateImgSplitBar();
    toggleValStrategy();
    toggleImgValStrategy();
  } catch (err) {
    outputLog.textContent = String(err.message || err);
  }
});
