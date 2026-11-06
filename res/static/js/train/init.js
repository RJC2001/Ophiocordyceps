document.addEventListener("DOMContentLoaded", async () => {
  try {
    disableResults();
    disableImgResults();
    await loadConfig();
    await loadLanguage();

    await api("/api/dataset/clear", { method: "POST", body: "{}" });
    await api("/api/output/clear", { method: "POST", body: "{}" });
    await api("/api/image_train/clear", { method: "POST", body: "{}" });

    await loadAlgorithms();
    await loadModels();
    await loadImageModels();
    startModelRefresh();
    startImageModelRefresh();
    updateSplitBar();
    toggleValStrategy();
    updateImgSplitBar();
    updateImgSplitCounts();
  } catch (err) {
    outputLog.textContent = String(err.message || err);
  }
});
