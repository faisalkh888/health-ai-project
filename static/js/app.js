function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function appendMessage(container, label, message, cssClass = "bot") {
  if (!container) {
    return;
  }
  const item = document.createElement("div");
  item.className = `chat-bubble ${cssClass}`;
  const title = document.createElement("strong");
  title.textContent = label;
  const body = document.createElement("p");
  body.textContent = message;
  item.appendChild(title);
  item.appendChild(body);
  container.appendChild(item);
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const icon = document.getElementById("theme-icon");
  if (icon) {
    icon.textContent = theme === "light" ? "\u263E" : "\u2600";
  }
  window.localStorage.setItem("health-ai-theme", theme);
}

function applySavedTheme() {
  const savedTheme = window.localStorage.getItem("health-ai-theme") || "dark";
  setTheme(savedTheme);
}

function buildChartOptions() {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  const textColor = isLight ? "#11223a" : "#eff7ff";
  const gridColor = isLight ? "rgba(15,23,42,0.08)" : "rgba(255,255,255,0.08)";
  return {
    plugins: {
      legend: {
        labels: { color: textColor },
      },
    },
    scales: {
      x: { ticks: { color: textColor }, grid: { color: gridColor } },
      y: { ticks: { color: textColor }, grid: { color: gridColor } },
    },
  };
}

function titleCaseDiseaseName(name) {
  return String(name || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function renderAnalyticsCharts() {
  const root = document.getElementById("analytics-root");
  if (!root) {
    return;
  }

  const analytics = JSON.parse(root.dataset.analytics || "{}");
  const diseaseCounts = analytics.disease_counts || {};
  const riskCounts = analytics.risk_counts || {};
  const trend = analytics.trend || [];

  const chartOptions = buildChartOptions();

  const diseaseCanvas = document.getElementById("diseaseChart");
  if (diseaseCanvas) {
    new Chart(diseaseCanvas, {
      type: "doughnut",
      data: {
        labels: Object.keys(diseaseCounts),
        datasets: [{
          data: Object.values(diseaseCounts),
          backgroundColor: ["#38bdf8", "#22c55e", "#f97316", "#a855f7"],
        }],
      },
      options: { plugins: chartOptions.plugins },
    });
  }

  const riskCanvas = document.getElementById("riskChart");
  if (riskCanvas) {
    new Chart(riskCanvas, {
      type: "bar",
      data: {
        labels: Object.keys(riskCounts),
        datasets: [{
          label: "Cases",
          data: Object.values(riskCounts),
          backgroundColor: ["#22c55e", "#f97316", "#ef4444"],
        }],
      },
      options: chartOptions,
    });
  }

  const trendCanvas = document.getElementById("trendChart");
  if (trendCanvas) {
    new Chart(trendCanvas, {
      type: "line",
      data: {
        labels: trend.map((item) => item.label),
        datasets: [{
          label: "Prediction Probability %",
          data: trend.map((item) => item.probability),
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.2)",
          tension: 0.35,
          fill: true,
        }],
      },
      options: chartOptions,
    });
  }
}

function renderAdminMetricsCharts() {
  const root = document.getElementById("admin-metrics-root");
  if (!root) {
    return;
  }
  const metrics = JSON.parse(root.dataset.metrics || "{}");
  const labels = Object.keys(metrics).filter((key) => metrics[key]);
  const accuracyValues = labels.map((key) => Number(metrics[key].accuracy || 0) * 100);
  const rocValues = labels.map((key) => Number(metrics[key].roc_auc || 0) * 100);
  const chartOptions = buildChartOptions();

  const accuracyCanvas = document.getElementById("accuracyChart");
  if (accuracyCanvas) {
    new Chart(accuracyCanvas, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Accuracy %",
          data: accuracyValues,
          backgroundColor: "#38bdf8",
        }],
      },
      options: chartOptions,
    });
  }

  const rocCanvas = document.getElementById("rocChart");
  if (rocCanvas) {
    new Chart(rocCanvas, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: "ROC AUC %",
          data: rocValues,
          borderColor: "#22c55e",
          backgroundColor: "rgba(34, 197, 94, 0.2)",
          tension: 0.35,
          fill: true,
        }],
      },
      options: chartOptions,
    });
  }
}

function setDiseaseFormState(disease) {
  document.querySelectorAll("[id$='-fields']").forEach((section) => {
    const isActive = section.id === `${disease}-fields`;
    section.classList.toggle("d-none", !isActive);
    section.querySelectorAll(".feature-input").forEach((input) => {
      input.disabled = !isActive;
      input.required = isActive;
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  applySavedTheme();
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || "";
  const themeToggle = document.getElementById("theme-toggle");
  const tabs = document.querySelectorAll(".disease-tab");
  const diseaseInput = document.getElementById("disease");

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      setTheme(current === "dark" ? "light" : "dark");
      window.location.reload();
    });
  }

  if (diseaseInput) {
    setDiseaseFormState(diseaseInput.value);
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const disease = tab.dataset.disease;
      diseaseInput.value = disease;
      setDiseaseFormState(disease);

      tabs.forEach((item) => {
        item.classList.toggle("btn-primary", item.dataset.disease === disease);
        item.classList.toggle("btn-outline-primary", item.dataset.disease !== disease);
      });
    });
  });

  const form = document.getElementById("prediction-form");
  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const disease = diseaseInput.value;
      const submitButton = form.querySelector('button[type="submit"]');
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.textContent = "Predicting...";
      }
      const features = Array.from(
        document.querySelectorAll(`.feature-input[data-disease="${disease}"]`)
      ).map((input) => input.value);

      try {
        const response = await fetch(`/api/predict/${disease}`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken },
          body: JSON.stringify({ features }),
        });

        const data = await response.json();
        if (!response.ok) {
          alert(data.error || "Prediction failed");
          if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = "Predict Risk";
          }
          return;
        }

        localStorage.setItem("result", JSON.stringify(data));
        window.location.href = "/result";
      } catch (_error) {
        alert("Prediction request failed. Please try again.");
        if (submitButton) {
          submitButton.disabled = false;
          submitButton.textContent = "Predict Risk";
        }
      }
    });
  }

  const uploadForm = document.getElementById("upload-form");
  if (uploadForm) {
    uploadForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const disease = diseaseInput.value;
      const formData = new FormData();
      const file = document.getElementById("upload-file").files[0];
      if (!file) {
        alert("Choose a file first.");
        return;
      }
      formData.append("file", file);

      const response = await fetch(`/api/predict/${disease}/upload`, {
        method: "POST",
        headers: { "X-CSRFToken": csrfToken },
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || "Upload prediction failed");
        return;
      }

      if (data.batch) {
        localStorage.setItem("batchResult", JSON.stringify(data.results));
        window.location.href = "/batch-result";
      } else {
        localStorage.setItem("result", JSON.stringify(data));
        window.location.href = "/result";
      }
    });
  }

  if (document.getElementById("risk")) {
    const data = JSON.parse(localStorage.getItem("result") || "null");
    if (!data || typeof data.probability !== "number") {
      alert("No prediction result found.");
      window.location.href = "/dashboard";
      return;
    }

    setText("disease-name", `${titleCaseDiseaseName(data.disease)} Prediction`);
    setText("risk", `Risk: ${data.risk}`);
    setText("prob", `Probability: ${(data.probability * 100).toFixed(2)}%`);
    setText("confidence", `Confidence: ${((data.confidence || 0) * 100).toFixed(2)}%`);
    setText("model-accuracy", data.model_metrics ? `${(Number(data.model_metrics.accuracy || 0) * 100).toFixed(2)}%` : "N/A");
    setText("metric-explainer", data.model_metrics ? `Dataset accuracy: ${(Number(data.model_metrics.accuracy || 0) * 100).toFixed(2)}%, ROC AUC: ${(Number(data.model_metrics.roc_auc || 0) * 100).toFixed(2)}%, test size: ${data.model_metrics.test_size}.` : "Model metrics are not available yet.");
    setText("result-summary", data.model_metrics ? `This ${titleCaseDiseaseName(data.disease)} prediction uses a trained model with ${(Number(data.model_metrics.accuracy || 0) * 100).toFixed(2)}% validation accuracy.` : "Review the predicted risk, model confidence, and feature-level explanation below.");

    const reportLink = document.getElementById("report-link");
    if (reportLink && data.id) {
      reportLink.href = `/api/report/${data.id}`;
      reportLink.classList.remove("d-none");
    }

    const emailReport = document.getElementById("email-report");
    if (emailReport && data.id) {
      emailReport.classList.remove("d-none");
      emailReport.addEventListener("click", async () => {
        emailReport.disabled = true;
        const response = await fetch(`/api/email-report/${data.id}`, {
          method: "POST",
          headers: { "X-CSRFToken": csrfToken },
        });
        const payload = await response.json();
        alert(payload.message || payload.error || "Email request finished.");
        emailReport.disabled = false;
      });
    }

    const explanationList = document.getElementById("explain");
    Object.entries(data.explanation || {}).forEach(([key, value]) => {
      const item = document.createElement("li");
      const numericValue = Number(value);
      item.textContent = `${key}: ${Number.isFinite(numericValue) ? numericValue.toFixed(3) : value}`;
      explanationList.appendChild(item);
    });

    const chartOptions = buildChartOptions();

    new Chart(document.getElementById("chart"), {
      type: "bar",
      data: {
        labels: ["Risk Probability", "Confidence"],
        datasets: [{
          label: "Score %",
          data: [data.probability * 100, (data.confidence || 0) * 100],
          backgroundColor: ["#38bdf8", "#22c55e"],
        }],
      },
      options: chartOptions,
    });

    if (document.getElementById("modelChart") && data.model_metrics) {
      new Chart(document.getElementById("modelChart"), {
        type: "radar",
        data: {
          labels: ["Accuracy", "ROC AUC", "Confidence"],
          datasets: [{
            label: "Model Quality",
            data: [
              (Number(data.model_metrics.accuracy || 0) * 100),
              (Number(data.model_metrics.roc_auc || 0) * 100),
              ((data.confidence || 0) * 100),
            ],
            borderColor: "#f97316",
            backgroundColor: "rgba(249, 115, 22, 0.18)",
          }],
        },
        options: chartOptions,
      });
    }

    const explanationEntries = Object.entries(data.explanation || {});
    if (document.getElementById("explanationChart") && explanationEntries.length) {
      new Chart(document.getElementById("explanationChart"), {
        type: "bar",
        data: {
          labels: explanationEntries.map(([key]) => key),
          datasets: [{
            label: "Feature contribution",
            data: explanationEntries.map(([, value]) => Number(value)),
            backgroundColor: explanationEntries.map(([, value]) => Number(value) >= 0 ? "#ef4444" : "#22c55e"),
          }],
        },
        options: {
          ...chartOptions,
          indexAxis: "y",
        },
      });
    }

    const selected = data.guidance || { diet: [], medical: [] };
    (selected.diet || []).forEach((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      document.getElementById("diet").appendChild(item);
    });
    (selected.medical || []).forEach((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      document.getElementById("med").appendChild(item);
    });
  }

  const chatBox = document.getElementById("chat-box");
  const msgInput = document.getElementById("msg");
  const sendBtn = document.getElementById("send-btn");
  const clearChatBtn = document.getElementById("clear-chat-btn");

  if (chatBox && !chatBox.children.length) {
    appendMessage(
      chatBox,
      "Health AI",
      "Hello. I can help with symptom questions, report summaries, healthy habits, and next steps after a prediction.",
      "bot"
    );
  }

  async function sendChatMessage() {
    if (!msgInput || !msgInput.value.trim()) {
      return;
    }
    const message = msgInput.value.trim();
    appendMessage(chatBox, "You", message, "user");
    msgInput.value = "";
    if (sendBtn) {
      sendBtn.disabled = true;
      sendBtn.textContent = "Sending...";
    }

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      appendMessage(chatBox, "Health AI", data.reply || data.error || "I could not respond.", "bot");
    } catch (_error) {
      appendMessage(chatBox, "Health AI", "The chatbot is unavailable right now. Please try again.", "bot");
    } finally {
      if (sendBtn) {
        sendBtn.disabled = false;
        sendBtn.textContent = "Send";
      }
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  }

  if (sendBtn) {
    sendBtn.addEventListener("click", sendChatMessage);
  }

  if (msgInput) {
    msgInput.addEventListener("keypress", async (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        await sendChatMessage();
      }
    });
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", async () => {
      clearChatBtn.disabled = true;
      try {
        await fetch("/api/chat/clear", {
          method: "POST",
          headers: { "X-CSRFToken": csrfToken },
        });
        if (chatBox) {
          chatBox.innerHTML = "";
          appendMessage(
            chatBox,
            "Health AI",
            "Chat cleared. Start a new symptom check or ask a health question any time.",
            "bot"
          );
        }
      } finally {
        clearChatBtn.disabled = false;
      }
    });
  }

  const micButton = document.getElementById("mic-btn");
  if (micButton) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      micButton.disabled = true;
      micButton.textContent = "Voice Unsupported";
    } else {
      const recognition = new SpeechRecognition();
      recognition.lang = "en-IN";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      micButton.addEventListener("click", () => {
        recognition.start();
        micButton.textContent = "Listening...";
      });

      recognition.addEventListener("result", (event) => {
        msgInput.value = event.results[0][0].transcript;
        micButton.textContent = "Voice";
      });

      recognition.addEventListener("end", () => {
        micButton.textContent = "Voice";
      });
    }
  }

  const retrain = document.getElementById("retrain");
  if (retrain) {
    retrain.addEventListener("click", async () => {
      retrain.disabled = true;
      retrain.textContent = "Retraining...";

      const response = await fetch("/api/retrain", {
        method: "POST",
        headers: { "X-CSRFToken": csrfToken },
      });
      const data = await response.json();
      alert(data.message || data.error || "Retraining finished.");
      window.location.reload();
    });
  }

  const batchTable = document.getElementById("batch-results");
  if (batchTable) {
    const results = JSON.parse(localStorage.getItem("batchResult") || "[]");
    results.forEach((item) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${item.id}</td>
        <td>${item.disease}</td>
        <td>${(item.probability * 100).toFixed(2)}%</td>
        <td>${((item.confidence || 0) * 100).toFixed(2)}%</td>
        <td>${item.risk}</td>
        <td><a href="/api/report/${item.id}">PDF</a></td>
      `;
      batchTable.appendChild(row);
    });
  }

  renderAnalyticsCharts();
  renderAdminMetricsCharts();
});

