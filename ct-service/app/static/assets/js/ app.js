import { postAnalyze } from "./api.js";
import { setProgress, show, hide, renderResult } from "./ui.js";

const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");
const progressCard = document.getElementById("progress");
const resultCard = document.getElementById("result");

let file = null;

function validateAndSet(f) {
  if (!f) return;
  if (!f.name.toLowerCase().endsWith(".zip")) { alert("Только .zip"); return; }
  if (f.size > 1024 ** 3) { alert("Файл больше 1 ГБ"); return; }
  file = f;
  sendBtn.disabled = false;
  dropzone.querySelector("p").textContent = `Выбран файл: ${f.name} (${Math.round(f.size/1e6)} МБ)`;
}

dropzone.addEventListener("dragover", (e) => { e.preventDefault(); });
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  validateAndSet(e.dataTransfer.files?.[0]);
});
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => validateAndSet(e.target.files?.[0]));

sendBtn.addEventListener("click", async () => {
  if (!file) return;
  hide(resultCard);
  show(progressCard);
  setProgress(0);

  try {
    const resp = await postAnalyze(file, (p) => setProgress(p));
    setProgress(1, resp.label === "normal");
    renderResult(resp);
    show(resultCard);
  } catch (err) {
    alert(`Ошибка: ${err.message}`);
  }
});

resetBtn.addEventListener("click", () => {
  file = null;
  fileInput.value = "";
  document.getElementById("report-link").href = "#";
  document.getElementById("warnings").innerHTML = "";
  document.getElementById("result-json").textContent = "";
  dropzone.querySelector("p").textContent = "Перетащи ZIP с DICOM сюда или выбери файл";
  sendBtn.disabled = true;
  hide(progressCard);
  hide(resultCard);
});
