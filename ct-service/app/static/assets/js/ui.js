export function setProgress(v, good) {
  const pct = Math.max(0, Math.min(1, v));
  const bar = document.getElementById("bar-fill");
  const txt = document.getElementById("bar-text");
  bar.style.width = `${Math.round(pct * 100)}%`;
  txt.textContent = `${Math.round(pct * 100)}%`;

  if (good === true) bar.style.background = "linear-gradient(90deg,#4caf50,#81c784)";
  else if (good === false) bar.style.background = "linear-gradient(90deg,#e53935,#ef5350)";
}

export function show(el) { el.classList.remove("hidden"); }
export function hide(el) { el.classList.add("hidden"); }

export function renderResult(resp) {
  const pre = document.getElementById("result-json");
  pre.textContent = JSON.stringify({
    job_id: resp.job_id,
    series_uid: resp.series_uid,
    label: resp.label,
    score: resp.score,
    routed_to_3d: resp.routed_to_3d
  }, null, 2);

  const link = document.getElementById("report-link");
  link.href = resp.report_path;

  const warn = document.getElementById("warnings");
  warn.innerHTML = resp.warnings && resp.warnings.length
    ? "Предупреждения:<br>" + resp.warnings.map(w => `• ${w}`).join("<br>")
    : "";
}
