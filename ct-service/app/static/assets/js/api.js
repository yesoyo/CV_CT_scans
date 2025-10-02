export async function postAnalyze(file, onProgress) {
  const url = `${window.API_BASE}/analyze`;
  const form = new FormData();
  form.append("file", file, file.name);

  const xhr = new XMLHttpRequest();
  return await new Promise((resolve, reject) => {
    xhr.open("POST", url, true);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && typeof onProgress === "function") {
        onProgress(e.loaded / e.total);
      }
    };
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); }
          catch { reject(new Error("bad json")); }
        } else {
          reject(new Error(`HTTP ${xhr.status}: ${xhr.responseText}`));
        }
      }
    };
    xhr.onerror = () => reject(new Error("network error"));
    xhr.send(form);
  });
}
