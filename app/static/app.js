const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#file");
const sampleButton = document.querySelector("#sample");
const prompt = document.querySelector("#prompt");
const preview = document.querySelector("#preview");
const status = document.querySelector("#status");
const meta = document.querySelector("#meta");
const lines = document.querySelector("#lines");
const placeholder = document.querySelector("#placeholder");
const copyButton = document.querySelector("#copy");

let transcript = "";

dropzone.addEventListener("submit", (event) => event.preventDefault());

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (file) submit(file);
});

sampleButton.addEventListener("click", async () => {
  setBusy("Loading the sample page…");
  try {
    const response = await fetch("/samples/angled_page.png");
    if (!response.ok) throw new Error("Sample page is missing.");
    const blob = await response.blob();
    await submit(new File([blob], "angled_page.png", { type: "image/png" }));
  } catch (error) {
    showError(error.message);
  }
});

["dragenter", "dragover"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag");
  });
});

["dragleave", "drop"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) submit(file);
});

copyButton.addEventListener("click", async () => {
  if (!transcript) return;
  await navigator.clipboard.writeText(transcript);
  copyButton.textContent = "Copied";
  setTimeout(() => {
    copyButton.textContent = "Copy";
  }, 1200);
});

async function submit(file) {
  fileInput.value = "";
  setBusy("Finding the page and reading it…");
  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch("/api/read", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(messageFrom(payload));
    render(payload);
  } catch (error) {
    showError(error.message);
  }
}

function render(payload) {
  status.hidden = true;
  prompt.hidden = true;
  preview.hidden = false;
  dropzone.classList.add("has-preview");
  preview.src = payload.preview;
  transcript = payload.text || "";
  placeholder.hidden = true;
  copyButton.disabled = !transcript;
  const pageNote = payload.page_found ? "Page flattened" : "No separate page found, so the whole photo was read";
  const count = payload.lines.length;
  meta.textContent = `${pageNote} · ${count} line${count === 1 ? "" : "s"} · ${payload.elapsed_ms} ms`;
  if (!transcript) placeholder.hidden = false;
  if (!transcript) placeholder.textContent = "No words were found on this page.";
  lines.replaceChildren(
    ...payload.lines.map((line) => {
      const item = document.createElement("li");
      const score = document.createElement("span");
      score.className = "confidence";
      score.textContent = `${Math.round(line.confidence * 100)}%`;
      const wording = document.createElement("span");
      wording.className = "line-text";
      wording.textContent = line.text;
      item.append(score, wording);
      return item;
    }),
  );
}

function messageFrom(payload) {
  if (payload && typeof payload.detail === "string") return payload.detail;
  if (payload && Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg).filter(Boolean).join(" ");
  }
  return "Paige could not read that image.";
}

function setBusy(message) {
  status.hidden = false;
  status.textContent = message;
  copyButton.disabled = true;
}

function showError(message) {
  status.hidden = false;
  status.textContent = message;
  meta.textContent = "Try another photo.";
}
