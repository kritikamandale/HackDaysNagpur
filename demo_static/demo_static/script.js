// Simple client-side mock segmentation: quantize brightness into classes and colorize
const NUM_CLASSES = 10;
const PALETTE = [
  [0,0,0],
  [167,199,231],
  [205,180,219],
  [181,234,215],
  [210,180,140],
  [139,90,43],
  [128,128,0],
  [139,69,19],
  [128,128,128],
  [135,206,235]
];

const fileInput = document.getElementById('fileInput');
const runBtn = document.getElementById('runBtn');
const downloadBtn = document.getElementById('downloadBtn');
const overlayToggle = document.getElementById('overlayToggle');
const geminiBtn = document.getElementById('geminiBtn');
const geminiPrompt = document.getElementById('geminiPrompt');
const geminiOutput = document.getElementById('geminiOutput');

const origCanvas = document.getElementById('origCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const origCtx = origCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');

let currentImage = null;
let currentMask = null;

async function geminiAnalyzeCurrentImage(promptText){
  if(!fileInput.files[0]) return 'Choose an image first';
  const imageDataUrl = await new Promise((resolve, reject)=>{
    const reader = new FileReader();
    reader.onload = ()=>resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(fileInput.files[0]);
  });
  const resp = await fetch('/gemini/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: promptText,
      image_data_url: imageDataUrl,
      mime_type: fileInput.files[0].type || 'image/png',
    }),
  });
  const data = await resp.json();
  if(!resp.ok) throw new Error(data.detail || 'Gemini request failed');
  return data.response || 'Gemini returned no text.';
}

fileInput.addEventListener('change', async (e)=>{
  const f = e.target.files[0];
  if(!f) return;
  const img = new Image();
  img.src = URL.createObjectURL(f);
  await img.decode();
  currentImage = img;
  // size canvases
  origCanvas.width = img.naturalWidth;
  origCanvas.height = img.naturalHeight;
  maskCanvas.width = img.naturalWidth;
  maskCanvas.height = img.naturalHeight;
  origCtx.drawImage(img,0,0);
  maskCtx.clearRect(0,0,maskCanvas.width,maskCanvas.height);
  currentMask = null;
  downloadBtn.disabled = true;
});

runBtn.addEventListener('click', async ()=>{
  if(!currentImage) return alert('Choose an image first');
  geminiOutput.textContent = 'Running segmentation...';
  // Try server inference first
  const file = fileInput.files[0];
  let handled = false;
  if(file){
    try{
      const fd = new FormData(); fd.append('file', file);
      const resp = await fetch('/predict', { method: 'POST', body: fd });
      if(resp.ok){
        const blob = await resp.blob();
        const img = new Image();
        img.src = URL.createObjectURL(blob);
        await img.decode();
        // draw returned colorized preview onto mask canvas
        maskCanvas.width = img.naturalWidth; maskCanvas.height = img.naturalHeight;
        maskCtx.clearRect(0,0,maskCanvas.width,maskCanvas.height);
        maskCtx.drawImage(img,0,0);
        // set currentMask to null (raw mask saved on server)
        currentMask = null;
        downloadBtn.disabled = true; // server saves raw mask in backend/outputs
        handled = true;
      }
    }catch(e){
      console.warn('Server inference failed, falling back to client mock:', e);
      handled = false;
    }
  }

  if(!handled){
    // Client-side mock segmentation: compute grayscale, quantize into NUM_CLASSES
    origCtx.drawImage(currentImage,0,0);
    const w = origCanvas.width, h = origCanvas.height;
    const imgData = origCtx.getImageData(0,0,w,h);
    const data = imgData.data;
    const mask = new Uint8ClampedArray(w*h);
    for(let i=0;i<w*h;i++){
      const r = data[i*4], g = data[i*4+1], b = data[i*4+2];
      const lum = Math.round(0.2126*r + 0.7152*g + 0.0722*b);
      const cid = Math.floor(lum * NUM_CLASSES / 256);
      mask[i] = Math.min(NUM_CLASSES-1, cid);
    }
    currentMask = mask;
    // draw colorized mask
    const out = maskCtx.createImageData(w,h);
    for(let i=0;i<w*h;i++){
      const c = PALETTE[mask[i]];
      out.data[i*4] = c[0];
      out.data[i*4+1] = c[1];
      out.data[i*4+2] = c[2];
      out.data[i*4+3] = 255;
    }
    maskCtx.putImageData(out,0,0);
    downloadBtn.disabled = false;
    if(overlayToggle.checked){
      drawOverlay();
    }
  }

  try{
    geminiOutput.textContent = 'Analyzing the image with Gemini...';
    const summary = await geminiAnalyzeCurrentImage(
      'Look at this image and tell me what is in it in one short sentence. If it looks like vegetation, bushes, or plants, say that clearly. Mention any obvious terrain or obstacle risks.'
    );
    geminiOutput.textContent = summary;
  }catch(err){
    geminiOutput.textContent = `Gemini error: ${err.message}`;
  }
});

overlayToggle.addEventListener('change', ()=>{
  if(!currentImage || !currentMask) return;
  if(overlayToggle.checked) drawOverlay();
  else { origCtx.drawImage(currentImage,0,0); }
});

function drawOverlay(){
  if(!currentImage || !currentMask) return;
  const w = origCanvas.width, h = origCanvas.height;
  // draw orig then mask with alpha
  origCtx.drawImage(currentImage,0,0);
  maskCtx.globalAlpha = 0.6;
  // copy mask onto orig canvas
  const maskData = maskCtx.getImageData(0,0,w,h);
  origCtx.putImageData(maskData,0,0);
  maskCtx.globalAlpha = 1.0;
}

downloadBtn.addEventListener('click', ()=>{
  if(!currentMask) return;
  // export raw mask as PNG where pixel value is class id
  const w = maskCanvas.width, h = maskCanvas.height;
  const rawCanvas = document.createElement('canvas');
  rawCanvas.width = w; rawCanvas.height = h;
  const rawCtx = rawCanvas.getContext('2d');
  const img = rawCtx.createImageData(w,h);
  for(let i=0;i<w*h;i++){
    const v = currentMask[i];
    img.data[i*4] = v; img.data[i*4+1] = v; img.data[i*4+2] = v; img.data[i*4+3] = 255;
  }
  rawCtx.putImageData(img,0,0);
  const url = rawCanvas.toDataURL('image/png');
  const a = document.createElement('a'); a.href = url; a.download = 'mask_raw.png'; a.click();
});

geminiBtn.addEventListener('click', async ()=>{
  if(!fileInput.files[0]) return alert('Choose an image first');
  geminiBtn.disabled = true;
  geminiOutput.textContent = 'Asking Gemini...';
  try{
    geminiOutput.textContent = await geminiAnalyzeCurrentImage(geminiPrompt.value || 'Describe this image.');
  }catch(err){
    geminiOutput.textContent = `Gemini error: ${err.message}`;
  }finally{
    geminiBtn.disabled = false;
  }
});

// Provide a small sample image if user wants one via drag+drop placeholder
window.addEventListener('load', ()=>{
  // nothing; user supplies image
});