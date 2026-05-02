import React, {useState, useRef} from 'react'
import axios from 'axios'

export default function UploadAndDisplay(){
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [maskUrl, setMaskUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [overlay, setOverlay] = useState(false)
  const canvasRef = useRef(null)

  function onFileChange(e){
    const f = e.target.files[0]
    if(!f) return
    setFile(f)
    setMaskUrl(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  async function runSegmentation(){
    if(!file) return alert('Choose an image first')
    setLoading(true)
    const form = new FormData()
    form.append('file', file)
    try{
      const resp = await axios.post('http://localhost:8000/predict', form, {
        responseType: 'blob',
        headers: {'Content-Type':'multipart/form-data'}
      })
      const blob = resp.data
      const url = URL.createObjectURL(blob)
      setMaskUrl(url)
      setTimeout(()=>{
        if(overlay) drawOverlay()
      }, 200)
    }catch(err){
      console.error(err)
      alert('Error during inference')
    }finally{
      setLoading(false)
    }
  }

  function clearAll(){
    setFile(null); setPreview(null); setMaskUrl(null)
  }

  function downloadMask(){
    if(!maskUrl) return
    const a = document.createElement('a')
    a.href = maskUrl
    a.download = 'mask.png'
    a.click()
  }

  function drawOverlay(){
    if(!preview || !maskUrl) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const img = new Image()
    const mask = new Image()
    img.src = preview
    mask.src = maskUrl
    img.onload = ()=>{
      canvas.width = img.width
      canvas.height = img.height
      ctx.clearRect(0,0,canvas.width,canvas.height)
      ctx.drawImage(img,0,0)
      if(overlay){
        mask.onload = ()=>{
          // draw mask with 50% alpha and multiply colors
          ctx.globalAlpha = 0.6
          ctx.drawImage(mask,0,0,canvas.width,canvas.height)
          ctx.globalAlpha = 1.0
        }
      }
    }
  }

  return (
    <div>
      <div className="upload-row">
        <label className="dropbox">
          <input type="file" accept="image/*" onChange={onFileChange} />
          <div className="drop-text">Drag & drop or click to select image</div>
        </label>
        <div className="controls">
          <button className="btn" onClick={runSegmentation} disabled={loading}>Run Segmentation</button>
          <button className="btn ghost" onClick={clearAll}>Clear</button>
          <button className="btn" onClick={downloadMask} disabled={!maskUrl}>Download Mask</button>
        </div>
      </div>

      <div className="result-row">
        <div className="panel">
          <h3>Original</h3>
          {preview ? <img src={preview} alt="preview" className="preview"/> : <div className="placeholder">No image</div>}
        </div>

        <div className="panel">
          <h3>Prediction</h3>
          {loading && <div className="loader"/>}
          {!loading && maskUrl && <img src={maskUrl} alt="mask" className="preview" onLoad={()=>{ if(overlay) drawOverlay()}}/>}
          {!loading && !maskUrl && <div className="placeholder">No prediction</div>}

          <div className="overlay-row">
            <label className="toggle">
              <input type="checkbox" checked={overlay} onChange={(e)=>{setOverlay(e.target.checked); setTimeout(drawOverlay,50)}} /> Overlay
            </label>
          </div>
        </div>
      </div>

      <canvas ref={canvasRef} style={{display: overlay ? 'block' : 'none'}} />
    </div>
  )
}
