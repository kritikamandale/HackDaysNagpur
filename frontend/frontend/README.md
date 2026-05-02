# Frontend (Vite + React) for Segmentation Demo

This is a lightweight React app built with Vite. It provides a drag-and-drop upload UI, preview, "Run Segmentation" button and displays the returned mask.

Quickstart

1. Install Node.js (v16+ recommended)
2. Install deps and run dev server:

```bash
cd frontend
npm install
npm run dev
```

3. Open the URL printed by Vite (usually http://localhost:5173).

Configuration

- The frontend posts to `http://localhost:8000/predict`. If your backend runs on a different host/port, edit `src/components/UploadAndDisplay.jsx` and update the URL.

Notes

- The UI includes an overlay toggle to blend prediction masks over the original image and a download button for the mask.
- Styling uses a glassmorphism aesthetic with soft pastel colors.
