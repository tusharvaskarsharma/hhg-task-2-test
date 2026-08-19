# HHG Engine - Frontend

This is the production-ready React frontend for the HHG (Hitchhiker's Guide) Semantic Search Engine. It acts as the final consumer for the Stage B RAG pipeline.

## Stack
- React 18
- Vite
- Tailwind CSS
- GSAP (GreenSock Animation Platform)
- Lucide React (Icons)

## Setup & Development

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Environment Variables:**
   Create a `.env` file containing the backend origin. (Defaults to `http://localhost:8000`).
   ```bash
   VITE_API_BASE_URL=http://localhost:8000
   ```
   *Note: This must not contain sensitive API keys. All inference/STT logic is strictly server-side.*

3. **Start local server:**
   ```bash
   npm run dev
   ```

4. **Production Build:**
   ```bash
   npm run build
   ```
   The compiled assets will be available in the `/dist` directory.

## Design Philosophy

The UI heavily leans on the **Antigravity Design** guidelines:
- **Spatial Depth:** Employs CSS 3D perspectives to tilt retrieval panels isometrically on interaction.
- **Glassmorphism:** Inputs and panels utilize heavy `backdrop-filter: blur`, floating over a dark radial-gradient abyss.
- **Smooth Staggers:** GSAP coordinates component mount phases so everything cascades onto the screen smoothly, completely avoiding instant snaps.

## Voice Functionality
Voice query functionality delegates actual processing to the backend's `/api/voice` route via `multipart/form-data`. The browser simply manages `MediaRecorder` buffers and cleanly handles hardware permission denial states.

## Accessibility
The application ensures that screen readers can cleanly interpret text interfaces and that focus states exist for keyboard navigators, especially around voice and query submission boundaries.
