# Afghanistan Crop AI — complete deployment
Upload the contents of this ZIP to a GitHub repository. Import that repository in Vercel, leave Root Directory at repository root and Framework Preset Other. The included vercel.json builds frontend and routes API calls to the Python function. No external API URL is required. Python dependencies and bundled trained models are included. Vercel deployment has not been tested; Python function size limits may require hosting the backend on Render instead.

Local: `pip install -r requirements.txt`, then `uvicorn backend.main:app --port 8000` from root. In another terminal `cd frontend`, `npm install`, `npm run dev`.

Alternative: deploy backend with Render (root backend, pip install -r requirements.txt, uvicorn main:app --host 0.0.0.0 --port $PORT). Set VITE_API_URL to the backend URL in Vercel and deploy frontend. 

Three models retrained from bundled raw CSV using commented script. Historical reports are preserved separately. Synthetic data and provisional rules are not field validation. Crop illustrations are emoji, not photographs.
