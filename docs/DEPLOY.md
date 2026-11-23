# Deploying the web demo (Streamlit Community Cloud)

**Live:** https://pcb-repair-assistant-b7yfndd6wrb2vbxtzvpszz.streamlit.app/

Streamlit redeploys automatically on every push to `main`.

The repo is ready to deploy as-is. This needs your own sign-in, about 10 minutes, once.

1. Go to https://share.streamlit.io and choose **Continue with GitHub**. Sign in as `xasxasxas`.
2. Accept Streamlit's terms, then allow access to **private repositories** when GitHub asks. The repo is private.
3. Click **Create app** → **Deploy a public app from GitHub** and fill in:
   - Repository: `xasxasxas/pcb-repair-assistant`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: pick something short, e.g. `pcb-repair-assistant-demo`
4. Open **Advanced settings** and set **Python version 3.12**. No secrets are needed.
5. Click **Deploy**. The first build takes about 5 minutes: Streamlit installs Tesseract from `packages.txt` and the Python packages from `requirements.txt`.

## Before the class demo

- **Warm the cache.** Open the app about 10 minutes before presenting and click through the boards you'll show (board_8601, board_1686, board_9387). Chip reading takes a few seconds per chip on the free CPU the first time, then results are cached.
- **Wake the app.** Free apps go to sleep after a few days without visitors. If you see "This app has gone to sleep", click the wake-up button and wait about a minute.
- **Expect about 0.5–0.6 GB of memory use,** within the free tier's limit (measured locally with the full bench flow and the YOLOv8s detector).
- **Expect a few seconds per new board.** Tiled detection with YOLOv8s takes about 1.5 s on a laptop CPU and may be slower on the free cloud CPU; chip reading adds a few seconds.

## Checked locally before deployment

- 50 automated tests pass (`pytest`), and the GitHub CI run passes.
- Headless run of the whole bench flow on three boards: no errors, 8–11 s per new board (chip reading included).
