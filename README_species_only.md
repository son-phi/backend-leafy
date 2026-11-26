# Houseplant prediction API

This API loads `houseplant_head_best.zip` from the workspace root and serves
`POST /predict`, the multipart endpoint used by the Android app. The response
includes the top-five `predictions` plus `id`, `label`, and `confidence`.

## Run

Run these commands from the `houseplant_api` directory (the directory containing
`requirements.txt`):

```powershell
python -m pip install -r requirements.txt
$env:HF_HOME = "D:\DATN\.cache\huggingface"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The checkpoint is expected in the parent workspace directory
(`..\houseplant_head_best.zip`).
`HF_HOME` keeps the pretrained DINOv2 weights used by `timm` on the D: drive.
Set it before starting Uvicorn in each new PowerShell window. The first start
needs internet access to download the DINOv2 ViT-L/14 backbone.
The API uses CUDA when available and otherwise runs on CPU.

The preprocessing resizes the shorter image side to 604, then center-crops to
the input size recorded in the checkpoint (336 for the current model).
`POST /predict` accepts multipart field `file` and optional `top_k`. The
`/predict_url` endpoint accepts public HTTPS image URLs only.

Open `http://127.0.0.1:8000/docs` to try the endpoint. Point
`ApiClient.BASE_URL` at this server (or its current ngrok URL). On a physical
Android device, use the computer's LAN IP and port `8000`; for an Android
emulator, use `http://10.0.2.2:8000/`.
