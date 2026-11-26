# Houseplant API — loài + bệnh (CPU)

## Cài đặt (một lần) — mọi thứ trên ổ D
Mở **PowerShell**:
```powershell
cd D:\DATN\houseplant_api
$env:PIP_CACHE_DIR = "D:\DATN\.cache\pip"; $env:TMP = "D:\DATN\.tmp"; $env:TEMP = "D:\DATN\.tmp"
mkdir D:\DATN\.tmp -Force | Out-Null
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

## Chạy
```powershell
cd D:\DATN\houseplant_api
.venv\Scripts\python main.py            # thêm --port 8000 --host 0.0.0.0 nếu cần
```
Cửa sổ in đường dẫn Python, thư mục cache, địa chỉ cho app mobile, rồi log từng request kèm thời gian xử lý (ms). Dừng: `Ctrl + C`.
`main.py` tự đặt cache (Hugging Face, torch, ultralytics, matplotlib) vào `D:\DATN\.cache` và file tạm vào `D:\DATN\.tmp`.

Lần đầu khởi động tự giải nén `..\disease_artifacts.zip` → `..\disease_artifacts\`. Trọng số DINOv2 ViT-L lấy từ `..\.cache\huggingface` (đã có sẵn, không cần mạng).
Khi log hiện `Sẵn sàng sau ...s` là gọi được.

## Endpoint
| Method | Path | Ghi chú |
|---|---|---|
| POST | `/predict` | Loài (giữ nguyên như cũ: `predictions`, `id`, `label`, `confidence`) |
| POST | `/predict_disease` | form-data `file`; `?return_image=true` để nhận ảnh vùng bệnh (base64 JPEG) |
| POST | `/predict_disease_url` | JSON `{"url": "https://...", "return_image": false}` |
| POST | `/analyze` | form-data `file`; `?top_k=3` → loài + bệnh trong 1 lần gọi |
| GET | `/diseases`, `/species`, `/health` | |

Kết quả bệnh: `{"disease": "Phấn trắng", "disease_id": "powdery_coating", "confidence": 99.8, "severity": {"percent_area": 27.1, "level": "Trung bình"}}` — `note` chỉ có khi không chắc chắn.

## Tuỳ chỉnh tốc độ (biến môi trường)
`QUANTIZE=0` tắt int8 · `MAX_LEAVES` (mặc định 4) · `MAX_SIDE` (mặc định 1280) · `TORCH_THREADS`.

## App mobile
Thiết bị thật: `http://<IPv4 máy tính>:8000/` (cùng Wi-Fi; cho phép cổng 8000 trong Windows Firewall). Emulator Android: `http://10.0.2.2:8000/`. Ngoài mạng LAN: `ngrok http 8000`.

## Gọi từ điện thoại ở mạng khác (ngrok)

1. Đăng ký tài khoản miễn phí tại ngrok.com, tải `ngrok.exe` vào `D:\DATN\tools\` (hoặc đặt trong PATH).
2. Lưu token (chỉ làm một lần):
   ```powershell
   D:\DATN\tools\ngrok.exe config add-authtoken <token>
   ```
3. Trên dashboard ngrok, mục **Domains**, lấy tên miền tĩnh miễn phí, ví dụ `leafy-abc.ngrok-free.app`.
4. Chạy server kèm tunnel bằng một lệnh:
   ```powershell
   .venv\Scripts\python main.py --ngrok leafy-abc.ngrok-free.app
   ```
   Hoặc đặt biến một lần cho cửa sổ PowerShell: `$env:NGROK_DOMAIN = "leafy-abc.ngrok-free.app"`.
5. Kiểm tra trên trình duyệt điện thoại: `https://leafy-abc.ngrok-free.app/health`.
6. Trong app: `BASE_URL = "https://leafy-abc.ngrok-free.app/"` (HTTPS nên không cần network_security_config).
   Nên thêm header `ngrok-skip-browser-warning: true` vào OkHttp để tránh trang cảnh báo của ngrok.

Log của ngrok ghi ở `logs\ngrok.log`; request đi qua ngrok hiện trong log server với IP thật của điện thoại và nhãn `(ngrok)`.

### Khóa API (nên bật khi dùng ngrok vì địa chỉ là công khai)

```powershell
$env:LEAFY_API_KEY = "mot-chuoi-bi-mat"
.venv\Scripts\python main.py --ngrok leafy-abc.ngrok-free.app
```

Khi bật, mọi request trừ `/`, `/health`, `/docs` phải gửi header `X-API-Key: mot-chuoi-bi-mat`, nếu không server trả 401.
Không đặt biến thì không kiểm tra khóa (chạy như cũ).
