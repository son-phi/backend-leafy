# House Plant API — nhận diện LOÀI (DINOv2 ViT-L + head) và BỆNH (pipeline v8), tối ưu cho CPU
# Chạy (từ D:\DATN\houseplant_api):  .venv\Scripts\python main.py   [--port 8000] [--host 0.0.0.0]
import http.client
import ipaddress
import io
import os
import socket
import sys
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import urlsplit

API_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = next(
    (
        directory
        for directory in (API_DIR.parent, API_DIR.parent.parent)
        if (directory / "houseplant_head_best.zip").is_file()
    ),
    API_DIR.parent,
)
CHECKPOINT_PATH = WORKSPACE_DIR / "houseplant_head_best.zip"
DISEASE_ZIP = WORKSPACE_DIR / "disease_artifacts.zip"
DISEASE_ARTIFACTS = Path(os.environ.get("DISEASE_ARTIFACTS", WORKSPACE_DIR / "disease_artifacts"))

# ---- MỌI cache / file tạm / cấu hình đặt trong D:\\DATN (không ghi gì ra ổ C) ----
_D_CACHE = WORKSPACE_DIR / ".cache"
_D_TMP = WORKSPACE_DIR / ".tmp"
for _k, _v in {
    "HF_HOME": _D_CACHE / "huggingface",
    "TORCH_HOME": _D_CACHE / "torch",
    "YOLO_CONFIG_DIR": _D_CACHE / "ultralytics",
    "MPLCONFIGDIR": _D_CACHE / "matplotlib",
    "XDG_CACHE_HOME": _D_CACHE,
    "PIP_CACHE_DIR": _D_CACHE / "pip",
    "TMP": _D_TMP, "TEMP": _D_TMP, "TMPDIR": _D_TMP,
}.items():
    os.environ[_k] = str(_v)          # ghi đè: Windows luôn có TMP/TEMP trỏ về ổ C
    Path(_v).mkdir(parents=True, exist_ok=True)
import tempfile
tempfile.tempdir = os.environ["TMP"]
import warnings
warnings.filterwarnings("ignore", message=".*quantiz.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Trọng số DINOv2 ViT-L nằm sẵn trong D:\DATN\.cache\huggingface → không cần đặt HF_HOME, không gọi mạng khi khởi động
if (Path(os.environ["HF_HOME"]) / "hub" / "models--timm--vit_large_patch14_dinov2.lvd142m").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, HttpUrl
from starlette.concurrency import run_in_threadpool
from torchvision import transforms

sys.path.insert(0, str(API_DIR))
import disease_pipeline as dp

# ---- tối ưu CPU ----
N_THREADS = int(os.environ.get("TORCH_THREADS", os.cpu_count() or 1))
QUANTIZE = os.environ.get("QUANTIZE", "1") == "1"            # ViT-L: lưu trọng số int8 (chỉ trọng số) → ~0,3 GB RAM, độ chính xác như fp32
SPECIES_IMG_SIZE = int(os.environ.get("SPECIES_IMG_SIZE", "518"))   # giống notebook Kaggle (Resize 604 → CenterCrop 518); 336 nhanh ~2,4 lần
MAX_LEAVES = int(os.environ.get("MAX_LEAVES", "4"))          # số lá tối đa model bệnh xét trong 1 ảnh
MAX_SIDE = int(os.environ.get("MAX_SIDE", "1280"))           # thu nhỏ ảnh lớn ngay khi giải mã
INFER_LOCK = threading.Lock()                                # CPU: xử lý tuần tự cho độ trễ ổn định


def set_threads() -> None:
    # ultralytics tự giảm số luồng torch khi import / lần chạy đầu → đặt lại
    if torch.get_num_threads() != N_THREADS:
        torch.set_num_threads(N_THREADS)


def quantize(model: torch.nn.Module) -> torch.nn.Module:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*quantize_per_tensor.*")
    try:
        return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as exc:
        print("Không lượng tử hoá được, dùng fp32:", repr(exc))
        return model
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
RESIZE_SIZE = 604
DEFAULT_TOP_K = 5


class PredictionItem(BaseModel):
    rank: int
    species: str
    confidence: float
    percent: float


class PredictionResponse(BaseModel):
    success: bool
    filename: str | None
    predictions: list[PredictionItem]
    id: int
    label: str
    confidence: float


class URLRequest(BaseModel):
    url: HttpUrl
    top_k: int = DEFAULT_TOP_K


class SpeciesClassifier(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        num_species: int = 47,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(embed_dim, hidden_dim)
        self.act = torch.nn.GELU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim, num_species)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.fc1(features)
        features = self.act(features)
        features = self.dropout1(features)
        return self.fc2(features)


class WeightOnlyInt8Linear(torch.nn.Module):
    """Linear lưu trọng số int8 (scale theo từng kênh ra), giải nén về fp32 khi tính → activation giữ fp32."""

    def __init__(self, lin: torch.nn.Linear) -> None:
        super().__init__()
        w = lin.weight.data.float()
        scale = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
        self.register_buffer("w_int8", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale)
        self.register_buffer("bias", lin.bias.data.float().clone() if lin.bias is not None else None)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.w_int8.to(x.dtype) * self.scale.to(x.dtype), self.bias)


_ST_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def _vitl_weight_file() -> Path:
    snaps = Path(os.environ["HF_HOME"]) / "hub" / "models--timm--vit_large_patch14_dinov2.lvd142m" / "snapshots"
    files = sorted(snaps.glob("*/model.safetensors")) if snaps.is_dir() else []
    if files:
        return files[0]
    # chưa có trong cache → tải 1 lần (cần mạng), file nằm trong D:\\DATN\\.cache\\huggingface
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("timm/vit_large_patch14_dinov2.lvd142m", "model.safetensors"))


def load_vitl_lowmem(image_size: int, quant: bool) -> torch.nn.Module:
    """Nạp DINOv2 ViT-L với bộ nhớ tối thiểu: dựng model rỗng (meta), đọc từng tensor từ file safetensors
    (không memory-map cả file 1,2 GB). quant=True: trọng số mỗi lớp Linear lưu int8 theo từng kênh, phép tính vẫn fp32
    (WeightOnlyInt8Linear) → RAM ~0,3 GB, kết quả như fp32.
    KHÔNG dùng quantize_dynamic: lượng tử hoá cả activation làm hỏng đặc trưng DINOv2 (mọi ảnh ra 'Ctenanthe')."""
    import json
    import struct
    from timm.layers import resample_abs_pos_embed

    with torch.device("meta"):
        model = timm.create_model("vit_large_patch14_dinov2", pretrained=False, num_classes=0, img_size=image_size)
    path = _vitl_weight_file()
    print(f"Đọc trọng số ViT-L từng phần{' + int8' if quant else ''}: {path}")
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    base = 8 + n

    def read(key: str) -> torch.Tensor:
        meta = header[key]
        start, end = meta["data_offsets"]
        f.seek(base + start)
        t = torch.frombuffer(bytearray(f.read(end - start)), dtype=_ST_DTYPES[meta["dtype"]]).reshape(meta["shape"]).float()
        if key == "pos_embed" and t.shape[1] != (image_size // 14) ** 2 + 1:      # 518 px → image_size
            g = image_size // 14
            t = resample_abs_pos_embed(t, new_size=(g, g), num_prefix_tokens=1)
        return t

    def owner(name: str):
        parent, _, attr = name.rpartition(".")
        return (model.get_submodule(parent) if parent else model), attr

    count = 0
    try:
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, torch.nn.Linear):
                continue
            lin = torch.nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None, device="meta")
            lin.weight = torch.nn.Parameter(read(name + ".weight"), requires_grad=False)
            if mod.bias is not None:
                lin.bias = torch.nn.Parameter(read(name + ".bias"), requires_grad=False)
            if quant:
                lin = WeightOnlyInt8Linear(lin)
            parent, attr = owner(name)
            setattr(parent, attr, lin)
            count += 2 if mod.bias is not None else 1
        for pname, p in list(model.named_parameters()):
            if p.is_meta:
                parent, attr = owner(pname)
                setattr(parent, attr, torch.nn.Parameter(read(pname), requires_grad=False))
                count += 1
    finally:
        f.close()
    left = [k for k, v in list(model.named_parameters()) + list(model.named_buffers()) if v.is_meta]
    if left:
        raise RuntimeError(f"Thiếu trọng số ViT-L: {left[:5]}")
    print(f"ViT-L: nạp {count} tensor")
    return model.eval()


class HouseplantPredictor:
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: SpeciesClassifier,
        labels: list[str],
        device: torch.device,
        image_size: int,
    ) -> None:
        self.backbone = backbone
        self.head = head
        self.labels = labels
        self.device = device
        self.image_size = image_size
        self.backbone_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    RESIZE_SIZE,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @classmethod
    def load(cls, checkpoint_path: Path = CHECKPOINT_PATH) -> "HouseplantPredictor":
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a dictionary.")

        label_map = checkpoint.get("label_map")
        head_state_dict = checkpoint.get("head_state_dict")
        if not isinstance(label_map, dict) or not isinstance(head_state_dict, dict):
            raise ValueError("Checkpoint is missing label_map or head_state_dict.")

        labels_by_id = sorted(label_map.items(), key=lambda item: item[1])
        labels = [str(label) for label, _ in labels_by_id]
        class_ids = [class_id for _, class_id in labels_by_id]
        if not labels or class_ids != list(range(len(labels))):
            raise ValueError("Checkpoint label IDs must be contiguous and start at 0.")

        config = checkpoint.get("config", {})
        image_size = SPECIES_IMG_SIZE      # notebook Kaggle chạy tốt với 518 (checkpoint ghi 336)
        if image_size <= 0 or image_size % 14 != 0:
            raise ValueError("Checkpoint input_size must be a positive multiple of 14.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading DINOv2 backbone...")
        quant = device.type == "cpu" and QUANTIZE
        backbone = load_vitl_lowmem(image_size, quant)
        backbone = backbone.to(
            device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).eval()
        if quant:
            print("Species backbone: trọng số int8, tính fp32")
        import gc
        gc.collect()

        print("Loading classifier checkpoint...")
        head = SpeciesClassifier(num_species=len(labels))
        head.load_state_dict(head_state_dict)
        head = head.to(device=device, dtype=torch.float32).eval()
        print(f"Model loaded. Device: {device}; image size: {image_size}; species: {len(labels)}")

        return cls(backbone, head, labels, device, image_size)

    def predict(
        self,
        image: Image.Image,
        filename: str | None,
        top_k: int,
    ) -> PredictionResponse:
        image_tensor = self.transform(image).unsqueeze(0).to(
            device=self.device,
            dtype=self.backbone_dtype,
        )

        with INFER_LOCK, torch.inference_mode():
            set_threads()
            features = self.backbone(image_tensor).float()
            probabilities = torch.softmax(self.head(features), dim=1)[0]
            top_probabilities, top_indices = torch.topk(
                probabilities,
                min(top_k, len(self.labels)),
            )

        predictions = [
            PredictionItem(
                rank=rank,
                species=self.labels[int(class_id)],
                confidence=round(float(probability), 4),
                percent=round(float(probability) * 100, 2),
            )
            for rank, (probability, class_id) in enumerate(
                zip(top_probabilities.tolist(), top_indices.tolist()),
                start=1,
            )
        ]
        top_class_id = int(top_indices[0].item())
        return PredictionResponse(
            success=True,
            filename=filename,
            predictions=predictions,
            id=top_class_id,
            label=self.labels[top_class_id],
            confidence=float(top_probabilities[0].item()),
        )


def decode_image(contents: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large.")
            image.draft("RGB", (MAX_SIDE, MAX_SIDE))      # JPEG: giải mã thẳng ở độ phân giải nhỏ hơn → nhanh
            image = ImageOps.exif_transpose(image).convert("RGB")   # xoay đúng chiều ảnh chụp điện thoại
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Cannot read image. Please upload a valid image file.",
        ) from exc


def validate_top_k(top_k: int, species_count: int) -> None:
    if top_k < 1 or top_k > species_count:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be between 1 and {species_count}.",
        )


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, address: str, timeout: int) -> None:
        super().__init__(hostname, port=443, timeout=timeout)
        self.address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self.address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


def download_public_image(url: str) -> bytes:
    parsed_url = urlsplit(url)
    if (
        parsed_url.scheme != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.port not in (None, 443)
    ):
        raise HTTPException(status_code=400, detail="Only public HTTPS image URLs are allowed.")

    hostname = parsed_url.hostname
    try:
        addresses = {
            record[4][0]
            for record in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        }
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Unable to resolve image URL.") from exc

    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise HTTPException(status_code=400, detail="Image URL must resolve to a public IP address.")

    target = parsed_url.path or "/"
    if parsed_url.query:
        target = f"{target}?{parsed_url.query}"

    connection = PinnedHTTPSConnection(hostname, sorted(addresses)[0], timeout=15)
    try:
        connection.request("GET", target, headers={"User-Agent": "HouseplantClassifier/1.0"})
        response = connection.getresponse()
        if response.status != 200:
            raise HTTPException(status_code=400, detail="Image URL did not return HTTP 200.")
        if not response.getheader("Content-Type", "").lower().startswith("image/"):
            raise HTTPException(status_code=400, detail="URL did not return an image.")

        contents = response.read(MAX_IMAGE_BYTES + 1)
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
        return contents
    except HTTPException:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise HTTPException(status_code=400, detail="Unable to download image from URL.") from exc
    finally:
        connection.close()


def load_disease_pipeline() -> dict:
    if not (DISEASE_ARTIFACTS / "pipeline_config.json").is_file():
        if not DISEASE_ZIP.is_file():
            raise FileNotFoundError(f"Không thấy {DISEASE_ARTIFACTS} hoặc {DISEASE_ZIP}")
        print(f"Giải nén {DISEASE_ZIP.name} → {DISEASE_ARTIFACTS}")
        with zipfile.ZipFile(DISEASE_ZIP) as z:
            z.extractall(DISEASE_ARTIFACTS)
    info = dp.load(str(DISEASE_ARTIFACTS), workdir=str(DISEASE_ARTIFACTS / "_restored"))
    # model bệnh nhỏ (ViT-S) → giữ fp32, không lượng tử hoá (tránh sai lệch như ViT-L)
    dp.CFG["MAX_LEAVES"] = MAX_LEAVES
    return info


def predict_disease_sync(image: Image.Image, return_image: bool) -> dict:
    with INFER_LOCK:
        set_threads()
        return dp.predict(image, return_image)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    t0 = time.time()
    set_threads()
    app.state.predictor = HouseplantPredictor.load()
    app.state.disease_info = load_disease_pipeline()
    print("Disease pipeline:", app.state.disease_info)
    dummy = Image.new("RGB", (640, 480), (60, 140, 60))     # chạy nóng: request đầu tiên không bị chậm
    app.state.predictor.predict(dummy, None, 1)
    predict_disease_sync(dummy, False)
    set_threads()
    print(f"Sẵn sàng sau {time.time() - t0:.1f}s | torch threads: {torch.get_num_threads()} | int8: {QUANTIZE}")
    yield


app = FastAPI(
    title="House Plant Classifier API",
    description="Plant species + leaf disease recognition (DINOv2), CPU-optimized.",
    version="2.0.0",
    lifespan=lifespan,
)
# ============================================================
# LOG LƯỢT GỌI + KẾT QUẢ: in ra cửa sổ và ghi file logs/requests_YYYYMMDD.log (mỗi dòng 1 JSON)
# ============================================================
import json as _json
from fastapi import Request
from fastapi.responses import Response

LOG_DIR = API_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
_LOG_LOCK = threading.Lock()


def _summary(path: str, data: dict) -> dict:
    """Rút gọn kết quả để log (bỏ ảnh base64)."""
    out: dict = {}
    sp = data.get("species") if "species" in data else (data if "predictions" in data else None)
    if isinstance(sp, dict) and sp.get("label") is not None:
        out["species"] = sp.get("label")
        out["species_conf"] = round(float(sp.get("confidence", 0)) * 100, 1)
    dis = data.get("disease") if isinstance(data.get("disease"), dict) else data.get("result")
    if isinstance(dis, dict) and "disease_id" in dis:
        out.update(disease=dis.get("disease"), disease_conf=dis.get("confidence"),
                   severity=(dis.get("severity") or {}).get("percent_area"), level=(dis.get("severity") or {}).get("level"))
        if dis.get("note"):
            out["note"] = dis["note"]
    if "detail" in data:
        out["error"] = data["detail"]
    return out


def _write_log(rec: dict) -> None:
    with _LOG_LOCK, open(LOG_DIR / f"requests_{time.strftime('%Y%m%d')}.log", "a", encoding="utf-8") as fh:
        fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")


# Khóa API (tùy chọn): đặt biến môi trường LEAFY_API_KEY thì mọi request (trừ các đường dẫn mở) phải gửi header X-API-Key
API_KEY = os.environ.get("LEAFY_API_KEY", "").strip()
OPEN_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}


def _client_of(request: Request) -> str:
    """IP thật của client; khi đi qua ngrok lấy từ X-Forwarded-For."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip() + " (ngrok)"
    return request.client.host if request.client else "?"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    client = _client_of(request)
    size = request.headers.get("content-length", "")
    print(f"[{time.strftime('%H:%M:%S')}] <- {client} {request.method} {request.url.path}"
          f"{'?' + request.url.query if request.url.query else ''} {size + ' bytes' if size else ''}", flush=True)
    if API_KEY and request.method != "OPTIONS" and request.url.path not in OPEN_PATHS \
            and request.headers.get("x-api-key", "") != API_KEY:
        print(f"[{time.strftime('%H:%M:%S')}] -> {client} {request.url.path} 401 sai/thiếu X-API-Key", flush=True)
        _write_log(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), client=client, method=request.method,
                        path=request.url.path, status=401, error="invalid api key"))
        return Response(content=_json.dumps({"success": False, "detail": "Thiếu hoặc sai X-API-Key"}),
                        status_code=401, media_type="application/json")
    try:
        response = await call_next(request)
    except Exception as exc:
        ms = int((time.time() - t0) * 1000)
        print(f"[{time.strftime('%H:%M:%S')}] -> {client} {request.url.path} LỖI {exc!r} ({ms} ms)", flush=True)
        _write_log(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), client=client, method=request.method,
                        path=request.url.path, status=500, ms=ms, error=repr(exc)))
        raise
    body = b"".join([chunk async for chunk in response.body_iterator])
    ms = int((time.time() - t0) * 1000)
    summ: dict = {}
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            data = _json.loads(body)
            if isinstance(data, dict):
                summ = _summary(request.url.path, data)
        except ValueError:
            pass
    if request.method == "POST" or response.status_code >= 400:
        print(f"[{time.strftime('%H:%M:%S')}] -> {client} {request.url.path} {response.status_code} ({ms} ms) "
              + _json.dumps(summ, ensure_ascii=False), flush=True)
        _write_log(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), client=client, method=request.method,
                        path=request.url.path, query=request.url.query, bytes=size, status=response.status_code,
                        ms=ms, **summ))
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return Response(content=body, status_code=response.status_code, headers=headers, media_type=response.media_type)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, object]:
    predictor: HouseplantPredictor = app.state.predictor
    return {
        "success": True,
        "message": "House Plant Classifier API",
        "device": str(predictor.device),
        "image_size": predictor.image_size,
        "species_count": len(predictor.labels),
        "disease_classes": len(dp.CLASS_NAMES),
        "disease_version": app.state.disease_info["version"],
    }


@app.get("/health")
def health() -> dict[str, object]:
    predictor: HouseplantPredictor = app.state.predictor
    return {"status": "healthy", "device": str(predictor.device), "model_loaded": True,
            "disease_model_loaded": dp.PIPE is not None}


@app.get("/species")
def get_species() -> dict[str, object]:
    predictor: HouseplantPredictor = app.state.predictor
    return {
        "success": True,
        "species_count": len(predictor.labels),
        "species": predictor.labels,
    }


@app.post("/predict")
@app.post("/predict_plant", include_in_schema=False)
async def predict(
    file: UploadFile = File(...),
    top_k: int = DEFAULT_TOP_K,
) -> PredictionResponse:
    predictor: HouseplantPredictor = app.state.predictor
    validate_top_k(top_k, len(predictor.labels))

    if file.content_type and not file.content_type.lower().startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read(MAX_IMAGE_BYTES + 1)
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

        image = decode_image(contents)
        return await run_in_threadpool(predictor.predict, image, file.filename, top_k)
    finally:
        await file.close()


@app.post("/predict_url")
def predict_url(data: URLRequest) -> PredictionResponse:
    predictor: HouseplantPredictor = app.state.predictor
    validate_top_k(data.top_k, len(predictor.labels))
    url = str(data.url)
    contents = download_public_image(url)
    image = decode_image(contents)
    filename = Path(urlsplit(url).path).name or None
    return predictor.predict(image, filename, data.top_k)


# ============================================================
# BỆNH LÁ
# ============================================================

class DiseaseURLRequest(BaseModel):
    url: HttpUrl
    return_image: bool = False


async def read_upload(file: UploadFile) -> Image.Image:
    if file.content_type and not file.content_type.lower().startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    return decode_image(contents)


@app.get("/diseases")
def get_diseases() -> dict[str, object]:
    return {
        "success": True,
        "version": app.state.disease_info["version"],
        "diseases": [{"id": c, "name": dp.NAME_VI.get(c, c)} for c in dp.CLASS_NAMES],
        "severity_levels": [lab for _, _, lab in dp.CFG["SEVERITY_SCALE"]],
    }


@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...), return_image: bool = False) -> dict[str, object]:
    """Bệnh, độ tin cậy, mức độ. return_image=true: thêm ảnh vùng bệnh (base64 JPEG)."""
    try:
        image = await read_upload(file)
        result = await run_in_threadpool(predict_disease_sync, image, return_image)
        return {"success": True, "filename": file.filename, "result": result}
    finally:
        await file.close()


@app.post("/predict_disease_url")
async def predict_disease_url(data: DiseaseURLRequest) -> dict[str, object]:
    contents = await run_in_threadpool(download_public_image, str(data.url))
    image = decode_image(contents)
    result = await run_in_threadpool(predict_disease_sync, image, data.return_image)
    return {"success": True, "url": str(data.url), "result": result}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), top_k: int = 3, return_image: bool = False) -> dict[str, object]:
    """Một lần gọi: loài (top_k) + bệnh."""
    predictor: HouseplantPredictor = app.state.predictor
    validate_top_k(top_k, len(predictor.labels))
    try:
        image = await read_upload(file)
        t0 = time.time()
        species = await run_in_threadpool(predictor.predict, image, file.filename, top_k)
        disease = await run_in_threadpool(predict_disease_sync, image, return_image)
        return {"success": True, "filename": file.filename, "species": species, "disease": disease,
                "elapsed_ms": int((time.time() - t0) * 1000)}
    finally:
        await file.close()


# ============================================================
# CHẠY TRỰC TIẾP: python main.py
# ============================================================

_VIRTUAL = ("vmware", "virtualbox", "vbox", "vethernet", "hyper-v", "wsl", "loopback", "docker", "bluetooth", "vpn", "tap", "zerotier", "hamachi")


def _lan_ips() -> list[tuple[str, str, bool]]:
    """(ip, tên card mạng, là card ảo?) — app trên điện thoại phải dùng IP của card Wi-Fi/Ethernet thật."""
    main_ip = None
    try:
        s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s_.connect(("8.8.8.8", 80)); main_ip = s_.getsockname()[0]; s_.close()
    except OSError:
        pass
    out = []
    try:
        import psutil
        stats = psutil.net_if_stats()
        for name, addrs in psutil.net_if_addrs().items():
            if name in stats and not stats[name].isup:
                continue
            for a in addrs:
                if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")):
                    out.append((a.address, name, any(v in name.lower() for v in _VIRTUAL) and a.address != main_ip))
    except Exception:
        if main_ip:
            out.append((main_ip, "?", False))
    out.sort(key=lambda t: (t[0] != main_ip, t[2]))
    return out


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Houseplant API (loài + bệnh), CPU")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ngrok", metavar="DOMAIN", default=os.environ.get("NGROK_DOMAIN", ""),
                    help="Mở tunnel ngrok với tên miền tĩnh, vd leafy-abc.ngrok-free.app (hoặc đặt biến NGROK_DOMAIN)")
    args = ap.parse_args()
    ngrok_proc = None
    if args.ngrok:
        import atexit
        import shutil
        import subprocess
        _la = Path(os.environ.get("LOCALAPPDATA", ""))
        _cands = [WORKSPACE_DIR / "tools" / "ngrok.exe", WORKSPACE_DIR / "ngrok.exe",
                  _la / "Microsoft" / "WinGet" / "Links" / "ngrok.exe"]
        _cands += sorted((_la / "Microsoft" / "WinGet" / "Packages").glob("*grok*/ngrok.exe"))
        exe = shutil.which("ngrok") or next((str(c) for c in _cands if c.is_file()), None)
        if not exe:
            print("!! Không tìm thấy ngrok.exe (PATH, thư mục WinGet, D:\\DATN\\tools). Mở PowerShell mới sau khi cài, "
                  "hoặc chạy riêng: ngrok http --url=<domain> 8000")
        else:
            domain = args.ngrok.replace("https://", "").replace("http://", "").strip("/")
            ngrok_proc = subprocess.Popen([exe, "http", f"--url={domain}", str(args.port), "--log", "stdout"],
                                          stdout=open(LOG_DIR / "ngrok.log", "a", encoding="utf-8"),
                                          stderr=subprocess.STDOUT)
            atexit.register(ngrok_proc.terminate)
            args.ngrok = domain
    print("=" * 60)
    print(f"Python      : {sys.executable}")
    print(f"Thư mục     : {WORKSPACE_DIR}")
    print(f"Cache       : {os.environ['HF_HOME']} | tạm: {os.environ['TMP']}")
    print(f"Docs        : http://127.0.0.1:{args.port}/docs")
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        print(f"RAM         : trống {vm.available / 2**30:.1f} / {vm.total / 2**30:.1f} GB | bộ nhớ ảo (pagefile) {sw.total / 2**30:.1f} GB")
        others = [p for p in psutil.process_iter(["name", "pid", "memory_info", "cmdline"])
                  if p.info["pid"] != os.getpid() and "python" in (p.info["name"] or "").lower()
                  and any("main.py" in c or "uvicorn" in c for c in (p.info["cmdline"] or []))]
        for p in others:
            print(f"  !! Server cũ còn chạy: PID {p.info['pid']} ({p.info['memory_info'].rss / 2**30:.1f} GB) — tắt bằng: taskkill /PID {p.info['pid']} /F")
        if vm.available < 2.5 * 2**30:
            print("  !! RAM trống < 2,5 GB: đóng bớt ứng dụng (Android Studio, emulator, Chrome...) hoặc tăng pagefile trên ổ D")
    except Exception:
        pass
    print(f"Log         : {LOG_DIR}\\requests_YYYYMMDD.log")
    for ip, name, virtual in _lan_ips():
        tag = "card ảo — KHÔNG dùng cho điện thoại" if virtual else "dùng cho app (điện thoại cùng mạng)"
        print(f"App mobile  : http://{ip}:{args.port}/   [{name}] {tag}")
    print(f"Thử từ trình duyệt điện thoại: http://<IP ở trên>:{args.port}/health")
    if ngrok_proc:
        print(f"Internet    : https://{args.ngrok}/   (BASE_URL cho app, mọi mạng) | log: {LOG_DIR}\\ngrok.log")
        print(f"Thử         : https://{args.ngrok}/health")
    print(f"Khóa API    : {'BẬT — app phải gửi header X-API-Key' if API_KEY else 'tắt (đặt LEAFY_API_KEY để bật)'}")
    print("=" * 60, flush=True)
    # proxy_headers: nhận đúng scheme/IP client khi đi qua ngrok
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                proxy_headers=True, forwarded_allow_ips="*")
