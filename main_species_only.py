import http.client
import ipaddress
import io
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import urlsplit

import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, HttpUrl
from torchvision import transforms


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
        image_size = int(config.get("input_size", 336))
        if image_size <= 0 or image_size % 14 != 0:
            raise ValueError("Checkpoint input_size must be a positive multiple of 14.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading DINOv2 backbone...")
        backbone = timm.create_model(
            "vit_large_patch14_dinov2",
            pretrained=True,
            num_classes=0,
            img_size=image_size,
        )
        backbone = backbone.to(
            device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).eval()

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

        with torch.inference_mode():
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
            return image.convert("RGB")
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.predictor = HouseplantPredictor.load()
    yield


app = FastAPI(
    title="House Plant Classifier API",
    description="Plant species classification using DINOv2.",
    version="1.0.0",
    lifespan=lifespan,
)
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
    }


@app.get("/health")
def health() -> dict[str, object]:
    predictor: HouseplantPredictor = app.state.predictor
    return {"status": "healthy", "device": str(predictor.device), "model_loaded": True}


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
        return predictor.predict(image, file.filename, top_k)
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
