# disease_pipeline.py — suy luận bệnh lá cây cảnh (pipeline v8), độc lập với notebook train
# Dùng:  import disease_pipeline as dp; dp.load('/đường/dẫn/artifacts'); res = dp.predict(ảnh_RGB_numpy_hoặc_PIL)
# Thư mục artifacts: leaf_yolo11n_seg.pt, lesion_dinov2.pt, stacker.pkl, ood_stats.npz, pipeline_config.json
import os, io, json, pickle, zipfile, tempfile, threading, base64
import numpy as np, pandas as pd, cv2, torch, timm
import torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageOps

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CFG = {}
_LOCK = threading.Lock()
PIPE = None

# ---------- khôi phục file .pt/.npz bị Kaggle giải nén thành thư mục ----------
def _as_file(path, workdir):
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    root = os.path.splitext(os.path.basename(path.rstrip('/')))[0]
    out = os.path.join(workdir, root + '.pt'); os.makedirs(workdir, exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:     # torch.load cần mọi file nằm trong 1 thư mục gốc
        for r, _, fs in os.walk(path):
            for f in fs:
                full = os.path.join(r, f)
                zi = zipfile.ZipInfo(root + '/' + os.path.relpath(full, path).replace(os.sep, '/'), (1980, 1, 1, 0, 0, 0))
                with open(full, 'rb') as fh:
                    z.writestr(zi, fh.read(), zipfile.ZIP_STORED)
    return out

def _find(art, name):
    for cand in [name, os.path.splitext(name)[0]]:
        p = os.path.join(art, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Không thấy {name} trong {art}')

# ================= tiện ích ảnh =================
IMG_MAX = 1024

def load_rgb(src):
    im = src if isinstance(src, Image.Image) else Image.open(src)
    return np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(im).convert('RGB')))

def resize_max(a, m=IMG_MAX, interp=cv2.INTER_AREA):
    h, w = a.shape[:2]; s = m / max(h, w)
    if s >= 1:
        return a, 1.0
    return cv2.resize(a, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp), s

def mask_box(m):
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

def box_mask(box, H, W):
    m = np.zeros((H, W), bool); x1, y1, x2, y2 = [int(round(v)) for v in box]
    m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    return m

def square_box(box, pad=0.08):
    x1, y1, x2, y2 = box; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(32.0, max(x2 - x1, y2 - y1) * (1 + 2 * pad))
    return int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))

def crop_square(a, sq, out, interp, fill=0):
    x0, y0, side = sq; H, W = a.shape[:2]
    canvas = np.full((side, side) + a.shape[2:], fill, a.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
    if xb > xa and yb > ya:
        canvas[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]
    return cv2.resize(canvas, (out, out), interpolation=interp)

def paste_back(small, sq, H, W, interp):
    x0, y0, side = sq
    big = cv2.resize(small, (side, side), interpolation=interp)
    out = np.zeros((H, W) + small.shape[2:], small.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
    if xb > xa and yb > ya:
        out[ya:yb, xa:xb] = big[ya - y0:yb - y0, xa - x0:xb - x0]
    return out

def quality_check(rgb):
    g = cv2.cvtColor(resize_max(rgb, 512)[0], cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var()); mean = float(g.mean() / 255)
    reasons = []
    if blur < CFG['BLUR_MIN']: reasons.append('blur')
    if mean < CFG['DARK_MEAN']: reasons.append('too_dark')
    if mean > CFG['BRIGHT_MEAN']: reasons.append('too_bright')
    ok = not reasons
    if min(rgb.shape[:2]) < 320: reasons.append('low_res')     # chỉ cảnh báo
    return dict(blur=round(blur, 1), brightness=round(mean, 3), ok=ok, reasons=reasons)

def color_mask(sym, lesion, nec_label='leaf_spot'):
    # sym: kênh loại tổn thương mỗi pixel; vùng hoại tử tô màu của bệnh mà lá được kết luận (đốm/cháy lá/cháy mép)
    out = np.zeros(sym.shape + (3,), np.uint8)
    for t, ch in TYPE_CH.items():
        col = COLOR.get(nec_label if t == 'necrosis' else TYPE2SYM.get(t, nec_label), (200, 200, 200))
        out[(sym == ch) & lesion] = col
    return out

def blend(img, cm, m, a=0.6):
    out = img.astype(np.float32).copy()
    out[m] = (1 - a) * out[m] + a * cm[m]
    return out.astype(np.uint8)

def draw_contours(img, mask, color=(255, 255, 255), th=2):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy(); cv2.drawContours(out, cs, -1, color, th)
    return out

# ================= GĐ1: tìm lá (YOLO11n-seg) =================
def postprocess_leaves(r, img):
    H, W = img.shape[:2]; out = []
    if r.masks is None:
        return out
    for poly, cf in zip(r.masks.xy, r.boxes.conf.cpu().numpy()):
        if len(poly) < 3:
            continue
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1); m = m.astype(bool)
        a = int(m.sum())
        if a >= CFG['LEAF_MIN_AREA'] * H * W:
            out.append(dict(mask=m, conf=float(cf), area=a))
    out.sort(key=lambda d: -d['conf'] * d['area'])
    keep = []
    for d in out:
        if any((d['mask'] & k['mask']).sum() / d['area'] > 0.8 for k in keep):
            continue
        keep.append(d)
        if len(keep) >= CFG['MAX_LEAVES']:
            break
    edge = np.zeros((H, W), bool); edge[[0, -1], :] = True; edge[:, [0, -1]] = True
    for d in keep:
        d['box'] = mask_box(d['mask']); x1, y1, x2, y2 = d['box']
        d['cut_by_border'] = bool((d['mask'] & edge).sum() > 0.15 * 2 * ((x2 - x1) + (y2 - y1)))
    return keep

def detect_leaves(imgs):
    res = LEAF_MODEL.predict([cv2.cvtColor(a, cv2.COLOR_RGB2BGR) for a in imgs], imgsz=CFG['YOLO_IMGSZ'],
                             conf=CFG['LEAF_CONF'], verbose=False, device=0 if DEVICE == 'cuda' else 'cpu')
    return [postprocess_leaves(r, a) for r, a in zip(res, imgs)]

# ================= GĐ2: LesionNet =================
def conv_bn(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.GELU())

def torch_geom(les, leaf4, x, steps=16):
    # les, leaf4: (B,1,h,w) → (B,4): [tỉ lệ bệnh ở dải mép, khoảng cách TB tới mép, tỉ lệ diện tích bệnh, tỉ lệ vàng]
    lf = (leaf4 > 0.5).float(); cur = 1 - lf; dist = torch.zeros_like(lf)
    for _ in range(steps):
        cur = F.max_pool2d(cur, 3, 1, 1); dist = dist + (1 - cur)
    area = lf.sum((2, 3), keepdim=True).clamp(min=1); sz = area.sqrt()
    band = (dist <= CFG['MARGIN_ALPHA'] * sz).float() * lf
    le = les * lf; tot = le.sum((2, 3)).clamp(min=1e-3)
    r_margin = (le * band).sum((2, 3)) / tot
    mdist = (le * dist / sz).sum((2, 3)) / tot
    frac = le.sum((2, 3)) / area[:, :, 0, 0]
    rgb = F.interpolate(x * STD_T + MEAN_T, size=les.shape[-2:], mode='area')
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    yel = ((r > 0.45) & (g > 0.40) & (b < 0.55 * g) & (r > 0.75 * g)).float() * lf
    return torch.cat([r_margin, mdist, frac, yel.sum((2, 3)) / area[:, :, 0, 0]], 1)

class LesionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bb = timm.create_model(cfg['BACKBONE'], pretrained=cfg['PRETRAINED'], num_classes=0, img_size=cfg['IMG_SIZE'])
        D = self.bb.embed_dim; self.npt = self.bb.num_prefix_tokens; nb = len(self.bb.blocks)
        self.k_frozen = nb - cfg['TRAIN_BLOCKS']; self.mid = min(cfg['MID_BLOCK'], nb - 1)
        for n, p in self.bb.named_parameters():
            p.requires_grad = any(n.startswith(f'blocks.{i}.') for i in range(self.k_frozen, nb)) or n.startswith('norm.')
        self.lat = conv_bn(2 * D, 256, 1); self.up1 = conv_bn(256, 128); self.up2 = conv_bn(128, 64)
        self.stem = nn.Sequential(conv_bn(3, 32), nn.MaxPool2d(2), conv_bn(32, 48), nn.MaxPool2d(2))
        self.fuse = nn.Sequential(conv_bn(64 + 48, 64), nn.Conv2d(64, NSEG, 1))
        self.use_cls, self.leaf_drop = cfg['USE_CLS_TOKEN'], cfg['LEAF_DROP']
        self.proj = nn.Sequential(nn.Linear((3 if self.use_cls else 2) * D + 4 + len(LESION_TYPES), 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 256))
        self.protos = nn.Parameter(torch.randn(NCLS, 256) * 0.02); self.scale = cfg['COS_SCALE']
    def tokens(self, x):
        bb = self.bb
        x = bb.norm_pre(bb.patch_drop(bb._pos_embed(bb.patch_embed(x)))); mid = None
        for i, blk in enumerate(bb.blocks):
            if i < self.k_frozen:
                with torch.no_grad():
                    x = blk(x)
            else:
                x = blk(x)
            if i == self.mid:
                mid = F.layer_norm(x, x.shape[-1:])
        return bb.norm(x), mid
    def forward(self, x, leafg, leaf4):
        tok, mid = self.tokens(x)
        cls, pt, mt = tok[:, 0], tok[:, self.npt:], mid[:, self.npt:]
        B, N, D = pt.shape; g = int(round(N ** 0.5))
        f = torch.cat([pt, mt], -1).transpose(1, 2).reshape(B, 2 * D, g, g)
        d = self.lat(f)
        d = self.up1(F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False))
        d = self.up2(F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False))
        s = self.stem(x)
        d = F.interpolate(d, size=s.shape[-2:], mode='bilinear', align_corners=False)
        seg = self.fuse(torch.cat([d, s], 1))
        if seg.shape[-1] != leaf4.shape[-1]:
            seg = F.interpolate(seg, size=leaf4.shape[-2:], mode='bilinear', align_corners=False)
        les = seg.float().softmax(1)[:, 2:].sum(1, keepdim=True)
        w = (F.adaptive_avg_pool2d(les.detach(), g) * leafg).flatten(1)
        lw = leafg.flatten(1)
        ptf = pt.float()
        z_les = (w[..., None] * ptf).sum(1) / (w.sum(1, keepdim=True) + 1e-4)
        z_leaf = (lw[..., None] * ptf).sum(1) / (lw.sum(1, keepdim=True) + 1e-4)
        geo = torch_geom(les.detach(), leaf4, x)
        lf4 = (leaf4 > 0.5).float()
        tshare = (seg.float().softmax(1)[:, 2:].detach() * lf4).sum((2, 3)) / lf4.sum((2, 3)).clamp(min=1)   # tỉ lệ từng loại tổn thương
        if self.training and self.leaf_drop > 0:     # buộc head dựa vào vùng bệnh thay vì nhận dạng loài cây
            z_leaf = z_leaf * (torch.rand(B, 1, device=z_leaf.device) >= self.leaf_drop).float()
        z = self.proj(torch.cat([z_les, z_leaf] + ([cls.float()] if self.use_cls else []) + [geo, tshare], -1))
        logits = self.scale * F.normalize(z, dim=-1) @ F.normalize(self.protos, dim=-1).T
        return dict(seg=seg, les=les, logits=logits, z=z, geo=geo)

# ================= GĐ3–GĐ4: đặc trưng, tầng 2, OOD =================
def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

FEAT_ALPHA = 0.08

MORPH_KEYS = ['lesion_frac', 'mean_les', 'necrosis_share', 'powdery_share', 'rust_share', 'chlorosis_share', 'n_comp', 'big_frac',
              'mean_comp_frac', 'small_comp_share', 'solidity', 'touch', 'big_touch', 'r_margin', 'tip_share', 'mean_dist', 'r_yellow',
              'leaf_px', 'big_r_margin', 'nec_frac']

COLOR_KEYS = ['c_white', 'c_orange', 'c_brown', 'c_red', 'c_green', 'c_hsin', 'c_hcos', 'c_s', 'c_v', 'c_vstd',
              'l_white', 'l_orange', 'l_brown', 'l_yellow', 'l_v']

FEAT_KEYS = MORPH_KEYS + COLOR_KEYS

def leaf_geometry(leaf):
    lf = leaf.astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(lf, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    sz = np.sqrt(max(int(lf.sum()), 1)); tip = np.zeros(leaf.shape, bool)
    ys, xs = np.nonzero(lf)
    if len(xs) > 10:
        c = np.stack([xs, ys], 1).astype(np.float32); c -= c.mean(0)
        t = c @ np.linalg.svd(c, full_matrices=False)[2][0]
        lo, hi = np.quantile(t, [CFG['TIP_FRAC'], 1 - CFG['TIP_FRAC']]); sel = (t <= lo) | (t >= hi)
        tip[ys[sel], xs[sel]] = True
    return dist, sz, tip

def _frac(m):
    return float(m.mean()) if m.size else 0.0

def extend_leaf(leaf, les):
    # thêm vùng bệnh dính vào lá nhưng nằm ngoài mask lá (mô chết ở mép/chóp thường bị bộ tìm lá bỏ sót)
    leaf = leaf.astype(bool); L = (les >= CFG['LESION_PIX_TH']) & ~leaf
    if not L.any() or not leaf.any():
        return leaf
    n, lab = cv2.connectedComponents(L.astype(np.uint8))
    near = cv2.dilate(leaf.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    hit = np.unique(lab[near & L]); hit = hit[hit > 0]
    if not len(hit):
        return leaf
    add = np.isin(lab, hit)
    return leaf | add if add.sum() <= 0.8 * leaf.sum() else leaf

def morph_feats(les, sym, leaf, rgb):
    # (đặc trưng, mask vùng bệnh, mask vàng tuyệt đối, hình học lá) — les: xác suất bệnh 0..1, sym: kênh loại tổn thương
    leaf = extend_leaf(leaf, les); A = max(int(leaf.sum()), 1)
    N = (les >= CFG['LESION_PIX_TH']) & leaf; nN = max(int(N.sum()), 1)
    nec = N & (sym == TYPE_CH['necrosis']); n_nec = int(nec.sum())
    dist, sz, tip = leaf_geometry(leaf)
    f = dict(lesion_frac=N.sum() / A, mean_les=float(les[leaf].mean()) if leaf.any() else 0.0, leaf_px=np.log1p(A), nec_frac=n_nec / A)
    for t in LESION_TYPES:
        f[f'{t}_share'] = float((N & (sym == TYPE_CH[t])).sum() / nN)
    if n_nec:
        n, lab, st, _ = cv2.connectedComponentsWithStats(nec.astype(np.uint8)); ar = st[1:, cv2.CC_STAT_AREA]
        bigm = lab == (int(ar.argmax()) + 1)
        cs, _ = cv2.findContours(bigm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea))) if cs else 0.0
        edge = leaf & (dist <= 1.5); band = (dist <= FEAT_ALPHA * sz) & leaf
        hit = np.unique(lab[edge & nec]); hit = hit[hit > 0]
        f.update(n_comp=np.log1p(len(ar)), big_frac=ar.max() / A, mean_comp_frac=ar.mean() / A,
                 small_comp_share=float((ar < 0.01 * A).mean()), solidity=ar.max() / max(hull, 1.0),
                 touch=float(st[hit, cv2.CC_STAT_AREA].sum() / n_nec) if len(hit) else 0.0, big_touch=float((bigm & edge).any()),
                 r_margin=float((nec & band).sum() / n_nec), tip_share=float((nec & tip).sum() / n_nec),
                 mean_dist=float(dist[nec].mean() / sz), big_r_margin=float((bigm & band).sum() / max(bigm.sum(), 1)))
    else:
        f.update({k: 0.0 for k in ['n_comp', 'big_frac', 'mean_comp_frac', 'small_comp_share', 'solidity', 'touch', 'big_touch',
                                   'r_margin', 'tip_share', 'mean_dist', 'big_r_margin']})
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    yel = leaf & (h >= 18) & (h <= 34) & (s > 70) & (v > 90)                # vàng TUYỆT ĐỐI (cả lá vàng toàn phiến)
    white = (s < 45) & (v > 165); orange = (h >= 5) & (h < 18) & (s > 110) & (v > 110)
    brown = (h <= 25) & (s > 50) & (v < 140) & (v > 30); red = ((h >= 150) | (h < 5)) & (s > 80); green = (h > 34) & (h <= 90) & (s > 50)
    ang = h[leaf] * (2 * np.pi / 180)
    f.update(r_yellow=float(yel.sum() / A), c_white=_frac(white[leaf]), c_orange=_frac(orange[leaf]), c_brown=_frac(brown[leaf]),
             c_red=_frac(red[leaf]), c_green=_frac(green[leaf]), c_hsin=float(np.sin(ang).mean()) if ang.size else 0.0,
             c_hcos=float(np.cos(ang).mean()) if ang.size else 0.0, c_s=float(s[leaf].mean() / 255) if leaf.any() else 0.0,
             c_v=float(v[leaf].mean() / 255) if leaf.any() else 0.0, c_vstd=float(v[leaf].std() / 255) if leaf.any() else 0.0,
             l_white=_frac(white[N]), l_orange=_frac(orange[N]), l_brown=_frac(brown[N]), l_yellow=_frac(yel[N]),
             l_v=float(v[N].mean() / 255) if N.any() else 0.0)
    return f, N, yel, (dist, sz, tip)

def make_X(P, Fd, use_p=True):
    Xf = Fd[FEAT_KEYS].values.astype(np.float32)
    return np.hstack([np.log(P + 1e-6), Xf]) if use_p else Xf

def _full_proba(proba, classes):
    out = np.full((len(proba), NCLS), 1e-6); out[:, classes] = proba
    return out / out.sum(1, keepdims=True)

def seg_consistency(P, Fd):
    # bệnh phải khớp loại tổn thương mà bản đồ bệnh tìm thấy: nhân xác suất với (0,15 + 0,85 × tỉ lệ loại tương ứng)
    sh = np.stack([np.ones(len(Fd)) if t is None else Fd[f'{t}_share'].values for t in CLS_TYPE], 1)
    act = (Fd['lesion_frac'].values >= 0.03)[:, None]
    w = np.where(act, 0.15 + 0.85 * sh, 1.0); w[:, H_IDX] = 1.0
    Q = P * w
    return Q / Q.sum(1, keepdims=True)

def final_probs(P, Fd):
    # xác suất cuối của mỗi lá: tầng 2 (nếu dùng) hoặc head đã hiệu chỉnh, rồi ràng buộc nhất quán (nếu dùng)
    if USE_STACK:
        P = softmax_np(np.log(_full_proba(STACK.predict_proba(make_X(P, Fd, STACK_USE_P)), STACK.classes_)) / STACK_T)
    return seg_consistency(P, Fd) if USE_CONSIST else P

def ood_score(z):
    d = z[:, None, :] - MU[None]
    return np.einsum('ncd,de,nce->nc', d, PREC, d).min(1)

# ================= pipeline =================
def severity_grade(pct):
    for g, up, lab in CFG['SEVERITY_SCALE']:
        if pct <= up:
            return g, lab
    return CFG['SEVERITY_SCALE'][-1][0], CFG['SEVERITY_SCALE'][-1][2]

class PlantDiseasePipeline:
    def __init__(self, lesion_model, T, version='v8.0.0'):
        self.m, self.T, self.version = lesion_model, T, version

    def _prep_leaf(self, rgb, d):
        sq = square_box(d['box'])
        ci = crop_square(rgb, sq, S, cv2.INTER_AREA); cl = crop_square(d['mask'].astype(np.uint8), sq, CS, cv2.INTER_NEAREST).astype(np.float32)
        x = ((ci.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1)
        return sq, ci, torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(cv2.resize(cl, (G, G), interpolation=cv2.INTER_AREA)[None]), \
            torch.from_numpy(cv2.resize(cl, (S4, S4), interpolation=cv2.INTER_AREA)[None])

    @torch.no_grad()
    def _forward(self, xs, lgs, l4s, tta):
        self.m.eval(); x, lg, l4 = torch.stack(xs).to(DEVICE), torch.stack(lgs).to(DEVICE), torch.stack(l4s).to(DEVICE)
        with torch.autocast('cuda', dtype=torch.float16, enabled=DEVICE == 'cuda'):
            o = self.m(x, lg, l4)
            o2 = self.m(x.flip(-1), lg.flip(-1), l4.flip(-1)) if tta else None
        prob = o['seg'].float().softmax(1); logits = o['logits'].float()
        if tta:
            prob = 0.5 * (prob + o2['seg'].float().softmax(1).flip(-1)); logits = 0.5 * (logits + o2['logits'].float())
        return logits.cpu().numpy(), prob.cpu().numpy(), o['z'].float().cpu().numpy()

    @staticmethod
    def _result(did, conf, pct, note=None):
        g, glab = severity_grade(pct)
        out = dict(disease=NAME_VI.get(did, 'Không xác định') if did else 'Không xác định', disease_id=did,
                   confidence=round(float(conf), 1), severity=dict(percent_area=round(float(pct), 1), level=glab))
        if note:
            out['note'] = note
        return out

    def _leaf_from_model(self, rgb):
        # YOLO không thấy lá → cho LesionNet tách phiến lá trên toàn ảnh (kênh 0 = ngoài lá), chạy 2 lượt
        H, W = rgb.shape[:2]; d = dict(mask=np.ones((H, W), bool), box=(0, 0, W, H), conf=0.0, area=H * W)
        for _ in range(2):
            p = self._prep_leaf(rgb, d); sq = p[0]
            _, prob, _ = self._forward([p[2]], [p[3]], [p[4]], False)
            lf = (prob[0, 0] < 0.5).astype(np.uint8)
            if lf.mean() < 0.03:
                return None
            n, lab, st, _ = cv2.connectedComponentsWithStats(lf)
            keep = 1 + np.where(st[1:, cv2.CC_STAT_AREA] >= 0.1 * st[1:, cv2.CC_STAT_AREA].max())[0]
            lf = np.isin(lab, keep).astype(np.uint8)
            m = paste_back(lf, sq, H, W, cv2.INTER_NEAREST).astype(bool)
            if m.sum() < 0.01 * H * W:
                return None
            d = dict(mask=m, box=mask_box(m), conf=0.0, area=int(m.sum()))
        return d

    def predict(self, src, tta=None):
        tta = CFG['TTA'] if tta is None else tta
        rgb = resize_max(src if isinstance(src, np.ndarray) else load_rgb(src))[0]; H, W = rgb.shape[:2]
        q = quality_check(rgb); leaves = detect_leaves([rgb])[0]; fallback = False
        if not leaves:      # thử lại với ngưỡng thấp hơn
            c0 = CFG['LEAF_CONF']; CFG['LEAF_CONF'] = max(0.05, c0 / 3)
            try:
                leaves = detect_leaves([rgb])[0]
            finally:
                CFG['LEAF_CONF'] = c0
        if not leaves:      # vẫn không thấy (lá đỏ/tím, chụp quá cận…) → LesionNet tự tách phiến lá
            d = self._leaf_from_model(rgb); leaves = [d] if d is not None else []; fallback = bool(leaves)
        dbg = dict(rgb=rgb, quality=q, detected=leaves, leaves=[], lesion_full=np.zeros((H, W), bool), fallback=fallback,
                   color_full=np.zeros((H, W, 3), np.uint8), status='no_leaf', legend=[])
        if not leaves:
            return self._result(None, 0, 0, 'Không thấy lá rõ trong ảnh — hãy chụp gần lá hơn, đủ sáng.'), dbg
        packs = [self._prep_leaf(rgb, d) for d in leaves]
        logits, prob, z = self._forward([p[2] for p in packs], [p[3] for p in packs], [p[4] for p in packs], tta)
        P = softmax_np(logits / self.T); oods = ood_score(z); leaf_union = np.zeros((H, W), bool)
        for j, (d, (sq, ci, _, _, l4)) in enumerate(zip(leaves, packs)):
            les = prob[j, 2:].sum(0); sym = (prob[j, 2:].argmax(0) + 2).astype(np.uint8); lf0 = l4[0].numpy() > 0.5
            lf = extend_leaf(lf0, les)          # thêm mô chết dính mép/chóp mà bộ tìm lá bỏ sót
            mfull = d['mask'] | (paste_back((lf & ~lf0).astype(np.uint8), sq, H, W, cv2.INTER_NEAREST).astype(bool) if (lf & ~lf0).any() else False)
            f, N, yel, geom = morph_feats(les, sym, lf, cv2.resize(ci, (S4, S4), interpolation=cv2.INTER_AREA))
            Pf = final_probs(P[j:j + 1], pd.DataFrame([f]))[0]; pred = int(Pf.argmax()); lab = CLASS_NAMES[pred]
            lesion = np.zeros_like(N) if pred == H_IDX else (N | yel if lab == 'chlorosis_discoloration' else N)
            sym_l = np.where(yel & ~N, TYPE_CH['chlorosis'], sym).astype(np.uint8) if lab == 'chlorosis_discoloration' else sym
            col = color_mask(sym_l, lesion, lab if lab in NEC_SYMS else 'leaf_spot')
            les_full = paste_back(lesion.astype(np.uint8), sq, H, W, cv2.INTER_NEAREST).astype(bool) & mfull
            col_full = paste_back(col, sq, H, W, cv2.INTER_NEAREST)
            dbg['lesion_full'] |= les_full; dbg['color_full'][les_full] = col_full[les_full]; leaf_union |= mfull; d['mask_ext'] = mfull
            if lesion.any():
                for t in LESION_TYPES:
                    if ((sym_l == TYPE_CH[t]) & lesion).sum() >= 0.05 * lesion.sum():
                        nm = (lab if lab in NEC_SYMS else 'leaf_spot') if t == 'necrosis' else TYPE2SYM.get(t)
                        if nm and nm not in dbg['legend']:
                            dbg['legend'].append(nm)
            pct = 100 * les_full.sum() / max(mfull.sum(), 1)
            dbg['leaves'].append(dict(pred=pred, conf=100 * float(Pf[pred]), pct=pct, P=Pf, crop=ci, leaf=lf, lesion=lesion,
                                      color=col, ood=float(oods[j]), feats=f))
        L = dbg['leaves']; dis = [l for l in L if l['pred'] != H_IDX]
        pct_plant = 100 * (dbg['lesion_full'] & leaf_union).sum() / max(leaf_union.sum(), 1)
        if dis:
            main = max(dis, key=lambda l: (l['pct'] + 0.1) * l['conf']); did, conf, Pm, ood = CLASS_NAMES[main['pred']], main['conf'], main['P'], main['ood']
        else:
            Pm = np.mean([l['P'] for l in L], 0); did, conf, ood = HEALTHY, 100 * float(Pm[H_IDX]), float(np.median(oods)); pct_plant = 0.0
        top2 = np.sort(Pm)[::-1][:2]
        note = None
        if not q['ok']:
            note = 'Ảnh mờ hoặc thiếu/thừa sáng — kết quả chỉ để tham khảo, nên chụp lại.'
        elif ood > OOD_TH or conf < CFG['UNCERTAIN_CONF'] or 100 * (top2[0] - top2[1]) < CFG['CLOSE_TOP2']:
            note = 'Chưa chắc chắn — nên chụp cận cảnh lá có dấu hiệu bất thường, đủ sáng.'
        dbg['status'] = 'uncertain' if note else ('diseased' if dis else 'healthy')
        return self._result(did, conf, pct_plant, note), dbg


def load(art_dir, workdir=None):
    """Nạp toàn bộ artifacts. art_dir: thư mục chứa 5 file artifacts (chấp nhận .pt bị giải nén thành thư mục)."""
    global CFG, CLASS_NAMES, HEALTHY, H_IDX, SYMS, LESION_TYPES, SYM2TYPE, TYPE_CH, NCLS, NSEG, SEG_CH, NAME_VI, COLOR
    global NEC_SYMS, TYPE2SYM, CLS_TYPE, MEAN, STD, S, S4, G, CS, MEAN_T, STD_T, LEAF_MODEL, TEMP
    global STACK, STACK_T, USE_STACK, STACK_USE_P, USE_CONSIST, MU, PREC, OOD_TH, PIPE
    workdir = workdir or tempfile.mkdtemp(prefix='disease_art_')
    pc = json.load(open(_find(art_dir, 'pipeline_config.json'), encoding='utf-8'))
    ck = torch.load(_as_file(_find(art_dir, 'lesion_dinov2.pt'), workdir), map_location='cpu', weights_only=False)
    CFG = dict(ck['cfg']); CFG['PRETRAINED'] = False; CFG['TTA'] = False
    CLASS_NAMES = list(ck['class_names']); HEALTHY = 'healthy'; H_IDX = CLASS_NAMES.index(HEALTHY)
    SYMS = [c for c in CLASS_NAMES if c != HEALTHY]
    LESION_TYPES = list(pc['lesion_types']); SYM2TYPE = dict(pc['symptom_to_type'])
    TYPE_CH = {t: 2 + i for i, t in enumerate(LESION_TYPES)}
    NCLS, NSEG = len(CLASS_NAMES), 2 + len(LESION_TYPES)
    SEG_CH = {s: TYPE_CH[SYM2TYPE.get(s, 'necrosis')] for s in SYMS}
    NAME_VI = dict(pc['name_vi']); COLOR = {k: tuple(v) for k, v in pc['color'].items()}
    NEC_SYMS = [c for c in SYMS if SYM2TYPE.get(c, 'necrosis') == 'necrosis']
    TYPE2SYM = {SYM2TYPE[c]: c for c in SYMS if SYM2TYPE.get(c, 'necrosis') != 'necrosis'}
    CLS_TYPE = [None if c == HEALTHY else SYM2TYPE.get(c, 'necrosis') for c in CLASS_NAMES]
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
    S = CFG['IMG_SIZE']; S4 = S // 4; G = S // 14; CS = CFG['CROP_STORE']
    MEAN_T = torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1); STD_T = torch.tensor(STD, device=DEVICE).view(1, 3, 1, 1)
    from ultralytics import YOLO
    LEAF_MODEL = YOLO(_as_file(_find(art_dir, 'leaf_yolo11n_seg.pt'), workdir))
    model = LesionNet(CFG); model.load_state_dict(ck['state_dict']); model = model.to(DEVICE).eval()
    TEMP = float(pc['temperature'])
    st = pickle.load(open(_find(art_dir, 'stacker.pkl'), 'rb'))
    STACK, STACK_T, USE_STACK = st['model'], st['T'], st['used']
    STACK_USE_P, USE_CONSIST = st.get('use_head_probs', True), st.get('use_consistency', False)
    op = _find(art_dir, 'ood_stats.npz')
    if os.path.isdir(op):
        MU, PREC, OOD_TH = (np.load(os.path.join(op, k + '.npy')) for k in ['mu', 'prec', 'threshold'])
    else:
        z = np.load(op); MU, PREC, OOD_TH = z['mu'], z['prec'], z['threshold']
    OOD_TH = float(OOD_TH)
    PIPE = PlantDiseasePipeline(model, TEMP, pc.get('version', 'v8'))
    return dict(version=PIPE.version, classes=CLASS_NAMES, device=DEVICE, stacker=USE_STACK, consistency=USE_CONSIST)


def overlay(dbg, max_side=768):
    """Ảnh minh hoạ: viền lá trắng + vùng bệnh tô màu theo bệnh (RGB numpy)."""
    ov = dbg['rgb'].copy()
    for d in dbg['detected']:
        ov = draw_contours(ov, d.get('mask_ext', d['mask']), (255, 255, 255), 2)
    ov = blend(ov, dbg['color_full'], dbg['lesion_full'], 0.65)
    return resize_max(ov, max_side)[0]


def predict(img, return_image=False):
    """img: PIL.Image hoặc numpy RGB. Trả về {disease, disease_id, confidence, severity{percent_area, level}[, note][, image]}."""
    if PIPE is None:
        raise RuntimeError('Chưa gọi load(artifacts_dir)')
    rgb = load_rgb(img) if isinstance(img, Image.Image) else np.ascontiguousarray(img)
    with _LOCK, torch.inference_mode():
        res, dbg = PIPE.predict(rgb)
    if return_image:
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(overlay(dbg), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        res['image'] = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()
    return res
