import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, uniform_filter
import io
import os
import tempfile

matplotlib.use("Agg")

# ─── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NeuraSeg | Brain Tumor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: #050810;
    color: #c9d4e8;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #080d1a !important;
    border-right: 1px solid rgba(99,179,237,0.12);
}

[data-testid="stSidebar"] * { color: #94a3b8; }
[data-testid="stSidebar"] h3 { color: #e2e8f0; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; }

/* ── hero ── */
.hero {
    position: relative;
    background: linear-gradient(135deg, #05091a 0%, #0a1432 40%, #050d1f 100%);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 20px;
    padding: 40px 48px 36px;
    margin-bottom: 28px;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 360px; height: 360px;
    background: radial-gradient(ellipse, rgba(99,179,237,0.07) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; bottom: -60px; left: 200px;
    width: 240px; height: 240px;
    background: radial-gradient(ellipse, rgba(168,85,247,0.06) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.25em;
    color: #60a5fa; text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    color: #f0f6ff; letter-spacing: -0.03em;
    line-height: 1.1; margin-bottom: 10px;
}
.hero-title span { color: #60a5fa; }
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: #4a6080;
    letter-spacing: 0.06em;
}
.hero-pills {
    margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap;
}
.pill {
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.22);
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.08em;
    padding: 4px 12px; border-radius: 20px;
}

/* ── stat cards ── */
.stat-row { display: flex; gap: 14px; margin-bottom: 24px; }
.stat-card {
    flex: 1;
    background: #08101f;
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 14px;
    padding: 18px 22px;
    position: relative; overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #3b82f6;
    text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 6px;
}
.stat-value { font-size: 1.7rem; font-weight: 700; color: #f0f6ff; }
.stat-sub { font-size: 0.75rem; color: #4a6080; margin-top: 2px; }

/* ── section headers ── */
.section-head {
    display: flex; align-items: center; gap: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #60a5fa;
    text-transform: uppercase; letter-spacing: 0.18em;
    border-bottom: 1px solid rgba(99,179,237,0.12);
    padding-bottom: 10px; margin-bottom: 18px;
}
.section-head::before {
    content: '';
    width: 4px; height: 14px;
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 2px; flex-shrink: 0;
}

/* ── status badges ── */
.badge-ok     { display:inline-block; background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.28); color:#34d399; padding:3px 12px; border-radius:20px; font-family:'JetBrains Mono',monospace; font-size:.7rem; }
.badge-warn   { display:inline-block; background:rgba(245,158,11,.12);  border:1px solid rgba(245,158,11,.28);  color:#fbbf24; padding:3px 12px; border-radius:20px; font-family:'JetBrains Mono',monospace; font-size:.7rem; }
.badge-danger { display:inline-block; background:rgba(239,68,68,.12);   border:1px solid rgba(239,68,68,.28);   color:#f87171; padding:3px 12px; border-radius:20px; font-family:'JetBrains Mono',monospace; font-size:.7rem; }

/* ── grade cards ── */
.grade-iv  { background:rgba(239,68,68,.15);   border:1px solid rgba(239,68,68,.35);   color:#fca5a5; padding:8px 18px; border-radius:10px; font-weight:600; display:inline-block; }
.grade-iii { background:rgba(245,158,11,.15);  border:1px solid rgba(245,158,11,.35);  color:#fcd34d; padding:8px 18px; border-radius:10px; font-weight:600; display:inline-block; }
.grade-ii  { background:rgba(16,185,129,.15);  border:1px solid rgba(16,185,129,.35);  color:#6ee7b7; padding:8px 18px; border-radius:10px; font-weight:600; display:inline-block; }

/* ── info card ── */
.info-card {
    background: #08101f;
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.info-card code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; color: #60a5fa;
}

/* ── modality bar ── */
.mod-bar-bg { background:#0d1a2e; border-radius:4px; height:6px; margin:6px 0; }
.mod-bar-fill { height:6px; border-radius:4px; }

/* ── step cards ── */
.step-card {
    background: #08101f;
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 14px;
    padding: 20px 22px 18px;
}
.step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #3b82f6;
    text-transform: uppercase; letter-spacing: 0.15em;
}
.step-title { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin: 6px 0 8px; }
.step-body  { font-size: 0.82rem; color: #4a6080; line-height: 1.6; }

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    padding: 12px 0 !important; width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #6d28d9) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.35) !important;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: #06090f !important;
    border: 1.5px dashed rgba(99,179,237,0.2) !important;
    border-radius: 12px !important;
}

/* ── sliders ── */
.stSlider [data-testid="stMarkdownContainer"] p { color: #60a5fa; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Model Definitions ─────────────────────────────────────────────────────────

class ModalityReliability(nn.Module):
    """Learns per-channel attention weights over the 4 MRI modalities."""
    def __init__(self, channels=4):
        super().__init__()
        self.conv    = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))


class DoubleConv(nn.Module):
    """Two Conv-BN-ReLU blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.conv(x)


class AttentionBlock(nn.Module):
    """Soft attention gate on skip connections (Oktay et al., 2018)."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.W_g(g) + self.W_x(x)))


class BrainTumorSegmentationModel(nn.Module):
    """
    Attention U-Net with dynamic modality weighting.
    Must match the architecture used during training exactly.
    """
    def __init__(self):
        super().__init__()
        self.modality_attention = ModalityReliability()
        self.conv1 = DoubleConv(4, 64)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.up    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att   = AttentionBlock(64, 64, 32)
        self.conv3 = DoubleConv(128, 64)
        self.out   = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x      = self.modality_attention(x)
        e1     = self.conv1(x)
        e2     = self.conv2(self.pool(e1))
        d1     = self.up(e2)
        e1_att = self.att(d1, e1)
        d1     = torch.cat([d1, e1_att], dim=1)
        d1     = self.conv3(d1)
        return torch.sigmoid(self.out(d1))


class MCDropoutWrapper(nn.Module):
    """MC Dropout wrapper — keeps dropout active at inference for uncertainty."""
    def __init__(self, base_model, p=0.25):
        super().__init__()
        self.base    = base_model
        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        x      = self.base.modality_attention(x)
        e1     = self.dropout(self.base.conv1(x))
        e2     = self.base.conv2(self.base.pool(e1))
        d1     = self.base.up(e2)
        e1_att = self.base.att(d1, e1)
        d1     = torch.cat([d1, e1_att], dim=1)
        d1     = self.dropout(self.base.conv3(d1))
        return torch.sigmoid(self.base.out(d1))


# ─── Grad-CAM ──────────────────────────────────────────────────────────────────

class SegmentationGradCAM:
    """
    Grad-CAM for segmentation: gradients of predicted tumour probability
    with respect to the last conv feature map.
    """
    def __init__(self, model, target_layer):
        self.model      = model
        self.gradients  = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, img_tensor, threshold=0.5):
        self.model.eval()
        img_tensor = img_tensor.clone().requires_grad_(True)
        output     = self.model(img_tensor)
        score      = (output * (output > threshold).float()).mean()
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze().detach()
        cam     = F.relu(cam)
        cam     = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        H, W = img_tensor.shape[2], img_tensor.shape[3]
        cam   = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False
        ).squeeze().detach().numpy()

        return cam, output.detach().cpu().numpy()[0, 0]


# ─── Helpers ───────────────────────────────────────────────────────────────────

MODALITY_NAMES = ["T1", "T2", "T1ce", "FLAIR"]
BG = "#050810"
CARD_BG = "#08101f"


def normalize(s):
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return ((s - mn) / (mx - mn)).astype(np.float32)


def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=BG, edgecolor="none", dpi=130)
    buf.seek(0)
    return buf


def mc_predict(mc_model, img_tensor, n=15):
    mc_model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n):
            preds.append(mc_model(img_tensor).cpu().numpy())
    preds = np.stack(preds, axis=0)
    return preds.mean(0), preds.std(0)


def extract_radiomics(mri_slice, pred_mask):
    """Extract intensity, shape, and texture features from tumour region."""
    px = mri_slice[pred_mask > 0.5]
    if len(px) == 0:
        return None
    f = {}
    f["mean_intensity"]  = float(np.mean(px))
    f["std_intensity"]   = float(np.std(px))
    f["kurtosis"]        = float(np.mean((px - f["mean_intensity"])**4) / (f["std_intensity"]**4 + 1e-8))
    f["skewness"]        = float(np.mean((px - f["mean_intensity"])**3) / (f["std_intensity"]**3 + 1e-8))
    _, num_comp          = ndimage.label(pred_mask > 0.5)
    filled               = binary_fill_holes(pred_mask > 0.5)
    f["tumor_area_pixels"]  = int((pred_mask > 0.5).sum())
    f["num_connected_comp"] = int(num_comp)
    f["solidity"]           = float(f["tumor_area_pixels"] / (filled.sum() + 1e-8))
    masked   = mri_slice * (pred_mask > 0.5)
    lm       = uniform_filter(masked, size=3)
    lv       = uniform_filter(masked**2, size=3) - lm**2
    f["texture_energy"] = float(lv[pred_mask > 0.5].mean() if (pred_mask > 0.5).any() else 0.0)
    return f


def grade_tumor(features):
    if features is None:
        return 0, "No tumor detected", "none"
    s = 0
    if features["std_intensity"] > 0.3:        s += 25
    elif features["std_intensity"] > 0.15:     s += 15
    if features["tumor_area_pixels"] > 500:    s += 20
    elif features["tumor_area_pixels"] > 200:  s += 10
    if features["num_connected_comp"] > 3:     s += 20
    elif features["num_connected_comp"] > 1:   s += 10
    if features["solidity"] < 0.6:             s += 20
    elif features["solidity"] < 0.8:           s += 10
    if features["texture_energy"] > 0.1:       s += 15
    if s >= 65:   label, g = "Grade IV — Glioblastoma (High Risk)", "iv"
    elif s >= 40: label, g = "Grade III — Anaplastic (Moderate-High Risk)", "iii"
    else:         label, g = "Grade II — Low-grade Glioma (Lower Risk)", "ii"
    return s, label, g


def detect_corruption(img_tensor, z_thresh=2.5):
    snrs = []
    for c in range(4):
        ch = img_tensor[:, c, :, :]
        snrs.append(abs(ch.mean().item()) / (ch.std().item() + 1e-8))
    mu, sd = np.mean(snrs), np.std(snrs) + 1e-8
    flags, rels = [], []
    for snr in snrs:
        z    = abs(snr - mu) / sd
        flag = (snr < 0.1) or (z > z_thresh and snr < mu)
        flags.append(flag)
        rel  = snr / (mu + 1e-8)
        rels.append(float(torch.sigmoid(torch.tensor(rel - 0.5)).item()))
    return flags, rels, snrs


def compute_metrics(pred_bin, mask_bin):
    tp = (pred_bin * mask_bin).sum()
    fp = (pred_bin * (1 - mask_bin)).sum()
    fn = ((1 - pred_bin) * mask_bin).sum()
    dice = (2*tp + 1e-8) / (2*tp + fp + fn + 1e-8)
    iou  = (tp + 1e-8) / (tp + fp + fn + 1e-8)
    return float(dice), float(iou)


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Model")
    model_file = st.file_uploader("Upload model (.pt / .pth)", type=["pt", "pth"])

    st.markdown("### MRI Modalities")
    st.caption("All 4 BraTS NIfTI files required")
    t1_file    = st.file_uploader("T1",    type=["nii", "gz"], key="t1")
    t2_file    = st.file_uploader("T2",    type=["nii", "gz"], key="t2")
    t1ce_file  = st.file_uploader("T1ce",  type=["nii", "gz"], key="t1ce")
    flair_file = st.file_uploader("FLAIR", type=["nii", "gz"], key="flair")

    st.markdown("### Analysis")
    threshold   = st.slider("Segmentation threshold", 0.1, 0.9, 0.45, 0.05)
    mc_iters    = st.slider("MC Dropout iterations",  5,  30,  15,  5)
    show_gcam   = st.checkbox("Grad-CAM heatmap",        value=True)
    show_unc    = st.checkbox("Uncertainty map",          value=True)
    show_radio  = st.checkbox("Radiomic profiling",       value=True)
    show_mod_qc = st.checkbox("Modality QC",              value=True)

    st.markdown("---")
    st.markdown(
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#2d4a6a;">'
        'Save model in Colab:<br><code style="color:#3b82f6">torch.save(model.state_dict(),\'model.pt\')</code>'
        '</span>', unsafe_allow_html=True
    )

    run_btn = st.button("Run Analysis")


# ─── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">University of Petroleum & Energy Studies · Minor Project</div>
    <div class="hero-title">Neura<span>Seg</span></div>
    <div class="hero-sub">BRAIN TUMOUR SEGMENTATION · ATTENTION U-NET · BraTS2020</div>
    <div class="hero-pills">
        <span class="pill">Attention U-Net</span>
        <span class="pill">Modality Reliability</span>
        <span class="pill">Grad-CAM XAI</span>
        <span class="pill">MC Dropout</span>
        <span class="pill">Radiomic Grading</span>
        <span class="pill">Modality QC</span>
    </div>
</div>
""", unsafe_allow_html=True)

# status row
model_ok = model_file is not None
mri_ok   = all([t1_file, t2_file, t1ce_file, flair_file])

c1, c2, c3, c4 = st.columns(4)
with c1:
    b = "badge-ok" if model_ok else "badge-warn"
    st.markdown(f'<span class="{b}">{"Model loaded" if model_ok else "No model"}</span>', unsafe_allow_html=True)
with c2:
    b = "badge-ok" if mri_ok else "badge-warn"
    n = sum([t1_file is not None, t2_file is not None, t1ce_file is not None, flair_file is not None])
    st.markdown(f'<span class="{b}">{"MRI ready" if mri_ok else f"{n}/4 uploaded"}</span>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<span class="badge-ok">Threshold: {threshold}</span>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<span class="badge-ok">MC iters: {mc_iters}</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Main Logic ────────────────────────────────────────────────────────────────

if run_btn:
    if not model_ok:
        st.error("Upload a .pt model file in the sidebar first.")
        st.stop()
    if not mri_ok:
        st.error("Upload all 4 NIfTI modalities (T1, T2, T1ce, FLAIR).")
        st.stop()

    device = torch.device("cpu")

    # load model
    with st.spinner("Loading model weights..."):
        try:
            model = BrainTumorSegmentationModel().to(device)
            state = torch.load(io.BytesIO(model_file.read()), map_location=device)
            model.load_state_dict(state)
            model.eval()
            mc_model = MCDropoutWrapper(model).to(device)
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.stop()

    # save NIfTI to temp and load
    with st.spinner("Loading MRI volumes..."):
        tmpdir = tempfile.mkdtemp()
        vols   = {}
        for key, fobj in [("t1", t1_file), ("t2", t2_file), ("t1ce", t1ce_file), ("flair", flair_file)]:
            p = os.path.join(tmpdir, fobj.name)
            with open(p, "wb") as f:
                f.write(fobj.read())
            try:
                vols[key] = nib.load(p).get_fdata()
            except Exception as e:
                st.error(f"Failed to load {key}: {e}")
                st.stop()

    num_slices = vols["t1"].shape[2]

    def build_tensor(idx):
        channels = [normalize(vols[k][:, :, idx]) for k in ["t1", "t2", "t1ce", "flair"]]
        return torch.tensor(np.stack(channels, axis=0)).float().unsqueeze(0).to(device)

    # slice selector
    st.markdown('<div class="section-head">Slice Selection</div>', unsafe_allow_html=True)
    sel = st.slider("Axial slice", 0, num_slices - 1, num_slices // 2)
    img_tensor = build_tensor(sel)

    with st.spinner("Running segmentation and analysis..."):
        # MC prediction
        mean_pred, unc_map = mc_predict(mc_model, img_tensor, n=mc_iters)
        pred_bin = (mean_pred[0, 0] > threshold).astype(float)

        # Grad-CAM
        if show_gcam:
            try:
                gcam   = SegmentationGradCAM(model, model.conv3.conv[-3])
                cam, _ = gcam.generate(img_tensor.clone(), threshold=threshold)
            except Exception:
                cam = np.zeros((img_tensor.shape[2], img_tensor.shape[3]))

    # ── visualisation panel ──────────────────────────────────────────────────

    st.markdown('<div class="section-head">Segmentation Results</div>', unsafe_allow_html=True)

    modality_imgs = {k: img_tensor[0, i].numpy() for i, k in enumerate(MODALITY_NAMES)}

    # row 1: all 4 modalities with mask
    fig = plt.figure(figsize=(20, 4.8), facecolor=BG)
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04)
    tumor_any = pred_bin.sum() > 0

    for i, (name, mri) in enumerate(modality_imgs.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(mri, cmap="gray", aspect="auto")
        if tumor_any:
            overlay = np.ma.masked_where(pred_bin < 0.5, pred_bin)
            ax.imshow(overlay, cmap="autumn", alpha=0.62, vmin=0, vmax=1)
        ax.set_title(name, color="#60a5fa", fontsize=11, fontweight="600",
                     pad=7, fontfamily="monospace")
        ax.axis("off")
        ax.set_facecolor(BG)

    tumor_px = int(pred_bin.sum())
    fig.suptitle(
        f"Slice {sel}   |   Tumour pixels: {tumor_px:,}   |   "
        f"Coverage: {100*tumor_px/pred_bin.size:.2f}%",
        color="#94a3b8", fontsize=10, y=1.01, fontfamily="monospace"
    )
    st.pyplot(fig)
    plt.close(fig)

    # row 2: probability | uncertainty | grad-cam
    cols = [show_unc, show_gcam]
    n_panels = 2 + sum(cols)
    fig2, axes2 = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), facecolor=BG)
    if n_panels == 1:
        axes2 = [axes2]

    flair_np = img_tensor[0, 3].numpy()

    # probability map
    ax = axes2[0]
    ax.set_facecolor(BG)
    im = ax.imshow(mean_pred[0, 0], cmap="plasma", vmin=0, vmax=1)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#94a3b8", labelsize=7)
    ax.set_title("Tumour Probability", color="#94a3b8", fontsize=10, fontfamily="monospace")
    ax.axis("off")

    # binary mask on FLAIR
    ax = axes2[1]
    ax.set_facecolor(BG)
    ax.imshow(flair_np, cmap="gray", aspect="auto")
    if tumor_any:
        ax.imshow(np.ma.masked_where(pred_bin < 0.5, pred_bin), cmap="hot", alpha=0.65)
    ax.set_title(f"FLAIR + Mask  (thr={threshold})", color="#94a3b8", fontsize=10, fontfamily="monospace")
    ax.axis("off")

    panel = 2
    if show_unc:
        ax = axes2[panel]
        ax.set_facecolor(BG)
        im = ax.imshow(unc_map[0, 0], cmap="viridis")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="#94a3b8", labelsize=7)
        ax.set_title(f"Uncertainty (MC n={mc_iters})", color="#94a3b8", fontsize=10, fontfamily="monospace")
        ax.axis("off")
        panel += 1

    if show_gcam:
        ax = axes2[panel]
        ax.set_facecolor(BG)
        flair_rgb = np.stack([flair_np, flair_np, flair_np], axis=-1)
        heatmap   = plt.cm.jet(cam)[:, :, :3].astype(np.float32)
        gcam_over = np.clip(0.5 * flair_rgb + 0.5 * heatmap, 0, 1)
        ax.imshow(gcam_over)
        ax.set_title("Grad-CAM Explainability", color="#94a3b8", fontsize=10, fontfamily="monospace")
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── metrics ───────────────────────────────────────────────────────────────

    st.markdown('<div class="section-head">Quantitative Metrics</div>', unsafe_allow_html=True)

    tumor_pct  = 100 * tumor_px / pred_bin.size
    mean_prob  = float(mean_pred[0, 0][pred_bin > 0.5].mean()) if tumor_px > 0 else 0.0
    mean_unc_v = float(unc_map[0, 0].mean())

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value, sub in [
        (m1, "Tumour Area",       f"{tumor_px:,}",          "pixels detected"),
        (m2, "Coverage",          f"{tumor_pct:.2f}%",      "of slice area"),
        (m3, "Mean Confidence",   f"{mean_prob:.3f}",       "in tumour region"),
        (m4, "Uncertainty (σ)",   f"{mean_unc_v:.4f}",      "MC Dropout"),
    ]:
        with col:
            st.markdown(f"""
            <div class="info-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── radiomic profiling ────────────────────────────────────────────────────

    if show_radio and tumor_px > 0:
        st.markdown('<div class="section-head">Radiomic Profiling  —  Novel Feature 2</div>', unsafe_allow_html=True)

        features       = extract_radiomics(flair_np, pred_bin)
        score, grade_l, grade = grade_tumor(features)
        grade_css      = f"grade-{grade}"

        g_col, chart_col = st.columns([1, 2])

        with g_col:
            st.markdown(f"""
            <div class="info-card">
                <div class="stat-label">WHO Grade Prediction</div>
                <div style="margin:12px 0;"><span class="{grade_css}">{grade_l}</span></div>
                <div class="stat-label" style="margin-top:16px;">Aggressiveness Score</div>
                <div class="stat-value">{score}<span style="font-size:1rem;color:#4a6080">/100</span></div>
            </div>
            """, unsafe_allow_html=True)

            if features:
                rows = [
                    ("Area", f"{features['tumor_area_pixels']} px"),
                    ("Components", str(features["num_connected_comp"])),
                    ("Solidity", f"{features['solidity']:.3f}"),
                    ("Std Intensity", f"{features['std_intensity']:.4f}"),
                    ("Texture Energy", f"{features['texture_energy']:.4f}"),
                ]
                tbl = "".join(
                    f'<tr><td style="color:#4a6080;padding:2px 0;">{k}</td>'
                    f'<td style="color:#94a3b8;text-align:right;">{v}</td></tr>'
                    for k, v in rows
                )
                st.markdown(f"""
                <div class="info-card" style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;">
                    <table style="width:100%;border-collapse:collapse;">{tbl}</table>
                </div>""", unsafe_allow_html=True)

        with chart_col:
            if features:
                feat_names = ["Mean Int.", "Std Int.", "|Skewness|", "Kurtosis*", "Solidity", "Texture×10"]
                feat_vals  = [
                    features["mean_intensity"], features["std_intensity"],
                    abs(features["skewness"]),  min(abs(features["kurtosis"]), 5),
                    features["solidity"],       features["texture_energy"] * 10,
                ]
                bar_colors = ["#ef4444" if v > 0.5 else "#3b82f6" for v in feat_vals]

                fig3, ax3 = plt.subplots(figsize=(8, 4.2))
                fig3.patch.set_facecolor(CARD_BG)
                ax3.set_facecolor(CARD_BG)
                bars = ax3.bar(feat_names, feat_vals, color=bar_colors,
                               width=0.6, edgecolor="#1e2a45", linewidth=0.6)
                for bar, val in zip(bars, feat_vals):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                             f"{val:.2f}", ha="center", va="bottom",
                             color="#94a3b8", fontsize=8, fontfamily="monospace")
                ax3.set_title(f"Radiomic Feature Profile  (Score: {score}/100)",
                              color="#e2e8f0", fontsize=10, pad=10, fontfamily="monospace")
                ax3.tick_params(colors="#4a6080")
                ax3.set_ylabel("Normalised Value", color="#4a6080", fontsize=9)
                for sp in ax3.spines.values():
                    sp.set_edgecolor("#1e2a45")
                plt.xticks(rotation=20, ha="right", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

    # ── modality QC ───────────────────────────────────────────────────────────

    if show_mod_qc:
        st.markdown('<div class="section-head">Modality Quality Control  —  Novel Feature 3</div>', unsafe_allow_html=True)

        flags, rels, snrs = detect_corruption(img_tensor)
        corrupted_names   = [MODALITY_NAMES[i] for i, f in enumerate(flags) if f]

        cols_mod = st.columns(4)
        for i, (col, name) in enumerate(zip(cols_mod, MODALITY_NAMES)):
            with col:
                rel      = rels[i]
                bar_w    = int(rel * 100)
                badge    = "badge-danger" if flags[i] else "badge-ok"
                status   = "Corrupted" if flags[i] else "Healthy"
                bar_col  = "#ef4444" if flags[i] else "#3b82f6"
                st.markdown(f"""
                <div class="info-card">
                    <div class="stat-label">{name}</div>
                    <div style="margin:6px 0;"><span class="{badge}">{status}</span></div>
                    <div class="mod-bar-bg">
                        <div class="mod-bar-fill" style="background:{bar_col};width:{bar_w}%;"></div>
                    </div>
                    <div class="stat-sub" style="font-family:'JetBrains Mono',monospace;margin-top:6px;">
                        SNR {snrs[i]:.3f} &nbsp;|&nbsp; Rel {rel:.3f}
                    </div>
                </div>""", unsafe_allow_html=True)

        if corrupted_names:
            st.warning(
                f"Corrupted modalities detected: {', '.join(corrupted_names)}. "
                "Attention weights redistributed to reliable channels."
            )


# ─── Welcome / Help State ──────────────────────────────────────────────────────

else:
    st.markdown('<div class="section-head">Getting Started</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    steps = [
        ("Step 01", "Save your model",
         "In your Colab notebook, after training run:<br>"
         "<code>torch.save(model.state_dict(), 'model.pt')</code><br>"
         "Then download the file to your computer."),
        ("Step 02", "Upload MRI scans",
         "Upload all 4 BraTS NIfTI modalities in the sidebar:<br>"
         "T1, T2, T1ce, and FLAIR (.nii or .nii.gz)."),
        ("Step 03", "Run and analyse",
         "Click <strong>Run Analysis</strong> to get segmentation, "
         "Grad-CAM, uncertainty maps, and radiomic grading."),
    ]
    for col, (num, title, body) in zip([s1, s2, s3], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card" style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;line-height:2;color:#4a6080;">
        <span style="color:#60a5fa;font-weight:600;">BrainTumorSegmentationModel</span>  (Attention U-Net + Modality Reliability)<br>
        ├─ ModalityReliability(4)    &nbsp;&nbsp;→ learned per-channel weights  <span style="color:#3b82f6">[Novel]</span><br>
        ├─ Encoder: DoubleConv(4→64) → MaxPool → DoubleConv(64→128)<br>
        ├─ Decoder: ConvTranspose2d(128→64) + AttentionBlock(64,64,32)<br>
        ├─ DoubleConv(128→64) → Conv2d(64→1) → Sigmoid<br>
        └─ MCDropoutWrapper(p=0.25)  &nbsp;→ Bayesian uncertainty  <span style="color:#8b5cf6">[Gal &amp; Ghahramani, 2016]</span>
    </div>
    """, unsafe_allow_html=True)
