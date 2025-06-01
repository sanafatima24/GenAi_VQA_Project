# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoTokenizer, ViTConfig, ViTModel, BertModel
import torch.nn as nn
import matplotlib.pyplot as plt

st.set_page_config(page_title="Radiology VQA Demo", layout="centered")

# --- 12.1) Define FastVQA ---
class FastVQA(nn.Module):
    def __init__(self, num_answers, answer2id, id2answer):
        super().__init__()
        self.vit = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            output_attentions=False
        )
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            output_attentions=False
        )
        H = self.vit.config.hidden_size
        assert H == self.bert.config.hidden_size

        self.fusion     = nn.Linear(H*2, H)
        self.activation = nn.GELU()
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(H, num_answers)

        # Freeze all but last blocks
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.bert.parameters():
            p.requires_grad = False

        vit_last = self.vit.encoder.layer[-1]
        for p in vit_last.parameters():
            p.requires_grad = True
        bert_last = self.bert.encoder.layer[-1]
        for p in bert_last.parameters():
            p.requires_grad = True

        self.answer2id = answer2id
        self.id2answer = id2answer

    def forward(self, pixel_values, input_ids, attention_mask):
        v_out = self.vit(pixel_values=pixel_values).pooler_output
        t_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x = torch.cat([v_out, t_out], dim=-1)
        x = self.activation(self.fusion(x))
        x = self.dropout(x)
        return self.classifier(x)

# --- 12.2) Load label mappings + pretrained checkpoint ---
labels_list = [
    "pleural effusion", "cardiomegaly", "pneumonia", "pneumothorax", "atelectasis",
    "opacity", "nodule", "mass", "edema", "fibrosis", "other"
]
answer2id = {ans:i for i, ans in enumerate(labels_list)}
id2answer = {i:ans for ans,i in answer2id.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FastVQA(num_answers=len(answer2id), answer2id=answer2id, id2answer=id2answer).to(device)
model.load_state_dict(torch.load('fast_vqa_best.pth', map_location=device))
model.eval()

# --- 12.3) Validators: image preprocess + question tokenizer + transforms ---
def preprocess_uploaded_image(uploaded_file) -> Image.Image:
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

val_image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
max_question_length = 30

# --- 12.4) Predict top-3 with temperature scaling (hardcode your T) ---
T = torch.tensor(1.0).to(device)  # ← replace 1.0 with your learned temperature

def predict_top3(image: Image.Image, question: str):
    pix = val_image_transform(image).unsqueeze(0).to(device)  # [1,3,224,224]
    enc = tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=max_question_length,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        raw_logits = model(pix, enc.input_ids, enc.attention_mask)  # [1,11]
        calib_logits = raw_logits / T
        probs = torch.softmax(calib_logits, dim=1).cpu().numpy().flatten()
    top3_inds    = probs.argsort()[-3:][::-1]
    top3_scores  = probs[top3_inds].tolist()
    top3_answers = [id2answer[i] for i in top3_inds]
    return top3_answers, top3_scores

# --- 12.5) Generate attention overlay (FIXED) ---
def get_attention_overlay(image: Image.Image, layer: int = -1) -> Image.Image:
    """
    Resize any input image to 224×224, run ViT to get CLS attention,
    upsample 14×14→224×224, and overlay as a heatmap.
    """
    # 1) Resize to exactly 224×224
    img_resized = image.resize((224, 224))

    # 2) Load ViT that outputs attentions
    vit_cfg = ViTConfig.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        output_attentions=True,
        attn_implementation="eager"
    )
    attn_vit = ViTModel.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        config=vit_cfg
    ).to(device)
    attn_vit.load_state_dict(model.vit.state_dict(), strict=False)
    attn_vit.eval()

    # 3) Convert to tensor
    pix = val_image_transform(img_resized).unsqueeze(0).to(device)  # [1,3,224,224]

    # 4) Forward pass for attentions
    with torch.no_grad():
        outputs = attn_vit(pixel_values=pix)
    attn = outputs.attentions[layer][0]   # [heads, seq_len, seq_len]

    # 5) Compute CLS→patch attention (drop CLS→CLS)
    cls_attn = attn[:, 0, 1:]             # [heads, 14*14]
    cls_mean = cls_attn.mean(0).cpu().numpy()  # [196]

    # 6) Reshape to 14×14 grid
    G = int(np.sqrt(cls_mean.shape[0]))   # → 14
    attn_map = cls_mean.reshape(G, G)     # [14,14]

    # 7) Upsample to 224×224 by tiling each patch’s attention
    up = 224 // G                          # → 16
    attn_up = np.kron(attn_map, np.ones((up, up)))  # [224,224]

    # 8) Normalize to [0,1]
    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

    # 9) Convert resized image to [0,1]
    overlay_rgb = np.array(img_resized).astype(np.float32) / 255.0  # [224,224,3]

    # 10) Build heatmap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(attn_norm)[:, :, :3]  # [224,224,3]

    # 11) Blend
    blended = 0.5 * overlay_rgb + 0.5 * heatmap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    # 12) Return PIL image
    return Image.fromarray(blended)

# --- 12.6) Streamlit UI ---
st.title("🔍 Radiology VQA Demo")
st.markdown(
    "Upload a chest X-ray (PNG/JPG) below, "
    "type a radiology question, and click ▶ Run VQA. "
    "You’ll see the top-3 answers with confidences + an attention overlay. "
    "Low-confidence answers (<0.50) will be flagged."
)

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["png","jpg","jpeg"])
question      = st.text_input("Ask your question:")
run_inference = st.button("▶ Run VQA")

if run_inference:
    if uploaded_file is None:
        st.error("Please upload an X-ray image first.")
    elif question.strip() == "":
        st.error("Please type a question first.")
    else:
        try:
            # a) Preprocess image
            user_img = preprocess_uploaded_image(uploaded_file)
            st.image(user_img, caption="Uploaded X-ray", use_container_width=True)

            # b) Model inference
            top3, confidences = predict_top3(user_img, question)
            st.subheader("Top-3 Predictions:")
            for idx, (ans, conf) in enumerate(zip(top3, confidences), start=1):
                st.write(f"**{idx}.** {ans}   (confidence {conf:.2f})")

            # c) Confidence check
            if confidences[0] < 0.50:
                st.warning("⚠ Low confidence (<0.50). Please consult a radiologist.")

            # d) Attention overlay
            st.subheader("Attention Overlay (ViT CLS Token):")
            overlay_img = get_attention_overlay(user_img)
            st.image(overlay_img, caption="Attention Overlay", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()
else:
    st.info("Upload an X-ray and type a question, then click ▶ Run VQA.")
