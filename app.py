from io import BytesIO
from urllib.parse import urlparse

import requests
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from model_def import NeuralNetwork

MAX_IMAGE_BYTES = 5 * 1024 * 1024

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

@st.cache_resource
def load_model():
    model = NeuralNetwork().to(device)
    weights = torch.load(
        "cifar10_cnn.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image or paste an image URL to see the model's top predictions.")
st.subheader("Classes this model can predict")
st.markdown("\n".join(
    f"- **{class_name.capitalize()}**"
    for class_name in CLASSES
))

source = st.radio("Image source", ["Upload an image", "Paste an image URL"])
image = None

if source == "Upload an image":
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            st.error(f"Could not read that image: {error}")
else:
    with st.form("image_url_form"):
        image_url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        submitted = st.form_submit_button("Load image")

    if submitted:
        parsed_url = urlparse(image_url.strip())
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            st.error("Enter a valid image URL starting with http:// or https://.")
        else:
            try:
                with requests.get(
                    image_url.strip(), stream=True, timeout=(5, 15)
                ) as response:
                    response.raise_for_status()
                    image_bytes = bytearray()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        image_bytes.extend(chunk)
                        if len(image_bytes) > MAX_IMAGE_BYTES:
                            raise ValueError("Image must be 5 MB or smaller.")

                image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except requests.RequestException as error:
                st.error(f"Could not download the image: {error}")
            except (UnidentifiedImageError, OSError) as error:
                st.error(f"The URL did not return a readable image: {error}")
            except ValueError as error:
                st.error(str(error))

if image is not None:
    st.image(image, caption="Uploaded image", use_container_width=True)

    model = load_model()
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = model(input_tensor).softmax(dim=1)[0]
        scores, indices = probabilities.topk(3)

    st.subheader("Top predictions")
    for score, index in zip(scores.tolist(), indices.tolist()):
        st.write(f"{CLASSES[index]} — {score:.1%}")