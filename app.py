import streamlit as st
from PIL import Image
import io
import base64
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from reportlab.lib.utils import ImageReader

# Function to automatically crop the document from an image
def auto_crop_document(image):
    # ... (rest of the auto_crop_document function remains the same)
    try:
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img_np, M, (maxWidth, maxHeight))
                return Image.fromarray(warped)
        return image
    except Exception as e:
        print(f"Error during auto-cropping: {e}")
        return image

def display_processed_images(processed_images):
    st.subheader("Processed Images - Assign Order")
    if "image_order" not in st.session_state:
        st.session_state["image_order"] = {i: i + 1 for i in range(len(processed_images))}

    cols = st.columns(min(len(processed_images), 3))
    for i, img in enumerate(processed_images):
        with cols[i % 3]:
            st.image(img, caption=f"Image {i + 1}", use_container_width=True)
            order_key = f"order_{i}"
            new_order = st.number_input("Order:", min_value=1, max_value=len(processed_images), value=st.session_state["image_order"].get(i, i + 1), key=order_key)
            st.session_state["image_order"][i] = new_order

def reorder_images_based_on_input():
    if st.session_state["processed_images"] and "image_order" in st.session_state:
        ordered_images = [None] * len(st.session_state["processed_images"])
        used_indices = set()
        for index, order in st.session_state["image_order"].items():
            if 1 <= order <= len(ordered_images) and order - 1 not in used_indices:
                ordered_images[order - 1] = st.session_state["processed_images"][index]
                used_indices.add(order - 1)
            else:
                st.warning(f"Invalid or duplicate order number: {order}. Please correct.")
                return None
        if all(img is not None for img in ordered_images):
            st.session_state["processed_images"] = ordered_images
            st.success("Images reordered!")
            return ordered_images
        else:
            st.warning("Please ensure all images have a valid and unique order number.")
            return None
    return st.session_state["processed_images"]

def convert_to_pdf(ordered_images):
    st.subheader("Generate PDF")
    if st.button("Generate PDF"):
        if not ordered_images:
            st.warning("Please upload and order some images first.")
            return

        st.info("Standardizing size and generating PDF...")
        pdf_bytes = generate_pdf_from_processed_images(ordered_images)

        if pdf_bytes:
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="combined_documents.pdf">Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Failed to generate PDF. Please check the images.")

def standardize_size_portrait(image, width_inches=8.5, height_inches=11, dpi=300):
    # ... (rest of the standardize_size_portrait function remains the same)
    target_width = int(width_inches * dpi)
    target_height = int(height_inches * dpi)
    img_width, img_height = image.size
    img_ratio = img_width / img_height
    tar
