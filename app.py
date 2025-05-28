import streamlit as st
from PIL import Image
import io
import base64
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from reportlab.lib.utils import ImageReader # Import ImageReader

# Function to automatically crop the document from an image
def auto_crop_document(image):
    """
    Attempts to automatically crop a document from an image using OpenCV.
    It converts the PIL Image to a NumPy array for OpenCV processing,
    finds contours, approximates a rectangle, performs a perspective transform,
    and returns the cropped PIL Image. If no document is found, the original
    image is returned.
    """
    try:
        # Convert PIL Image to NumPy array (RGB to BGR for OpenCV)
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200) # Adjust thresholds if needed for better edge detection

        # Find contours in the edged image
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Sort contours by area in descending order and take the largest one
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate the contour to a polygon (ideally a rectangle)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # If the approximated contour has 4 points (a rectangle)
            if len(approx) == 4:
                # Reshape points for easier handling
                pts = approx.reshape(4, 2)
                # Create a new array to store the ordered points (top-left, top-right, bottom-right, bottom-left)
                rect = np.zeros((4, 2), dtype="float32")

                # The sum of the points gives top-left (smallest sum) and bottom-right (largest sum)
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)] # Top-left
                rect[2] = pts[np.argmax(s)] # Bottom-right

                # The difference between the points gives top-right (smallest difference) and bottom-left (largest difference)
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)] # Top-right
                rect[3] = pts[np.argmax(diff)] # Bottom-left

                # Now that we have the ordered corners, perform perspective transformation
                (tl, tr, br, bl) = rect

                # Compute the width of the new image (maximum of the two bottom-top widths)
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))

                # Compute the height of the new image (maximum of the two left-right heights)
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                # Define the destination points for the warped image (a perfect rectangle)
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                # Get the perspective transform matrix
                M = cv2.getPerspectiveTransform(rect, dst)
                # Apply the perspective transform
                warped = cv2.warpPerspective(img_np, M, (maxWidth, maxHeight))

                # Convert the OpenCV (NumPy array) image back to PIL Image
                return Image.fromarray(warped)
        # If no suitable contour is found or approximated, return the original image
        return image
    except Exception as e:
        # Print error for debugging purposes
        print(f"Error during auto-cropping: {e}")
        # Return original image in case of any error during processing
        return image

# Function to display processed images in columns
def display_processed_images(processed_images):
    """Displays a list of PIL Images in a grid layout within Streamlit."""
    st.subheader("Processed Images")
    # Determine the number of columns (max 3 for better display)
    cols = st.columns(min(len(processed_images), 3))
    for i, img in enumerate(processed_images):
        # Use use_container_width for responsive image display
        cols[i % 3].image(img, caption=f"Processed {i+1}", use_container_width=True)

# Function to provide UI for reordering images
def reorder_processed_images_ui(processed_images):
    """
    Provides a Streamlit multiselect widget for users to reorder images.
    The order of selection in the multiselect determines the new order.
    """
    st.subheader("Select and Reorder Images")
    # The default value ensures all images are initially selected in their current order
    # The format_func helps display meaningful names for each image
    reordered_images = st.multiselect("Select the images in the order you want them in the PDF:",
                                      processed_images,
                                      default=processed_images,
                                      format_func=lambda img: f"Image {st.session_state['processed_images'].index(img) + 1}")
    return reordered_images

# Function to trigger PDF conversion and provide download link
def convert_to_pdf(ordered_images):
    """
    Displays a button to generate the PDF. Once clicked, it calls
    generate_pdf_from_processed_images and provides a download link.
    """
    st.subheader("Generate PDF")
    if st.button("Generate PDF"):
        if not ordered_images:
            st.warning("Please upload and process some images first.")
            return

        st.info("Standardizing size and generating PDF...")
        # Call the PDF generation function with the ordered images
        pdf_bytes = generate_pdf_from_processed_images(ordered_images)

        if pdf_bytes:
            # Encode PDF bytes to base64 for download link
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="combined_documents.pdf">Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Failed to generate PDF. Please check the images.")

# Function to standardize image size to 8.5 x 11 inches portrait
def standardize_size_portrait(image, width_inches=8.5, height_inches=11, dpi=300):
    """
    Resizes and/or crops a PIL Image to fit 8.5 x 11 inches in portrait
    orientation without stretching. It maintains aspect ratio by cropping
    or adding white padding.
    """
    target_width = int(width_inches * dpi)
    target_height = int(height_inches * dpi)
    img_width, img_height = image.size
    img_ratio = img_width / img_height
    target_ratio = target_width / target_height

    # Create a new blank white image of the target size
    new_img = Image.new('RGB', (target_width, target_height), color='white')

    if img_ratio > target_ratio:
        # Image is wider than target page aspect ratio (e.g., landscape photo on portrait page)
        # Resize based on height, then center horizontally
        scaled_height = target_height
        scaled_width = int(scaled_height * img_ratio)
        resized_img = image.resize((scaled_width, scaled_height), Image.LANCZOS)
        # Calculate horizontal offset to center the image
        x_offset = (target_width - scaled_width) // 2
        new_img.paste(resized_img, (x_offset, 0))
    else:
        # Image is taller or similar aspect ratio as target page
        # Resize based on width, then center vertically
        scaled_width = target_width
        scaled_height = int(scaled_width / img_ratio)
        resized_img = image.resize((scaled_width, scaled_height), Image.LANCZOS)
        # Calculate vertical offset to center the image
        y_offset = (target_height - scaled_height) // 2
        new_img.paste(resized_img, (0, y_offset))
    
    return new_img


# Function to generate the final PDF from processed images
def generate_pdf_from_processed_images(processed_images):
    """
    Takes a list of processed PIL Images, standardizes their size,
    and combines them into a single PDF document.
    """
    buffer = BytesIO()
    # Create a PDF canvas with letter size (8.5 x 11 inches)
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter # Get dimensions of the letter page

    for img in processed_images:
        # Standardize each image to fit the portrait page
        standardized_img = standardize_size_portrait(img)
        
        # Convert PIL Image to a byte stream (PNG format) for reportlab
        img_byte_arr = io.BytesIO()
        standardized_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0) # Rewind the buffer to the beginning

        # Use ImageReader to explicitly tell reportlab that this is image data
        image_reader = ImageReader(img_byte_arr)

        # Draw the standardized image onto the PDF page
        # Pass the ImageReader object
        c.drawImage(image_reader, 0, 0, width=width, height=height)
        # Add a new page for the next image
        c.showPage()

    # Save the PDF to the BytesIO buffer
    c.save()
    # Return the PDF content as bytes
    return buffer.getvalue()

# --- Streamlit UI Layout ---
st.title("Image to PDF Converter with Auto-Cropping")
st.write("Upload photos, they will be automatically cropped to focus on the document, then you can select and reorder them to convert to a single PDF (standardized 8.5 x 11 portrait pages).")

# File uploader widget
uploaded_files = st.file_uploader("Upload Photos", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Initialize session state for processed images if not already present
if "processed_images" not in st.session_state:
    st.session_state["processed_images"] = []

# Process uploaded files when new files are detected
if uploaded_files:
    # Clear previous processed images if new files are uploaded
    st.session_state["processed_images"] = []
    # Iterate through each uploaded file
    for uploaded_file in uploaded_files:
        try:
            # Open the image using PIL
            img = Image.open(uploaded_file)
            # Display a spinner while processing each image
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Auto-crop the document from the image
                cropped_img = auto_crop_document(img)
                # Add the processed image to session state
                st.session_state["processed_images"].append(cropped_img)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    # If there are processed images, display them and allow reordering/PDF generation
    if st.session_state["processed_images"]:
        display_processed_images(st.session_state["processed_images"])
        # Get the reordered list of images from the UI
        reordered_images = reorder_processed_images_ui(st.session_state["processed_images"])
        # Pass the reordered list to the PDF conversion function
        convert_to_pdf(reordered_images)
else:
    st.info("Please upload some image files to begin.")
