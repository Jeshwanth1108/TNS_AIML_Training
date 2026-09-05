import streamlit as st
import os
import tempfile

from tracking import process_video


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Multi-Object Tracking",
    page_icon="🎥",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🎥 Multi-Object Tracking & Counting")

st.markdown(
    """
    ### YOLOv8 + ByteTrack + OpenCV

    Upload a video to detect, track and count people
    crossing the counting line.
    """
)


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("⚙️ Settings")

st.sidebar.info(
    """
    **Detection Model:** YOLOv8n

    **Tracker:** ByteTrack

    **Object:** Person

    **Counting:** Line Crossing
    """
)


# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "📁 Upload a video",
    type=["mp4", "avi", "mov"]
)


# -----------------------------
# Process Video
# -----------------------------

if uploaded_file is not None:

    st.success(
        f"Video uploaded: {uploaded_file.name}"
    )

    # Save uploaded video temporarily
    temp_input = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_input.write(uploaded_file.read())
    temp_input.close()

    # Output directory
    os.makedirs("output", exist_ok=True)

    output_video = os.path.join(
        "output",
        "tracked_output.mp4"
    )

    # Process button
    if st.button(
        "🚀 Start Tracking",
        type="primary"
    ):

        with st.spinner(
            "Processing video... Please wait."
        ):

            try:

                results = process_video(
                    temp_input.name,
                    output_video
                )

                st.success(
                    "Video processing completed!"
                )

                # -----------------------------
                # Metrics
                # -----------------------------

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Count",
                        results["total"]
                    )

                with col2:
                    st.metric(
                        "Left → Right",
                        results["left_to_right"]
                    )

                with col3:
                    st.metric(
                        "Right → Left",
                        results["right_to_left"]
                    )

                # -----------------------------
                # Video
                # -----------------------------

                st.subheader(
                    "🎬 Tracked Video"
                )

                st.video(
                    output_video
                )

                # -----------------------------
                # Processing Information
                # -----------------------------

                st.subheader(
                    "📊 Processing Information"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        f"**Frames processed:** "
                        f"{results['frames']}"
                    )

                with col2:

                    st.write(
                        f"**Unique counted objects:** "
                        f"{results['total']}"
                    )

                # -----------------------------
                # Download
                # -----------------------------

                with open(
                    output_video,
                    "rb"
                ) as video_file:

                    st.download_button(
                        label="⬇️ Download Tracked Video",
                        data=video_file,
                        file_name="tracked_output.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:

                st.error(
                    f"Error while processing video: {e}"
                )