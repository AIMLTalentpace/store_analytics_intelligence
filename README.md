# store_analytics_intelligence
# Steps to setup the environment to run the model
1. Install the wsl 22.04 version
2. Install the GStreamer
   ```
   sudo apt update
   sudo apt install -y \
   python3-gi python3-gi-cairo \
   gir1.2-gst-plugins-base-1.0 \
   gir1.2-gstreamer-1.0 \
   gstreamer1.0-tools \
   gstreamer1.0-plugins-base \
   gstreamer1.0-plugins-good \
   gstreamer1.0-plugins-bad \
   gstreamer1.0-plugins-ugly \
   gstreamer1.0-libav
   ```

3. Check the GStreamer is installed correctly or not.
   ```
   python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print(Gst.version())"
   ```
5. Create the python environment
   ```
    python3 -m venv venv --system-site-packages
   ```
7. Activate the environment
   ```
   source venv/bin/activate
   ```
8. Install the required python libraries:
   ```
   pip install ultralytics
   pip install opencv-python
   ```
   


