# Multi-Feed Demo Setup (4 laptops, RTSP)

## Topology
- **Laptop A (Viewer)**: runs your app + RTSP server (MediaMTX).
- **Laptops B/C/D (Producers)**: push their webcam (or video file) to A.

Find A's IP (viewer):
- Windows: `ipconfig` → e.g., 192.168.1.50
- Linux/Mac: `ifconfig`/`ip a`

## 1) Install & run MediaMTX on Laptop A
Download [MediaMTX (rtsp-simple-server)](https://github.com/bluenviron/mediamtx/releases), place mediamtx.yml next to the binary:

```yaml
# mediamtx.yml (minimal)
rtspDisable: false
hlsDisable: true
paths:
  cam1: {}
  cam2: {}
  cam3: {}
  cam4: {}
```

Start the server:

```bash
# Windows (PowerShell)
.\mediamtx.exe

# Linux/Mac
./mediamtx
```

Open firewall on A for TCP/UDP 8554 (RTSP) if prompted.

## 2) Push streams from B/C/D (and A itself if you want a 4th)
Install ffmpeg on each producer laptop.

### Windows (PowerShell)
List camera name (often "Integrated Camera") in Devices → Cameras, then:

```powershell
ffmpeg -f dshow -i video="Integrated Camera" `
 -vcodec libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p `
 -r 15 -g 30 -b:v 1500k -f rtsp rtsp://192.168.1.50:8554/cam1
```

### Linux
```bash
ffmpeg -f v4l2 -i /dev/video0 \
 -vcodec libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
 -r 15 -g 30 -b:v 1500k -f rtsp rtsp://192.168.1.50:8554/cam2
```

### Mac
```bash
ffmpeg -f avfoundation -i "0" \
 -vcodec libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
 -r 15 -g 30 -b:v 1500k -f rtsp rtsp://192.168.1.50:8554/cam3
```

### (Optional) Use a file as a live feed
```bash
ffmpeg -re -stream_loop -1 -i sample.mp4 \
 -vcodec libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
 -r 15 -g 30 -b:v 1500k -f rtsp rtsp://192.168.1.50:8554/cam4
```

Replace 192.168.1.50 with Laptop A's IP.
You can mix any OS/command for cam1–cam4; just use distinct path names.

## 3) Wire feeds in your app (on Laptop A)
Use the Add Camera UI with these URLs:

```
rtsp://192.168.1.50:8554/cam1
rtsp://192.168.1.50:8554/cam2
rtsp://192.168.1.50:8554/cam3
rtsp://192.168.1.50:8554/cam4
```

Set Resolution ~ 1280×720 and FPS cap 15.

Start each feed; you should see 4 independent live streams.

## 4) Quick verification (optional)
On Laptop A:

```bash
ffplay rtsp://192.168.1.50:8554/cam1
```

Repeat for cam2–cam4. If ffplay shows video, your app can, too.

## 5) Notes & Troubleshooting
- **Latency**: `-tune zerolatency`, `-r 15`, `-g 30` keep it snappy.
- **CPU**: If high, drop to 720p/10-15 FPS per feed; your app drops frames when overloaded.
- **Webcam busy**: Close other camera apps. On Windows, try different device names via `ffmpeg -list_devices true -f dshow -i dummy`.
- **Firewall**: Allow MediaMTX and ffmpeg on all machines.
- **Network**: Keep all laptops on the same Wi-Fi/LAN; avoid VPNs.

## 6) What to show in the demo
- Add each RTSP URL from the UI → "Connecting → Live" status.
- Run different tasks per feed (e.g., Detection on cam1, Segmentation on cam2).
- (Stretch) Duplicate cam1 inside the app → two models on the same feed side-by-side.
