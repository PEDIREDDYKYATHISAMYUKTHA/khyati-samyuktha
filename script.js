/* =========================================
   ELEMENTS
========================================= */

const imageInput         = document.getElementById("imageInput");
const previewImage       = document.getElementById("previewImage");
const detectBtn          = document.getElementById("detectBtn");
const faceShapeText      = document.getElementById("faceShape");
const outputImage        = document.getElementById("outputImage");
const hairstyleList      = document.getElementById("hairstyleList");
const resultBox          = document.getElementById("resultBox");
const hairstyleContainer = document.getElementById("hairstyleContainer");

const videoFeed = document.getElementById("videoFeed");

/* Missing person elements */
const referenceInput   = document.getElementById("referenceImage");
const referencePreview = document.getElementById("referencePreview");
const videoInput       = document.getElementById("videoInput");
const videoPreview     = document.getElementById("videoPreview");
const videoDetection   = document.getElementById("videoDetection");
const detectedImage    = document.getElementById("detectedImage");
const detectedBox      = document.getElementById("detectedBox");
const alertSound       = document.getElementById("alertSound");

/* Progress elements */
const progressBar      = document.getElementById("progressBar");
const progressText     = document.getElementById("progressText");
const progressContainer= document.getElementById("progressContainer");


/* =========================================
   IMAGE PREVIEW
========================================= */

if (imageInput) {

    imageInput.onchange = () => {

        const file = imageInput.files[0];
        if (!file) return;

        previewImage.src           = URL.createObjectURL(file);
        previewImage.style.display = "block";

        // Hide old result
        if (resultBox)          resultBox.style.display          = "none";
        if (outputImage)        outputImage.style.display        = "none";
        if (hairstyleContainer) hairstyleContainer.style.display = "none";

    };

}


/* =========================================
   FACE SHAPE DETECTION
========================================= */

if (detectBtn) {

    detectBtn.onclick = async () => {

        if (!imageInput.files.length) {
            alert("Please select an image first");
            return;
        }

        detectBtn.innerText = "Detecting...";
        detectBtn.disabled  = true;

        const formData = new FormData();
        formData.append("image", imageInput.files[0]);

        try {

            const response = await fetch("/detect", {
                method: "POST",
                body:   formData
            });

            if (!response.ok) throw new Error("Server error");

            const data = await response.json();

            if (data.error) {
                alert("❌ " + data.error);
                detectBtn.innerText = "🔍 Detect Face Shape";
                detectBtn.disabled  = false;
                return;
            }

            // Show result box
            faceShapeText.innerText = data.face_shape + " (" + data.confidence + "%)";
            if (resultBox) resultBox.style.display = "block";

            // Show output image
            outputImage.src           = data.image_url + "?t=" + Date.now();
            outputImage.style.display = "block";

            // Show hairstyles
            hairstyleList.innerHTML = "";
            data.hairstyles.forEach(style => {
                const li     = document.createElement("li");
                li.innerText = "💈 " + style;
                hairstyleList.appendChild(li);
            });
            if (hairstyleContainer) hairstyleContainer.style.display = "block";

            // Scroll to result
            if (resultBox) resultBox.scrollIntoView({ behavior: "smooth" });

        } catch (err) {
            alert("Detection failed: " + err.message);
        }

        detectBtn.innerText = "🔍 Detect Face Shape";
        detectBtn.disabled  = false;

    };

}


/* =========================================
   LIVE CAMERA
========================================= */

function startCamera() {
    if (!videoFeed) return;
    videoFeed.src           = "/video_feed";
    videoFeed.style.display = "block";
}

function stopCamera() {
    if (!videoFeed) return;
    videoFeed.src           = "";
    videoFeed.style.display = "none";
}


/* =========================================
   REFERENCE IMAGE PREVIEW
========================================= */

if (referenceInput) {

    referenceInput.onchange = () => {

        const file = referenceInput.files[0];
        if (!file) return;

        referencePreview.src           = URL.createObjectURL(file);
        referencePreview.style.display = "block";

    };

}


/* =========================================
   UPLOAD REFERENCE IMAGE
========================================= */

async function uploadReference() {

    if (!referenceInput || !referenceInput.files.length) {
        alert("Please select a reference image first");
        return;
    }

    const btn    = document.getElementById("uploadReferenceBtn");
    const status = document.getElementById("referenceStatus");

    btn.innerText = "Uploading...";
    btn.disabled  = true;
    if (status) status.innerText = "";

    const formData = new FormData();
    formData.append("image", referenceInput.files[0]);

    try {

        const response = await fetch("/upload_reference", {
            method: "POST",
            body:   formData
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();

        if (data.error) {
            alert("❌ " + data.error);
            if (status) status.innerText = "❌ " + data.error;
        } else {
            if (status) status.innerText = "✅ " + data.message;
        }

    } catch (err) {
        alert("Reference upload failed");
        if (status) status.innerText = "❌ Upload failed";
    }

    btn.innerText = "⬆ Upload Reference Image";
    btn.disabled  = false;

}


/* =========================================
   VIDEO PREVIEW
========================================= */

if (videoInput) {

    videoInput.onchange = () => {

        const file = videoInput.files[0];
        if (!file) return;

        videoPreview.src           = URL.createObjectURL(file);
        videoPreview.style.display = "block";

    };

}


/* =========================================
   UPLOAD VIDEO THEN START FULL SCAN
   FIX: Scans ENTIRE video at once
        Only detects reference person — ignores others
========================================= */

let savedVideoPath = null;

async function uploadVideo() {

    const refStatus = document.getElementById("referenceStatus");

    if (!refStatus || !refStatus.innerText.includes("✅")) {
        alert("Please upload a reference image first (Step 1)");
        return;
    }

    if (!videoInput || !videoInput.files.length) {
        alert("Please select a video first");
        return;
    }

    const btn       = document.getElementById("uploadVideoBtn");
    const vidStatus = document.getElementById("videoStatus");

    btn.innerText = "Uploading...";
    btn.disabled  = true;
    if (vidStatus) vidStatus.innerText = "";

    // Hide old results
    if (detectedBox) detectedBox.style.display = "none";
    const msg = document.getElementById("detectMessage");
    if (msg) msg.innerText = "";

    const formData = new FormData();
    formData.append("video", videoInput.files[0]);

    try {

        // Step 1: Upload video
        const uploadRes  = await fetch("/upload_video", {
            method: "POST",
            body:   formData
        });

        if (!uploadRes.ok) throw new Error("Upload failed");

        const uploadData = await uploadRes.json();

        if (uploadData.error) {
            alert("❌ " + uploadData.error);
            btn.innerText = "▶ Start Detection";
            btn.disabled  = false;
            return;
        }

        savedVideoPath = uploadData.video_path;
        if (vidStatus) vidStatus.innerText = "✅ Video uploaded — scanning entire video...";

        // Show progress bar
        if (progressContainer) progressContainer.style.display = "block";
        if (progressBar)  progressBar.style.width  = "0%";
        if (progressText) progressText.innerText   = "0% — Scanning...";

        btn.innerText = "⏳ Scanning Entire Video...";

        // Step 2: Start full video scan in background
        const scanRes  = await fetch("/start_scan", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ video_path: savedVideoPath })
        });

        const scanData = await scanRes.json();

        if (scanData.error) {
            alert("❌ " + scanData.error);
            btn.innerText = "▶ Start Detection";
            btn.disabled  = false;
            return;
        }

        // Step 3: Poll for result every second
        pollScanStatus();

    } catch (err) {
        alert("Failed: " + err.message);
        btn.innerText = "▶ Start Detection";
        btn.disabled  = false;
    }

}


/* =========================================
   POLL SCAN STATUS EVERY SECOND
========================================= */

function pollScanStatus() {

    const interval = setInterval(async () => {

        try {

            const res  = await fetch("/scan_status");
            const data = await res.json();

            // Update progress bar
            if (progressBar)  progressBar.style.width  = data.progress + "%";
            if (progressText) progressText.innerText   =
                data.progress + "% — " + (data.message || "Scanning...");

            // Done — person found
            if (data.status === "done") {

                clearInterval(interval);
                playAlert();
                showResult(data);

            // Done — person NOT found
            } else if (data.status === "not_found") {

                clearInterval(interval);
                showNotFound(data);

            }

        } catch (err) {
            console.error("Poll error:", err);
        }

    }, 1000);

}


/* =========================================
   SHOW RESULT WHEN PERSON IS FOUND
========================================= */

function showResult(data) {

    const btn       = document.getElementById("uploadVideoBtn");
    const vidStatus = document.getElementById("videoStatus");
    const msg       = document.getElementById("detectMessage");

    if (progressText) progressText.innerText =
        `✅ Scan complete — Found in ${data.match_count} frames!`;

    if (vidStatus) vidStatus.innerText =
        `✅ Person found! Best match at frame ${data.best_frame} with ${data.accuracy}% accuracy`;

    if (msg) {
        msg.innerText   = `⚠️ MISSING PERSON DETECTED! (${data.accuracy}% accuracy)`;
        msg.style.color = "red";
    }

    // Show detected frame image
    if (detectedImage) {
        detectedImage.src = "/static/detected.jpg?t=" + Date.now();
    }
    if (detectedBox) {
        detectedBox.style.display = "block";
        detectedBox.scrollIntoView({ behavior: "smooth" });
    }

    if (btn) {
        btn.innerText = "▶ Start Detection";
        btn.disabled  = false;
    }

}


/* =========================================
   SHOW RESULT WHEN PERSON NOT FOUND
========================================= */

function showNotFound(data) {

    const btn       = document.getElementById("uploadVideoBtn");
    const vidStatus = document.getElementById("videoStatus");
    const msg       = document.getElementById("detectMessage");

    if (progressText) progressText.innerText =
        `✅ Scan complete — ${data.total_frames} frames checked`;

    if (vidStatus) vidStatus.innerText =
        `❌ Person not found in video (${data.total_frames} frames scanned)`;

    if (msg) {
        msg.innerText   = "❌ Person not found in this video";
        msg.style.color = "#888";
    }

    if (btn) {
        btn.innerText = "▶ Start Detection";
        btn.disabled  = false;
    }

}


/* =========================================
   ALERT SOUND
========================================= */

function playAlert() {
    if (alertSound) {
        alertSound.currentTime = 0;
        alertSound.play().catch(() => {});
    }
}


/* =========================================
   CONNECT BUTTONS ON PAGE LOAD
========================================= */

window.addEventListener("load", () => {

    const uploadReferenceBtn = document.getElementById("uploadReferenceBtn");
    const uploadVideoBtn     = document.getElementById("uploadVideoBtn");

    if (uploadReferenceBtn) uploadReferenceBtn.onclick = uploadReference;
    if (uploadVideoBtn)     uploadVideoBtn.onclick     = uploadVideo;

});