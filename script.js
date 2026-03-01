// --- CONFIGURATION ---
const scriptURL = 'https://script.google.com/macros/s/AKfycbz6kvy4Wn8dmmXVbcx2gg-PI8D6a30l7x5Z7X6Xn4FwrfycrJ3A403_wm1batb39_8N/exec';
// REPLACE with your PC's IP address (e.g., http://192.168.1.5:5000/scan)
const AI_SERVER_URL = "http://192.168.241.1:5501/scan"; 

let stream = null;
let flash = false;

// --- AI INITIALIZATION (Updated for YOLOv8) ---
console.log("SMC YOLOv8 Scanner System Ready");

// --- PARKING LOGIC (NO CHANGES) ---

function updatePrice() {
    const duration = document.getElementById('duration').value;
    let price = 20;
    if (duration == "2") price = 40;
    if (duration == "4") price = 70;
    if (duration == "12") price = 150;
    document.getElementById('priceLabel').innerText = "₹" + price;
}

function processParking() {
    const vehicle = document.getElementById('vehNo').value; 
    const duration = document.getElementById('duration').value;

    if (vehicle.length < 4) {
        alert("Please enter a valid Surat Vehicle Number");
        return;
    }

    document.getElementById('payBtn').innerText = "Processing...";
    document.getElementById('payBtn').disabled = true;

    setTimeout(() => {
        let expiry = new Date();
        expiry.setHours(expiry.getHours() + parseInt(duration));

        document.getElementById('receipt').classList.remove('hidden');
        document.getElementById('recVehicle').innerText = vehicle.toUpperCase();
        document.getElementById('recTime').innerText = expiry.toLocaleTimeString();
        document.getElementById('payBtn').innerText = "PAID & SAVED";
        
        fetch(scriptURL, {
            method: 'POST',
            mode: 'no-cors',
            body: JSON.stringify({ 
                vehicle: vehicle.toUpperCase(), 
                expiry: expiry.toISOString(),
                amount: document.getElementById('priceLabel').innerText 
            })
        });

        alert("Parking ticket generated for " + vehicle);
    }, 1500);
}

// --- HIGH-SPEED CAMERA & SCANNER FUNCTIONS ---

async function openCam() {
    const overlay = document.getElementById('camOverlay');
    overlay.style.display = 'flex';
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: "environment", 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        document.getElementById('video').srcObject = stream;
    } catch (err) { 
        alert("Camera Error: Check Permissions"); 
        closeCam(); 
    }
}

function closeCam() {
    if(stream) stream.getTracks().forEach(t => t.stop());
    document.getElementById('camOverlay').style.display = 'none';
    flash = false;
}

async function toggleFlash() {
    const track = stream.getVideoTracks()[0];
    flash = !flash;
    try { await track.applyConstraints({ advanced: [{ torch: flash }] }); } catch(e) {}
}

// --- UPDATED SNAP FUNCTION (NOW USES YOLOV8) ---

async function snap() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const btn = document.getElementById('snapBtn');
    const ctx = canvas.getContext('2d');

    // 1. Capture full resolution for the AI to see clearly
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    btn.innerText = "YOLOv8 THINKING...";
    btn.disabled = true;

    // 2. Convert to Blob and send to your Python Server
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'plate.jpg');

        try {
            const response = await fetch(AI_SERVER_URL, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            
            // 3. Process result from YOLOv8
            if(result.plate && result.plate !== "Not Found") {
                // Only updates the input box for manual confirmation
                document.getElementById('vehNo').value = result.plate;
                
                const beep = document.getElementById('beepSound');
                if(beep) beep.play();
                
                closeCam();
            } else {
                alert("YOLOv8 could not find the plate. Move closer.");
            }
        } catch (err) {
            console.error(err);
            alert("Connection Error: Make sure Python app.py is running on your PC.");
        } finally {
            btn.innerText = "SCAN NOW";
            btn.disabled = false;
        }
    }, 'image/jpeg', 0.9); 
}