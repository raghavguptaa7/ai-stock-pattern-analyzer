import io
import matplotlib
# Force matplotlib to run headlessly without opening window popups
matplotlib.use('Agg')

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import yfinance as yf
import mplfinance as mpf
import pandas as pd

app = FastAPI(
    title="QuantVision AI Live Trading Engine",
    description="YOLOv8 Real-Time Streaming Market Analyzer Backend.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load global custom model weights
MODEL_PATH = "weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print("🚀 QuantVision Live Streaming Engine Active!")
except Exception as e:
    raise RuntimeError(f"Could not load weights at {MODEL_PATH}: {str(e)}")


def calculate_trading_verdict(detections):
    """Weighs detections to generate actionable signals."""
    if not detections:
        return {
            "verdict": "HOLD / NEUTRAL",
            "reasoning": "No high-probability technical candlestick patterns were identified on this streaming timeline."
        }

    bullish_score, bearish_score, neutral_score = 0.0, 0.0, 0.0

    for det in detections:
        pattern_name = det["pattern"].lower()
        conf = det["confidence_value"]

        if "rise" in pattern_name or "bullish" in pattern_name or "hammer" in pattern_name:
            bullish_score += conf
        elif "fall" in pattern_name or "bearish" in pattern_name or "hanging" in pattern_name or "shooting" in pattern_name:
            bearish_score += conf
        else:
            neutral_score += conf

    if bullish_score > bearish_score and bullish_score > neutral_score:
        margin = "STRONG" if (bullish_score - bearish_score) > 40 else "WEAK"
        return {"verdict": f"{margin} BUY", "reasoning": f"Bullish signals dominate at {round(bullish_score, 1)}% aggregate weight."}
    elif bearish_score > bullish_score and bearish_score > neutral_score:
        margin = "STRONG" if (bearish_score - bullish_score) > 40 else "WEAK"
        return {"verdict": f"{margin} SELL", "reasoning": f"Bearish distribution structures dominate at {round(bearish_score, 1)}% weight."}
    else:
        return {"verdict": "HOLD / WAIT", "reasoning": "The streaming profile exhibits perfectly balanced market indecision."}


@app.get("/")
def root():
    return {"status": "online", "engine": "QuantVision Engine v3.0-Live"}


@app.get("/predict/live/{ticker}")
def predict_live_ticker(ticker: str, timeframe: str = "1mo", interval: str = "1d"):
    """
    Fetches real-time live market asset OHLC data, draws a clean 
    candlestick chart programmatically, and feeds it into the custom AI model.
    """
    try:
        # 1. Fetch live asset streams from Yahoo Finance API
        formatted_ticker = ticker.upper().strip()
        stock = yf.Ticker(formatted_ticker)
        df = stock.history(period=timeframe, interval=interval)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"Ticker symbol '{formatted_ticker}' returned no live market stream arrays.")

        # 2. Render clean chart directly to an in-memory byte buffer (No storage footprint)
        buf = io.BytesIO()
        
        # 'charles' style uses industry standard Green/Red candlesticks matching your dataset formatting
        mpf.plot(
            df, 
            type='candle', 
            style='charles', 
            savefig=dict(fname=buf, bbox_inches='tight'), 
            volume=False, 
            axisoff=True  # Strips background text lines so YOLO focuses purely on pure geometric candle anatomy
        )
        buf.seek(0)
        
        # 3. Convert bytes structure to standard format for inference
        image = Image.open(buf).convert("RGB")

        # 4. Trigger model evaluation
        results = model.predict(source=image, conf=0.25, imgsz=640)
        
        detections = []
        result_object = results[0]
        
        for box in result_object.boxes:
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            class_name = result_object.names[class_id]
            confidence_score = round(float(box.conf[0].item()) * 100, 2)

            detections.append({
                "pattern": class_name,
                "confidence_percentage": f"{confidence_score}%",
                "confidence_value": confidence_score,
                "bounding_box": {
                    "xmin": round(xyxy[0], 1), "ymin": round(xyxy[1], 1),
                    "xmax": round(xyxy[2], 1), "ymax": round(xyxy[3], 1)
                }
            })

        # 5. Process data arrays through financial algorithm constraints
        decision = calculate_trading_verdict(detections)

        for d in detections:
            d.pop("confidence_value", None)

        return {
            "success": True,
            "ticker": formatted_ticker,
            "stream_timeframe": timeframe,
            "stream_interval": interval,
            "total_patterns_found": len(detections),
            "market_verdict": decision["verdict"],
            "verdict_reasoning": decision["reasoning"],
            "detections": detections
        }

    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"Live Engine Pipeline Failure: {str(e)}")


@app.post("/predict")
async def predict_uploaded_file(file: UploadFile = File(...)):
    """Keeps support for manual snapshot image uploads intact."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid format.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model.predict(source=image, conf=0.25, imgsz=640)
        detections = []
        result_object = results[0]
        for box in result_object.boxes:
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            class_name = result_object.names[class_id]
            confidence_score = round(float(box.conf[0].item()) * 100, 2)
            detections.append({
                "pattern": class_name, "confidence_percentage": f"{confidence_score}%", "confidence_value": confidence_score,
                "bounding_box": {"xmin": round(xyxy[0], 1), "ymin": round(xyxy[1], 1), "xmax": round(xyxy[2], 1), "ymax": round(xyxy[3], 1)}
            })
        decision = calculate_trading_verdict(detections)
        for d in detections: d.pop("confidence_value", None)
        return {"success": True, "filename": file.filename, "total_patterns_found": len(detections), "market_verdict": decision["verdict"], "verdict_reasoning": decision["reasoning"], "detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))