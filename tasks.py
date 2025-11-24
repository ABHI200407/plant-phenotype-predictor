import os
import torch
import torch.nn as nn
import pandas as pd
import joblib
import smtplib
import zipfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from celery import Celery
from sklearn.preprocessing import StandardScaler

# --- 1. EMAIL CONFIGURATION ---
SENDER_EMAIL = 'phenopredictg@gmail.com'
SENDER_PASSWORD = 'maed ijmq xfle jrjg'

# --- 2. SYSTEM CONFIG ---
LITE_MODEL_PATH = 'lite_model.pth'
LITE_SCALER_PATH = 'lite_scaler.pkl'
HEAVY_MODEL_PATH = 'best_model.pth'
HEAVY_DATA_PATH = 'merged_rice_data.csv'

# --- 3. CELERY SETUP ---
celery = Celery('tasks', 
    broker=os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')
)

# --- 4. GLOBAL MODEL STORAGE ---
LITE_MODEL = None
LITE_SCALER = None
HEAVY_MODEL = None

# --- 5. HELPERS ---
def get_lite_model():
    global LITE_MODEL, LITE_SCALER
    if LITE_MODEL is None:
        print("⚡ Loading Lite Model into RAM...")
        LITE_SCALER = joblib.load(LITE_SCALER_PATH)
        
        model = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        model.load_state_dict(torch.load(LITE_MODEL_PATH, map_location='cpu'))
        model.eval()
        LITE_MODEL = model
    return LITE_MODEL, LITE_SCALER

def get_heavy_model(input_dim):
    global HEAVY_MODEL
    if HEAVY_MODEL is None:
        print("🐘 Loading Heavy Model into RAM...")
        model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        model.load_state_dict(torch.load(HEAVY_MODEL_PATH, map_location='cpu'))
        model.eval()
        HEAVY_MODEL = model
    return HEAVY_MODEL

def send_email_with_zip(recipient_email, result_csv_path):
    print(f"📧 Preparing to email processed file: {result_csv_path}")
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = 'PhenoAI: Prediction Results (GRWT100)'

        body = "Hello,\n\nYour analysis is complete. The attached ZIP file contains your original data WITH the new 'Predicted_GRWT100' column added at the end.\n\n- PhenoAI Team"
        msg.attach(MIMEText(body, 'plain'))

        # COMPRESS TO ZIP
        zip_filename = result_csv_path + ".zip"
        print(f"🗜️ Zipping {result_csv_path} -> {zip_filename}")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(result_csv_path, arcname=os.path.basename(result_csv_path))

        # ATTACH ZIP
        with open(zip_filename, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(zip_filename))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(zip_filename)}"'
            msg.attach(part)

        # SEND
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        
        # CLEANUP
        try:
            os.remove(zip_filename)
        except:
            pass
            
        print("✅ Email sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Email Failed: {e}")
        return False

# --- 6. TASKS ---

@celery.task(bind=True)
def predict_lite_task(self, user_inputs):
    try:
        model, scaler = get_lite_model()
        
        dna_map = {'A': 1.0, 'T': 2.0, 'G': 3.0, 'C': 4.0}
        processed_inputs = []
        
        for item in user_inputs:
            if isinstance(item, str):
                processed_inputs.append(dna_map.get(item.upper().strip(), 0.0))
            else:
                processed_inputs.append(float(item))
        
        input_scaled = scaler.transform([processed_inputs])
        
        with torch.no_grad():
            pred = model(torch.FloatTensor(input_scaled))
            
        return {"result": pred.item(), "model": "Lite"}
    except Exception as e:
        return {"error": str(e)}

@celery.task(bind=True)
@celery.task(bind=True)
def process_csv_task(self, filepath, user_email):
    try:
        print(f"📂 Processing CSV: {filepath}")
        user_df = pd.read_csv(filepath)
        user_df.fillna(0, inplace=True)
        
        # Load Reference Data
        ref_df = pd.read_csv(HEAVY_DATA_PATH)
        ref_df.fillna(0, inplace=True)
        
        drop_cols = ['Sample_ID', 'GRWT100', 'GRLT', 'GRWD', 'HDG_80HEAD', 
                     'LIGLT', 'LLT', 'LWD', 'PLT_POST', 'SDHT', 'CUNO_REPRO']
        
        existing_drops_ref = [c for c in drop_cols if c in ref_df.columns]
        X_ref = ref_df.drop(existing_drops_ref, axis=1).values
        
        scaler = StandardScaler()
        scaler.fit(X_ref)

        # Scale User Data
        existing_drops_user = [c for c in drop_cols if c in user_df.columns]
        X_user = user_df.drop(existing_drops_user, axis=1).values
        X_user_scaled = scaler.transform(X_user)
        
        # Predict
        input_dim = X_ref.shape[1]
        model = get_heavy_model(input_dim)
        
        with torch.no_grad():
            outputs = model(torch.FloatTensor(X_user_scaled))
            predictions = outputs.numpy().flatten()
            
        # 1. Add column to dataframe
        user_df['Predicted_GRWT100'] = predictions
        
        # 2. Rename safely
        base_name, _ = os.path.splitext(filepath)
        result_path = f"{base_name}_RESULT.csv"
        
        # --- NEW LOGIC: SAVE LIGHTWEIGHT FILE ---
        print(f"📉 Saving LITE results (ID + Prediction only) to: {result_path}")
        
        # We only want to save the ID (if it exists) and the Prediction
        columns_to_save = ['Predicted_GRWT100']
        if 'Sample_ID' in user_df.columns:
            columns_to_save.insert(0, 'Sample_ID')
            
        # Save only the necessary columns to keep file size small
        user_df[columns_to_save].to_csv(result_path, index=False)
        
        # 3. Send the LITE file
        email_success = send_email_with_zip(user_email, result_path)
        
        return {"status": "Success", "email_sent": email_success}

    except Exception as e:
        print(f"❌ Error in Heavy Task: {e}")
        return {"status": "Error", "message": str(e)}