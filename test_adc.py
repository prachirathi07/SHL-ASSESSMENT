# test_adc.py
import google.auth

try:
    credentials, project = google.auth.default()
    print("✅ ADC is set up correctly.")
    print(f"Project ID: {project}")
except Exception as e:
    print("❌ ADC is NOT set up correctly.")
    print(e)
