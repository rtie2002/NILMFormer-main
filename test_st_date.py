"""
Quick test to see what format st_date should be
"""
import sys
sys.path.append('.')

from src.helpers.preprocessing import UKDALE_DataBuilder

# Check what get_nilm_dataset actually returns
print("Checking st_date format from UKDALE_DataBuilder...")
print("Expected format: Likely a DatetimeIndex or simple list")
print("\nOur current format: pd.DataFrame({'start_date': ...})")
print("\nThis mismatch could cause training issues!")
