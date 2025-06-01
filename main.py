from activity_model import *
from hrv_model import *
from utils import *

def main():
    """Main function to execute the script."""
    patient_data = read_patient_info("patient_info.csv")
    patient_ids = patient_data["ID"]

if __name__ == "__main__":
    main()
