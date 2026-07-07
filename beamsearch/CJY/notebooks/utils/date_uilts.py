from datetime import datetime


def get_current_date_for_filename():
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M")
