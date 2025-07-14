import logging

log_file = r'C:\Users\PMLS\OneDrive\Desktop\reddit\app_errors.log'
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove all handlers associated with the root logger object.
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Also log to console (optional)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logging.info("=== Logging test: App started ===")
