import sys
import logging
from src import calculation
from src import utils

def main():
    utils.setup_logging()
    logger = logging.getLogger("MAIN")

    logger.info(">>> STARTING CLIMATE INJUSTICE INDEX CALCULATION <<<")

    try:
        calculation.run()

        logger.info(">>> PROCESS COMPLETED SUCCESSFULLY! <<<")
        logger.info("Check the 'data/outputs/results/' folder for generated files.")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (Ctrl+C).")
        sys.exit(0)

    except Exception as e:
        logger.critical(f"UNHANDLED ERROR: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()