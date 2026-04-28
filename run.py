import sys
import logging
from src import pipeline
from src import utils # Importing utils to access the logging setup

def main():
    # 1. ACTIVATE LOGS FIRST!
    utils.setup_logging()
    logger = logging.getLogger("MAIN")
    
    logger.info(">>> STARTING CLIMATE INJUSTICE INDEX CALCULATION <<<")

    try:
        pipeline.run()

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