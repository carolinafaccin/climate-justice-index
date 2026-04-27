import sys
import logging
from src import pipeline
from src import utils # Importing utils to access the logging setup

def main():
    # 1. ACTIVATE LOGS FIRST!
    utils.setup_logging()
    logger = logging.getLogger("MAIN")
    
    logger.info(">>> INICIANDO CÁLCULO DO ÍNDICE DE INJUSTIÇA CLIMÁTICA <<<")

    try:
        pipeline.run()

        logger.info(">>> PROCESSO CONCLUÍDO COM SUCESSO! <<<")
        logger.info("Verifique a pasta 'data/outputs/results/' para os arquivos gerados.")

    except KeyboardInterrupt:
        logger.warning("Processo interrompido pelo usuário (Ctrl+C).")
        sys.exit(0)

    except Exception as e:
        logger.critical(f"ERRO NÃO TRATADO: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()