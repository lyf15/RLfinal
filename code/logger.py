import logging
import os

def set_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=f'./log/log{name}', 
        filemode='a',               
    )

# set_logger(args.name)
# logger = logging.getLogger(__name__)
# logger.info