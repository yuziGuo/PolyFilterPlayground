import logging
import time
import os
import sys 


def _get_logger(file_mode=True, 
                detailed_console_mode=False,
                logger_name="detail", 
                log_dir_if_file_mode='runs/Logs24'
                ):

    """
    Some notes for my logger:
    
    There are two handlers for the logger named `logger_name`:
    a **console handler**, and **possibly, a file handler**.
    
    The levels of the handlers are determined by the argument `file_mode` and `detailed_console_mode`:
    - The console always shows at least WARNING level messages, but can show more detailed DEBUG level messages if needed.
    - The file handler logs INFO level messages and above when file_mode is enabled, allowing you to store records in files.

    The levels of the handlers are determined by the arguments `file_mode` and `detailed_console_mode`:
    - The console handler is **always** active to ensure monitoring, especially during debug codes.
        - If `detailed_console_mode` is True, it logs DEBUG level messages and above.
        - If `detailed_console_mode` is False, it logs WARNING level messages and above.
    - The file handler logs INFO level messages and above when `file_mode` is enabled, allowing you to store records in files.


    Reminder:
    The relationship between the levels of the handlers and the levels of the messages is as follows:
    --------------------|  ------  |   -----   |   ------- |
    Information level→  |  debug   |   info    |  warning  | 
                        |          |           |           |
    Handler level↓      |          |           |           |
    --------------------|  ------  |   -----   |   ------- |
            debug       |    √     |     √     |    √      |
            info        |    x     |     √     |    √      |
            warning     |    x     |     x     |    √      |
    --------------------|  ------  |   -----   |   ------- |

    """
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)  

    # set formatter
    trial_starttime = time.strftime('%m%d-%H:%M:%S', time.localtime(time.time()))
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")

    # Set up the file handler if `file_mode` is `True` 
    # The file handler logs INFO level messages and above to a file 
    if file_mode == True:
        if not os.path.exists(log_dir_if_file_mode) or os.path.isfile(log_dir_if_file_mode):
            os.makedirs(log_dir_if_file_mode)
        fname = os.path.join(log_dir_if_file_mode, trial_starttime) + '.log'
        
        fh = logging.FileHandler(fname, mode='w')
        fh.setLevel(logging.INFO)  
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Set up the console handler
    # The console handler is **always active**.
    # - If `detailed_console_mode` is `True`, log DEBUG level messages and above to the console
    # - If `detailed_console_mode` is `False`, log WARNING level messages and above to the console
    ch = logging.StreamHandler(stream=sys.stdout)
    chlevel = logging.DEBUG if detailed_console_mode else logging.WARNING  
    ch.setLevel(chlevel) 
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def _set_logger(args):
    file_log_dir = None
    if args.file_logging:
        if args.file_log_id > 0:
            # If a log ID is provided, create a directory with the log ID
            file_log_dir = 'runs/Logs' + str(args.file_log_id)
        else:
            file_log_dir = 'runs/Logs' + 'temp_records'

    logger = _get_logger(file_mode=args.file_logging, 
                         detailed_console_mode=args.detailed_console_logging,
                         log_dir_if_file_mode=file_log_dir
                        )
    
    return logger