import logging  # 引入logging模块
import time
import os
import sys 


def get_logger(file_mode=True, logger_name="detail", dir_name='runs/Logs13', detailedConsoleHandler=False):
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)  # Log等级总开关
    # logger.handlers = []

    # set formatter
    rq = time.strftime('%m%d-%H:%M:%S', time.localtime(time.time()))
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")

    if file_mode == True:
        # I. add a file handler  (第二详细用 info | debug × ,info √, warning √)
        if not os.path.exists(dir_name) or os.path.isfile(dir_name):
            os.makedirs(dir_name)
        detail_log_name = os.path.join(dir_name, rq) + '.log'
        fh = logging.FileHandler(detail_log_name, mode='w')
        fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # II. add a console handler (最详细 | debug √ , info √, warning √)
    ch = logging.StreamHandler(stream=sys.stdout)
    chlevel = logging.DEBUG if detailedConsoleHandler else logging.WARNING  
    ch.setLevel(chlevel) 
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def _set_logger(args):
    if args.id_log > 0:
        log_d = 'runs/Logs'+str(args.id_log)
        logger = get_logger(file_mode=args.logging, dir_name=log_d)
    else:
        logger = get_logger(file_mode=args.logging, detailedConsoleHandler=args.log_detailedCh)
    return logger