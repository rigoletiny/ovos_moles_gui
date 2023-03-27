import logging
import os
import datetime

def init_logger():
    # create logger with 'spam_application'
    logger = logging.getLogger('Cookie_defect')
    logger.setLevel(logging.DEBUG)

    #limit number of stored logs
    save_path_log = os.path.join("workspace", "log")
    limit_img = 365
    list_dir_log = os.listdir(save_path_log)
    if len(list_dir_log) > limit_img:
        os.remove(os.path.join(save_path_log, list_dir_log[0]))

    #Storing log with date
    date = datetime.datetime.now()
    month = "0" + str(date.month) if date.month < 10 else str(date.month)
    day = "0" + str(date.day) if date.day < 10 else str(date.day)
    file_name = "cookie_defect_" + str(date.year) + "_" + month + "_" + day + ".log"
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join("workspace", "log", file_name))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Init logger')
    return logger
