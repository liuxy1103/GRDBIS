# -*- coding: utf-8 -*-
# @Time : 2021/3/30
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm

import logging
import time
import os

class Log():
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # log_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"logs","Running_logs") #存放打印的日志的目录
        log_path = os.path.join('./logs', 'Running_logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        timestr = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        self.log_name = os.path.join(log_path,timestr+".log")  #log's name includes time
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s]: %(message)s')  # [2019-05-15 14:48:52,947] - test.py] - ERROR: this is error
        fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')
        # 设置日志等级
        fh.setLevel(logging.INFO)
        # 设置handler的格式对象
        fh.setFormatter(self.formatter)
        # 将handler增加到logger中
        self.logger.addHandler(fh)

        # 创建一个StreamHandler,用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

        # # 关闭打开的文件
        fh.close()

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

if __name__ == '__main__':
    def printlog():
        log = Log()
        log.info("hubo")
    printlog()





