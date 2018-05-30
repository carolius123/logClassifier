#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : config.py
# @Author: Sui Huafeng
# @Date  : 2017/12/28
# @Desc  : 读取配置文件、初始化工作目录、日志设置，生成cfg、log全局对象
#
"""
Initialize log setting, working folders, and set global variables
"""

import logging
import os
import re
import sys
# noinspection PyCompatibility
from configparser import ConfigParser
from platform import system


class Workspaces(object):
    """
    设置日志和全局变量
    """
    # 初始化类变量
    cfg = ConfigParser()
    cfg.read('./config.ini', encoding='UTF-8')

    # 赋值并初始化各类工作目录
    __workspace = cfg.get('General', 'Workspace')  # 数据处理的工作目录
    logsPath = os.path.join(__workspace, 'logs')  # 运行日志
    transitPath = os.path.join(__workspace, 'transits')  # 原始压缩样本文件应上传至该目录中
    inboxPath = os.path.join(__workspace, 'inbox')  # 原始样本文件应保存在该目录及下级子目录中
    mergedFilePath = os.path.join(__workspace, 'merged')  # 某进程生成的多个日志文件合并后保存在该目录及下级子目录中
    classifiedFilePath = os.path.join(__workspace, 'classified')  # 同类日志文件合并后保存在该目录下
    modelPath = os.path.join(__workspace, 'models')  # 保存模型及配置数据
    for __folder in [logsPath, transitPath, inboxPath, mergedFilePath, classifiedFilePath, modelPath]:
        if not os.path.exists(__folder):
            os.mkdir(__folder)

    anchorTimeMargin = cfg.getint('General', 'AnchorTimeMargin')
    anchorDateMargin = cfg.getint('General', 'AnchorDateMargin')
    last_update_seconds = cfg.getint('General', 'LastUpdateHours') * 3600
    fileMergePattern = re.compile(cfg.get('General', 'FileMergePattern'))
    fileCheckPattern = re.compile(cfg.get('General', 'FileCheckPattern'))

    win = (system() == 'Windows')  # 本机操作系统类型，用于处理路径分隔符等Win/Unix的差异

    # 初始化日志设置
    log = logging.getLogger() if cfg.getboolean('General', 'IncludeLibs') else logging.getLogger('ailog')
    logLevel = cfg.getint('General', 'LogLevel')
    log.setLevel(logLevel)
    __formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(filename)s %(lineno)d\t%(message)s')
    __fh = logging.FileHandler(os.path.join(logsPath, os.path.splitext(os.path.split(sys.argv[0])[1])[0]) + '.log',
                               encoding='GBK')
    __fh.setFormatter(__formatter)
    log.addHandler(__fh)
    if cfg.getboolean('General', 'StdOut'):
        __fh = logging.StreamHandler()
        __fh.setFormatter(__formatter)
        log.addHandler(__fh)
    log.debug("stared!")
