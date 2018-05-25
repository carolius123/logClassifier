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
    inbox = os.path.join(__workspace, 'inbox')  # 原始压缩样本文件应上传至该目录中
    l0_inputs = os.path.join(__workspace, 'l0inputs')  # 原始样本文件应保存在该目录及下级子目录中
    l1_cache = os.path.join(__workspace, 'l1cache')  # 某进程生成的多个日志文件合并后保存在该目录及下级子目录中
    l2_cache = os.path.join(__workspace, 'l2cache')  # 同类日志文件合并后保存在该目录下
    productModelPath = os.path.join(__workspace, cfg.get('General', 'ProductID'))  # 保存产品及通用模型数据
    projectModelPath = os.path.join(__workspace, cfg.get('General', 'ProjectID'))  # 保存生成的最终及中间模型及配置文件
    modelPaths = [productModelPath, productModelPath]
    outputs = os.path.join(__workspace, 'outputs')  # 保存输出数据文件

    for __folder in [logsPath, inbox, l0_inputs, l1_cache, l2_cache, productModelPath, projectModelPath, outputs]:
        if not os.path.exists(__folder):
            os.mkdir(__folder)

    timeMargin = cfg.getint('General', 'TimeMargin')
    dateMargin = cfg.getint('General', 'DateMargin')
    last_update_seconds = cfg.getint('General', 'LastUpdateHours') * 3600
    '''
    fileDescriptor文件保存如下列表: [[l0_filenames], [fd_origin], [fd_common], [fd_category]]. 其中,
        l0_filenames 是$l0_input中所有文件全路径名列表, 目录分割符为/
        fd_origin = [ori_name, gather_time, last_update_time, ori_size] 在被管服务器上的全路径名,采集时间等
        fd_common = [common_name, anchor] 在$l1_cache中全路径名(去掉数字后合并的文件)
        fd_category = [category_name, confidences, distances] 分类, 置信度和中心距离
    '''
    fileDescriptor = os.path.join(projectModelPath, 'FileDescriptor.dbf')  # 样本文件描述信息
    fd_origin_none = ['', 0, -last_update_seconds * 2, 0]  # [ori_name, gather_time, last_update_time, ori_size]
    fd_common_none = ['', '']  # [common_name, anchor]
    fd_category_none = ['', 0, 0]  # [category_name, confidences, distances]

    productFileClassifierModel = os.path.join(productModelPath, 'FileClassifier.Model')
    projectFileClassifierModel = os.path.join(projectModelPath, 'FileClassifier.Model')
    fileMergePattern = re.compile(cfg.get('General', 'FileMergePattern'))
    fileCheckPattern = re.compile(cfg.get('General', 'FileCheckPattern'))
    minFileConfidence = cfg.getfloat('FileClassifier', 'MinConfidence')

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
