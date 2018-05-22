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
    outputs = os.path.join(__workspace, 'outputs')  # 保存输出数据文件

    for __folder in [logsPath, inbox, l0_inputs, l1_cache, l2_cache, productModelPath, projectModelPath, outputs]:
        if not os.path.exists(__folder):
            os.mkdir(__folder)

    timeMargin = cfg.getint('General', 'TimeMargin')
    dateMargin = cfg.getint('General', 'DateMargin')
    last_update_seconds = cfg.getint('Classifier', 'LastUpdateHours') * 3600
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
    fileMergePattern = re.compile(cfg.get('Classifier', 'FileMergePattern'))
    fileCheckPattern = re.compile(cfg.get('Classifier', 'FileCheckPattern'))
    minConfidence = cfg.getfloat('Classifier', 'MinConfidence')
    maxClassifyLines = cfg.getint('Classifier', 'MaxLines')

    win = (system() == 'Windows')  # 本机操作系统类型，用于处理路径分隔符等Win/Unix的差异

    # 初始化日志设置
    log = logging.getLogger() if cfg.getboolean('General', 'IncludeLibs') else logging.getLogger('classifier')
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

    # 对一个样本(字符串)进行处理，返回词表[word]
    @classmethod
    def getWords(cls, document, rule_set):
        """

        :param document:
        :param rule_set:
        :return:
        """
        # 按照replace_rules对原文进行预处理，替换常用变量，按标点拆词、滤除数字等等
        keep_words = []
        for (replace_from, replace_to) in rule_set[1]:
            if replace_to == 'KEEP':  # 保留变量原值，防止被分词、去掉数字等后续规则破坏
                keep_word = replace_from.findall(document)
                if not keep_word:  # 未找到，无需进行后续替换
                    continue
                keep_words += [word[0] for word in keep_word]  # 保存找到的原值
                replace_to = ''  # 让后续替换在原文中去掉原值
            document = replace_from.sub(replace_to, document)
        words = [w for w in document.split() if len(w) > 1 and w.lower() not in rule_set[2]]  # 分词,滤除停用词和单字母

        # 实现k-shingle逻辑，把连续多个词合并为一个词
        k_shingles = []
        for k in rule_set[3]:
            if k == 1:
                k_shingles = words
                continue
            for i in range(len(words) - k):
                k_shingle = ''
                for j in range(k):
                    k_shingle += words[i + j]
                k_shingles += ([k_shingle])

        # 合并返回单词列表
        return keep_words + k_shingles
