#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : config.py
# @Author: Sui Huafeng
# @Date  : 2017/12/28
# @Desc  : 读取配置文件、初始化工作目录、日志设置，生成cfg、log全局对象
#

import logging
from configparser import ConfigParser
from os import path, mkdir
from platform import system


class Workspaces(object):
    # 初始化类变量
    cfg = ConfigParser()
    cfg.read('./config.ini', encoding='UTF-8')

    # 赋值并初始化各类工作目录
    __workspace = cfg.get('General', 'Workspace')  # 数据处理的工作目录
    logsPath = path.join(__workspace, 'logs')  # 运行日志
    inputs = path.join(__workspace, 'inputs')  # 原始样本文件应保存在该目录及下级子目录中
    l1_cache = path.join(__workspace, 'l1cache')  # 某进程生成的多个日志文件合并后保存在该目录及下级子目录中
    l2_cache = path.join(__workspace, 'l2cache')  # 同类日志文件合并后保存在该目录下
    models = path.join(__workspace, cfg.get('General', 'SiteID') + cfg.get('General', 'Version'))  # 保存生成的最终及中间模型及配置文件
    outputs = path.join(__workspace, 'outputs')  # 保存输出数据文件

    for __folder in [logsPath, inputs, l1_cache, l2_cache, models, outputs]:
        if not path.exists(__folder):
            mkdir(__folder)

    # 赋值各类文件路径
    l2FilesList = path.join(models, 'l2file_info.csv')  # l2_cache中文件的元数据：包括文件名称、定界时间位置、日志类型等

    win = (system() == 'Windows')  # 本机操作系统类型，用于处理路径分隔符等Win/Unix的差异

    # 初始化日志设置
    log = logging.getLogger() if cfg.getboolean('General', 'IncludeLibs') else logging.getLogger('classifier')
    logLevel = cfg.getint('General', 'LogLevel')
    log.setLevel(logLevel)
    for __Handler in cfg.get('General', 'Handlers').split():
        __formatter = logging.Formatter('%(asctime)s\t%(levelno)s\t%(filename)s %(lineno)d\t%(message)s')
        __fh = logging.StreamHandler() if __Handler == 'STDOUT' else logging.FileHandler(path.join(logsPath, __Handler))
        __fh.setFormatter(__formatter)
        log.addHandler(__fh)
    log.debug("stared!")

    # 对一个样本(字符串)进行处理，返回词表[word]
    @classmethod
    def getWords( cls, document, rule_set ):

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
