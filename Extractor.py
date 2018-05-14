#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Extractor.py
# @Author: Sui Huafeng
# @Date  : 2018/5/10
# @Desc  : 
#

import os
import tarfile
import threading

from sklearn.externals import joblib

from config import Workspaces as G


# 
class Extractor(object):
    """
    
    """

    def __init__(self, from_=G.inbox, to_=G.l0_inputs):
        self.fds = G.loadFds()
        self.interval = G.cfg.getfloat('Extractor', 'IntervalMinutes') * 60
        self.extractFiles(from_, to_)

    # from_目录下压缩文件,解压释放到to_目录下
    def extractFiles(self, from_, to_):
        c_ = {'tar files': 0, 'files': 0, 'err': 0}
        G.log.info('Extracting files from %s to %s', from_, to_)
        for tar_file in os.listdir(from_):
            if not tar_file.endswith('.tar.gz'):
                continue
            filename = ''
            tar_file_fullname = os.path.join(from_, tar_file)
            c_['tar files'] += 1
            try:
                # 解压文件
                tar = tarfile.open(tar_file_fullname)
                file_names = tar.getnames()
                path_to = os.path.join(to_, tar_file[:-7])
                for filename in file_names:
                    tar.extract(filename, path=path_to)
                    c_['files'] += 1
                tar.close()

                # 收集元数据文件
                descriptor_file = os.path.join(path_to, 'descriptor.csv')
                if not os.path.isfile(descriptor_file):
                    continue
                for line in open(descriptor_file, encoding='utf-8'):
                    gather_time, last_update_time, ori_size, ori_name, archive_name = line.strip().split('\t')
                    l0_filenames = os.path.join(path_to, archive_name).replace('\\', '/')
                    fd_origin = [ori_name, float(gather_time), float(last_update_time), float(ori_size)]
                    if l0_filenames in self.fds[0]:
                        idx = self.fds[0].index(l0_filenames)
                        self.fds[1][idx] = fd_origin
                    else:
                        self.fds[0].append(l0_filenames)
                        self.fds[1].append(fd_origin)
                        self.fds[2].append(G.fd_common_none)
                        self.fds[3].append(G.fd_category_none)

                os.remove(tar_file_fullname)
                os.remove(descriptor_file)
                G.log.debug('%s extracted.', tar_file_fullname)
            except Exception as err:
                G.log.warning('Extract %s:%s error:%s', tar_file_fullname, filename, str(err))
                c_['err'] += 1
                continue

        joblib.dump(self.fds, G.fileDescriptor)
        G.log.info('Extract %d files from %d tar.gz(err:%d) to %s, metadata recorded in %s'
                   , c_['files'], c_['tar files'], c_['err'], to_, G.fileDescriptor)

        global timer
        timer = threading.Timer(self.interval, self.extractFiles, (from_, to_))
        timer.start()


if __name__ == '__main__':
    Extractor()
