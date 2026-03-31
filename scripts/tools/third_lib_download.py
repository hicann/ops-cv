#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import urllib.request
import os
import subprocess
import shutil


def clone_git_repo(git_cmd, url, repo_path):
    """克隆 Git 仓库"""
    try:
        subprocess.run([git_cmd, 'clone', url, repo_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"克隆失败: {e}")


def update_git_repo(git_cmd, repo_path):
    """更新已存在的 Git 仓库"""
    git_dir = os.path.join(repo_path, '.git')
    if not os.path.isdir(git_dir):
        print(f"警告: {repo_path} 目录存在但不是 git 仓库，跳过更新。")
        return

    print(f"目录 {repo_path} 已存在，执行 git pull 更新。")
    try:
        subprocess.run(
            [git_cmd, '-C', repo_path, 'pull', '--ff-only'],
            check=True,
            timeout=600
        )
        print(f"更新成功: {repo_path}")
    except subprocess.TimeoutExpired:
        print(f"更新超时: {repo_path}")
    except subprocess.CalledProcessError as e:
        print(f"更新失败: {e}")


def download_file(url, file_path):
    """下载普通文件（带临时文件机制，防止下载失败后残留不完整文件）

    如果文件已存在，则重新下载最新版本
    """
    # 解析文件名，处理 URL 以 / 结尾的情况
    file_name = url.rstrip('/').split('/')[-1]
    if not file_name:
        file_name = "downloaded_file"

    # 使用与目标文件相同目录的临时文件，确保在同一文件系统
    temp_file_path = file_path + ".tmp"

    if os.path.exists(file_path):
        print(f"{file_path} 已存在，重新下载最新版本")
        os.remove(file_path)

    # 清理可能存在的残留临时文件
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        print(f"清理残留临时文件: {temp_file_path}")

    print(f"正在下载 {url} 到 {file_path}")
    try:
        # 设置超时，防止长时间阻塞
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=300) as response:
            with open(temp_file_path, 'wb') as f:
                shutil.copyfileobj(response, f)
        # 下载成功后重命名到目标文件
        os.rename(temp_file_path, file_path)
        print(f"下载成功: {file_path}")
    except Exception as e:
        # 下载失败时清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        print(f"下载失败: {url}, {e}")


def get_repo_name(url):
    """从 Git URL 提取仓库名"""
    repo_name = url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    return repo_name


def is_git_url(url):
    """判断 URL 是否为 Git 仓库"""
    return url.strip().endswith('.git')


def parse_file_name(url):
    """从 URL 解析文件名"""
    file_name = url.rstrip('/').split('/')[-1]
    return file_name if file_name else "downloaded_file"


def ensure_git_repo(git_cmd, url, repo_path):
    """确保 git 仓库存在，如果不存在则克隆，如果存在则更新或重新克隆"""
    if not os.path.exists(repo_path):
        clone_git_repo(git_cmd, url, repo_path)
        return

    git_dir = os.path.join(repo_path, '.git')
    if os.path.isdir(git_dir):
        update_git_repo(git_cmd, repo_path)
    else:
        print(f"警告: {repo_path} 不是 git 仓库，删除后重新克隆")
        shutil.rmtree(repo_path)
        clone_git_repo(git_cmd, url, repo_path)


def handle_git_url(git_cmd, url, current_dir):
    """处理 Git 仓库 URL"""
    if not git_cmd:
        print("git 命令未找到，请安装 git。")
        return
    repo_name = get_repo_name(url)
    repo_path = os.path.join(current_dir, repo_name)
    ensure_git_repo(git_cmd, url, repo_path)


def handle_file_url(url, current_dir):
    """处理普通文件 URL"""
    file_name = parse_file_name(url)
    file_path = os.path.join(current_dir, file_name)
    download_file(url, file_path)


def down_files_native(url_list):
    """下载第三方库文件（如果文件已存在，则重新下载最新版本）"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    git_cmd = shutil.which('git')

    for url in url_list:
        if is_git_url(url):
            handle_git_url(git_cmd, url, current_dir)
        else:
            handle_file_url(url, current_dir)


if __name__ == "__main__":
    third_urls = [
        "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz",
        "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip",
        ("https://gitcode.com/cann-src-third-party/makeself/releases/download/"
         "release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz"),
        "https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz",
        "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz",
        ("https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/"
         "20230802.1/abseil-cpp-20230802.1.tar.gz"),
        "https://gitcode.com/cann/opbase.git"   # Git 仓库
    ]

    down_files_native(third_urls)