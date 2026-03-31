#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default='219.223.182.106', help='Server IP')
args = parser.parse_args()

def run_test_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, -1)
    
    address = f"tcp://{args.ip}:5555"
    print(f"[Client] 正在连接到机器人 Server: {address}")
    socket.connect(address)

    # 模拟 GraspNet 的输出字典形式
    # translation: [x, y, z]
    # rotation: 3x3 旋转矩阵
    # width: 夹爪目标张开宽度
    test_grasps = [
        {
            "translation": [0.4, 0.0, 0.55],
            "rotation": [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ], # 这是一个典型的“手心向下”的姿态
            "width": 0.06,  # 6cm
            "score": 0.98
        }
    ]

    for grasp in test_grasps:
        print(f"\n[Client] >>> 发送抓取指令!")
        print(f"  位置: {grasp['translation']}, 宽度: {grasp['width']}")
        socket.send_json(grasp)
        
        reply = socket.recv_json()
        print(f"[Client] 机器人回复: {reply['status']}")

    print("\n[Client] 测试结束")
    socket.close()

if __name__ == "__main__":
    run_test_client()
