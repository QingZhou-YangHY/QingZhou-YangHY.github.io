---
title: "ray.init()初始化挂起/失败问题的解决"
description: "Pass@k论文复现启动Ray分布式训练时出现挂起/失败问题的解决方法"
date: 2025-09-20T10:00:00+08:00
slug: 
image:  'Ray.jpg'
math: false
license: 
hidden: false
comments: true
draft: false
categories:
    - Notes

--- 

## Abstract

此文章发布约一周前就已经发现该问题了，但是由于专注看官方文档和仓库进行规范 + 课内事情，一直没有得到解决。  

在发文前一天发现此问题需要重点解决（无法避免），询问了师兄（论文作者）并咨询了相关团队，未果，所以花了两整天才解决这一个bug。  

---

## Issue

### What happened + What you expected to happen

Running the following snippet will hang indefinitely

```
>>> import  ray
>>> ray.init()
2025-09-20 11:44:47,741 INFO worker.py:1538 -- Started a local Ray instance.  
```

Sometimes it will fail instead
```
[2025-09-20 11:50:22,050 E 31652 31652] core_worker.cc:179: Failed to register worker 01000000ffffffffffffffffffffffffffffffffffffffffffffffff to Raylet. IOError: [RayletClient] Unable to register worker with raylet. No such file or directory
```
### Versions/Dependencies
Python 3.10.18  
Ray 2.49.1  
grpcio 1.75.0  
Ubuntu 24.04.2 LTS  

### Reproduction script
```
import ray
ray.init()
```

### Issue Severity
High: It blocks me from completing my task.

---
上面全英是因为当时要提issue或者给Ray框架作者发邮件，但是后来解决了，就打算留下来了，这样后来的人可以模仿一下这个写法。

## 可能的问题
- 1.workers实际上并没有启动。（可以看一下```/tmp/ray/session_latest/raylet.out```,如果在```/tmp/ray/session_latest/```看到有前缀```python-core-worker-``` 的可以看一下，因为这个能了解工作进程可能发生了什么）  
- 2.系统中进程数/线程数设置错误（可以通过cat /proc/sys/kernel/threads-max查看系统中一个进程可以创建多少个线程）

## 可能的方法
- 1.在```import ray```之后加上```ray.init(num_cpus=56, num_gpus=2)```。具体参数需要根据服务器进行自定义。
作者根据这个方法对自己进行了适配解决了问题。  
##### 具体操作：```/yhy/verl/trainer/config/ppo_trainer.yaml```配置文件中对```num_cpus=0```修改成```num_cpus=10```, ```num_gpus=1```进行定义。
- 2.升级```grpcio```,2023年的时候安装```grpcio 1.48.1``` 版本是有用的,相应的```venv```是 ```CentOS 7,Python 3.7.11，Ray 2.5.1,grpcio1.48.1```。  
但我进行升级的时候，无法解决该问题，并且会导致包之间的冲突。（是一个opencv的包，已经```pip install```了）  
- 3.添加```ulimit -n 65536```语句，因为分布式训练一开始可能会开成千上万个进程，默认是4096，所以会导致线程创建失败。  
##### 感觉这点也是有用的，但可能不是主要因素？
- 4.一定要设置参数,只是用```ray.init()```就会崩溃。需要手动设置```num_cpus```。  
这点和1重复了，可以说是大家实验得到的结论？（也许）

## 近期其他人也遇到过该问题
2024.1.16也有在```Ubuntu 20.04```上遇到同样的问题 ```venv: ray == 2.7.1,grpcio == 1.59.2,python == 3.11.5```  
2024.4.2有在```ubuntu 22.04.3```（docker内部）上遇到同样的问题，但是他只是失败，而不是挂起。在docker之外运行良好（他的M1 MacBook上）  

2024.4.14，2024.4.17，2024.7.11等等太多人遇到同样的问题了

## 至此，问题的解决方案已经讲述完毕。

---
### 回顾解决问题的流程
刚遇到这个问题，我先看了一下是不是自己遇到过的，发现没有就交给了copilot，发现copilot无法解决，给了chatgpt5，同样无法解决，又给了Gemini看看能不能有些新意（其实这步可以忽略），上述方法都不行，问了师兄是否遇到过。  

发现他们都没有遇到过，我只能去Ray官方仓库里面的issue进行查看。感觉现在人们都不怎么用StackOverflow等等论坛了，所以就只能去issue里面找了。  

幸运的是发现了很多人遇到了同样的问题和报错，我就开始追根溯源，发现从17.18年就有人提出了这个问题，当时也有相应的解决办法，但是随着版本更新变得不适用。  
我就开始收集所有对这个问题的理解和解决方案，逐个尝试，很幸运的是我debug成功了！  

因为论文中没有常见的问题的解决方案，如果有的话应该是第一步先去看的。

这就是我整个解决这个问题的流程，大体上看似乎没有太大问题。但是还是可以优化一下，下次遇到类似比较“偏”的问题可以更快，心态更平和地解决这个问题。

### 看到这里了，祝你遇到像我遇到的这样比较“偏门”的问题时，也可以顺利并更快地解决！