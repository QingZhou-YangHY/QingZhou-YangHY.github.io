---
title: "SJTU-research-intern"
description: "remote research intern"
date: 2025-10-23T10:00:00+08:00
slug: 
image:  'SJTU.jpg'
math: false
license: 
hidden: false
comments: true
draft: false
categories:
    - Notes

---
#### 2025/10/23

实习第一天  

#### 2025/10/27

周六凌晨开始跑的论文复现，当时为了图速度就没有顾虑太多。今天学姐问我一些实验过程中的指标变化等等来对比idea的实验是否正常工作，只能回去手动复制tmux里面的内容。  

每跑一次实验都要记得能留日志就留日志，能上wandb/Swanlab就上，跑完的实验数据等等都要保存，留着。后续可能会用到。  

千万不要一味地追求速度，图省事！  

解决了submodule问题，具体的流程为删掉子仓库中的.git，移除本地缓存（觉得他是子module），从.gitmodule中移除（当然我本地没有）。  

解决了关于代码冲突的问题（需要手动进行修改），在git pull中是常见问题，总结关于这种多人协作常出现的问题，注意事项及解决方案