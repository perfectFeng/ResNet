import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow
import cv2
import os
import numpy as np
import random

'''

checkpoint_path = "./chckPts/save4040.ckpt"
# checkpoint_path = os.path.join(model_dir, "model.ckpt-9999")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# print(var_to_shape_map)
for key in var_to_shape_map:
    if len(key.split('/')) == 8 and key.split('/')[1]=='block1':
        # print(reader.get_tensor(key))
        print("tensor_name: ", key)
'''

'''
# 原始数据筛选

txts = []

num=1
segment = 32
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/original_result/32'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/original_result/32/' + file, 'r')
    lines = list(f)
    f.close()
    for l in lines:
        p = float(l.strip('\n').split(' ')[4])
        # if label != '6' and float(p) > 0.5:
        if p>0.5:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/32/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()

txts = []

num=1
segment = 64
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/original_result/64'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/original_result/64/' + file, 'r')
    lines = list(f)
    f.close()
    for l in lines:
        p = float(l.strip('\n').split(' ')[4])
        # if label != '6' and float(p) > 0.5:
        if p>0.5:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/64/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()
'''
# 原始数据合并
'''
txts = []

num=1
segment = 32
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/original_result/32'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/original_result/32/' + file, 'r')
    lines = list(f)
    f.close()
    pv = lines[0].strip('\n').split(' ')[0]
    ps = int(lines[0].strip('\n').split(' ')[1])
    pe = lines[0].strip('\n').split(' ')[2]
    plabel = lines[0].strip('\n').split(' ')[3]
    pp = float(lines[0].strip('\n').split(' ')[4])
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        s = int(l.strip('\n').split(' ')[1])
        e = int(l.strip('\n').split(' ')[2])
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        # if label != '6' and float(p) > 0.5:
        if s-ps<=num*segment*0.5 and label == plabel:
            pe = l.strip('\n').split(' ')[2]
            num = num+1
            if p > pp:
                pp = p
        else:
            event.append(pv+' '+str(ps)+' '+pe+' '+plabel+' '+str(pp)+'\n')
            if lines.index(l) == len(lines)-1:
                flag = 1
            pv = l.strip('\n').split(' ')[0]
            ps = int(l.strip('\n').split(' ')[1])
            pe = l.strip('\n').split(' ')[2]
            plabel = l.strip('\n').split(' ')[3]
            pp = float(l.strip('\n').split(' ')[4])
            num=1
        if flag:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/32/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()

txts = []

num=1
segment = 64
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/original_result/64'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/original_result/64/' + file, 'r')
    lines = list(f)
    f.close()
    pv = lines[0].strip('\n').split(' ')[0]
    ps = int(lines[0].strip('\n').split(' ')[1])
    pe = lines[0].strip('\n').split(' ')[2]
    plabel = lines[0].strip('\n').split(' ')[3]
    pp = float(lines[0].strip('\n').split(' ')[4])
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        s = int(l.strip('\n').split(' ')[1])
        e = int(l.strip('\n').split(' ')[2])
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        # if label != '6' and float(p) > 0.5:
        if s-ps<=num*segment*0.5 and label == plabel:
            pe = l.strip('\n').split(' ')[2]
            num = num+1
            if p > pp:
                pp = p
        else:
            event.append(pv+' '+str(ps)+' '+pe+' '+plabel+' '+str(pp)+'\n')
            if lines.index(l) == len(lines)-1:
                flag = 1
            pv = l.strip('\n').split(' ')[0]
            ps = int(l.strip('\n').split(' ')[1])
            pe = l.strip('\n').split(' ')[2]
            plabel = l.strip('\n').split(' ')[3]
            pp = float(l.strip('\n').split(' ')[4])
            num=1
        if flag:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/64/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()
'''

t='foul'

# 筛选
# 32
txts = []

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/32'):
    txts = files
    break
for file in txts:
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/32/' + file, 'r')
    lines = list(f)
    f.close()
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        if label=='5' and p>0.99:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/32/'+v.split('.')[0]+'.txt', 'w')
    for e in event:
        f.write(e)
    f.close()
# 64
txts = []

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/64'):
    txts = files
    break
for file in txts:
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/64/' + file, 'r')
    lines = list(f)
    f.close()
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        if label=='5' and p>0.99:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/64/'+v.split('.')[0]+'.txt', 'w')
    for e in event:
        f.write(e)
    f.close()

# 再次合并（中间被其他事件阻断的）
# 32
txts = []

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/32'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/32/' + file, 'r')
    lines = list(f)
    f.close()
    if len(lines) == 0:
        continue
    pv = lines[0].strip('\n').split(' ')[0]
    ps = int(lines[0].strip('\n').split(' ')[1])
    pe = int(lines[0].strip('\n').split(' ')[2])
    plabel = lines[0].strip('\n').split(' ')[3]
    pp = float(lines[0].strip('\n').split(' ')[4])
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        s = int(l.strip('\n').split(' ')[1])
        e = int(l.strip('\n').split(' ')[2])
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        if s < int(pe):
            if e > int(pe):
                pe = e
            if p > pp:
                pp = p
        else:
            event.append(pv+' '+str(ps)+' '+str(pe)+' '+plabel+' '+str(pp)+'\n')
            if lines.index(l) == len(lines) - 1:
                flag = 1
            pv = l.strip('\n').split(' ')[0]
            ps = int(l.strip('\n').split(' ')[1])
            pe = l.strip('\n').split(' ')[2]
            plabel = l.strip('\n').split(' ')[3]
            pp = float(l.strip('\n').split(' ')[4])
        if flag:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/32/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()


# 64
txts = []

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/64'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/64/' + file, 'r')
    lines = list(f)
    f.close()
    if len(lines) == 0:
        continue
    pv = lines[0].strip('\n').split(' ')[0]
    ps = int(lines[0].strip('\n').split(' ')[1])
    pe = int(lines[0].strip('\n').split(' ')[2])
    plabel = lines[0].strip('\n').split(' ')[3]
    pp = float(lines[0].strip('\n').split(' ')[4])
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        s = int(l.strip('\n').split(' ')[1])
        e = int(l.strip('\n').split(' ')[2])
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        if s<int(pe):
            if e>int(pe):
                pe = e
            if p > pp:
                pp = p
        else:
            event.append(pv+' '+str(ps)+' '+str(pe)+' '+plabel+' '+str(pp)+'\n')
            if lines.index(l) == len(lines) - 1:
                flag = 1
            pv = l.strip('\n').split(' ')[0]
            ps = int(l.strip('\n').split(' ')[1])
            pe = l.strip('\n').split(' ')[2]
            plabel = l.strip('\n').split(' ')[3]
            pp = float(l.strip('\n').split(' ')[4])
        if flag:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/64/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()

# 不同长度合并
dir_32 = []
dir_64 = []

flag = 0
num = []
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/32'):
    dir_32 = files
    break
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/64'):
    dir_64 = files
    break
for i in range(len(dir_64)):
    event = []
    file_32 = dir_32[i]
    file_64 = dir_64[i]
    f_32 = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/32/' + file_32)
    f_64 = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/'+t+'/64/' + file_64)
    l_32 = list(f_32)
    l_64 = list(f_64)
    f_32.close()
    f_64.close()
    if len(l_32) == 0 and len(l_64) == 0:
        continue
    if len(l_32) == 0:
        for p in range(len(l_64)):
            event.append(l_64[p])
            continue
    if len(l_64) == 0:
        for p in range(len(l_32)):
            event.append(l_32[p])
            continue
    for j in range(len(l_32)):
            s_32 = int(l_32[j].split(' ')[1])
            e_32 = int(l_32[j].split(' ')[2])
            p_32 = float(l_32[j].split(' ')[4])
            for k in range(len(l_64)):
                    s_64 = int(l_64[k].split(' ')[1])
                    e_64 = int(l_64[k].split(' ')[2])
                    p_64 = float(l_64[k].split(' ')[4])
                    if e_32<=s_64 or e_64<=s_32:
                        if k<len(l_64)-1:
                            continue
                    else:
                        if p_32>p_64:
                            event.append(l_32[j])
                            num.append(k)
                            break
                        else:
                            event.append(l_64[k])
                            num.append(k)
                            break
                    event.append(l_32[j])
    for p in range(len(l_64)):
        if p not in num:
            event.append(l_64[p])

    d_event = []
    for i in range(len(event)):
        e1_s = int(event[i].split(' ')[1])
        e1_e = int(event[i].split(' ')[2])
        e1_p = float(event[i].split(' ')[4])
        for j in range(len(event)):
            e2_s = int(event[j].split(' ')[1])
            e2_e = int(event[j].split(' ')[2])
            e2_p = float(event[j].split(' ')[4])
            if i != j:
                if e1_e<=e2_s or e1_s>=e2_e:
                    continue
                else:
                    over = min(e1_e, e2_e) - max(e1_s, e2_s)
                    union = max(e1_e, e2_e) - min(e1_s, e2_s)
                    IOU = float(over)/union
                    if IOU > 0.3 and e1_p>=e2_p:
                        if j not in d_event:
                            d_event.append(j)
                    elif IOU > 0.3 and e1_p<e2_p:
                        if i not in d_event:
                            d_event.append(i)
    if len(d_event)>0:
        for d in range(len(d_event)):
            event.pop(d_event[d]-d)

    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/final_result/'+t+'/'+file_32, 'w')
    for e in event:
        f.write(e)


# 计算P、R
file = []
num = 0
predict_num = 0
label_num = 0

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/final_result/'+ t):
    file = files
    break
for name in file:
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/final_result/'+t+'/' + name)
    rf = open('D:/冯咩咩的文件夹/soccer_detection/dataset/test/' + name.split('.')[0] + '_event.txt')
    lf = list(f)
    lrf = list(rf)
    f.close()
    rf.close()
    for j in range(len(lf)):
        s_f = int(lf[j].split(' ')[1])
        e_f = int(lf[j].split(' ')[2])
        for k in range(len(lrf)):
            s_r = int(lrf[k].split(' ')[1])
            e_r = int(lrf[k].split(' ')[2])
            l_r = lrf[k].split(' ')[0]
            replay = lrf[k].split(' ')[3]
            if replay != '1' and l_r in [t]:
                if s_f<s_r<e_f or s_f<e_r<e_f or (s_r<s_f and e_r>e_f):
                    num = num+1
                    print(lf[j])
                    break

for name in file:
    f = open('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/final_result/final_result/'+t+'/' + name)
    lf = list(f)
    f.close()
    predict_num = predict_num + len(lf)

file = []
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/ResNet/prepredict/original_result/32/'):
    file = files
    break
for name in file:
    rf = open('D:/冯咩咩的文件夹/soccer_detection/dataset/test/' + name.split('_')[0] + '_event.txt')
    lrf = list(rf)
    rf.close()
    for k in range(len(lrf)):
        l_r = lrf[k].split(' ')[0]
        replay = lrf[k].strip('\n').split(' ')[3]
        if replay != '1' and l_r in [t]:
            label_num = label_num+1
            print(name, lrf[k])
print(num)


print('R:', float(num)/float(label_num))
print('P:', float(num)/float(predict_num))
