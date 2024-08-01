import re
import random
from typing import Set
import argparse
from attr import attr
import networkx as nx
import time
import shutil
import os
import matplotlib.pyplot as plt

def disrupt(filePath):
    # 打乱顺序
    f1 = open(filePath, 'r', encoding='utf-8')
    con = f1.readlines()
    random.shuffle(con)
    len_data = len(con)  # 数据集长度、行
    f1.close()
    f2 = open(filePath, 'w', encoding='utf-8')
    for i in range(len_data):
        f2.write(con[i])
    f2.close()

class Event:
    def __init__(self, name, hap_num, type_num):
        self.name = name
        self.hap_num = hap_num
        self.type_num = type_num

class Generator:

    def __init__(self, 
                 initPath= "data/train.txt",
                 event_frequence= "high",
                 event_threshold= -1,
                 event_strategy= -1,
                #  type_threshold= -1,
                 isPAE= 0,
                 pae_link= "tail",
                 pae_threshold= -1,
                 pae_select= "big",
                 generPath="../data/trainset.txt",
                 link_dict= None,    # 存放事件-实体对
                 link_ent_dict= None,# 存放实体-事件对
                 event_dict= None,   # 存放事件及其信息
                 entity_dict= None  # 存放实体及其信息
                 ):
        self.initPath = initPath
        self.event_frequence = event_frequence
        self.event_threshold = event_threshold
        self.event_strategy = event_strategy
        # self.type_threshold = type_threshold
        self.isPAE = isPAE
        self.pae_link = pae_link
        self.pae_threshold = pae_threshold
        self.pae_select = pae_select
        self.generPath = generPath
        self.link_dict = dict()
        self.event_dict = dict()
        self.link_ent_dict = dict()
        self.entity_dict = dict()

    def addEvent(self):
        # 二元排序
        def high_sort(x_obj):
            return (-x_obj.hap_num, -x_obj.type_num)
        def low_sort(x_obj):
            return (x_obj.hap_num, x_obj.type_num)
        
        # 创建一个空字典来存储数据
        # event_num_dict = {} # 存放事件和事件数
        # link_dict = {}  # 存放事件和实体
        find_T = re.compile(r"<\d+-\d+-\d+>")
        find_E = re.compile(r"<http://")  # 实体带http

        # 打开文件并逐行读取数据
        with open('events.txt', 'r', encoding='utf-8') as file:
            for line in file:
                if len(re.findall(find_T, line)) != 0:  # 找到带T的三元组(实体-关系-事件)
                    # 分割每行数据，假设数据由制表符分隔
                    parts = line.strip().split('\t')
                    if len(re.findall(find_E, parts[0])) != 0:
                        key, value = parts[2], parts[0]

                        # 检查link_dict字典中是否已经有该键
                        if key in self.link_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            if value not in self.link_dict[key]:
                                self.link_dict[key].add(value)
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.link_dict[key] = {value}

                        # 检查event_dict字典中是否已经有该键
                        if key in self.event_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            self.event_dict[key].hap_num += 1
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.event_dict[key] = Event(key, 1, 0)
        
        # 统计事件对应的类型数量
        for key in self.event_dict:
            if key in self.link_dict:
                self.event_dict[key].type_num = len(self.link_dict[key])
            else:
                print("the dict is wrong!")
        

        # 提取字典中的值
        events = list(self.event_dict.values())

        # 使用sorted函数对字典进行排序，并生成一个列表
        if self.event_frequence == "high":
            sorted_list = sorted(events, key=high_sort)
        elif self.event_frequence == "low":
            sorted_list = sorted(events, key=low_sort)

        # for event in sorted_list:
        #     print(event.hap_num, event.type_num, event.name)

        shutil.copy(self.initPath, self.generPath)
        if self.isPAE == 1:
            TKGtempPath = self.generPath.replace(".txt", "_temp.txt")
            shutil.copy("TKG.txt", TKGtempPath)
            TKGtempFile = open(TKGtempPath, 'a', encoding='utf-8')
        with open(self.generPath, 'a', encoding='utf-8') as file:
            num_elements_to_traverse = int(len(sorted_list) * self.event_threshold * 0.01)
            add_num = 0
            for i in range(num_elements_to_traverse):
                for value in self.link_dict[sorted_list[i].name]:
                    file.write(value + "\t" + "entity_to_thing_" + str(add_num) + "\t" + sorted_list[i].name + "\n")
                    if self.isPAE == 1:
                        TKGtempFile.write(value + "\t" + "entity_to_thing" + str(add_num) + "\t" + sorted_list[i].name + "\n")
                    add_num += 1
        if self.isPAE == 1:
            TKGtempFile.close()

    def addPAE(self):
        initPath = "TKG.txt"
        # filename = "TKG-or"
        TKGtempPath = self.generPath.replace(".txt", "_temp.txt")
        countPath = initPath
        anchorPath = self.generPath+"_anchor.txt"

        if self.event_strategy != -1:
            countPath = TKGtempPath

        countG = nx.Graph()
        G = nx.Graph()                                  # 创建一个没有节点和边的空图形

        print("锚点寻找中~~~已找到0个锚点，，，")
        # 锚点找取
        # 统计图
        with open(countPath, encoding='utf-8') as file:
            dataline = file.readlines()
            for line in dataline:
                line = line.rstrip().split("\t")
                a = line[0]
                b = line[2]  # a,b为实体
                c = line[1]  # c为关系
                countG.add_node(a, iscovered='no', color='blue')
                if c == "entity_to_thing":
                    countG.add_node(b, iscovered='no', color='green')
                else:
                    countG.add_node(b, iscovered='no', color='blue')
                countG.add_edge(a, b, attr=c)

        # 操作图
        with open(initPath, encoding='utf-8') as file:
            dataline = file.readlines()
            for line in dataline:
                line = line.rstrip().split("\t")
                a = line[0]
                b = line[2]  # a,b为实体
                c = line[1]  # c为关系
                G.add_node(a, iscovered='no', color='blue')
                G.add_node(b, iscovered='no', color='blue')
                G.add_edge(a, b, attr=c)
        # print(G._node)              # 查看节点属性
        # print(G.edges(data=True))  # 查看边及所有属性
        print("原图的节点数：", G.number_of_nodes())  # 节点的数量
        print("原图的边数：", G.number_of_edges())  # 边的数量

        order_2_degree = {}                                 # 字典存储所有结点的度
        for node in G.nodes:
            if node in countG:
                order_2_degree[node] = countG.degree[node]
        # order_out_degree = {}                                 # 字典存储所有结点的出度
        # for node in G1.nodes:
        #     order_out_degree[node] = G1.out_degree[node]
        # order_in_degree = {}                                 # 字典存储所有结点的入度
        # for node in G1.nodes:
        #     order_in_degree[node] = G1.in_degree[node]

        list_degree = {}                                  # 存储本轮次剩余结点的度
        list_least_nodes = []                   # 存储锚点
        num = 0
        flag = True

        # 选取锚点（多类型）
        while flag:
            print("锚点寻找中~~~,已找到" + str(num + 1) + "个锚点！",
                time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time())))
            Max_degree = 0
            for key, value in order_2_degree.items():
                if value > Max_degree and G.nodes[key]['iscovered'] == 'no':
                    Max_degree = value                      # 找出图节点的最大度值
            print("此时度的最大值", Max_degree)
            if Max_degree == 5:
                break
            # 度相同的情况下多类型优先
            list_type = []                              # 存储现轮次度最大的节点
            for key, value in order_2_degree.items():
                if (value == Max_degree and G.nodes[key]['iscovered'] == 'no'):
                    list_type.append(key)
            # print(list_type)

            edge_type = []                                      # 存储边类型
            maxtype = list_type                                # 存储边类型最多的节点
            maxtype = []                                # 存储边类型最多的节点

            typenumber = 0                                      # 边类型最多数
            for node_degree in list_type:                       # 遍历最大度节点
                for ed in G.edges(data=True):
                    if node_degree in ed:
                        edge_type.append(ed[2]['attr'])
                if len(set(edge_type)) >= typenumber:
                    typenumber = len(set(edge_type))
                    edge_type = []
                    maxtype = []
                    maxtype.append(node_degree)
                else:
                    edge_type = []
            print("度最大且边类型最多的节点", maxtype)
            print("此时锚点边的最大类型数", typenumber)

            for key, value in order_2_degree.items():
                if (value == Max_degree and key == maxtype[0] and G.nodes[key]['iscovered'] == 'no'):
                    list_least_nodes.append(key)
                    G.nodes[key]['iscovered'] = 'yes'
                    G.nodes[key]['color'] = 'red'  # 锚点color属性标记红色
                    break
            for node_2 in G.neighbors(list_least_nodes[num]):
                G.nodes[node_2]['iscovered'] = 'yes'  # 锚点周围邻居打上cover
            list_iscovered = list(nx.get_node_attributes(G, 'iscovered').values())
            # list_iscoverednode = list(nx.get_node_attributes(G, 'iscovered').keys())
            num_no = list_iscovered.count('no')
            print("未被覆盖节点数：", num_no)
            num = num + 1
            if num_no > 0:
                flag = True
            else:
                flag = False

        fo = open(anchorPath, 'w')
        for node in list_least_nodes:
            fo.write(str(node)+"\t"+str(order_2_degree[node])+"\n")
        fo.close()
        print("锚点已找全~~~~~~~")
        # print(G_rebuild._node)              # 查看节点属性
        # print(G_rebuild.edges(data=True))  # 查看边及所有属性
        print("原图的节点数：", G.number_of_nodes())  # 节点的数量
        print("原图的边数：", G.number_of_edges())  # 边的数量

        pot_anchor_set = []
        # 选取锚点（潜在锚点）
        print("选取潜在锚点中，，")
        for Potential_anchor in G.nodes:
            if Potential_anchor not in list_least_nodes:
                anchor_inLink = 0  # 记录节点的入度
                for edg in G.edges():
                    if Potential_anchor == edg[1] and edg[0] in list_least_nodes:  # 锚点->潜在锚点
                        anchor_inLink += 1
                if anchor_inLink >= self.pae_threshold:  # 潜在锚点的阈值(tag pae_threshold)
                    pot_anchor_set.append(Potential_anchor)
        print("选取完成")

        def find12Nei(G, node):  # 获取一阶二阶邻居
            nei1_li = []
            nei2_li = []
            for FNs in list(nx.neighbors(G, node)):  # find 1_th neighbors
                nei1_li .append(FNs)

            for n1 in nei1_li:
                for SNs in list(nx.neighbors(G, n1)):  # find 2_th neighbors
                    nei2_li.append(SNs)
            nei2_li = list(set(nei2_li) - set(nei1_li))
            if node in nei2_li:
                nei2_li.remove(node)
            return nei1_li, nei2_li

        def anchor_max_degree(anchor_list):  # 二阶邻居中度最大的锚点
            list_max_degree = 0
            anchor_index = 0
            for i in range(len(anchor_list)):
                an_degree = order_2_degree.get(anchor_list[i])
                if an_degree >= list_max_degree:
                    list_max_degree = an_degree
                    anchor_index = i
            return anchor_index

        def anchor_min_degree(anchor_list):  # 二阶邻居中度最小的锚点
            list_min_degree = 9999
            anchor_index = 0
            for i in range(len(anchor_list)):
                an_degree = order_2_degree.get(anchor_list[i])
                if an_degree <= list_min_degree:
                    list_min_degree = an_degree
                    anchor_index = i
            return anchor_index

        # fa = open(filename+"_anchorlink.txt", 'w')
        # shutil.copy2(self.initPath, self.generPath)
        fa = open(self.generPath, 'a')

        # 实体链接
        print("根据锚点开始实体链接。。。")
        # 二阶邻居链接
        for node in G.nodes():
            entity_nei2anchor = []  # 实体的二阶邻居锚点
            _, nei2 = find12Nei(G, node)
        # print("加速重构中。。。" + time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time())))
            if node in list_least_nodes:  # 锚点连边
                for anchor in list_least_nodes:
                    if node != anchor:
                        # anchor属于二阶邻居且node的度大于另一个锚点的度，则锚点间链接。
                        # 度大锚点->度小锚点
                        if anchor in nei2 and order_2_degree.get(node) >= order_2_degree.get(anchor):
                            fa.write(node + "\t" + "anchor_to_anchor" + "\t" + anchor+"\n")
            else:  # 实体连边锚点
                for anchor1 in list_least_nodes:
                    if anchor1 in nei2:
                        entity_nei2anchor.append(anchor1)
                if len(entity_nei2anchor) != 0:   # 锚点->实体
                    # index = random.random()*len(entity_nei2anchor)  #二阶邻居中随机一个锚点
                    index = anchor_max_degree(entity_nei2anchor)
                    fa.write(entity_nei2anchor[int(index)] + "\t" + "anchor_to_entity" + "\t"+node+"\n")

        # 潜在锚点链接. 实体->锚点
        for node in G.nodes():
            entity_nei2anchor1 = []  # 实体的二阶邻居锚点
            _, nei2p = find12Nei(G, node)
            if node not in list_least_nodes and node not in pot_anchor_set:
                for pot_anchor in pot_anchor_set:
                    if pot_anchor in nei2p:
                        entity_nei2anchor1.append(pot_anchor)
                if len(entity_nei2anchor1) != 0:   # 实体->潜在锚点(tag pae_select)
                    if self.pae_select == "big":
                        index1 = anchor_max_degree(entity_nei2anchor1)
                    elif self.pae_select == "small":
                        index1 = anchor_min_degree(entity_nei2anchor1)
                    elif self.pae_select == "random":
                        index1 = random.random()*len(entity_nei2anchor1)    #二阶邻居中随机一个潜在锚点
                    if self.pae_link == "tail":   # (tag pae_link)  
                        fa.write(node+"\t" + "entity_to_anchor" + "\t" + entity_nei2anchor1[int(index1)] + "\n")
                    elif self.pae_link == "head":
                        fa.write(entity_nei2anchor1[int(index1)] +"\t" + "anchor_to_entity" + "\t" + node+"\n")


        fa.close()
        print("链接完成！")
        try:
            os.remove(TKGtempPath)
            print(f"文件 {TKGtempPath} 已成功删除")
        except FileNotFoundError:
            print(f"文件 {TKGtempPath} 不存在")
        except Exception as e:
            print(f"删除文件时发生错误: {str(e)}")
        try:
            os.remove(anchorPath)
            print(f"文件 {anchorPath} 已成功删除")
        except FileNotFoundError:
            print(f"文件 {anchorPath} 不存在")
        except Exception as e:
            print(f"删除文件时发生错误: {str(e)}")
        # disrupt(self.generPath)

    def addEvent_Entity(self):
        # 二元排序
        def high_sort(x_obj):
            return (-x_obj.hap_num, -x_obj.type_num)
        def low_sort(x_obj):
            return (x_obj.hap_num, x_obj.type_num)
        
        # 创建一个空字典来存储数据
        find_T = re.compile(r"<\d+-\d+-\d+>")
        find_E = re.compile(r"<http://")  # 实体带http

        # 打开文件并逐行读取数据
        with open('haveT/haveT.handle.txt', 'r', encoding='utf-8') as file:
            for line in file:
                if len(re.findall(find_T, line)) != 0:  # 找到带T的三元组(实体-关系-事件)
                    # 分割每行数据，假设数据由制表符分隔
                    parts = line.strip().split('\t')
                    if len(re.findall(find_E, parts[0])) != 0:
                        key, value = parts[0], parts[2]

                        # 检查link_ent_dict字典中是否已经有该键
                        if key in self.link_ent_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            if value not in self.link_ent_dict[key]:
                                self.link_ent_dict[key].add(value)
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.link_ent_dict[key] = {value}

                        # 检查entity_dict字典中是否已经有该键
                        if key in self.entity_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            self.entity_dict[key].hap_num += 1
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.entity_dict[key] = Event(key, 1, 0)
        
        # 统计实体对应的类型数量
        for key in self.entity_dict:
            if key in self.link_ent_dict:
                self.entity_dict[key].type_num = len(self.link_ent_dict[key])
            else:
                print("the dict is wrong!")
        
        # 提取字典中的值
        entities = list(self.entity_dict.values())
        # for i in range(len(entities)):
        #     print(entities[i].name, ":", entities[i].hap_num, ",", entities[i].type_num)

        # 使用sorted函数对字典进行排序，并生成一个列表
        if self.event_frequence == "high":
            sorted_list = sorted(entities, key=high_sort)
        elif self.event_frequence == "low":
            sorted_list = sorted(entities, key=low_sort)

        shutil.copy(self.initPath, self.generPath)
        if self.isPAE == 1:
            TKGtempPath = self.generPath.replace(".txt", "_temp.txt")
            shutil.copy("noT/TKG-or.txt", TKGtempPath)
            TKGtempFile = open(TKGtempPath, 'a', encoding='utf-8')
        with open(self.generPath, 'a', encoding='utf-8') as file:
            num_elements_to_traverse = int(len(sorted_list) * self.event_threshold * 0.01)
            add_num = 0
            for i in range(num_elements_to_traverse):
                for value in self.link_ent_dict[sorted_list[i].name]:
                    file.write(sorted_list[i].name + "\t" + "entity_to_thing_" + str(add_num) + "\t" + value + "\n")
                    if self.isPAE == 1:
                        TKGtempFile.write(sorted_list[i].name + "\t" + "entity_to_thing_" + str(add_num) + "\t" + value + "\n")
                    add_num += 1
        if self.isPAE == 1:
            TKGtempFile.close()
        
    def addEvent3(self):
        # 按时间重复引入
        # 二元排序
        def high_sort(x_obj):
            return (-x_obj.hap_num, -x_obj.type_num)
        def low_sort(x_obj):
            return (x_obj.hap_num, x_obj.type_num)
        
        # 创建一个空字典来存储数据
        find_T = re.compile(r"<\d+-\d+-\d+>")
        find_E = re.compile(r"<http://")  # 实体带http

        
        # 打开文件并逐行读取数据
        thing_tmp_file = open(self.generPath.replace(".txt", "_thing_tmp.txt"), 'w', encoding= 'utf-8')
        with open('haveT/haveT.handle.txt', 'r', encoding='utf-8') as file:
            for line in file:
                if len(re.findall(find_T, line)) != 0:  # 找到带T的三元组(实体-关系-事件)
                    # 分割每行数据，假设数据由制表符分隔
                    parts = line.strip().split('\t')
                    if len(re.findall(find_E, parts[0])) != 0:
                        key, value = parts[0], parts[2]
                        thing_tmp_file.write(line)
                        # 检查link_ent_dict字典中是否已经有该键
                        if key in self.link_ent_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            if value not in self.link_ent_dict[key]:
                                self.link_ent_dict[key].add(value)
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.link_ent_dict[key] = {value}

                        # 检查entity_dict字典中是否已经有该键
                        if key in self.entity_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            self.entity_dict[key].hap_num += 1
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.entity_dict[key] = Event(key, 1, 0)
        thing_tmp_file.close()

        # 统计实体对应的类型数量
        for key in self.entity_dict:
            if key in self.link_ent_dict:
                self.entity_dict[key].type_num = len(self.link_ent_dict[key])
            else:
                print("the dict is wrong!")
        
        # 提取字典中的值
        entities = list(self.entity_dict.values())
        # for i in range(len(entities)):
        #     print(entities[i].name, ":", entities[i].hap_num, ",", entities[i].type_num)

        # 使用sorted函数对字典进行排序，并生成一个列表
        if self.event_frequence == "high":
            sorted_list = sorted(entities, key=high_sort)
        elif self.event_frequence == "low":
            sorted_list = sorted(entities, key=low_sort)

        shutil.copy(self.initPath, self.generPath)
        if self.isPAE == 1:
            TKGtempPath = self.generPath.replace(".txt", "_temp.txt")
            shutil.copy("noT/TKG-or.txt", TKGtempPath)
            TKGtempFile = open(TKGtempPath, 'a', encoding='utf-8')

        generFile = open(self.generPath, 'a', encoding='utf-8')
        with open(self.generPath.replace(".txt", "_thing_tmp.txt"), 'r', encoding='utf-8') as file:
            num_elements_to_traverse = int(len(sorted_list) * self.event_threshold * 0.01)
            sorted_list = sorted_list[:num_elements_to_traverse]

            for line in file:
                parts = line.strip().split('\t')
                if len(re.findall(find_E, parts[0])) != 0:
                    ent, eve = parts[0], parts[2] # 提取实体-事件
                    # 如果实体在待选序列中且事件在对应的待选实体-事件对中，将三元组放入最终文件中
                    if any(obj.name == ent for obj in sorted_list) and eve in self.link_ent_dict[ent]:
                        generFile.write(line)
                        if self.isPAE == 1:
                            TKGtempFile.write(line)
        generFile.close()
        if self.isPAE == 1:
            TKGtempFile.close()
        os.remove(self.generPath.replace(".txt", "_thing_tmp.txt"))

    def addEvent4(self):
        # 在策略二的基础上不加入事件到训练集中
        # 二元排序
        def high_sort(x_obj):
            return (-x_obj.hap_num, -x_obj.type_num)
        def low_sort(x_obj):
            return (x_obj.hap_num, x_obj.type_num)
        
        # 创建一个空字典来存储数据
        find_T = re.compile(r"<\d+-\d+-\d+>")
        find_E = re.compile(r"<http://")  # 实体带http

        # 打开文件并逐行读取数据
        with open('haveT/haveT.handle.txt', 'r', encoding='utf-8') as file:
            for line in file:
                if len(re.findall(find_T, line)) != 0:  # 找到带T的三元组(实体-关系-事件)
                    # 分割每行数据，假设数据由制表符分隔
                    parts = line.strip().split('\t')
                    if len(re.findall(find_E, parts[0])) != 0:
                        key, value = parts[0], parts[2]

                        # 检查link_ent_dict字典中是否已经有该键
                        if key in self.link_ent_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            if value not in self.link_ent_dict[key]:
                                self.link_ent_dict[key].add(value)
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.link_ent_dict[key] = {value}

                        # 检查entity_dict字典中是否已经有该键
                        if key in self.entity_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            self.entity_dict[key].hap_num += 1
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.entity_dict[key] = Event(key, 1, 0)
        
        # 统计实体对应的类型数量
        for key in self.entity_dict:
            if key in self.link_ent_dict:
                self.entity_dict[key].type_num = len(self.link_ent_dict[key])
            else:
                print("the dict is wrong!")
        
        # 提取字典中的值
        entities = list(self.entity_dict.values())
        # for i in range(len(entities)):
        #     print(entities[i].name, ":", entities[i].hap_num, ",", entities[i].type_num)

        # 使用sorted函数对字典进行排序，并生成一个列表
        if self.event_frequence == "high":
            sorted_list = sorted(entities, key=high_sort)
        elif self.event_frequence == "low":
            sorted_list = sorted(entities, key=low_sort)

        shutil.copy(self.initPath, self.generPath)
        if self.isPAE == 1:
            TKGtempPath = self.generPath.replace(".txt", "_temp.txt")
            shutil.copy("noT/TKG-or.txt", TKGtempPath)
            TKGtempFile = open(TKGtempPath, 'a', encoding='utf-8')
        with open(self.generPath, 'a', encoding='utf-8') as file:
            num_elements_to_traverse = int(len(sorted_list) * self.event_threshold * 0.01)
            add_num = 0
            for i in range(num_elements_to_traverse):
                for value in self.link_ent_dict[sorted_list[i].name]:
                    # file.write(sorted_list[i].name + "\t" + "entity_to_thing_" + str(add_num) + "\t" + value + "\n")
                    if self.isPAE == 1:
                        TKGtempFile.write(sorted_list[i].name + "\t" + "entity_to_thing_" + str(add_num) + "\t" + value + "\n")
                    add_num += 1
        if self.isPAE == 1:
            TKGtempFile.close()

    def addEvent5(self):
        # 二元排序
        def high_sort(x_obj):
            return (-x_obj.hap_num, -x_obj.type_num)
        def low_sort(x_obj):
            return (x_obj.hap_num, x_obj.type_num)
        
        # 创建一个空字典来存储数据
        find_T = re.compile(r"<\d+-\d+-\d+>")
        find_E = re.compile(r"<http://")  # 实体带http

        # 打开文件并逐行读取数据
        with open('haveT/haveT.handle.txt', 'r', encoding='utf-8') as file:
            for line in file:
                if len(re.findall(find_T, line)) != 0:  # 找到带T的三元组(实体-关系-事件)
                    # 分割每行数据，假设数据由制表符分隔
                    parts = line.strip().split('\t')
                    if len(re.findall(find_E, parts[0])) != 0:
                        key, value = parts[0], parts[2]

                        # 检查link_ent_dict字典中是否已经有该键
                        if key in self.link_ent_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            if value not in self.link_ent_dict[key]:
                                self.link_ent_dict[key].add(value)
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.link_ent_dict[key] = {value}

                        # 检查entity_dict字典中是否已经有该键
                        if key in self.entity_dict:
                            # 如果值还没有在集合中，就将值添加到集合中，并追加到值列表中
                            self.entity_dict[key].hap_num += 1
                        else:
                            # 如果键还不存在，创建一个新的键值对，并初始化一个包含值的集合
                            self.entity_dict[key] = Event(key, 1, 0)
        
        # 统计实体对应的类型数量
        for key in self.entity_dict:
            if key in self.link_ent_dict:
                self.entity_dict[key].type_num = len(self.link_ent_dict[key])
            else:
                print("the dict is wrong!")
        
        # 提取字典中的值
        entities = list(self.entity_dict.values())
        # for i in range(len(entities)):
        #     print(entities[i].name, ":", entities[i].hap_num, ",", entities[i].type_num)

        # 使用sorted函数对字典进行排序，并生成一个列表
        if self.event_frequence == "high":
            sorted_list = sorted(entities, key=high_sort)
        elif self.event_frequence == "low":
            sorted_list = sorted(entities, key=low_sort)

        shutil.copy(self.initPath, self.generPath)
        if self.isPAE == 1:
            TKGtempPath = self.generPath.replace(".txt", "_temp.txt")
            shutil.copy("noT/TKG-or.txt", TKGtempPath)
            TKGtempFile = open(TKGtempPath, 'a', encoding='utf-8')
        with open(self.generPath, 'a', encoding='utf-8') as file:
            num_elements_to_traverse = int(len(sorted_list) * self.event_threshold * 0.01)
            add_num = 0
            for i in range(num_elements_to_traverse):
                for value in self.link_ent_dict[sorted_list[i].name]:
                    # file.write(sorted_list[i].name + "\t" + "entity_to_thing_" + str(add_num) + "\t" + value + "\n")
                    if self.isPAE == 1:
                        TKGtempFile.write(sorted_list[i].name + "\t" + "entity_to_thing_" + str(add_num) + "\t" + value + "\n")
                    add_num += 1
        if self.isPAE == 1:
            TKGtempFile.close()


    def generate(self):
        self.name()
        print("将生成文件", self.generPath)
        if self.event_threshold != -1:
            if self.event_strategy == 1:
                self.addEvent()
            elif self.event_strategy == 2:
                self.addEvent_Entity()
            elif self.event_strategy == 3:
                self.addEvent3()
            elif self.event_strategy == 4:
                self.addEvent4()
            elif self.event_strategy == 5:
                self.addEvent5()
        if self.isPAE == 1:
            self.addPAE()

    def name(self):
        initPath = self.initPath
        fileDir = "../data"
        fileName = "train"

        if self.isPAE == 0:
            fileName += ("_CE-GERL_" + "策略" + str(self.event_strategy) + "_" + self.event_frequence + "_" + str(self.event_threshold) + "%")
        elif self.isPAE == 1:
            fileName += ("_CP-GERL_" + "策略" + str(self.event_strategy) + "_" + self.event_frequence + "_" + str(self.event_threshold) + "%")
            fileName += ("_" + str(self.pae_threshold) + "_" + self.pae_link + "_" + self.pae_select)
        fileName += ".txt"
        self.generPath = fileDir + "/" + fileName



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainset", type=str, default="data/train.txt", nargs="?")
    parser.add_argument("--event_frequence", type=str, default="high", nargs="?", help="high or low ")
    parser.add_argument("--event_threshold", type=int, default=-1, nargs="?")
    parser.add_argument("--event_strategy", type=int, default=-1, nargs="?")
    # parser.add_argument("--type_threshold", type=int, default=-1, nargs="?")
    parser.add_argument("--ispae", type=int, default=0, nargs="?")
    parser.add_argument("--pae_link", type=str, default="tail", nargs="?")
    parser.add_argument("--pae_threshold", type=int, default=3, nargs="?")
    parser.add_argument("--pae_select", type=str, default="big", nargs="?")
    args = parser.parse_args()

    generator = Generator(initPath=args.trainset,
                          event_frequence=args.event_frequence,
                          event_threshold=args.event_threshold,
                          event_strategy=args.event_strategy,
                        #   type_threshold=args.type_threshold,
                          isPAE=args.ispae,
                          pae_link=args.pae_link,
                          pae_threshold=args.pae_threshold,
                          pae_select=args.pae_select,
                          generPath="")
    generator.generate()
    print("训练集已生成到：", generator.generPath)