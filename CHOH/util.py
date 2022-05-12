from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import math
import dgl
import torch as th
import random
from rotate.rotations import xrotate,yrotate,zrotate,vrotate
from rotate.exchange import change
from rotate.move import translation,moveToNode
from einops.einops import rearrange
def dimenhance(coordin):
    N=len(coordin)
    #初始图
    coordin=np.array(coordin)
    #(4,3) (node,3)
    #平移
    move=translation(coordin)
    #(graph,node,3)
    #群转图
    nodeArr=moveToNode(move)
    #(node,move,3)
    #旋转
    randXTheta=random.randint(0,360)
    randYTheta=random.randint(0,360)
    randZTheta=random.randint(0,360)
    randVecTheta=random.randint(0,360)

    # print("x rotate is"+str(randXTheta))
    # print("y rotate is"+str(randYTheta))
    # print("z rotate is"+str(randZTheta))
    # print("v rotate is"+str(randVecTheta))

    vec = [1,1,1]
    xro=xrotate(nodeArr,randXTheta)
    yro=yrotate(nodeArr,randYTheta)
    zro=zrotate(nodeArr,randZTheta)
    vrot = vrotate(nodeArr, randVecTheta, vec)
    # (4, 4, 3)  (n,n,3)
    # (4, 4, 4, 3) (4,n,n,3)
    rotates=[nodeArr,xro,yro,zro]
    rotates=np.float32(rotates)
    #(4,4,12)   (4,n,3n)
    rotates=rotates.reshape((N, 4, 3*N))
    #(n,4,3n)
    rotates = rearrange(rotates, "b n d -> n b d")
    rotates=rotates.reshape((N, 12*N))

    #(n,12n) (4,48)
    changelist=[1,2,3]
    times=3
    change1=change(rotates,changelist,times)
    #(n,12n) (4,48)

    final=np.concatenate((rotates,change1),axis=1)
    return final
    #rotates = rearrange(rotates, "b h n d -> n b h d")

def getEdge(nodenum):
    send=[]
    for i in range(0,nodenum):
        for j in range(0,nodenum-1):
            send.append(i)
    receive=[]
    for i in range(0,nodenum):
        for j in range(0,nodenum):
            if(i==j):
                continue
            receive.append(j)
    return send,receive
def getEdgeNormalOh3():
    send    = [0,0,0,1,2,3]
    receive = [1,2,3,0,0,0]
    return send,receive
def getEdgeNormalCh5():
    send    = [0,0,0,0,1,2,3,4]
    receive = [1,2,3,4,0,0,0,0]
    return send,receive
def ShowGraph(graph, nodeLabel, EdgeLabel):
    plt.figure(figsize=(8, 8))
    G = graph.to_networkx(node_attrs=nodeLabel.split(), edge_attrs=EdgeLabel.split())  # 转换 dgl graph to networks
    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True)  # 画图，设置节点大小
    node_data = nx.get_node_attributes(G, nodeLabel)  # 获取节点的desc属性
    node_labels = {index: "N:" + str(data) for index, data in
                   enumerate(node_data)}  # 重新组合数据， 节点标签是dict, {nodeid:value,nodeid2,value2} 这样的形式
    pos_higher = {}

    for k, v in pos.items():  # 调整下顶点属性显示的位置，不要跟顶点的序号重复了
        if (v[1] > 0):
            pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
        else:
            pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=12)  # 将desc属性，显示在节点上
    edge_labels = nx.get_edge_attributes(G, EdgeLabel)  # 获取边的weights属性，

    edge_labels = {(key[0], key[1]): "dis+:" + str(edge_labels[key].item()) for key in
                   edge_labels}  # 重新组合数据， 边的标签是dict, {(nodeid1,nodeid2):value,...} 这样的形式
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)  # 将Weights属性，显示在边上

    print(G.edges.data())
    plt.show()
def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))
def getDistance(pointa,pointb):
    dis1 = pointb[0] - pointa[0]
    dis2 = pointb[1] - pointa[1]
    dis3 = pointb[2] - pointa[2]
    dis=math.pow(dis1,2)+math.pow(dis2,2)+math.pow(dis3,2)
    dis=math.sqrt(dis)
    return dis
def getEdgeFeature(temp):
    point_H0 = temp[0]
    point_H1 = temp[1]
    point_H2 = temp[2]
    point_H3 = temp[3]
    dis=[]
    disCH1 = getDistance(point_H0, point_H1)
    disCH2 = getDistance(point_H0, point_H2)
    disCH3 = getDistance(point_H0, point_H3)
    disCH4 = getDistance(point_H1, point_H0)
    disCH5 = getDistance(point_H1, point_H2)
    disCH6 = getDistance(point_H1, point_H3)
    disCH7 = getDistance(point_H2, point_H0)
    disCH8 = getDistance(point_H2, point_H1)
    disCH9 = getDistance(point_H2, point_H3)
    disCH10 = getDistance(point_H3, point_H0)
    disCH11 = getDistance(point_H3, point_H1)
    disCH12 = getDistance(point_H3, point_H2)
    dis.append(disCH1)
    dis.append(disCH2)
    dis.append(disCH3)
    dis.append(disCH4)
    dis.append(disCH5)
    dis.append(disCH6)
    dis.append(disCH7)
    dis.append(disCH8)
    dis.append(disCH9)
    dis.append(disCH10)
    dis.append(disCH11)
    dis.append(disCH12)
    return dis
def inputToGraphCH5(origindata,device):
    graphData=[]
    length=len(origindata)
    for i in range(0, length):
        nodenum = len(origindata[i])
        send,receive=getEdgeNormalCh5()
        u, v = th.tensor(send).to(device), th.tensor(receive).to(device)
        g = dgl.graph((u, v))
        #enchancedata = enhance(origindata[i])
        temp = th.from_numpy(origindata[i]).to(device)
        g.ndata['pos'] = temp
        # edgeFeature = getEdgeFeature(origindata[i])
        # tensor = th.Tensor(edgeFeature).to(device)
        # g.edata['dis'] = tensor
        # ShowGraph(g, "pos", 'dis')
        # print(g)
        # g = dgl.to_bidirected(g)
        g = g.to(device)
        graphData.append(g)
    return graphData
def inputToGraphOH3(origindata,device):
    graphData=[]
    length=len(origindata)
    for i in range(0, length):
        nodenum = len(origindata[i])
        send,receive=getEdgeNormalOh3()
        u, v = th.tensor(send).to(device), th.tensor(receive).to(device)
        g = dgl.graph((u, v))
        #enchancedata = enhance(origindata[i])
        temp = th.from_numpy(origindata[i]).to(device)
        g.ndata['pos'] = temp
        # edgeFeature = getEdgeFeature(origindata[i])
        # tensor = th.Tensor(edgeFeature).to(device)
        # g.edata['dis'] = tensor
        # ShowGraph(g, "pos", 'dis')
        # print(g)
        # g = dgl.to_bidirected(g)
        g = g.to(device)
        graphData.append(g)
    return graphData
def inputToGraph(origindata,device):
    graphData=[]
    length=len(origindata)
    for i in range(0, length):
        nodenum = len(origindata[i])
        send,receive=getEdge(nodenum)
        u, v = th.tensor(send).to(device), th.tensor(receive).to(device)
        g = dgl.graph((u, v))
        #enchancedata = enhance(origindata[i])
        temp = th.from_numpy(origindata[i]).to(device)
        g.ndata['pos'] = temp
        edgeFeature = getEdgeFeature(origindata[i])
        tensor = th.Tensor(edgeFeature).to(device)
        g.edata['dis'] = tensor
        # ShowGraph(g, "pos", 'dis')
        # print(g)
        # g = dgl.to_bidirected(g)
        g = g.to(device)
        graphData.append(g)
    return graphData
def inputToGroupGraph(origindata,device):
    graphData=[]
    length=len(origindata)
    for i in range(0, length):
        nodenum = len(origindata[i])
        send,receive=getEdge(nodenum)
        u, v = th.tensor(send).to(device), th.tensor(receive).to(device)
        g = dgl.graph((u, v))
        #enchancedata = enhance(origindata[i])
        temp = th.from_numpy(origindata[i]).to(device)
        g.ndata['pos'] = temp
        edgeFeature = getEdgeFeature(origindata[i])
        tensor = th.Tensor(edgeFeature).to(device)
        g.edata['dis'] = tensor
        # ShowGraph(g, "pos", 'dis')
        # print(g)
        # g = dgl.to_bidirected(g)
        g = g.to(device)
        graphData.append(g)
    return graphData

#用法
# x_train=dataEnhance(x_train,y_train,is_rotate=False
#                 ,is_translation=False
#                 ,is_replacement=False
#                 ,is_addAtomQuality=True)
#
# x_test=dataEnhance(x_test,y_test,is_rotate=False
#                 ,is_translation=False
#                 ,is_replacement=False
#                 ,is_addAtomQuality=True)
def dataEnhance(ori_x,ori_y
                ,is_rotate=True
                ,is_translation=True
                ,is_replacement=True
                ,is_addAtomQuality=True):
    new_x = []
    new_y = []
    quality = [15.9994, 1.00794, 1.00794, 1.00794]
    quality = np.float32(quality)

    for x in ori_x:
        # print(x)
        if (is_rotate):
            randXTheta = random.randint(0, 360)
            xro = xrotate(x, randXTheta)
            randYTheta = random.randint(0, 360)
            yro = yrotate(xro, randYTheta)
            randZTheta = random.randint(0, 360)
            zro = zrotate(yro, randZTheta)
            x=zro
            #x = x.squeeze()
        #if (is_translation):
        if (is_replacement):
            changelist = [1, 2, 3]
            times = 3
            x = change(x, changelist, times)
        if (is_addAtomQuality):
            x = np.column_stack((quality, x))
        new_x.append(x)
    newx = np.float32(new_x)
    return newx
def dataEnhance2(x_train):
    new_train = []
    for x in x_train:
        newx = dimenhance(x)
        new_train.append(newx)
    return new_train
def addQualify(oh3_x):
    quality=[15.9994,1.00794,1.00794,1.00794]
    quality=np.float32(quality)
    new_x=[]
    for x in oh3_x:
        #print(x)
        x=np.column_stack((quality,x))
        new_x.append(x)
    return new_x
from matplotlib import pyplot as plt
import networkx as nx
def ShowGraph(graph, nodeLabel, EdgeLabel):
    plt.figure(figsize=(8, 8))
    G = graph.to_networkx(node_attrs=nodeLabel.split(), edge_attrs=EdgeLabel.split())  # 转换 dgl graph to networks
    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True)  # 画图，设置节点大小
    node_data = nx.get_node_attributes(G, nodeLabel)  # 获取节点的desc属性
    node_labels = {index: "N:" + str(data) for index, data in
                   enumerate(node_data)}  # 重新组合数据， 节点标签是dict, {nodeid:value,nodeid2,value2} 这样的形式
    pos_higher = {}

    for k, v in pos.items():  # 调整下顶点属性显示的位置，不要跟顶点的序号重复了
        if (v[1] > 0):
            pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
        else:
            pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=12)  # 将desc属性，显示在节点上
    edge_labels = nx.get_edge_attributes(G, EdgeLabel)  # 获取边的weights属性，

    edge_labels = {(key[0], key[1]): "dis+:" + str(edge_labels[key].item()) for key in
                   edge_labels}  # 重新组合数据， 边的标签是dict, {(nodeid1,nodeid2):value,...} 这样的形式
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)  # 将Weights属性，显示在边上

    print(G.edges.data())
    plt.show()