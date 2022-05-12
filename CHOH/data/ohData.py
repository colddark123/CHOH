from dgl.data import DGLDataset
import torch as th
import dgl
class MyGraphDataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    url : str
        下载原始数据集的url。
    raw_dir : str
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """



    def __init__(self,input,labels,device):
        super(MyGraphDataset, self).__init__(name='molcular')
        self.labels=labels
        graphData = []
        length = len(input)
        def getEdge(nodenum):
            send = []
            for i in range(0, nodenum):
                for j in range(0, nodenum - 1):
                    send.append(i)
            receive = []
            for i in range(0, nodenum):
                for j in range(0, nodenum):
                    if (i == j):
                        continue
                    receive.append(j)
            return send, receive
        for i in range(0, length):
            nodenum = len(input[i])
            send, receive = getEdge(nodenum)
            u, v = th.tensor(send).to(device), th.tensor(receive).to(device)
            g = dgl.graph((u, v))
            # enchancedata = enhance(origindata[i])
            temp = th.from_numpy(input[i]).to(device)
            g.ndata['R'] = temp
            g.ndata['Z'] = th.IntTensor([8,1,1,1]).to(device)

            # edgeFeature = getEdgeFeature(origindata[i])
            # tensor = th.Tensor(edgeFeature).to(device)
            # g.edata['dis'] = tensor
            # ShowGraph(g, "pos", 'dis')
            # print(g)
            # g = dgl.to_bidirected(g)
            g = g.to(device)
            graphData.append(g)
        self.graphs=graphData




    def download(self):
        # 将原始数据下载到本地磁盘
        pass

    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码
        pass

    def __getitem__(self, idx):
        return self.graphs[idx],self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        # 将处理后的数据保存至 `self.save_path`
        pass

    def load(self):
        # 从 `self.save_path` 导入处理后的数据
        pass

    def has_cache(self):
        # 检查在 `self.save_path` 中是否存有处理后的数据
        pass