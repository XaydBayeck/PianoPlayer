# 安装

安装`pytorch`作为神经网络框架和`pretty_mid`用于处理`midi`文件。
以及`numpy`等其他常用`python`库。

数据集使用`Tensorflow`[官方教程](https://tensorflow.google.cn/tutorials/audio/music_generation)中的 [`maestro-v2.0.0`]( https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip )


```python
import glob
import torch
import torch.utils.data as tud
import pretty_midi as pm
import numpy as np
from tqdm import  tqdm
import random
from typing import *
```

# 读取Midi文件

读取midi文件


```python
midifile = "/home/sid/Downloads/m_basho.mid"
pmf = pm.PrettyMIDI(midifile)
```

`instruments` 为乐器的列表

`instrument` 等于`pmf`的第一个乐器，类型为`pretty_midi.Instrument`, `instrument.notes` 即为该乐器的音符列表。


```python
instrument = pmf.instruments[0]
instrument.notes
```

`pretty_midi.Note`有4个属性

    - start    : 开始时间
    - end      : 结束时间
    - pitch    : 音高
    - velocity : 音符力度


创建midi文件使用`pmf.write(midi_file)`即可。

# 处理Midi数据

从 `instrument` 中读取到的音符列表中音符的三个特征`pitch`、`step`、`duration`将会被处理转换为`numpy`数组。


```python
def GetNoteSequence(instrument: pm.Instrument) -> np.ndarray:
    """
    先根据音符的开始时间排序;
    由音符的开始时间减去前一个音符的开始时间得到`step`;
    由音符的结束时间减去开始时间得到`duration`；
    """
    sorted_notes = sorted(instrument.notes, key=lambda x: x.start)
    assert len(sorted_notes) > 0

    prev_starts = [note.start for note in sorted_notes]
    prev_starts = [prev_starts[0]] + prev_starts[:-1]

    notes = [
        [note.pitch, note.start - prev_start, note.end - note.start]
        for note, prev_start in zip(sorted_notes, prev_starts)
    ]

    return np.array(notes)
```


```python
GetNoteSequence(instrument)
```

# 准备训练数据

pyTorch中数据处理的核心是`torch.utils.data.Dataset`和`torch.utils.data.DataLoader`
我们需要自定义一个Dataset来处理我们的midi数据
首先继承`torch.utils.data.Dataset`，然后实现`__init__`,`__getitem__`,`__len__`三个函数，功能分别为初始化，取出第i个数据，获得数据总数


在初始化中，首先用`glob`读出文件列表（`glob`可以使用通配符），然后遍历所有文件，用`pretty_midi`打开，得到它的音符序列，并对音符的`pitch`归一化（`tensorflow`官方教程这样干的）
将音符序列保存在类里面

（关于`np.append`的功能可以看这个[文章](https://blog.csdn.net/sinat_28576553/article/details/90247286)）


在读取第`i`个数据时，取出第`i`个音符开始的，长度为`seq_len`的序列作为输入数据，取出序列尾部的下一个音符为标签


最后加了`getendseq`功能，主要是方便测试时能够获得无标签的最后一个序列
<!-- ————————————————
版权声明：本文为CSDN博主「CaptainHarryChen」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/can919/article/details/122793127 -->


```python
class SequenceMIDI(tud.Dataset):
    def __init__(self, files, seq_len, max_file_num=None):
        notes = None

        filenames = glob.glob(files)
        print(f"Find {len(filenames)} files.")

        if max_file_num is None:
            max_file_num = len(filenames)
        print(f"Reading {max_file_num} files...")

        for f in tqdm(filenames[:max_file_num]): # tqdm 提供进度条        
            pmf = pm.PrettyMIDI(f)
            instrument = pmf.instruments[0]
            new_notes = GetNoteSequence(instrument)
            new_notes /= [128.0, 1.0, 1.0]
            if notes is not None:
                notes = np.append(notes, new_notes, axis=0)
            else:
                notes = new_notes

        self.seq_len = seq_len
        self.notes = np.array(notes, dtype=np.float32)
    
    def __len__(self):
        return len(self.notes) - self.seq_len

    def __getitem__(self, idx) -> Tuple[np.ndarray, dict]:
        label_note = self.notes[idx + self.seq_len]
        label = {
            'pitch': (label_note[0] * 128).astype(np.int64), 
            'step': label_note[1],
            'duration': label_note[2]
        }
        return self.notes[idx:idx+self.seq_len], label
    
    def getendseq(self) -> np.ndarray:
        return self.notes[-self.seq_len:]
```

# 加载数据

用刚刚构建的Dataset来构建一个DataLoader作为数据加载器
然后就可以像遍历数组一样遍历DataLoader来获取数据了


```python
# 一些资源
batch_size = 64
sequence_lenth = 25
max_file_num = 200
epochs = 200
learning_rate = 0.005

loss_weight = [0.1, 20.0, 1.0]

save_model_name = "music_producter.pth"

# 使用 GPU 训练如果可以
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

```


```python
trainning_data = SequenceMIDI(
    "maestro-v2.0.0/*/*.midi", 
    sequence_lenth,
    max_file_num=max_file_num
)
print(f"Read {len(trainning_data)} sequences.")
loader = tud.DataLoader(trainning_data, batch_size=batch_size)

for X, y in loader:
    print(f"X: {X.shape} {X.dtype}")
    print(f"y: {y}")
    break
```

# 模型构建

继承`torch.nn.Module`来构建自己的模型


一个`LSTM`处理输入的音符，再分别用三个全连接层算出`pitch`,`step`,`duration`，其中`pitch`的全连接层后使用`Sigmoid`将得到的值放在0~1之间


在这里`LSTM`在指定`batch_first=True`时，输入维度为 $(N,L,H_{in})$，分别为`batch`，序列长度，输入维度

`pitch`特征输出为128位，表示每个音高出现的权重
`step`和`duration`都是一维的标量

一般`torch.nn.`中的自带模型初始化都是`(输入维度，输出维度)`

在`forward`中，注意到`LSTM`的输出为两个，它的输出格式其实是 $output,(h_n,c_n)$，后面的**tuple**是我们用不到的隐藏状态
而`output`是一个序列，而我们只需要序列的最后一个，所以后面的`x`都是取`x[:-1]`
（具体可以去了解LSTM的原理）
<!-- ————————————————
版权声明：本文为CSDN博主「CaptainHarryChen」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/can919/article/details/122793127 -->


```python
class MusicProducter(torch.nn.Module):
    def __init__(self):
        super(MusicProducter, self).__init__()
        self.lstm = torch.nn.LSTM(3, 128, num_layers=1, batch_first=True)
        self.pitch_linear = torch.nn.Linear(128, 128)
        self.pitch_sigmoid = torch.nn.Sigmoid()
        self.step_linear = torch.nn.Linear(128, 1)
        self.duration_linear = torch.nn.Linear(128, 1)
    
    def forward(self, x, hidden_prev):
        x, hidden_prev = self.lstm(x, hidden_prev)
        pitch = self.pitch_sigmoid(self.pitch_linear(x[:, -1]))
        # print(f"row pitch predict data: {pitch}")
        step = self.step_linear(x[:, -1])
        duration = self.duration_linear(x[:, -1])
        return {'pitch': pitch, 'step': step, 'duration': duration}, hidden_prev
```


```python
class MusicProducterExport(torch.nn.Module):
    def __init__(self):
        super(MusicProducterExport, self).__init__()
        self.lstm = torch.nn.LSTM(3, 128, num_layers=1, batch_first=True)
        self.pitch_linear = torch.nn.Linear(128, 128)
        self.pitch_sigmoid = torch.nn.Sigmoid()
        self.step_linear = torch.nn.Linear(128, 1)
        self.duration_linear = torch.nn.Linear(128, 1)
    
    def forward(self, x, hidden_prev):
        x, hidden_prev = self.lstm(x)
        pitch = self.pitch_sigmoid(self.pitch_linear(x[:, -1]))
        step = self.step_linear(x[:, -1])
        duration = self.duration_linear(x[:, -1])
        return (pitch, step, duration), hidden_prev
```

# 损失函数

为了方便我把损失函数也写成一个模型
在这个损失函数中，分别计算三个特征的损失，并带权加和
`pitch`使用交叉熵（常用于分类器），而另外两个标量用均方差并带上使他变为正数的压力（毕竟时间都是正数）

注意到`torch`自带的`CrossEntropyLoss`既可以计算一个分类权重与下标的交叉熵，也可以两个分类权重的交叉熵
也就是下面两种都支持计算（这里我们使用第一种）
<!-- ————————————————
版权声明：本文为CSDN博主「CaptainHarryChen」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/can919/article/details/122793127 -->

```
[        1      ,        2       ,        3       ]
[[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.3, 0.7, 0.0]]
```


```
[[1.00, 0.00, 0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
[[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.3, 0.7, 0.0]]
```


```python
def mse_with_positive_pressure(pred, y):
    mse = (y-pred) ** 2
    positive_pressure = 10 * torch.maximum(-pred, torch.tensor(0))
    return torch.mean(mse + positive_pressure)
```


```python
class MPLoss(torch.nn.Module):
    def __init__(self, weight):
        super(MPLoss, self).__init__()
        self.weight = torch.Tensor(weight)
        self.pitch_loss = torch.nn.CrossEntropyLoss()
        self.step_loss = mse_with_positive_pressure
        self.duration_loss = mse_with_positive_pressure
    
    def forward(self, pred, y):
        losses = [
            self.pitch_loss(pred['pitch'], y['pitch']),
            self.step_loss(pred['step'], y['step']),
            self.duration_loss(pred['duration'], y['duration'])
            ]
        return sum([l * w for l, w in zip(losses, self.weight)])
```

关于loss权重的设置
可以先预先运行一下，得到几个损失值，然后手动设置权重使他们比较相近

# 训练模型

首先加载模型到GPU(如果可行)

设置损失后汉书，优化器

循环每个**epoch**，用`model.train()`将模型设置为训练模式

在每个**epoch**中，遍历`loader`来获取数据`batch`

将数据放在GPU/CPU上后，输入进模型，计算损失

将优化器的导数记录清零，再对`loss`求导，然后用`optimizer.step()`优化模型参数
<!-- 
————————————————
版权声明：本文为CSDN博主「CaptainHarryChen」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/can919/article/details/122793127 -->


```python
model = MusicProducter().to(device)
loss_fn = MPLoss(loss_weight).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)
print(loss_fn)
```


```python
print("Start trainning...")
size = len(loader.dataset)
for t in range(epochs):
    model.train()
    avg_loss = 0.0
    print(f"Epoch {t+1}\n--------------------------")
    for batch,(X, y) in enumerate(tqdm(loader)):
        X = X.to(device)
        for feature in y.keys():
            y[feature] = y[feature].to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss /= len(loader)
    print(f"average loss = {avg_loss}")
    if (t + 1) % 10 == 0:
        torch.save(model.state_dict(), save_model_name)
print("Done!")
```


```python
torch.save(model.state_dict(), save_model_name)
print(f"Saved PyTorch Model State to {save_model_name}")
```

# 生成音乐

## 预测下一个音符

首先用`model.eval()`将模型设置为测试模式，用`torch.no_grad()`让`pyTorch`不记录导数节约内存

读入的音符序列需要增加一个维度来代表**batch**，因为模型的输入是带有**batch**维度的

使用`torch.tensor.unsqueeze()`来增加维度（`[0,1,2]-->[[0,1,2]]`）
然后将输入数据扔进模型里得到**predictions**

根据**prediction**中音高`pitch`的128位权重输出，按权重随机产生音符，这里我手写了一个按权值随机

由于输出中`pitch`,`duration`,`step`都是带有一维**batch**的，所以使用`np.squeeze`把**batch**维度去掉（`[[2]]–>[2]`）
最后要将`step`与`duration`与`0`取`max`，防止输出负数时间
<!-- ————————————————
版权声明：本文为CSDN博主「CaptainHarryChen」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/can919/article/details/122793127 -->


```python
# def WeightedRandom(weight, k=100000) -> int:
#     sum = int(0)
#     for w in weight:
#         sum += int(k * w)
    
#     x = random.randint(1, sum)

#     sum = 0

#     for id, w in enumerate(weight):
#         sum += int(k * w)
#         if sum >= x:
#             return id
#     return x

def WeightedRandom(weight, k=100000) -> int:
    max = 0
    max_id = 0
    for id, w in enumerate(weight):
        if w > max:
            max = w
            max_id = id
    
    return max_id


# def PredictNextNote(model: MusicProducter, input: nd.ndarray):
def PredictNextNote(model: MusicProducter, input: np.ndarray, pre_hiddens):
    model.eval()
    with torch.no_grad():
        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
        pred, hiddens = model(input, pre_hiddens)
        pitch = WeightedRandom(np.squeeze(pred['pitch'], axis=0))
        step = np.maximum(np.squeeze(pred['step'], axis=0), 0)
        duration = np.maximum(np.squeeze(pred['duration'], axis=0), 0)
    return pitch, float(step), float(duration), hiddens
```

## 生成序列

首先需要一个起始的输入序列作为灵感

用`sample_file_name`中初始化一个`Dataset`，然后将他的最后一个序列作为输入

具体操作就是每预测一个音符，就先删除输入序列的第一个音符，并将生成的音符放进输入序列的末尾


```python
def CreateMIDIInstrumennt(notes: np.ndarray, instrument_name: str) -> pm.Instrument:
    instrument = pm.Instrument(pm.instrument_name_to_program(instrument_name))
    prev_start = 0
    for note in notes:
        prev_start += note[1]
        note = pm.Note(start=prev_start, end=prev_start +
                    note[2], pitch=note[0], velocity=100)
        instrument.notes.append(note)
    return instrument
```


```python
sample_file_name = "sample.midi"
output_file_name = "output0.midi"
# save_model_name = "music_producter.pth"
save_model_name = "model110.pth"
predict_length = 128
sequence_length = 25

model = MusicProducter()
model.load_state_dict(torch.load(save_model_name, map_location=torch.device(device)))

sample_data = SequenceMIDI(sample_file_name, sequence_length)
```


```python
filenames = glob.glob("sample.midi")
filenames
```


```python
sample_data
```


```python
cur = sample_data.getendseq()
res = []
prev_start = 0
```


```python
example = torch.tensor(cur, dtype=torch.float32).unsqueeze(0)
```


```python
export_model = MusicProducterExport()
export_model.load_state_dict(torch.load(save_model_name, map_location=torch.device(device)))
trace_cell = torch.jit.script(export_model, example)
trace_cell.save("music_producter.pt")
```


```python
print(trace_cell.code)
```


```python
torch.tensor(cur[0:3], dtype=torch.float32).unsqueeze(0)
```


```python
init_hiddens = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
```


```python
PredictNextNote(model, cur[0:3], init_hiddens)
```


```python
prev_hiddens = init_hiddens

for i in tqdm(range(predict_length)):
    pitch, step, duration, prev_hiddens = PredictNextNote(model, cur, pre_hiddens)
    res.append([pitch, step, duration])
    cur = cur[1:]
    cur = np.append(cur, [[pitch, step, duration]], axis=0)
    prev_start += step

pm_output = pm.PrettyMIDI()
pm_output.instruments.append(
    CreateMIDIInstrumennt(res, "Acoustic Grand Piano")
)
pm_output.write(output_file_name)
```
