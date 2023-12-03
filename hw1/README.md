# Photometric Stereo

## 项目结构
.
├── READMEmd
├── imread_datadir_re.py    # 读取原图像
├── load_datadir_re.py      # 读取图像数据
├── mainBaseline.py         # 程序入口
├── myIO.py                 # 保存图像
├── myPMS.py                # 计算 `normal` 和 `albedo`
├── pmsData                 # 原图像
│   ├── bearPNG
│   ├── buddhaPNG
│   ├── catPNG
│   └── potPNG
├── output                  # 输出图像, 从左至右以此为: 重建图像(处理阴影和高光), 重建图像(无处理), 原图像
│   ├── bearPNG
│   ├── buddhaPNG
│   ├── catPNG
│   └── potPNG
└── requirements.txt        # 依赖文件

## 运行方式
安装所需的依赖环境: (测试环境: `python=3.9`)
```shell
pip install -r requirements.txt
```

运行生成图像并保存: 
```shell
python mainBaseline.py
```