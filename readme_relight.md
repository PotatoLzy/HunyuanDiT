# 修改内容
## 数据集修改：
1. 在hydit/data_loader/arrow_load_stream.py中新增RelightingArrowStream类，继承TextImageArrowStream，修改__getitem__()函数使其可以对fg和bg数据的读取。

ps：在生成数据后，请检查生成数据中fg_path和bg_path路径伤的文件是否存在，防止运行代码时报错FileNotFoundError。
 
2. 新增hydit/data_loader/csv2arrow_relight.py，在make_arrow()函数中新增relight_mode参数，加入对fg_path和bg_path列进行处理，读取相应的图片文件至stream中。

a. 对csv文件的格式要求：

如果是不含背景生成的图像，需要在csv文件中保存image_path, text_zh, fg_path三列。（顺序如上）

如果是含背景生成的图像，需要在csv文件中保存image_path, text_zh, fg_path, bg_path四列。（顺序如上）

image_path, text_zh为hunyuan本身需要的数据参数，fg_path/bg_path为额外增加的数据参数。

0918 代码中增加了该文件缺少的依赖包。

 
3. 对hydit/config.py进行修改，增加参数args.relight_mode，choices=None, fg, bg。fg代表只使用前景图像进行relighting，bg代表使用前景+背景进行relighting。

 
训练模型修改：
4. 新增hydit/train_deepspeed_relight.py。
在其中修改dit模型的x_embedder.proj(Conv2d)的输入通道数，并对preparre_model_inputs函数进行调整，使其可以应对调整后的数据集RelightingArrowStream类。
 
5. 加入train_relight.sh、run_g_relight.sh/train_relight_v1.1.sh、run_g_relight_v1.1.sh（base: train.sh, run_g.sh），用于调用代码。

目前进度：目前train_relight.sh仍在fix，train_relight_v1.1.sh已测试到loss反向传播处，但由于单卡显存不够，还没有验证后续的代码是否可行。
 
6. 修改模型，只调整unet的参数。
 
7. 修改diffusion模型的输入（具体见hydit/diffusion文件夹中的修改信息。由于仅对前四个通道的信息进行加噪去噪，后面四个/八个维度只需要在模型中输入并作为信息生成真实图像即可。）
 

