from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning import loggers

from src.Nets import *
import multiprocessing as mp

if __name__ == '__main__':

    mp.freeze_support() #多进程支持

    tb_logger = loggers.TensorBoardLogger('logs/',name='tb_unet')
    # tb_logger.experiment
    model = LitModel(AttU_Net())

    #假设你需要的batch大小：16
    #当前 cur_batch = 2
    #accumulate_grad_batches = 16/2 #累计一定batch 后进行梯度更新
    trainer = Trainer(gpus=[0],max_epochs=20,logger=tb_logger,amp_level='O1',precision=16,use_amp=True
                      ,accumulate_grad_batches=8 //TRAIN_BATCH)
    trainer.fit(model)

    # model = LitModel(AttU_Net())
    # model.prepare_data()