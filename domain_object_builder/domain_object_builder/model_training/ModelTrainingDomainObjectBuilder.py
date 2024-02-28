import os
import pathlib
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from .ModelTrainingDomainObject import ModelTrainingDomainObject


class ModelTrainingDomainObjectBuilder:
    def __init__(self,
            config_model : DictConfig,
            datamodule   : pl.LightningDataModule
        ) -> None:
        self.config_model  = config_model
        self.domain_object = ModelTrainingDomainObject()
        self.domain_object.set_datamodule(datamodule)

    def build_tb_logger(self, version: str = None):
        from pytorch_lightning.loggers import TensorBoardLogger
        from paramsDIct_to_directoryName import ModelParameters2DirectoryName
        # ---
        params2dirName = ModelParameters2DirectoryName()
        tb_logger = TensorBoardLogger(
            version  = params2dirName.convert(self.config_model) if version is None else version,
            # version = log_dir_name,
            **self.config_model.logger
        )
        self.domain_object.set_tb_logger(tb_logger)
        # ---
        p = pathlib.Path(tb_logger.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config_model, tb_logger.log_dir + "/config.yaml")

    def build_trainer(self):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
        from cdsvae.domain.callbacks import CallbackTrainingTime
        # ---
        trainer = Trainer(
            logger    = self.domain_object.tb_logger,
            callbacks = [
                LearningRateMonitor(),
                ModelCheckpoint(
                    dirpath  = os.path.join(self.domain_object.tb_logger.log_dir , "checkpoints"),
                    filename = '{epoch}',
                    **self.config_model.checkpoint,
                )
            ] + [CallbackTrainingTime()],
            **self.config_model.trainer
        )
        self.domain_object.set_trainer(trainer)

    def get_domain_object(self):
        '''
            - return deepcopy(self.domain_object) はダメ
            - 並列化する場合などQueueを使っている場合deepcopyで直列化できない
        '''
        return self.domain_object
