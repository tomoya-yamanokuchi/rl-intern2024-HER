


class ModelTrainingDomainObject:
    def set_datamodule(self, datamodule):
        from pytorch_lightning import LightningDataModule
        self.datamodule : LightningDataModule = datamodule

    def set_tb_logger(self, tb_logger):
        from pytorch_lightning.loggers import TensorBoardLogger
        self.tb_logger : TensorBoardLogger = tb_logger

    def set_trainer(self, trainer):
        from pytorch_lightning import Trainer
        self.trainer : Trainer = trainer



