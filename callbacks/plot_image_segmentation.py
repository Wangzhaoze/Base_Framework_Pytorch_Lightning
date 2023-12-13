import pytorch_lightning as pl
import torch


class PlotImageSegmentation(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_validation_epoch_end(self, trainer, model):
        self.val_dataloader = trainer.datamodule.val_dataloader()

        with torch.no_grad():
            for images, targets in self.val_dataloader:
                images = images.to(device=torch.device('cuda'))

                pred = model(images)

                logger = trainer.logger.experiment

                logger.add_image(
                    'Segmentation/Inputs', images[0], global_step=trainer.global_step
                )
                logger.add_image(
                    'Segmentation/Output', pred[0], global_step=trainer.global_step
                )

                break  # Only visualize the first batch

        pass
