import os
import sys
from _ast import Lambda

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch import nn
from cytomine.models import Job
from neubiaswg5 import CLASS_PIXCLA
from neubiaswg5.helpers import get_discipline, NeubiasJob, prepare_data, upload_data, upload_metrics
from neubiaswg5.helpers.data_upload import imwrite, imread


from pspnet import PSPNet


MEAN = [0.78676176, 0.50835603, 0.78414893]
STD = [0.16071789, 0.24160224, 0.12767686]


def open_image(path):
    img = Image.open(path)
    trsfm = Compose([ToTensor(), Normalize(mean=MEAN, std=STD), Lambda(lambda x: x.unsqueeze(0))])
    return trsfm(img)


def predict_img(net, img_path, device, out_threshold=0.5):
    net.eval()

    with torch.no_grad():
        x = open_image(img_path)
        logits = net(x.to(device))
        y_pred = nn.Softmax(dim=1)(logits)
        proba = y_pred.detach().cpu().squeeze(0).numpy()[1, :, :]
        return proba > out_threshold


def load_model(filepath):
    net = PSPNet(pretrained=False)
    net.cpu()
    net.load_state_dict(torch.load(filepath, map_location='cpu'))
    return net


class Monitor(object):
    def __init__(self, job, iterable, start=0, end=100, period=None, prefix=None):
        self._job = job
        self._start = start
        self._end = end
        self._update_period = period
        self._iterable = iterable
        self._prefix = prefix

    def update(self, *args, **kwargs):
        return self._job.job.update(*args, **kwargs)

    def _get_period(self, n_iter):
        """Return integer period given a maximum number of iteration """
        if self._update_period is None:
            return None
        if isinstance(self._update_period, float):
            return max(int(self._update_period * n_iter), 1)
        return self._update_period

    def _relative_progress(self, ratio):
        return int(self._start + (self._end - self._start) * ratio)

    def __iter__(self):
        total = len(self)
        for i, v in enumerate(self._iterable):
            period = self._get_period(total)
            if period is None or i % period == 0:
                statusComment = "{} ({}/{}).".format(self._prefix, i + 1, len(self))
                relative_progress = self._relative_progress(i / float(total))
                self._job.job.update(progress=relative_progress, statusComment=statusComment)
            yield v

    def __len__(self):
        return len(list(self._iterable))


def main(argv):
    with NeubiasJob.from_cli(argv) as nj:
        problem_cls = get_discipline(nj, default=CLASS_PIXCLA)
        is_2d = True

        nj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")
        in_images, gt_images, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, nj, **nj.flags)

        # 2. Call the image analysis workflow
        nj.job.update(progress=10, statusComment="Load model...")
        net = load_model("/app/model.pth")
        device = torch.device("cpu")

        for in_image in Monitor(nj, in_images, start=20, end=75, period=0.05, prefix="Apply UNet to input images"):
            mask = predict_img(net, in_image.filepath, device="cpu", out_threshold=nj.parameters.threshold)

            imwrite(
                path=os.path.join(out_path, in_image.filename),
                image=mask.astype(np.uint8),
                is_2d=is_2d
            )

        # 4. Create and upload annotations
        nj.job.update(progress=70, statusComment="Uploading extracted annotation...")
        upload_data(problem_cls, nj, in_images, out_path, **nj.flags, is_2d=is_2d, monitor_params={
            "start": 70, "end": 90, "period": 0.1
        })

        # 5. Compute and upload the metrics
        nj.job.update(progress=90, statusComment="Computing and uploading metrics (if necessary)...")
        upload_metrics(problem_cls, nj, in_images, gt_path, out_path, tmp_path, **nj.flags)

        # 6. End the job
        nj.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])

