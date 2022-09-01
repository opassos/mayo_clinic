from datasets import Dataset, ClassLabel

from toolkit.adriano.classification import *
from toolkit.adriano_v0.modeling.datamodule import DataModule ## TODO: review this and move to main
from toolkit.adriano_v0.modeling.splitter import RandomSplitter ## TODO: review this and move to main
from toolkit.adriano.modeling.sampler import UnderSample
from toolkit.adriano.image.open import open_img_as_tensor

### CUSTOM TOOLKIT FUNCTIONS ###
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
def encode_img(x):
    return (ToPILImage()(make_grid(x, nrow=4))).resize((1024,1024))
class LogFirstBatch(CB.LogFirstBatch):
    def __init__(self, max_n = 64, img_keys=None, ignore_keys=None):
        self.tensor2img = encode_img
        self.max_n = max_n
        self.img_keys = img_keys or [Keys.input]
        self.ignore_keys = ignore_keys or []
#############

DATASET_PATH = '/mnt/c/Users/adria/Downloads/kaggle-mayo'

ds = Dataset.from_csv(DATASET_PATH + '/train.csv')
ds

label_encoder = ClassLabel(names = ['CE', 'LAA'])

def mayo_type_tfm(example):
    img_id = example['image_id']
    imgs = [open_img_as_tensor(f"{DATASET_PATH}/train/{img_id}_{i}.jpg") for i in range(16)]
    example[Keys.input] = torch.stack(imgs)
    example[Keys.target] = torch.tensor(label_encoder.str2int(example['label']))
    return example


dblock = BaseDataBlock(
    type_tfm=[mayo_type_tfm]
)
splitter = RandomSplitter(seed=42) # replace this with a proper splitter after an EDA
resampler = UnderSample('label')

ds_train, ds_test = splitter(ds)
ds_valid = UnderSample('label').sample(ds_test)

dm = DataModule(
    (ds_train, ds_valid, ds_test), dblock, None, resampler,
    num_workers=10, prefetch_factor=2, batch_size=4,
    shuffle=False, drop_last=False,
    train__shuffle=True, train__drop_last=True,
    )
print("Creating train dataloader for debugging ...")
dm.setup()
dl_train = dm.train_dataloader()

batch_tfm_train = K.VideoSequential(
    K.ColorJiggle(0.1, 0.2, 0.1, 0, p=0.5),
    K.RandomAffine(2, (0.1, 0.1), (0.95, 1.05), (0.05, 0.05), p=0.5),
)
batch_tfm_test = None
print('='*50)
print('Shapes of tensors in train_batch:')
batch = next(iter(dl_train))
batch[Keys.input] = batch_tfm_train(batch[Keys.input])
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k} -> {v.shape}")
print('='*50)


class MayoNeck(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        x = x.view(-1, self.n, x.shape[-1])
        x = self.pool(x.swapaxes(1, 2)).squeeze(-1)
        return x

preprocessor = module_dict_wraper(nn.Flatten(0, 1), input_key=Keys.input, output_key=Keys.input)
neck = module_dict_wraper(MayoNeck(16), input_key=Keys.embeddings, output_key=Keys.features)
head = module_dict_wraper(LinearHead(2), input_key=Keys.features, output_key=Keys.logits)
loss = module_dict_wraper(nn.CrossEntropyLoss(), input_key=[Keys.logits, Keys.target], output_key=Keys.loss)
model = BaseModel(
    preprocessor=preprocessor,
    backbone='resnet18',
    neck=neck,
    head=head,
    loss=loss,
    input_shape=(2, 16, 3, 1024, 1024),
)

with torch.no_grad():
    outputs = model.shared_step(batch)

print('='*50)
print('Checking model output shapes:')
for k, v in outputs.items():
    if 'loss' in k:
        print(k, '->', v.item())
    else:
        print(k, '->', v.shape)

optimizer = OneCycle(max_lr=1e-4, pct_start=0.2)
callbacks=[
    PLCB.StochasticWeightAveraging(),
    CB.BatchImgAugment(batch_tfm_train, batch_tfm_test),
    LogFirstBatch(),
    CB.ClassificationMetrics(2),
    CB.ClassificationReport(2),
    # CB.LogEmbeddings(
    #     max_n = 128,
    #     embedding_key=Keys.features,
    #     extra_keys=[Keys.conf+'_binary', Keys.pred+'_binary', 'id', Keys.input+'_ref']
    #     ),
]
trainer = Trainer(model, dm, optimizer, callbacks, {
    'max_epochs': 50,
    'precision': 16,
    'accumulate_grad_batches': 16,
    })

with wandb.init(
    job_type='classification',
    project='kaggle-mayo',
    entity='coldfir3',
    save_code=True,
    dir=str(DefaultPath.base),
    ) as run:
    run_name = run.dir.split('/')[-2].replace('run-', '')+'_baseline'
    checkpoint_path = DefaultPath.checkpoints/run.project_name()/run_name
    trainer.add_callback([
        PLCB.ModelCheckpoint(
            dirpath=checkpoint_path,
            save_last=True,
            save_top_k=0
        ),
        CB.ExportOnnx(checkpoint_path, opset=11)
        ])
    trainer.fit(run)