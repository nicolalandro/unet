from data import *
from model import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATASET_BASE_PATH = 'data/membrane'

train_dataset = '%s/train' % DATASET_BASE_PATH
test_dataset = "%s/test" % DATASET_BASE_PATH
results_folder = '%s/results' % DATASET_BASE_PATH
model_name = '%s/unet.hdf5' % DATASET_BASE_PATH

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(2, train_dataset, 'images', 'labels', data_gen_args, save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

testGene = testGenerator("%s" % test_dataset)
results = model.predict_generator(testGene, 30, verbose=1)
saveResult(results_folder, results)
