from SplicingModel import *
import os
import numpy as np
import h5py

### Error: without chunk, the result of model.predict is correct with CPU but wrong with GPU for large datasets.  Might be a bug! Investigate later.
def ModelPredict(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40_NWD1_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel', DS=['train', 'val', 'test'], batchSize=32, batchPerChunk=1000, prefetchSize=8):
    model =  tf.keras.models.load_model(inDir, custom_objects={"LossFunction": LossFunction})
    check_points = sorted([x.split('.index')[0] for x in os.listdir(inDir) if x.endswith('.ckpt.index')])
    for ckpt in check_points:
        model.load_weights(inDir + '/' + ckpt).expect_partial()
        for ds in DS:
            inF2 = inF.split('.tfds')[0] + f'_SMchr-{ds}.tfds'
            ouF = inDir + '/' + ckpt + f'.{ds}.ypred.h5'
            ouFile = h5py.File(ouF, 'w')
            dataset = tf.data.Dataset.load(inF2, compression='GZIP')
            chunkSize = batchSize * batchPerChunk
            for n in range(0, len(dataset), chunkSize):
                dataset2 = dataset.skip(n)
                dataset2 = dataset2.take(chunkSize)
                dataset2 = dataset2.batch(batchSize, drop_remainder=False)
                dataset2 = dataset2.prefetch(prefetchSize)
                y_pred = model.predict(dataset2)
                ouFile.create_dataset(f'y_pred_{n//chunkSize}', data=y_pred)
            ouFile.close()


ModelPredict(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank40_NWD1_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel', DS=['train', 'val'])
ModelPredict(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank200.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank200_NWD2_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel', DS=['train', 'val'])
ModelPredict(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank1000.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank1000_NWD3_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel', DS=['train', 'val'])
ModelPredict(inF='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank5000.tfds', inDir='Homo_sapiens.GRCh38.110.bed12_seq_nt5000_flank5000_NWD4_NRB4_NF32_SMchr_LR0.001_LFcc2d_NE10_SavedModel', DS=['train', 'val'])
