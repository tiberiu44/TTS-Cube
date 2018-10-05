#
# Author: Tiberiu Boros
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import dynet_config
import optparse
import sys


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--cleanup', action='store_true', dest='cleanup',
                      help='Cleanup temporary training files and start from fresh')
    parser.add_option('--phase', action='store', dest='phase',
                      choices=['1', '2', '3', '4', '5', '6', '7'],
                      help='select phase: 1 - prepare corpus; 2 - train vocoder; 3 - train encoder; 4 - end-to-end; '
                           '5 - test vocoder; 6 - convert to SparseLSTM; 7 - train SparseLSTM')
    parser.add_option("--batch-size", action='store', dest='batch_size', default='1000', type='int',
                      help='number of samples in a single batch (default=1000)')
    parser.add_option("--set-mem", action='store', dest='memory', default='2048', type='int',
                      help='preallocate memory for batch training (default 2048)')
    parser.add_option("--autobatch", action='store_true', dest='autobatch',
                      help='turn on/off dynet autobatching')
    parser.add_option("--resume", action='store_true', dest='resume',
                      help='resume from last checkpoint')
    parser.add_option("--no-guided-attention", action='store_true', dest='no_guided_attention',
                      help='disable guided attention')
    parser.add_option("--no-bounds", action='store_true', dest='no_bounds',
                      help='disable fixed synthesis length')
    parser.add_option("--use-gpu", action='store_true', dest='gpu',
                      help='turn on/off GPU support')
    parser.add_option('--train-folder', action='store', dest='train_folder',
                      help='Location of the training files')
    parser.add_option('--dev-folder', action='store', dest='dev_folder',
                      help='Location of the development files')
    parser.add_option('--out-folder', action='store', dest='out_folder',
                      help='Location of the output files')                      
    parser.add_option('--target-sample-rate', action='store', dest='target_sample_rate',
                      help='Resample input files at this rate (default=16000)', type='int', default=16000)
    parser.add_option('--mgc-order', action='store', dest='mgc_order', type='int',
                      help='Order of MGC parameters (default=60)', default=60)
    parser.add_option('--sparsity-target', action='store', type='int', default='95', dest='sparsity_target',
                      help='Target sparsity rate for LSTM')
    parser.add_option('--sparsity-step', action='store', type='int', default='5', dest='sparsity_step',
                      help='Step size when increasing sparsity')
    parser.add_option('--sparsity-increase-at', action='store', type='int', default='200', dest='sparsity_increase',
                      help='Number of files to train on between sparsity increase')

    (params, _) = parser.parse_args(sys.argv)

    memory = int(params.memory)
    if params.autobatch:
        autobatch = True
    else:
        autobatch = False
    dynet_config.set(mem=memory, random_seed=9, autobatch=autobatch)
    if params.gpu:
        dynet_config.set_gpu()




    
    ##############################        
    ## PHASE 1) Prepare corpus
    ##############################
    
    def phase_1_prepare_corpus(params):
        from io_modules.setup_training import Setup_Training
        if not params.out_folder:
            sys.stdout.write ("***\n\tWARNING! No OUTPUT folder set, using current location!\n***")
            out_folder = './'
        else:
            out_folder = params.out_folder    
        sys.stdout.write("Preparing the training and dev data! \n")

        st = Setup_Training(params.train_folder, params.dev_folder, out_folder)
        st.setup()


    ##############################        
    ## PHASE 2) Train vocoder
    ##############################
    def phase_2_train_vocoder(params):
        from io_modules.dataset import Dataset
        from models.vocoder import Vocoder
        from trainers.vocoder import Trainer
        vocoder = Vocoder(params)
        if params.resume:
            sys.stdout.write('Resuming from previous checkpoint\n')
            vocoder.load('data/models/rnn_vocoder')
        trainset = Dataset("data/processed/train")
        devset = Dataset("data/processed/dev")
        sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
            len(devset.files)) + ' development files\n')
        trainer = Trainer(vocoder, trainset, devset)
        trainer.start_training(20, params.batch_size, params.target_sample_rate)


    ##############################        
    ## PHASE 3) Train encoder
    ##############################
    def phase_3_train_encoder(params):
        from io_modules.setup_training import Setup_Training
        from io_modules.features import Feature_Set
        from models.encoder import Encoder
        from trainers.encoder import Trainer

        if not params.out_folder:
            sys.stdout.write ("***\n\tWARNING! No OUTPUT folder set, using current location!\n***")
            out_folder = './'
        else:
            out_folder = params.out_folder

        st = Setup_Training(params.train_folder, params.dev_folder, out_folder)
        trainset = st.train_file_list
        devset = st.dev_file_list
        sys.stdout.write('Found ' + str(len(trainset)) + ' training files and ' + str(
            len(devset)) + ' development files\n')
        features = Feature_Set('feat.config')
        count = 0
        if not params.resume:
            for train_file in trainset:
                count += 1
                if count % 100 == 0:
                    sys.stdout.write('\r' + str(count) + '/' + str(len(trainset.files)) + ' processed files')
                    sys.stdout.flush()
                from io_modules.dataset import DatasetIO
                dio = DatasetIO(features)
                lab_list = dio.read_input_feats(st.train_data_folder+'/'+ train_file + ".lab", features)
            sys.stdout.write('\r' + str(count) + '/' + str(len(trainset)) + ' processed files\n')
            features.store(st.model_folder+'/encoder.encodings')
        else:
            features.load(st.model_folder+'encoder.encodings')

        if params.resume:
            runtime = True  # avoid ortonormal initialization
        else:
            runtime = False


        encoder = Encoder(params, features, runtime=runtime)

        if params.resume:
            sys.stdout.write('Resuming from previous checkpoint\n')
            encoder.load(st.model_folder+'/rnn_encoder')

        if params.no_guided_attention:
            sys.stdout.write('Disabling guided attention\n')
        if params.no_bounds:
            sys.stdout.write('Using internal stopping condition for synthesis\n')

        trainer = Trainer(encoder, features, st)
        trainer.start_training(10, 1000, params)

    ##############################        
    ## PHASE 5) Test vocoder
    ##############################
    def phase_5_test_vocoder(params):
        from io_modules.dataset import Dataset
        from models.vocoder import Vocoder
        from trainers.vocoder import Trainer
        vocoder = Vocoder(params, runtime=True)
        vocoder.load('data/models/rnn')
        trainset = Dataset("data/processed/train")
        devset = Dataset("data/processed/dev")
        sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
            len(devset.files)) + ' development files\n')
        trainer = Trainer(vocoder, trainset, devset)
        trainer.synth_devset(params.batch_size, target_sample_rate=params.target_sample_rate, sample=True,
                             temperature=0.8)


    ##############################        
    ## PHASE 6) Sparse LSTM
    ##############################
    def phase_6_convert_to_sparse_lstm(params):
        sys.stdout.write('Converting existing vocoder to SparseLSTM...\n')
        sys.stdout.flush()
        f = open('data/models/rnn_vocoder.network')
        lines = f.readlines()
        f.close()
        f = open('data/models/rnn_vocoder_sparse.network', 'w')

        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.startswith('#Parameter# /vanilla-lstm-builder'):
                f.write(line)
            else:
                # write the standard LSTM part
                for zz in range(4):  # there are 4 lines per VanillaLSTM for the Weights
                    f.write(lines[index + zz])

                # get the size of P1
                sz = lines[index].split(" ")[2]
                sz = sz.replace("{", "").replace("}", "").replace(",", " ").split(" ")
                print("\tfound VanillaLSTM paramter with size", sz)
                f.write(lines[index].replace("/_0", "/_2"))
                p_size = int(sz[0]) * int(sz[1])
                for x in range(p_size):
                    f.write("+1.00000000e+00 ")
                f.write("\n")

                sz = lines[index + 2].split(" ")[2]
                sz = sz.replace("{", "").replace("}", "").replace(",", " ").split(" ")
                print("\tfound VanillaLSTM paramter with size", sz)
                f.write(lines[index + 2].replace("/_1", "/_3"))
                p_size = int(sz[0]) * int(sz[1])
                for x in range(p_size):
                    f.write("+1.00000000e+00 ")

                f.write("\n")

                index += 4

                # rename the bias parameter and write it to the file
                f.write(lines[index].replace("/_2", "/_4"))
                f.write(lines[index + 1])
                index += 1

            index += 1

        f.close()
        sys.stdout.write('done\n')


    def phase_7_train_sparse(params):
        sys.stdout.write("Starting sparsification for VanillaLSTM\n")
        from io_modules.dataset import Dataset
        from models.vocoder import Vocoder
        from trainers.vocoder import Trainer
        vocoder = Vocoder(params, use_sparse_lstm=True)

        sys.stdout.write('Resuming from previous checkpoint\n')
        vocoder.load('data/models/rnn_vocoder_sparse')
        sys.stdout.write("Reading datasets\n")
        sys.stdout.flush()

        trainset = Dataset("data/processed/train")
        devset = Dataset("data/processed/dev")
        sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
            len(devset.files)) + ' development files\n')
        sys.stdout.flush()
        trainer = Trainer(vocoder, trainset, devset)
        trainer.start_training(20, params.batch_size, params.target_sample_rate, params=params)


    if params.phase and params.phase == '1':
        phase_1_prepare_corpus(params)
    if params.phase and params.phase == '2':
        phase_2_train_vocoder(params)
    if params.phase and params.phase == '3':
        phase_3_train_encoder(params)
    if params.phase and params.phase == '4':
        print("Not yet implemented. Still wondering if this is really required")
    if params.phase and params.phase == '5':
        phase_5_test_vocoder(params)
    if params.phase and params.phase == '6':
        phase_6_convert_to_sparse_lstm(params)
    if params.phase and params.phase == '7':
        phase_7_train_sparse(params)
