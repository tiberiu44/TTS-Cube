#
# Author: Adriana Stan, october 2018
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

class Feature_Set:

    def __init__(self, config_file):
        '''
        Assume config file contains inputs of type
             IN: feat_name, category (binary, discrete, real, array), size
             OUT: feat_name, category (binary, discrete, real, array), size
        each on one line, separated by commas
        the config file should correspond to the label structure
        '''

        self.input_features_length = 0
        self.output_features_length = 0
        self._input_features= []
        self._output_features = []
        with open(config_file) as f:
            for line in f.readlines():
                feat_type = line.strip().split(':')[0]
                data = line.strip().split(':')[1].split(',')
                if feat_type == 'IN':
                    self._input_features.append(Feature_Item(data[0], data[1], int(data[2])))
                    self.input_features_length += int(data[2])
                elif (feat_type == 'OUT'):
                    self._output_features.append(Feature_Item(data[0], data[1], int(data[2])))
                    self.output_features_length += int(data[2])


    def get_input_feature_number(self):
        return len(self._input_features)

    def get_output_feature_number(self):
        return len(self._output_features)

    def get_input_features(self):
        return self._input_features

    def get_output_features(self):
        return self._output_features

    def get_input_feature_size(self):
        size = 0
        for feat in self._input_features:
            if feat.get_category() != 'D':
                size += feat.get_size()
            else:
                size += len(feat.discrete2int)
        return size


    def store(self, filename):
        with open(filename, 'w') as f:
            for feat in self._input_features:
                if feat.get_category()=='D':
                    f.write("---%s-%s-%s\n" %(feat.get_name(),feat.get_category(), feat.get_size()))
                    for key in feat.discrete2int :
                        f.write("%s %s\n" %(key, feat.discrete2int[key]))


class Feature_Item:
    '''
    Stores one feature info
        - name - feature name
        - category - binary, discrete, real
        - size - feature length if array, 1 for others
    '''

    def __init__(self, name, category, size = 1):
        self._name = name
        self._category = self._norm_cat(category)
        self._size = size
        if self._category == 'D':
            self.discrete2int = {}
            self.int2discrete = {}

    def _norm_cat(self, category):
        if category.lower().strip() == "binary":
            return 'B'
        elif category.lower().strip() == "discrete":
            return 'D'
        elif category.lower().strip() == "real":
            return 'R'
        elif category.lower().strip() =="array":
            return 'A'
        else:
            return 'U'

    def update_discrete2int(self, value):
        if value not in self.discrete2int:
            self.discrete2int[value] = len(self.discrete2int)
            self.int2discrete[len(self.discrete2int)] = value

    def get_one_hot (self, value):
        from numpy import zeros
        vector = zeros(len(self.discrete2int))
        vector[self.discrete2int[value]] = 1
        return vector

    def get_category(self):
        return self._category


    def get_name(self):
        return self._name


    def get_size(self):
        return self._size




