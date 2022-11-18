

"""
modify GreedyDecoder to adapt to MindSpore
"""

import numpy as np
from deepspeech_pytorch.decoder import GreedyDecoder

class MSGreedyDecoder(GreedyDecoder):
    """
    GreedyDecoder used for MindSpore
    """

    def process_string(self, sequence, size, remove_repetitions=False):
        """
        process string
        """
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, offsets

    def decode(self, probs, sizes=None):
        probs = probs.asnumpy()
        sizes = sizes.asnumpy()

        max_probs = np.argmax(probs, axis=-1)
        strings, offsets = self.convert_to_strings(max_probs, sizes, remove_repetitions=True, return_offsets=True)
        return strings, offsets
