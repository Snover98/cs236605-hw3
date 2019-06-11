import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # It's best if you also sort the chars before assigning indices, so that
    # they're in lexical order.
    # ====== YOUR CODE: ======
    unique_chars = list(set(text))
    unique_chars.sort()

    idx_to_char = {idx: character for idx, character in enumerate(unique_chars)}
    char_to_idx = {character: idx for idx, character in idx_to_char.items()}

    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # ====== YOUR CODE: ======
    remove_pattern = '[' + ''.join([re.escape(character) for character in chars_to_remove]) + ']'

    text_clean, n_removed = re.subn(remove_pattern, '', text)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # ====== YOUR CODE: ======
    char_indices = [char_to_idx[character] for character in text]

    result = torch.zeros(len(text), len(char_to_idx), dtype=torch.int8)
    result[range(len(text)), char_indices] = 1

    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # ====== YOUR CODE: ======
    result = ''.join([idx_to_char[idx.item()] for idx in (embedded_text == 1).nonzero()[:, 1]])
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO: Implement the labelled samples creation.
    # 1. Embed the given text.
    # 2. Create the samples tensor by splitting to groups of seq_len.
    #    Notice that the last char has no label, so don't use it.
    # 3. Create the labels tensor in a similar way and convert to indices.
    # Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    embedded_text = chars_to_onehot(text, char_to_idx)

    samples_split = embedded_text[:-1, :].split(seq_len)
    if samples_split[-1].shape[0] != seq_len:
        samples_split = samples_split[:-1]

    samples = torch.stack(samples_split).to(device)

    text_labels = (embedded_text == 1).nonzero()[:, 1]

    labels_split = text_labels[1:].split(seq_len)
    if labels_split[-1].shape[0] != seq_len:
        labels_split = labels_split[:-1]

    labels = torch.stack(labels_split).to(device)

    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======

    exponent = torch.exp(y/temperature)

    result = exponent / torch.sum(exponent, dim=dim)

    result[result != result] = 1.0

    if torch.sum(result) <= 0:
        result[torch.argmax(exponent)] = 1.0
        # print('y!')
        # print(y)
        # print('exponent!')
        # print(exponent)
        # print('result!')
        # print(result)
        # print('sum!')
        # print(torch.sum(exponent, dim=dim))
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO: Implement char-by-char text generation.
    # 1. Feed the start_sequence into the model.
    # 2. Sample a new char from the output distribution of the last output
    #    char. Convert output to probabilities first.
    #    See torch.multinomial() for the sampling part.
    # 3. Feed the new char into the model.
    # 4. Rinse and Repeat.
    #
    # Note that tracking tensor operations for gradient calculation is not
    # necessary for this. Best to disable tracking for speed.
    # See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        cur_seq = chars_to_onehot(start_sequence, char_to_idx).to(device)
        cur_seq = cur_seq.view(1, *cur_seq.shape)

        pred_sequence, hidden_state = model(cur_seq.to(torch.float))
        pred_sequence = torch.squeeze(pred_sequence)
        if len(pred_sequence.shape) > 1:
            pred_sequence = pred_sequence[-1]
        pred_probs = hot_softmax(pred_sequence, dim=0, temperature=T)

        for _ in range(n_chars-len(start_sequence)):
            sampled_char = idx_to_char[torch.multinomial(pred_probs, 1).item()]
            out_text += sampled_char

            prev_char = chars_to_onehot(sampled_char, char_to_idx).to(device)
            prev_char = prev_char.view(1, *prev_char.shape)

            pred_sequence, hidden_state = model(prev_char.to(torch.float), hidden_state)

            pred_probs = hot_softmax(torch.squeeze(pred_sequence), dim=0, temperature=T)
    # ========================

    return out_text


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model.
        # To implement the affine transforms you can use either nn.Linear
        # modules (recommended) or create W and b tensor pairs directly.
        # Create these modules or tensors and save them per-layer in
        # the layer_params list.
        # Important note: You must register the created parameters so
        # they are returned from our module's parameters() function.
        # Usually this happens automatically when we assign a
        # module/tensor as an attribute in our module, but now we need
        # to do it manually since we're not assigning attributes. So:
        #   - If you use nn.Linear modules, call self.add_module() on them
        #     to register each of their parameters as part of your model.
        #   - If you use tensors directly, wrap them in nn.Parameter() and
        #     then call self.register_parameter() on them. Also make
        #     sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        sequence = []
        self.GRU_layers = []

        prev_dim = in_dim
        for _ in range(n_layers):
            gru_layer = GRUBlock(prev_dim, h_dim)
            sequence.append(gru_layer)
            if dropout != 0:
                self.sequence.append(nn.Dropout(dropout))

            self.GRU_layers.append(gru_layer)

            prev_dim = h_dim

        sequence.append(nn.Linear(prev_dim, out_dim))
        self.sequence = nn.Sequential(*sequence)

        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO: Implement the model's forward pass.
        # You'll need to go layer-by-layer from bottom to top (see diagram).
        # Tip: You can use torch.stack() to combine multiple tensors into a
        # single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        layer_outputs = []

        for t in range(seq_len):
            for layer_state, layer in zip(layer_states, self.GRU_layers):
                layer.set_hidden_state(layer_state)

            layer_outputs.append(self.sequence(input[:,t,:]))
            layer_states = [layer.pop_hidden_state() for layer in self.GRU_layers]

        layer_output = torch.stack(layer_outputs, dim=1)
        hidden_state = torch.stack(layer_states, dim=1)
        # ========================
        return layer_output, hidden_state


class GRUBlock(nn.Module):
    """
    as single layer in the multilayer GRU
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.update_gate_in = nn.Linear(in_dim, out_dim, bias=False)
        self.update_gate_hidden = nn.Linear(out_dim, out_dim)
        self.update_gate_sigmoid = nn.Sigmoid()

        self.reset_gate_input = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_gate_hidden = nn.Linear(out_dim, out_dim)
        self.reset_gate_sigmoid = nn.Sigmoid()

        self.hidden_candidate_in = nn.Linear(in_dim, out_dim, bias=False)
        self.hidden_candidate_hidden = nn.Linear(out_dim, out_dim)
        self.hidden_candidate_tanh = nn.Tanh()

        self._hidden = None

    def update_gate(self, input: Tensor, hidden_state: Tensor):
        return self.update_gate_sigmoid(self.update_gate_in(input) + self.update_gate_hidden(hidden_state))

    def reset_gate(self, input: Tensor, hidden_state: Tensor):
        return self.reset_gate_sigmoid(self.reset_gate_input(input) + self.reset_gate_hidden(hidden_state))

    def hidden_candidate(self, input: Tensor, hidden_state: Tensor, reset_val):
        return self.hidden_candidate_tanh(
            self.hidden_candidate_in(input) + reset_val * self.hidden_candidate_hidden(hidden_state))

    def calc_hidden_state(self, hidden_state: Tensor, hidden_candidate: Tensor, update_val):
        return update_val * hidden_state + (1 - update_val) * hidden_candidate

    def set_hidden_state(self, hidden_state: Tensor = None):
        self._hidden = hidden_state.clone().detach_()

    def pop_hidden_state(self):
        hidden_state = self._hidden
        self._hidden = None
        return hidden_state

    def forward(self, input: Tensor):
        z = self.update_gate(input, self._hidden)
        r = self.reset_gate(input, self._hidden)
        g = self.hidden_candidate(input, self._hidden, r)

        hidden = self.calc_hidden_state(self._hidden, g, z)
        self.set_hidden_state(hidden)
        return hidden
