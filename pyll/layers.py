import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import LSTMCell
from torch.nn import Parameter


class LSTMCell(nn.Module):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** (batch, input_size): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** (batch. hidden_size): tensor containing the initial cell state
          for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> cx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True, a_i=torch.tanh, a_h=torch.tanh, a_ig=torch.sigmoid, a_og=torch.sigmoid, a_fg=torch.sigmoid):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # activations
        self.a_i, self.a_h, self.a_ig, self.a_og, self.a_fg = a_i, a_h, a_ig, a_og, a_fg
        # input weights
        self.w_fg, self.w_ig, self.w_og, self.w_ci = self.get_parameters((hidden_size, input_size))
        # recurrent weights
        self.r_fg, self.r_ig, self.r_og, self.r_ci = self.get_parameters((hidden_size, hidden_size))
        # cell state
        # self.ct = Variable(torch.zeros(hidden_size)).cuda()
        # bias
        if bias:
            self.b_fg, self.b_ig, self.b_og, self.b_ci = self.get_parameters((hidden_size,))
        else:
            self.bi_fg, self.bi_ig, self.bi_og, self.bi_ci = [Parameter(torch.zeros(hidden_size), requires_grad=False) for _ in range(4)]

        self.reset_parameters()

    def cuda(self, device=None):
        super(self, device)
        self.ct.cuda(device=device)
        self.ht.cuda(device=device)

    def cpu(self):
        super(self)
        self.ct.cpu()
        self.ht.cpu()

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def get_parameters(self, shape):
        return [Parameter(torch.Tensor(*shape)) for _ in range(4)]

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, xt, cx, hx):
        ft = self.a_fg(xt @ self.w_fg.t() + hx @ self.r_fg.t() + self.b_fg)
        it = self.a_ig(xt @ self.w_ig.t() + hx @ self.r_ig.t() + self.b_ig)
        ot = self.a_og(xt @ self.w_og.t() + hx @ self.r_og.t() + self.b_og)
        ct = ft * cx + it * self.a_i(xt @ self.w_ci.t() + hx @ self.r_ci.t() + self.b_ci)
        ht = ot * self.a_h(ct)
        return ht, ct


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())

class OwnConvLSTMCell(nn.Module):
    def __init__(self):
        z = tf.layers.conv2d(e, 128, 1, activation=tf.nn.tanh)

        if t == 0:
            c_old = tf.zeros_like(z)
            h_old = tf.zeros_like(z)

        gates = tf.layers.conv2d(h_old, 128 * 3, 1, activation=tf.sigmoid)
        i, f, o = tf.split(gates, 3, axis=3)
        c = c_old * f + z * i
        h = tf.nn.tanh(c) * o
        h_old, c_old = h, c

class OwnLSTMCell(nn.Module):
    def __init__(self):
        if t == 0:
            c_old = tf.zeros_like(z)
            h_old = tf.zeros_like(z)

        gates = tf.layers.dense(h_old, 128 * 3, activation=tf.sigmoid)
        i, f, o = tf.split(gates, 1, axis=1)
        c = c_old * f + z * i
        h = tf.nn.tanh(c) * o
        h_old, c_old = h, c