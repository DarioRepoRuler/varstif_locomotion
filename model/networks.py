from torch import nn


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None, using_norm=True):
        super().__init__()
        self.using_norm = using_norm
        self.linear = nn.Linear(in_features, out_features)
        if self.using_norm:
            self.norm = nn.BatchNorm1d(out_features)
        self.activation = activation

    def forward(self, x):
        if self.using_norm:
            x = self.norm(self.linear(x))
        else:
            x = self.linear(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, hidden_features, activation=None, using_norm=True):
        super().__init__()
        self.activation = activation
        self.dense1 = Dense(hidden_features, hidden_features, activation, using_norm)
        self.dense2 = Dense(hidden_features, hidden_features, using_norm=using_norm)

    def forward(self, x):
        y = self.dense1(x)
        y = self.dense2(y)
        out = y + x
        if self.activation is not None:
            out = self.activation(out)

        return out


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 n_layers,
                 act=nn.LeakyReLU(0.2),
                 output_act=None,
                 using_norm=True):
        super().__init__()
        n_blocks = (n_layers - 1) // 2
        input_layer = Dense(in_features, hidden_features, act, using_norm)
        layers = [input_layer]
        for i in range(n_blocks):
            layers.append(DenseBlock(hidden_features, act, using_norm))
        output_layer = Dense(hidden_features, out_features, output_act, using_norm)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_features, hidden_features, n_layers, using_norm=True):
        super().__init__()
        n_blocks = n_layers // 2
        input_layer = Dense(in_features, hidden_features, nn.LeakyReLU(0.2), using_norm)
        layers = [input_layer]
        for i in range(n_blocks):
            layers.append(DenseBlock(hidden_features, nn.LeakyReLU(0.2), using_norm))

        output_layer = Dense(hidden_features, 1, using_norm=using_norm)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x
