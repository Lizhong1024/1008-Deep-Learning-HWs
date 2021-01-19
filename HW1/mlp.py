import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        z1 = torch.matmul(self.parameters['W1'].unsqueeze(0), x.unsqueeze(-1)) + self.parameters['b1'].unsqueeze(-1)
        if self.f_function == 'relu':
            z2 = torch.relu(z1)
        elif self.f_function == 'sigmoid':
            z2 = torch.sigmoid(z1)
        else:
            z2 = z1
        
        z3 = torch.matmul(self.parameters['W2'].unsqueeze(0), z2) + self.parameters['b2'].unsqueeze(-1)
        if self.g_function == 'relu':
            y_hat = torch.relu(z3)
        elif self.g_function == 'sigmoid':
            y_hat = torch.sigmoid(z3)
        else:
            y_hat = z3

        self.cache['x'] = x
        self.cache['z1'] = z1
        self.cache['z2'] = z2
        self.cache['z3'] = z3
        return y_hat.squeeze(-1)
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        def activation_grads(function, v):
            if function == 'relu':
                dfdv = v.clone()
                dfdv[dfdv>0] = 1
                dfdv[dfdv<=0] = 0
            elif function == 'sigmoid':
                dfdv = torch.sigmoid(v) * (1-torch.sigmoid(v))
            else:
                dfdv = torch.ones(v.size())
            dfdv = torch.diag_embed(dfdv.squeeze(-1))
            return dfdv
        
        dz2dz1 = activation_grads(self.f_function, self.cache['z1'])
        dy_hatdz3 = activation_grads(self.g_function, self.cache['z3'])
        dJdy_hat = dJdy_hat.unsqueeze(-1)
        
        self.grads['dJdb2'] = torch.matmul(dJdy_hat, dy_hatdz3)
        self.grads['dJdW2'] = torch.matmul(self.grads['dJdb2'], torch.transpose(self.cache['z2'], -1, -2))
        self.grads['dJdb1'] = torch.matmul(torch.matmul(self.grads['dJdb2'], self.parameters['W2'].unsqueeze(0)), dz2dz1)
        self.grads['dJdW1'] = torch.matmul(torch.transpose(self.grads['dJdb1'],-1,-2), self.cache['x'].unsqueeze(-2))

        self.grads['dJdb2'] = torch.mean(self.grads['dJdb2'], dim=0).squeeze()
        self.grads['dJdW2'] = torch.mean(self.grads['dJdW2'], dim=0).squeeze()
        self.grads['dJdb1'] = torch.mean(self.grads['dJdb1'], dim=0).squeeze()
        self.grads['dJdW1'] = torch.mean(self.grads['dJdW1'], dim=0).squeeze()

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss = torch.mean(torch.square(y_hat - y))/2
    dJdy_hat = y_hat - y
    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    loss = torch.mean(-y*torch.log(y_hat) -(1-y)*torch.log(1-y_hat))
    dJdy_hat = (y_hat - y) / (y_hat * (y_hat - y))
    return loss, dJdy_hat











