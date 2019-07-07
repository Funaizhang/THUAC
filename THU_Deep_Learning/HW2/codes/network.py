class Network(object):
    def __init__(self, name):
        self.name = name
        self.layer_list = []
        self.params = []
        self.num_layers = 0

    def add(self, layer):
        self.layer_list.append(layer)
        self.num_layers += 1

    def forward(self, input, visualize=False, layer_name=None):
        output = input
        output_visualize = None
        for i in range(self.num_layers):
            output = self.layer_list[i].forward(output)
            # get the output from the layer that we wish to visualize
            if visualize and self.layer_list[i].name == layer_name:
                output_visualize = output
        if not visualize:
            return output
        else:
            return output, output_visualize

    def backward(self, grad_output):
        grad_input = grad_output
        for i in range(self.num_layers - 1, -1, -1):
            grad_input = self.layer_list[i].backward(grad_input)

    def update(self, config):
        for i in range(self.num_layers):
            if self.layer_list[i].trainable:
                self.layer_list[i].update(config)
    
    def save_weights(self, loss_name, epoch):
        for i in range(self.num_layers):
            if self.layer_list[i].trainable:
                self.layer_list[i].save_weights(self.name, loss_name, epoch)
