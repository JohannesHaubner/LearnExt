from torch_extension.networks import *

from torch_extension.loading import ModelLoader


model = ModelLoader("torch_extension/models/yankee")

print(model)
print(model[0].x_mean)
print(model[0].x_std)


from torch_extension.models.yankee.load_yankee import *

model_0 = load_yankee()
model_0.float()

print(model_0[1].layers[0].weight.data)
print(model[1].layers[0].weight.data)

print(torch.equal(model_0[1].layers[0].weight.data, model[1].layers[0].weight.data))
print(torch.equal(model_0[1].layers[-1].weight.data, model[1].layers[-1].weight.data))
print(torch.equal(model_0[0].x_mean, model[0].x_mean))
print(torch.equal(model_0[0].x_std, model[0].x_std))
