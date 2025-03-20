import eqnr
import torch
import math


def print_model_info(model):
    """Prints detailed information about model, such as how input shape is
    transformed to output shape and how many parameters are trained in each
    block.
    """
    print("Forward renderer")
    print("----------------\n")
    print_layer_summary(model.transform_3d, "3D Layers")
    print("\n")
    print_layer_summary(model.projection, "Projection")
    print("\n")
    print_layer_summary(model.transform_2d, "2D Layers")
    print("\n")

    print("Inverse renderer")
    print("----------------\n")
    print_layer_summary(model.inv_transform_2d, "Inverse 2D Layers")
    print("\n")
    print_layer_summary(model.inv_projection, "Inverse Projection")
    print("\n")
    print_layer_summary(model.inv_transform_3d, "Inverse 3D Layers")
    print("\n")

    print("Scene Representation:")
    print("\tShape: {}".format(model.scene_shape))
    # Size of scene representation corresponds to non zero entries of
    # spherical mask
    print("\tSize: {}\n".format(int(model.spherical_mask.mask.sum().item())))
    print("Number of parameters: {}\n".format(count_trainable_parameters(model)))
    

def print_layer_summary(model_layer, title):
    """Prints information about a model.

    Args:
        model (see get_layers_info)
        title (string): Title of model.
    """
    # Extract layers info for model
    layers_info = extract_layers_info(model_layer)
    # Print information in a nice format
    print(title)
    print("-" * len(title))
    print("{: <12} \t {: <14} \t {: <14} \t {: <10} \t {: <10}".format("name", "in_shape", "out_shape", "num_params", "feat_size"))
    print("---------------------------------------------------------------------------------------------")

    min_feat_size = 2 ** 20  # Some huge number
    for info in layers_info:
        feat_size =  math.prod(info["out_shape"])
        print("{: <12} \t {: <14} \t {: <14} \t {: <10} \t {: <10}".format(info["name"],
                                                                            str(info["in_shape"]),
                                                                            str(info["out_shape"]),
                                                                            info["num_params"],
                                                                            feat_size))
        if feat_size < min_feat_size:
            min_feat_size = feat_size
    print("---------------------------------------------------------------------------------------------")
    # Only print model info if model is not empty
    if len(layers_info):
        print("{: <12} \t {: <14} \t {: <14} \t {: <10} \t {: <10}".format("Total",
                                                                str(layers_info[0]["in_shape"]),
                                                                str(layers_info[-1]["out_shape"]),
                                                                count_trainable_parameters(model_layer),
                                                                min_feat_size))


def extract_layers_info(model_layer):
    """Returns information about input shapes, output shapes and number of
    parameters in every block of model.

    Args:
        model (torch.nn.Module): Model to analyse. This will typically be a
            submodel of models.neural_renderer.NeuralRenderer.
    """
    in_shape = model_layer.input_shape
    layers_info = []

    if isinstance(model_layer, eqnr.nn.Projection):
        out_shape = (in_shape[0] * in_shape[1], *in_shape[2:])
        layer_info = {"name": "Reshape", "in_shape": in_shape,
                      "out_shape": out_shape, "num_params": 0}
        layers_info.append(layer_info)
        in_shape = out_shape

    for layer in model_layer.forward_layers:
        if isinstance(layer, torch.nn.Conv2d):
            if layer.stride[0] == 1:
                out_shape = (layer.out_channels, *in_shape[1:])
            elif layer.stride[0] == 2:
                out_shape = (layer.out_channels, in_shape[1] // 2, in_shape[2] // 2)
            name = "Conv2D"
        elif isinstance(layer, torch.nn.ConvTranspose2d):
            if layer.stride[0] == 1:
                out_shape = (layer.out_channels, *in_shape[1:])
            elif layer.stride[0] == 2:
                out_shape = (layer.out_channels, in_shape[1] * 2, in_shape[2] * 2)
            name = "ConvTr2D"
        elif isinstance(layer, eqnr.nn.ResBlock2d):
            out_shape = in_shape
            name = "ResBlock2D"
        elif isinstance(layer, torch.nn.Conv3d):
            if layer.stride[0] == 1:
                out_shape = (layer.out_channels, *in_shape[1:])
            elif layer.stride[0] == 2:
                out_shape = (layer.out_channels, in_shape[1] // 2, in_shape[2] // 2, in_shape[3] // 2)
            name = "Conv3D"
        elif isinstance(layer, torch.nn.ConvTranspose3d):
            if layer.stride[0] == 1:
                out_shape = (layer.out_channels, *in_shape[1:])
            elif layer.stride[0] == 2:
                out_shape = (layer.out_channels, in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2)
            name = "ConvTr3D"
        elif isinstance(layer, eqnr.nn.ResBlock3d):
            out_shape = in_shape
            name = "ResBlock3D"
        else:
            # If layer is just an activation layer, skip
            continue

        num_params = count_trainable_parameters(layer)
        layer_info = {"name": name, "in_shape": in_shape,
                      "out_shape": out_shape, "num_params": num_params}
        layers_info.append(layer_info)

        in_shape = out_shape

    if isinstance(model_layer, eqnr.nn.InverseProjection):
        layer_info = {"name": "Reshape", "in_shape": in_shape,
                      "out_shape": model_layer.output_shape, "num_params": 0}
        layers_info.append(layer_info)

    return layers_info

def count_trainable_parameters(model):
    """Returns number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
