import numpy as np
import onnx
import torch
from onnx import shape_inference
from models import MobileNetSkipAdd
import onnxruntime as ort

def convert_onnx(net, output, opset=9, simplify=False):
    assert isinstance(net, torch.nn.Module)
    #img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = np.random.randint(0, 255, size=(224,224,3), dtype=np.int32)
    img = img.astype(np.float)
    # img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img = img.cuda()
    net.eval()
    print('pytorch result:', net((img)))
    torch.onnx.export(net, img, output, input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    ort_session = ort.InferenceSession(output)
    img = img.cpu()
    img = img.numpy()
    outputs = ort_session.run(None, {'data': img})
    print('onnx result:', outputs[0])
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    #onnx.save(model, output)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(output)), output)

    
if __name__ == '__main__':
    import os
    import argparse
   #from backbones import get_model

    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('--input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    #parser.add_argument('--network', type=str, default=None, help='backbone network')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)
    print(args)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(input_file)
    if type(checkpoint) is dict:
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
    else:
        model = checkpoint
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    convert_onnx(model, args.output, simplify=args.simplify)
