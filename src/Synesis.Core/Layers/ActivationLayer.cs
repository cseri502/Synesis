using System.Numerics.Tensors;
using Synesis.Core.Activations;
using Synesis.Core.Optimizers;

namespace Synesis.Core.Layers;

public sealed class ActivationLayer : ILayer
{
    private readonly IActivationFunction _activation;
    private Tensor<float> _lastInput;
    
    public ActivationLayer (IActivationFunction activation) => _activation = activation;
    
    public Tensor<float> Forward(Tensor<float> input, bool isTraining)
    {
        _lastInput = input;
        return _activation.Forward(input);
    }

    public Tensor<float> Backward(Tensor<float> gradOutput)
    {
        var derivative = _activation.Derivative(_lastInput);
        return Tensor.Multiply<float>(gradOutput.AsReadOnlyTensorSpan(), derivative.AsReadOnlyTensorSpan());
    }

    public void UpdateParameters(IOptimizer optimizer) {}
}