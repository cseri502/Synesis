using System.Numerics.Tensors;

namespace Synesis.Core.Activations;

public sealed class Sigmoid : IActivationFunction
{
    public Tensor<float> Forward(Tensor<float> input) => Tensor.Sigmoid(input.AsReadOnlyTensorSpan());

    public Tensor<float> Derivative(Tensor<float> input)
    {
        var s = Tensor.Sigmoid(input.AsReadOnlyTensorSpan());
        var diff = Tensor.Subtract(1f, s.AsReadOnlyTensorSpan());
        
        return Tensor.Multiply(s.AsReadOnlyTensorSpan(), diff.AsReadOnlyTensorSpan());
    }
}